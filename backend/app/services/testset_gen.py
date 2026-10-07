"""Testset generation pipeline based on ragas TestsetGenerator.

Design reference: design doc — the testset technical notes. That document is
kept out of the repository (see .gitignore), so it is referred to by name and
section, as elsewhere in this module.

Pipeline: language detection → seed selection → knowledge graph build/reuse
→ persona generation → per-type question generation → fallback top-up →
dedup & trim → persist items.

Heavy ragas interactions are isolated in module-level functions so tests can
monkeypatch them.
"""

import asyncio
import hashlib
import json
import logging
import math
import random
import re
import secrets
import shutil
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field
from ragas.prompt import PydanticPrompt

from app.core.config import get_settings
from app.core.host import host_gateway
from app.db.models import ModelConfig, Testset, TestsetItem
from app.db.session import get_session_factory

logger = logging.getLogger(__name__)

MIN_DOC_CHARS = 1500
MAX_DOC_CHARS = 8000
# Output-budget escalation ceiling for truncated LLM responses (see _Metered).
_MAX_OUTPUT_CEILING = 32768
SEED_RATIO = 1.3
MIN_SEEDS = 15
# Three. This was two, and the measurement behind two no longer applies: it was
# taken when every round paid for a knowledge-graph expansion, and a third round
# never repaid it (net gains of 0, 0 and -1 across the three types that reached
# one). Rounds now rotate the node pool, which costs no graph work at all — only
# the round after a barren one merges documents — so a third round costs one
# more generation pass and nothing else.
#
# Measured against that: in one run the second round came back gained=0, which
# is exactly the signal that asks for the merge, and the loop ended before it
# could act. In another, each of two rounds gained 1 and the run finished one
# sample short.
# Upper bound on the per-type top-up rounds, and the divisor for the slice of
# the bar each round advances (see round_slice in _pipeline).
#
# Five. The case for six did not survive being checked. Six was never reached in
# any run, and the caps before it — two, then three — produced 22 runs that
# finished short of target, every one of them with the round budget exhausted
# rather than the material, so the cap has to exceed three. Measured under the
# six-round cap, over the nine runs that followed it (27 run-by-type pairs): the
# most any type needed was three rounds, and none finished short. Five keeps
# headroom over the observed maximum without paying for rounds never needed.
#
# Small sample: those nine runs span two gate versions, and the distribution
# moves when the gate does.
FALLBACK_ROUNDS = 5
# How many suspects the screen/verify pass probes at once. Suspects are one
# call each; the screen is one call per batch.
VERIFY_CONCURRENCY = 8
# The top-up is sized from the yield this corpus has actually shown so far, not
# from a fixed multiplier: how much survives the quality gate and dedup varies
# with the material, so a constant either over-generates on a clean corpus or
# still falls short on a messy one. Clamped both ways: never below the gap
# itself, never more than MAX_TOPUP_RATIO × the gap.
MIN_YIELD_RATE = 0.2
# Personas are the viewpoints questions get asked from. Setting this is not
# enough on its own: ragas slices the list we hand the generator down to the
# first `num_personas` when it builds scenarios
# (`scenario.generate_scenarios(..., persona_list=self.persona_list[:num_personas])`
# in testset/synthesizers/generate.py), so `_generate_chunk` passes the same
# count to `gen.generate()` — without that the constant would only change how
# many personas get generated, not how many are used.
#
# Raising it is not free: the multi-hop abstract synthesizer sends every theme x
# persona pairing to the LLM, so its output grows with the count. Worth
# measuring before changing — two runs credited to 5 personas each had a call
# run past the output ceiling and burn 95-145 s, though those runs predate the
# num_personas pass-through and have not been re-measured.
NUM_PERSONAS = 3
RANDOM_SEED = 42
# How many extra clusters to ask the graph for so a rotating window has room to
# move. ragas picks clusters in a deterministic order, so asking for the same n
# every call returns the same n — the extra ones are what a fallback round moves
# onto. See _rotate_node_pool.
_CLUSTER_WINDOW_FACTOR = 3
# Per-request HTTP timeout for the LLM and embedding clients, in seconds.
#
# A product connects to whatever LLM the customer configured, so this has to
# hold for a model slower than ours and still cut a stalled call. Ten seconds
# does both: the distribution is bimodal — with DeepSeek, p50 is 1.0 s and p95
# is 2.0 s, then nothing until a tail at 8 s and beyond (>10 s is 1.48% of
# 1,953 calls, >5 s is 1.64%, so the exact threshold barely changes how many
# are cut) — and 10 s sits above anything a reasonably fast model produces per
# call while still catching the tail, which is what costs the time. The tail is
# what matters because ragas dispatches in batches: one call that sits for the
# timeout holds up its whole batch, so the batch's wall time is the slowest
# member's total. A call cut at 10 s and retried costs about 11 s; at 20 s it
# cost 22 s, and three of those were 66 s of a 209 s run.
#
# It is a fixed number rather than an escalating one on purpose. A ladder that
# starts short (2 s, from DeepSeek's p95) would cut every call twice on a model
# whose p95 is 8 s, and the retry loop it needs means taking over max_retries
# and the client's error handling — a lot of the request path to buy 5 s per
# stalled call. Short timeout + retry is the right shape; 10 s is where short
# stops being model-specific.
#
# The target RAG system is the opposite case — retrieval, reranking and
# generation can genuinely take a minute — and keeps its own generous,
# user-configurable timeout (RagSystemConfig.timeout). Do not fold the two
# together.
LLM_TIMEOUT = 10
QUESTION_TYPES = ("single", "multi_specific", "multi_abstract")

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "assets" / "testset_prompts"

_running: set[int] = set()


def is_running(testset_id: int) -> bool:
    """True while run_generation is working on this testset in this process.

    Callers that are about to mark a testset 'generating' check this first:
    run_generation returns immediately when the id is already in flight, so
    flipping the status anyway leaves the row stuck at generating with progress
    0 and nothing running — visible only as a spinner that never advances."""
    return testset_id in _running


# ---------------------------------------------------------------------------
# Paths & pure helpers
# ---------------------------------------------------------------------------


def testset_dir_for(name: str) -> Path:
    return get_settings().data_dir / "testset" / name


def _archive_drops(testset_name: str, type_key: str, round_no: int | None,
                   samples: list[dict], problems: dict[int, str],
                   drop: set[int]) -> None:
    """Append the rejected pairs, with the reason the verdict gave.

    The reason is the model's, not a person's, so it says why the check thought
    so rather than what was actually wrong — good enough to sample for false
    kills, which is what it is for.
    """
    path = dropped_path_for(testset_name)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for i in sorted(drop):
                s = samples[i - 1]
                fh.write(json.dumps({
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "type": type_key,
                    "round": round_no,
                    "question": s.get("user_input", ""),
                    "reference": s.get("reference", ""),
                    "contexts": s.get("reference_contexts") or [],
                    "reason": problems.get(i, ""),
                }, ensure_ascii=False) + "\n")
    except OSError as e:
        # An archive that cannot be written must not fail a run.
        logger.warning("Could not archive dropped pairs (%s)", e)


def dropped_path_for(testset_name: str) -> Path:
    """Where a run's rejected pairs go.

    The gate is the only thing standing between the generator and the customer,
    and until now what it dropped simply ceased to exist: the log quoted two
    examples and the items themselves were gone. That leaves no way to audit a
    drop (single-hop was dropping 69% of a batch on the last measured run, and
    at least one of the quoted drops named its company and looked like a false
    kill) and no pool of known defects to measure a change against — building
    one by hand is what took most of a day.
    """
    return get_settings().data_dir / "testset" / testset_name / "dropped.jsonl"


def kg_path_for(corpus_name: str) -> Path:
    return get_settings().data_dir / "corpus" / corpus_name / "kg.json"


def seeds_path_for(corpus_name: str) -> Path:
    return get_settings().data_dir / "corpus" / corpus_name / "seeds.json"


def kg_fingerprint_path_for(corpus_name: str) -> Path:
    return get_settings().data_dir / "corpus" / corpus_name / "kg_fingerprint.json"


def personas_path_for(corpus_name: str) -> Path:
    return get_settings().data_dir / "corpus" / corpus_name / "personas.json"


def _kg_fingerprint(llm_cfg: ModelConfig, emb_cfg: ModelConfig) -> dict:
    """KG reuse is only valid when built with the same models/chunk params.

    Seed-list growth is handled by incremental expansion, so seeds are NOT
    part of the fingerprint (per design doc §3.3 + reuse table). run_seed is
    stored alongside (not compared): reusing the KG reproduces the same
    scenario sampling; rebuilding gets a fresh random seed.
    """
    return {
        "llm_model": llm_cfg.model,
        "llm_base_url": llm_cfg.base_url,
        "emb_model": emb_cfg.model,
        "emb_base_url": emb_cfg.base_url,
        "max_doc_chars": MAX_DOC_CHARS,
    }


def corpus_md_files(corpus_name: str) -> list[Path]:
    corpus_dir = get_settings().data_dir / "corpus" / corpus_name
    if not corpus_dir.is_dir():
        return []
    return sorted(p for p in corpus_dir.rglob("*.md") if p.is_file())


def estimate_seed_count(n_total: int, doc_count: int, ratio: float = SEED_RATIO) -> int:
    """m = clamp(ceil(ratio * N), 15, doc_count)."""
    return max(1, min(max(math.ceil(ratio * n_total), MIN_SEEDS), doc_count))


_CJK = re.compile(r"[一-鿿]")


def cjk_ratio(text: str) -> float:
    return len(_CJK.findall(text)) / len(text) if text else 0.0


def language_matches(text: str, lang: str) -> bool:
    """Guardrail: QA language must match the corpus language.

    Models occasionally drift language on abstract questions; such samples are
    dropped and topped up by the fallback loop.
    """
    ratio = cjk_ratio(text)
    return ratio >= 0.2 if lang == "zh" else ratio <= 0.1


# ---------------------------------------------------------------------------
# QA text hygiene
# ---------------------------------------------------------------------------

_ZERO_WIDTH_RE = re.compile(r"[​‌‍﻿]")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Spaces between two CJK characters never carry meaning; models sometimes
# imitate the PDF-extraction style of the source ("香 港 联 合 ...").
_CJK_SPACE_RE = re.compile(r"(?<=[一-鿿])[ 　]+(?=[一-鿿])")
_Q_PREFIX_RE = re.compile(r"^(?:问题|question|q)\s*[:：]\s*", re.IGNORECASE)
_QUOTE_PAIRS = (("“", "”"), ("‘", "’"), ('"', '"'), ("'", "'"), ("「", "」"), ("『", "』"))


def normalize_qa_text(text: str, *, multiline: bool = False) -> str:
    """Clean up LLM output artifacts in a question/answer.

    Conservative: only removes things that never carry meaning in QA text —
    zero-width/control chars, spaces between CJK characters, "Question:"-style
    prefixes, and quotes wrapping the whole text. reference_contexts are
    source excerpts and must NOT be normalized (faithfulness to the corpus).
    """
    if not text:
        return ""
    text = _ZERO_WIDTH_RE.sub("", text)
    text = _CONTROL_CHAR_RE.sub("", text)
    text = text.replace("　", " ")  # full-width space
    text = _CJK_SPACE_RE.sub("", text)
    if multiline:
        lines = [ln.rstrip() for ln in text.split("\n")]
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
    else:
        text = re.sub(r"\s+", " ", text)
    text = _Q_PREFIX_RE.sub("", text.strip())
    for open_q, close_q in _QUOTE_PAIRS:
        if len(text) > 2 and text.startswith(open_q) and text.endswith(close_q):
            inner = text[1:-1]
            # Only strip when the pair wraps the whole text (no inner quotes).
            if open_q not in inner and close_q not in inner:
                text = inner.strip()
            break
    return text


_HOP_PREFIX_RE = re.compile(r"^<\d+-hop>\s*")


def strip_hop_markers(contexts: list[str]) -> list[str]:
    """Remove the '<N-hop>' prefixes ragas multi-hop synthesizers prepend.

    The marker is prompt scaffolding (it tells the LLM the hop structure);
    it is not corpus content, so stripping it makes stored reference_contexts
    more faithful to the source documents.
    """
    return [_HOP_PREFIX_RE.sub("", c) for c in contexts]


_META_REF_RE = re.compile(
    r"本文|该文|文中|上文|下文|这篇|这段(?:文字|内容|材料|文本|描述|话)"
    r"|所(?:提供|给)的?(?:材料|文本|上下文|文档|信息|段落|内容)"
    # "根据提供的信息" without the 所 prefix — but only when 根据 leads straight
    # into it, so that "根据公司提供的公告…" (a specific reference) is kept.
    r"|根据(?:所)?提供的?(?:材料|文本|上下文|文档|信息|段落|内容)"
    r"|the (?:given|provided|above) (?:text|passage|document|context|article)"
    r"|according to the (?:text|passage|context|document)"
    r"|based on the (?:provided|given) (?:information|material|text|context)",
    re.IGNORECASE,
)


def is_standalone_question(question: str) -> bool:
    """A good RAG-eval question is self-contained; one pointing at "this text"
    is useless without the source and gets dropped + topped up."""
    return not _META_REF_RE.search(question)


# Fragment references ("江阴银行", "130万元", "公司2017年年度股东大会") break
# AnswerCorrectness: the claim-level F1 needs the reference to entail the
# answer's claims, and a fragment entails nothing — a correct answer that
# literally contains the reference still scores ~0.2.
MIN_REFERENCE_CHARS = 10
_SHORT_REFERENCE_CHARS = 30
_SENTENCE_END = "。！？.!?"


def reference_is_complete(reference: str) -> bool:
    """A usable reference is a sentence, not a bare fragment.

    Length alone cannot separate "一种可转换为股票的债券。" (complete) from
    "公司2017年年度股东大会" (fragment) — both are ~13 chars — so short
    references must also end with sentence punctuation. Long ones are trusted.
    """
    ref = (reference or "").strip()
    if len(ref) < MIN_REFERENCE_CHARS:
        return False
    return len(ref) >= _SHORT_REFERENCE_CHARS or ref[-1] in _SENTENCE_END


# A reference that reports the material does not contain what was asked means
# the pair asks about something the corpus never says — unanswerable by any RAG
# system, so it measures nothing and only invites hallucination. Measured on the
# 205 stored items, this catches 2 (1.0%) and both are the same recurring
# generator defect (a question comparing two meetings' attendance when only one
# of the two announcements carries an attendance section); no item in the corpus
# pairs a genuine "was X disclosed?" question with a "not disclosed" answer, so
# the pattern has no false kill on the data to hand. It stays a bilingual
# backstop in the same spirit as _META_REF_RE, not a general classifier: a
# legitimate yes/no about disclosure would trip it, and the trade is the one the
# gate already accepts (a wrongly dropped pair costs one fallback regeneration).
_REFERENCE_GAP_RE = re.compile(
    r"未(?:在|予|被|能)?(?:提供|披露|提及|说明|列出|给出|包含|记载|明确|找到)"
    r"|没(?:有)?(?:提供|披露|提及|说明|列出|给出)"
    r"|无法(?:确定|回答|判断|得出)"
    r"|not (?:disclosed|provided|mentioned|stated|specified|given|available|found)"
    r"|no (?:information|mention|record)"
    r"|does not (?:disclose|provide|mention|state|specify|contain|include)",
    re.IGNORECASE,
)


def reference_admits_no_answer(reference: str) -> bool:
    """The reference says the source does not cover what the question asked."""
    return bool(_REFERENCE_GAP_RE.search(reference or ""))




# Ragas' own Faithfulness is the gate's factual check: measured on a real
# testset it flagged both defective references (0.00 and 0.40) that a hand-written
# one-shot judge passed. Using the same metric we score with keeps the two
# consistent. The alignment check stays a single call (ragas has no metric for
# "does the reference answer this question").
MIN_REFERENCE_FAITHFULNESS = 0.9
# How many rejected pairs to quote in the log (per gate call), so the failure
# mode is visible instead of just a count.
REJECTED_SAMPLE_COUNT = 2
# Note: a second, independent grading was tried here and removed. Measured, it
# helps only the ~15% of pairs that are unstable — and those are unstable
# because the pair itself is ill-defined (facts present, framing wrong), which
# no amount of re-grading resolves. The other 85% grade identically, so the
# second pass was pure cost. Prevention belongs in the generation prompt.


async def _run_faithfulness(llm, samples: list[dict]) -> list[float]:
    """One ragas Faithfulness pass; entries that fail to parse come back NaN."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset(samples=[
        SingleTurnSample(
            user_input=s.get("user_input", ""),
            response=s.get("reference", ""),
            retrieved_contexts=s.get("reference_contexts") or [],
            reference=s.get("reference", ""),
        )
        for s in samples
    ])
    try:
        result = await asyncio.to_thread(
            evaluate, dataset=dataset, metrics=[Faithfulness()], llm=llm,
            # ragas' max_retries counts attempts (tenacity's stop_after_attempt),
            # not retries, so 1 means "no retry". Its retry catches every
            # exception, which makes it a second network retry on top of the
            # client's — and nesting the two multiplied a doomed call.
            run_config=RunConfig(max_workers=8, timeout=180, max_retries=1),
            show_progress=False,
        )
        return list(result.to_pandas()["faithfulness"])
    except Exception as e:
        # Raised, not turned into NaN. NaN has one meaning here — "scored, but
        # the metric's own output could not be parsed" — and _faithfulness_scores
        # retries exactly those. Folding a failed call into the same value made a
        # dead endpoint look like a parse problem and bought each doomed request
        # two more passes. _filter_faithful catches this: the gate must never
        # break generation.
        logger.warning("Faithfulness scoring failed (%s)", e)
        raise


async def _faithfulness_scores(llm, samples: list[dict], attempts: int = 3) -> list[float]:
    """Faithfulness per reference, retrying the ones that fail to parse.

    ragas' statement-extraction prompt occasionally comes back in the wrong shape
    and the metric yields NaN. Measured: such an item scored 1.00 on every retry,
    so the failure is transient and worth another pass rather than surfacing as an
    unverifiable item."""
    scores = await _run_faithfulness(llm, samples)
    for _ in range(attempts - 1):
        retry_idx = [i for i, v in enumerate(scores) if v != v]
        if not retry_idx:
            break
        logger.info("Faithfulness parse failed for %d sample(s); retrying", len(retry_idx))
        retry = await _run_faithfulness(llm, [samples[i] for i in retry_idx])
        for i, v in zip(retry_idx, retry):
            if v == v:
                scores[i] = v
    return scores


_DEDUP_INSTRUCTION = """You check a set of questions generated for one benchmark testset.

Two questions CONFLICT only when they ask for the SAME fact: the same entity AND the same attribute of it. The attribute decides, not the phrasing.

  {c_a}
  {c_b}
Same entity, same attribute ({c_attr}) → conflict; one of them must go.

  {d_a}
  {d_b}
Same entity, DIFFERENT attributes → NOT a conflict, keep both.

  {e_a}
  {e_b}
Different entities → NOT a conflict, keep both.

Naming the entity or the file more fully in one wording does not make the attribute different:
{w_a} and {w_b} ask the same fact.

Report the 1-based indices of the questions to DROP, one per conflicting group. Report an
empty list when nothing conflicts."""

_DEDUP_EXAMPLES = {
    "en": {
        "c_a": '"How many outpatients did the East Hospital see in 2020?"',
        "c_b": '"How many outpatients did the East Hospital see in 2020, and how many were admitted that year?"',
        "c_attr": "the outpatient count",
        "d_a": '"How many outpatients did the East Hospital see in 2020?"',
        "d_b": '"How many patients were admitted to the East Hospital in 2020?"',
        "e_a": '"How many outpatients did the East Hospital see in 2020?"',
        "e_b": '"How many outpatients did the West Hospital see in 2020?"',
        "w_a": '"Where is device C on 2020-06-01?"',
        "w_b": '"According to North School\'s 2020 first-term report, where is device C on 2020-06-01?"',
    },
    "zh": {
        "c_a": "「东院2020年的门诊量是多少？」",
        "c_b": "「东院2020年的门诊量是多少，以及当年的住院人数是多少？」",
        "c_attr": "门诊量",
        "d_a": "「东院2020年的门诊量是多少？」",
        "d_b": "「东院2020年的住院人数是多少？」",
        "e_a": "「东院2020年的门诊量是多少？」",
        "e_b": "「西院2020年的门诊量是多少？」",
        "w_a": "「丙设备在2020年6月1日位于哪里？」",
        "w_b": "「根据北校2020年第一学期报告，丙设备在2020年6月1日位于哪里？」",
    },
}

_SEMANTIC_DEDUP_PROMPT_EN = _DEDUP_INSTRUCTION.format(**_DEDUP_EXAMPLES["en"])
_SEMANTIC_DEDUP_PROMPT_ZH = _DEDUP_INSTRUCTION.format(**_DEDUP_EXAMPLES["zh"])


class _SemanticDedupInput(BaseModel):
    questions: list[str] = Field(description="The generated questions, in order")


class _SemanticDedupOutput(BaseModel):
    drop: list[int] = Field(default_factory=list,
                            description="1-based indices of questions to drop")
    reason: str = Field(default="", description="Why they duplicate each other")


class _SemanticDedupBase(PydanticPrompt[_SemanticDedupInput, _SemanticDedupOutput]):
    """Catches duplicates the embedding pass misses.

    Measured: two questions asking the same fact scored only 0.835 cosine similarity
    because one was far more verbose, so the 0.95 threshold let both through.

    The test used to be "one and the same fact answers both questions", which a
    pair that only partly overlaps never satisfies — so a question that asked
    everything a second one asked *and more* passed both. Two shipped items did
    exactly that ("…2018年度股东大会的股权登记日是哪一天，以及该股权登记日与…有何
    关联？" next to "…2018年度股东大会的股权登记日是哪一天，以及现场会议召开的…").
    The test is now the shared fact itself, same entity and same attribute, which
    catches partial overlap and still keeps questions about different attributes
    or different entities.

    Measured over the shipped items with examples from nowhere: the four known
    pairs are each resolved, and the version that added "one fact per question"
    on top of this was dropped — it started discarding questions that ask
    different attributes of the same meeting.
    """
    input_model = _SemanticDedupInput
    output_model = _SemanticDedupOutput


class _SemanticDedupPromptEn(_SemanticDedupBase):
    instruction = _SEMANTIC_DEDUP_PROMPT_EN


class _SemanticDedupPromptZh(_SemanticDedupBase):
    instruction = _SEMANTIC_DEDUP_PROMPT_ZH


def _exact_duplicates(samples: list[dict]) -> tuple[list[dict], list[str]]:
    """Drop questions whose text is already in the batch, character for
    character. Keeps the first occurrence; returns it and what it dropped.

    In front of the reviewer because there is nothing here to judge: two
    identical strings are the same question in any corpus and any language, so
    this costs one set lookup and cannot false-kill. That matters — the reviewer
    was handed a byte-equal pair and let it through, and the pair shipped
    (#21 and #23 of one run). Questions reaching here are already run through
    normalize_qa_text, so equal text means equal question, not equal spacing.

    Empty questions are left alone: sample_quality_ok rejects those later, and
    dropping them here would misattribute the reason.
    """
    seen: set[str] = set()
    kept: list[dict] = []
    dropped: list[str] = []
    for s in samples:
        text = s.get("user_input", "")
        if text and text in seen:
            dropped.append(text)
            continue
        if text:
            seen.add(text)
        kept.append(s)
    return kept, dropped


def _semantic_dedup_prompt(language: str) -> type[_SemanticDedupBase]:
    return _SemanticDedupPromptZh if language == "zh" else _SemanticDedupPromptEn


async def _semantic_dedup(llm, samples: list[dict], testset_id: int | None,
                          type_key: str, language: str) -> list[dict]:
    """One call per batch: drop questions that ask what another already asks."""
    if len(samples) < 2:
        return samples
    try:
        verdict = await _semantic_dedup_prompt(language)().generate(
            llm=llm,
            data=_SemanticDedupInput(questions=[s.get("user_input", "") for s in samples]),
        )
    except Exception as e:
        # Never let this break generation.
        logger.warning("Semantic dedup failed (%s); keeping all samples", e)
        return samples
    reported = list(verdict.drop)
    drop = {i for i in reported if 1 <= i <= len(samples)}
    ignored = sorted(i for i in reported if not 1 <= i <= len(samples))
    # The verdict as returned, before the range filter. An index pointing
    # outside the batch is not a drop, but discarding it silently means a
    # duplicate the reviewer did find can vanish with nothing to show it
    # happened — which is how a byte-equal pair shipped (#21 and #23 of one run).
    logger.info("Dedup [%s] reviewed=%d reported=%s dropped=%s",
                type_key, len(samples), sorted(reported), sorted(drop))
    if ignored:
        logger.warning("Dedup [%s] reported indices outside 1..%d: %s",
                       type_key, len(samples), ignored)
    if drop:
        # Server log only. The customer reads one merged check count per round;
        # which mechanism took a pair, and what it was, is ours.
        logger.info("Dedup [%s] dropped=%d: %s", type_key, len(drop),
                    " | ".join(_excerpt(samples[i - 1].get("user_input"))
                               for i in sorted(drop)[:REJECTED_SAMPLE_COUNT]))
    return [s for i, s in enumerate(samples, start=1) if i not in drop]


def _excerpt(text: str | None, limit: int = 60) -> str:
    """Trim to `limit` characters, marking the cut.

    Cutting at a fixed width mid-sentence left the rejection examples reading
    as broken text — `参「…本次限售股份上市流通后股数（股）」` is half a
    clause — so a truncated one says so.
    """
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + "…"


# --- screening and verification: the two steps that replace one per-item judgement
#
# Measured over 148 shipped items against a hand reading of the same items: 10 of
# the 15 real defects caught, against 2 for the single judgement it replaces, at
# the same false-kill rate (6 of 133 against 7 of 133), for 76-86 LLM calls
# against 148. Three runs agreed on all ten.
#
# The split is the whole of it. Screening alone fires on 40-50% of a batch and
# still misses some; verifying alone *is* the per-item judgement, which cannot see
# that fifteen neighbours name a company and one does not. What works is having
# the screen raise the suspicion and having the check keep the batch and that
# suspicion in hand — measured with the batch dropped from the check, it
# confirmed 0 of 64 suspects; with the batch and the reason, 16.
_SCREEN_PROMPT_EN = """You are checking a set of questions generated for one benchmark testset. Read them together.

List the ones you SUSPECT are defective. Be generous: a second pass checks every one of them properly against its own answer and material, so a false alarm costs a little and a miss ships a bad question. Look for:

  - a question that does not stand on its own: the organisation, person, place, event or file it asks about is not named in the question itself — either a referring expression with no referent ("the company", "this meeting"), or the subject left out altogether, so nothing says whose figures, whose meeting or whose shareholders are meant;
  - a question that is not a well-formed benchmark question: no determinate answer, not a question at all, a broken sentence, or casual chat rather than a query someone would type into a search system;
  - a question that asks about the same fact as another question in the set.

Report the 1-based indices you suspect, one short reason each. Empty list if none."""

_SCREEN_PROMPT_ZH = """你在检查一整套为某个基准测试生成的问题。把它们放在一起读。

把**你怀疑有问题**的列出来。宁可多报：第二轮会对每一条单独取它自己的答案和素材去核实，所以误报的代价很小，而漏掉的会直接交付给客户。找这几类：

  - **不能独立成立的**：它问的机构、人物、地点、事件或文件没有在问题里点名——要么是指代没有所指（「该公司」「本次会议」），要么是主体整个缺席，看不出是谁的数字、谁的会议、谁的股东；
  - **不像一条基准问题的**：没有唯一确定的答案、根本不是问句、句子残缺，或者像随口的聊天而不是用户会打进检索系统的一句话；
  - **和同一批里另一条问的是同一个事实**的。

报出你怀疑的编号（从 1 开始），每条一句简短理由。没有就报空列表。"""

_VERIFY_PROMPT_EN = """A first pass looked at a whole set of benchmark questions and suspected ONE of them. You are given that question, its reference answer, its source material, the other questions in the same set, and what the first pass suspected. Decide whether the suspicion is right.

Keep the set in view: a question stands or falls with what the others look like. If fifteen of them name an organisation and this one does not, that is the defect, even though nothing in this question's own wording looks broken.

The pair is defective when:
  - the question does not name the organisation, person, place, event or file it asks about, so a search system that never saw this document could not answer it;
  - the question has no single determinate answer, is not a question at all, is a broken sentence, or reads like casual chat rather than a query someone would type;
  - the reference answer is not supported by the material, does not answer the question, or mixes up entities, years or events.

ok=false when the suspicion holds. ok=true when this pair is in fact fine."""

_VERIFY_PROMPT_ZH = """第一遍读过一整套基准问题，怀疑其中**某一条**有问题。下面给你那条问题、它的参考答案、它的素材、同一批的其他问题，以及第一遍怀疑的是什么。你判断这个怀疑对不对。

**把整批放在眼前**：一条问题成不成立，要看其他那些长什么样。如果十五条都点了名、就这一条没有，那就是缺陷——哪怕这条自己的措辞看不出任何毛病。

下面任一条成立，这条就有问题：
  - 问题没有点名它问的机构、人物、地点、事件或文件，没看过这份文档的检索系统答不出来；
  - 问题没有唯一确定的答案、根本不是问句、句子残缺，或者像随口的聊天而不是用户会打进检索系统的一句话；
  - 参考答案不被素材支持、没有回答所问，或者把主体、年份、事件弄混了。

怀疑成立就判 ok=false；这条其实没问题就判 ok=true。"""


class _ScreenInput(BaseModel):
    questions: list[str] = Field(description="The questions, in order")


class _ScreenOutput(BaseModel):
    suspect: list[int] = Field(default_factory=list,
                               description="1-based indices of the questions you suspect")
    reason: list[str] = Field(default_factory=list, description="One short reason each")


class _VerifyInput(BaseModel):
    question: str
    reference: str
    material: str
    other_questions: list[str] = Field(description="The rest of the set, in order")
    suspicion: str = Field(description="What the first pass suspected about this question")


class _VerifyOutput(BaseModel):
    ok: bool = Field(description="False when the pair is defective")
    problem: str = Field(default="", description="What is wrong, when ok is false")


class _ScreenBase(PydanticPrompt[_ScreenInput, _ScreenOutput]):
    input_model = _ScreenInput
    output_model = _ScreenOutput


class _ScreenPromptEn(_ScreenBase):
    instruction = _SCREEN_PROMPT_EN


class _ScreenPromptZh(_ScreenBase):
    instruction = _SCREEN_PROMPT_ZH


class _VerifyBase(PydanticPrompt[_VerifyInput, _VerifyOutput]):
    input_model = _VerifyInput
    output_model = _VerifyOutput


class _VerifyPromptEn(_VerifyBase):
    instruction = _VERIFY_PROMPT_EN


class _VerifyPromptZh(_VerifyBase):
    instruction = _VERIFY_PROMPT_ZH


def _screen_prompt(language: str) -> type[_ScreenBase]:
    return _ScreenPromptZh if language == "zh" else _ScreenPromptEn


def _verify_prompt(language: str) -> type[_VerifyBase]:
    return _VerifyPromptZh if language == "zh" else _VerifyPromptEn


async def _screen_and_verify(llm, samples: list[dict], testset_id: int | None,
                             type_key: str, language: str,
                             round_no: int | None = None,
                             others: list[str] | None = None,
                             testset_name: str | None = None) -> list[dict]:
    """One batch pass to raise suspicion, then one call per suspect to settle it.

    `others` are the questions already settled in earlier types. Only the verdict
    sees them — screening them again would be work for nothing — but a question
    here that repeats an earlier type's fact is what the verdict has to see.

    Never raises: a check that cannot run keeps its batch, as the rest of the
    gate does.
    """
    if len(samples) < 2:
        return samples
    if round_no is not None:
        # The bar is inside this type's segment, so the counter is the round.
        _update(testset_id, stage=f"checking_{type_key}",
                stage_done=round_no + 1, stage_total=0)
    questions = [s.get("user_input", "") for s in samples]
    # Only the check sees the settled types. Screening them again would be work
    # for nothing — they have already been through it — but a question in this
    # type that repeats an earlier type's fact is exactly what the check needs
    # to be able to see.
    beyond = list(others or [])
    try:
        screened = await _screen_prompt(language)().generate(
            llm=llm, data=_ScreenInput(questions=questions))
    except Exception as e:
        logger.warning("Screen failed (%s); keeping every sample", e)
        return samples

    reasons = list(screened.reason)
    suspects = [(i, str(reasons[k]) if k < len(reasons) else "")
                for k, i in enumerate(screened.suspect) if 1 <= i <= len(samples)]
    # Same reason as the dedup's: what the screen returned, before the range
    # filter takes its silently-discarded share.
    ignored = sorted(i for i in screened.suspect if not 1 <= i <= len(samples))
    logger.info("Screen [%s] reviewed=%d reported=%s suspects=%s",
                type_key, len(samples), sorted(screened.suspect),
                sorted(i for i, _w in suspects))
    if ignored:
        logger.warning("Screen [%s] reported indices outside 1..%d: %s",
                       type_key, len(samples), ignored)
    if not suspects:
        logger.info("Check [%s] screened=%d, no suspects", type_key, len(samples))
        return samples

    sem = asyncio.Semaphore(VERIFY_CONCURRENCY)

    async def verdict(i: int, why: str) -> tuple[bool, str]:
        sample = samples[i - 1]
        async with sem:
            try:
                v = await _verify_prompt(language)().generate(llm=llm, data=_VerifyInput(
                    question=questions[i - 1],
                    reference=sample.get("reference", ""),
                    material="\n\n".join(sample.get("reference_contexts") or [])[:MAX_DOC_CHARS],
                    other_questions=([q for j, q in enumerate(questions, 1) if j != i]
                                     + beyond),
                    suspicion=why))
                return bool(v.ok), str(v.problem)
            except Exception as e:
                # Keeping the pair is the safe direction everywhere else here too.
                logger.warning("Verify failed (%s); keeping sample", e)
                return True, ""

    got = await asyncio.gather(*(verdict(i, why) for i, why in suspects))
    drop = {i for (i, _w), (ok, _p) in zip(suspects, got) if not ok}
    if drop and testset_name:
        _archive_drops(testset_name, type_key, round_no, samples,
                       {i: p for (i, _w), (ok, p) in zip(suspects, got) if not ok}, drop)
    logger.info("Check [%s] screened=%d suspects=%d dropped=%d",
                type_key, len(samples), len(suspects), len(drop))
    return [s for i, s in enumerate(samples, 1) if i not in drop]


async def _filter_faithful(llm, samples: list[dict], testset_id: int | None,
                           type_key: str, language: str) -> list[dict]:
    """Drop QA pairs whose reference is unsupported by its material or off-topic.

    Two independent verdicts: ragas Faithfulness (factual support, including
    attributing facts to the right subject) and a single alignment call (does the
    reference answer the question). Either failing rejects the pair.
    """
    def _report(checked: int, dropped: int) -> None:
        # Always, even when nothing was dropped and even when there was nothing
        # to check. Reporting only the rejections meant a gate that ran and
        # passed everything left no trace at all, and a run whose generation
        # came back empty read as "generate -> fallback" with the gate
        # apparently skipped: one run hit this early return and logged nothing.
        #
        # Server log only. The customer sees the outcome on the genType line;
        # which of the three filters took a sample is our business.
        logger.info("Gate [%s] checked=%d dropped=%d", type_key, checked, dropped)

    if not samples:
        _report(0, 0)
        return samples

    try:
        faithful = await _faithfulness_scores(llm, samples)
    except Exception as e:
        # The gate must never break generation — but a skipped gate is not the
        # same as a passed one, and this line is the first time the customer gets
        # to see the difference. It is what would have explained a 15-minute run
        # that produced nothing extra for the wait.
        logger.warning("Gate [%s] skipped, batch kept unchecked: %s", type_key, e)
        _report(0, 0)
        if testset_id is not None:
            _log(testset_id, "gateSkipped", {"type": type_key})
        return samples

    kept: list[dict] = []
    rejected: list[tuple[dict, str, float | None]] = []
    for sample, score in zip(samples, faithful):
        if score == score and score < MIN_REFERENCE_FAITHFULNESS:
            rejected.append((sample, "lowFaithfulness", score))
        else:
            # NaN means the metric could not score the pair — keep it rather
            # than dropping on a failure to measure.
            kept.append(sample)
    _report(len(samples), len(rejected))
    # A couple of what was rejected, server log only: whether the fix belongs in
    # the prompt, the model or the threshold is our question, and a 60-character
    # excerpt of a question does not tell the customer anything they can act on.
    # They see the outcome on the referenceDrop and genType lines.
    for sample, reason, score in rejected[:REJECTED_SAMPLE_COUNT]:
        logger.info(
            "Gate [%s] rejected (%s, score=%s): Q %s | A %s",
            type_key, reason, "n/a" if score is None else round(score, 2),
            _excerpt(sample.get("user_input")), _excerpt(sample.get("reference")),
        )
    return kept


def sample_quality_ok(sample: dict) -> bool:
    """Non-empty question + an answerable reference; question stands alone."""
    return (
        bool(sample.get("user_input"))
        and reference_is_complete(sample.get("reference"))
        and not reference_admits_no_answer(sample.get("reference"))
        and is_standalone_question(sample["user_input"])
    )


def detect_language(corpus_name: str) -> str:
    """Sample up to 5 docs; >30% CJK chars in the first 2000 chars -> 'zh'."""
    files = corpus_md_files(corpus_name)[:5]
    if not files:
        return "en"
    total = 0
    cjk = 0
    for f in files:
        text = f.read_text(encoding="utf-8", errors="ignore")[:2000]
        total += len(text)
        cjk += len(_CJK.findall(text))
    return "zh" if total > 0 and cjk / total > 0.3 else "en"

def select_seeds(files: list[Path], m: int, exclude: set[str] | None = None,
                 seed: int = RANDOM_SEED) -> list[Path]:
    """Length-constrained (>=1500 chars) deterministic random sampling (layer 3)."""
    rng = random.Random(seed)
    candidates = _eligible_files(files, exclude)
    rng.shuffle(candidates)
    return candidates[:m]


def _eligible_files(files: list[Path], exclude: set[str] | None = None) -> list[Path]:
    exclude = exclude or set()
    candidates = []
    for f in files:
        if str(f) in exclude:
            continue
        try:
            n = len(f.read_text(encoding="utf-8", errors="ignore"))
        except OSError:
            continue
        if n >= MIN_DOC_CHARS:
            candidates.append(f)
    return candidates


def _round_robin_groups(groups: list[list[Path]], m: int, seed: int = RANDOM_SEED) -> list[Path]:
    """Interleave picks across groups (shuffled within each group)."""
    rng = random.Random(seed)
    pools = []
    for g in groups:
        pool = list(g)
        rng.shuffle(pool)
        pools.append(pool)
    picked: list[Path] = []
    while len(picked) < m and any(pools):
        for pool in pools:
            if pool and len(picked) < m:
                picked.append(pool.pop())
    return picked


DEEP_GROUP_SIZE = 6  # min docs per group to form multi-hop clusters


def _allocate_adaptive(
    groups: dict[str, list[Path]], m: int, multi_share: float, seed: int = RANDOM_SEED
) -> list[Path]:
    """Depth-breadth adaptive allocation driven by the question-type mix.

    multi_share = (n_multi_specific + n_multi_abstract) / N. A share of the
    seeds goes deep into a few groups (~6 docs each, enough for multi-hop
    clusters); the rest spreads across the remaining groups for single-hop
    diversity. Deterministic via fixed seed.
    """
    rng = random.Random(seed)
    pools: list[list[Path]] = []
    for g in groups.values():
        pool = list(g)
        rng.shuffle(pool)
        pools.append(pool)
    pools = [p for p in pools if p]
    if not pools:
        return []
    g_deep = min(math.ceil(m * multi_share / DEEP_GROUP_SIZE), len(pools))

    order = list(range(len(pools)))
    rng.shuffle(order)
    deep = [pools[i] for i in order[:g_deep]]
    breadth = [pools[i] for i in order[g_deep:]]

    picked: list[Path] = []
    # deep groups: round-robin, up to DEEP_GROUP_SIZE each
    deep_quota = min(g_deep * DEEP_GROUP_SIZE, m)
    counts = [0] * len(deep)
    while len(picked) < deep_quota and any(deep):
        for k, pool in enumerate(deep):
            if pool and counts[k] < DEEP_GROUP_SIZE and len(picked) < deep_quota:
                picked.append(pool.pop())
                counts[k] += 1
    # breadth groups: one round at a time across all remaining groups
    while len(picked) < m and any(breadth):
        for pool in breadth:
            if pool and len(picked) < m:
                picked.append(pool.pop())
    # top up from deep-group leftovers
    while len(picked) < m and any(deep):
        for pool in deep:
            if pool and len(picked) < m:
                picked.append(pool.pop())
    return picked[:m]


def _kmeans_labels(vectors: list[list[float]], k: int, iterations: int = 20,
                   seed: int = RANDOM_SEED) -> list[int]:
    """Plain numpy K-means on cosine-normalized vectors (fixed seed)."""
    import numpy as np

    X = np.array(vectors, dtype=float)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    rng = np.random.RandomState(seed)
    centers = X[rng.choice(len(X), k, replace=False)].copy()
    labels = np.zeros(len(X), dtype=int)
    for _ in range(iterations):
        labels = (X @ centers.T).argmax(axis=1)
        for j in range(k):
            members = X[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels.tolist()


async def select_seeds_smart(
    files: list[Path],
    m: int,
    exclude: set[str] | None,
    corpus_name: str,
    emb,
    testset_id: int | None = None,
    multi_share: float = 0.0,
    seed: int = RANDOM_SEED,
) -> list[Path]:
    """Layered seed selection (design doc §3.2 fallback chain):

    1. directory grouping (top-level subdirs of the corpus = company/topic)
    2. embedding clustering (any corpus, one embedding pass over a pool)
    3. length-constrained fixed-seed random (last resort, quality caveat logged)

    Within layers 1-2, allocation across groups is adaptive: multi_share of the
    seeds go deep into a few groups (multi-hop clusters need depth), the rest
    spread for single-hop diversity.
    """
    candidates = _eligible_files(files, exclude)
    if len(candidates) <= m:
        return candidates

    # layer 1: directory grouping
    corpus_dir = get_settings().data_dir / "corpus" / corpus_name

    def _group_by(base: Path) -> dict[str, list[Path]]:
        groups: dict[str, list[Path]] = {}
        for f in candidates:
            rel = f.relative_to(base)
            top = rel.parts[0] if len(rel.parts) > 1 else "_root_"
            groups.setdefault(top, []).append(f)
        return groups

    dir_groups = _group_by(corpus_dir)
    # Unwrap single wrapper directories (e.g. uploaded "data/<公司>/*.md"):
    # descend while the whole corpus sits under one directory.
    while len(dir_groups) == 1 and "_root_" not in dir_groups:
        wrapper = next(iter(dir_groups))
        dir_groups = _group_by(corpus_dir / wrapper)
    if len(dir_groups) > 1:
        return _allocate_adaptive(dir_groups, m, multi_share, seed)

    # layer 2: embedding clustering over a pre-sampled pool
    try:
        rng = random.Random(seed)
        pool = list(candidates)
        rng.shuffle(pool)
        pool = pool[: min(3 * m, 200)]
        texts = [p.read_text(encoding="utf-8", errors="ignore")[:1000] for p in pool]
        vecs = await emb.aembed_documents(texts)
        k = min(max(3, m // 5), len(pool))
        labels = _kmeans_labels(vecs, k, seed=seed)
        clusters: dict[int, list[Path]] = {}
        for p, lb in zip(pool, labels):
            clusters.setdefault(lb, []).append(p)
        # No row for a successful clustered pick, same as the directory path
        # above: the count says nothing the customer can act on. The random
        # fallback below keeps its row — that one warns multi-hop quality may
        # suffer, which is actionable.
        return _allocate_adaptive(clusters, m, multi_share, seed)
    except Exception as e:
        logger.warning("Cluster seed selection failed, falling back to random: %s", e)

    # layer 3: random
    picked = select_seeds(files, m, exclude, seed=seed)
    if testset_id is not None:
        _log(testset_id, "seedsRandom", {"count": len(picked)})
    return picked


# ---------------------------------------------------------------------------
# ragas interaction layer (monkeypatched in tests)
# ---------------------------------------------------------------------------


def _anthropic_sdk_base_url(base_url: str) -> str:
    """Base URL in the form the Anthropic SDK expects.

    The SDK appends /v1/messages itself, so a configured base_url that already
    ends in /v1 (our convention — see llm_http._anthropic_url) must be
    trimmed, otherwise requests go to /v1/v1/messages."""
    base = (base_url or "").rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base or "https://api.anthropic.com"


class _MeteredLLM:
    """Shared judge/generation client behaviour: token accounting and
    output-budget escalation on truncation. Mixed into ChatOpenAI or
    ChatAnthropic; concrete subclasses must declare `totals` as a model field
    (pydantic rejects assignment to undeclared attributes)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per instance: a class-level dict would be shared by every client.
        self.totals = {"prompt": 0, "completion": 0, "calls": 0}

    def _record(self, result) -> None:
        for gens in result.generations:
            for gen in gens if isinstance(gens, list) else [gens]:
                u = gen.message.response_metadata.get("token_usage") or {}
                self.totals["prompt"] += u.get("prompt_tokens", 0)
                self.totals["completion"] += u.get("completion_tokens", 0)
                self.totals["calls"] += 1

    @staticmethod
    def _hit_length_stop(result) -> bool:
        """Any generation cut off by the output budget. OpenAI reports
        finish_reason="length"; Anthropic reports stop_reason="max_tokens"."""
        for gens in result.generations:
            for gen in gens if isinstance(gens, list) else [gens]:
                info = getattr(gen, "generation_info", None) or {}
                meta = getattr(getattr(gen, "message", None),
                               "response_metadata", None) or {}
                if info.get("finish_reason") == "length" or meta.get("finish_reason") == "length":
                    return True
                if info.get("stop_reason") == "max_tokens" or meta.get("stop_reason") == "max_tokens":
                    return True
        return False

    def _next_budget(self, budget: int) -> int | None:
        """Doubled budget, or None when the ceiling is reached."""
        nxt = budget * 2
        if nxt > _MAX_OUTPUT_CEILING:
            logger.warning("Output still truncated at %d tokens; giving up", budget)
            return None
        logger.info("Output truncated at max_tokens=%d; retrying with %d", budget, nxt)
        return nxt

    def _clamped_budget(self, exc: Exception, budget: int, attempted: int) -> int | None:
        """Platforms often name their real output cap in a 400 message; retry
        once at that cap instead of giving up."""
        m = re.search(r"max(?:imum)?[^\d]{0,30}(\d{4,6})", str(exc), re.I)
        if m and budget < int(m.group(1)) < attempted:
            logger.info("Platform output cap is %s; final clamped retry", m.group(1))
            return int(m.group(1))
        logger.warning("Platform rejected max_tokens=%d: %s", attempted, exc)
        return None

    @staticmethod
    def _size_of(value) -> int:
        """Character count of a prompt or a completed generation, for the call
        log. Cheap, and unlike token counts it is always available — the
        provider's usage block comes back empty on exactly the truncated
        responses that matter most."""
        try:
            return len(str(value))
        except Exception:
            return 0

    @classmethod
    def _generated_chars(cls, result) -> int:
        total = 0
        for gens in getattr(result, "generations", []):
            for gen in gens if isinstance(gens, list) else [gens]:
                total += cls._size_of(getattr(gen, "text", ""))
        return total

    def _log_call(self, started: float, prompt_chars: int, result) -> None:
        """One line per logical LLM call: how long it took, how big the prompt
        was, how much it wrote back, and whether the output budget cut it off.

        A run that takes far longer than usual has to be traceable to a call:
        two runs had one call run past the output ceiling, retry at 32768 and
        give up after 95-145 s, and nothing said which prompt it was. Prompt and
        output sizes name it — a judge prompt is a few thousand characters, a
        corpus-fed generation prompt tens of thousands.

        Deliberately NOT reporting attempts or token deltas: the client is
        shared by every concurrent call in a phase, so a delta over
        `self.totals` blends in other in-flight calls and reads as nonsense
        (14 attempts, 0 tokens) under the gate's 8-way fan-out. Server log only:
        the UI shows the phase-level entries."""
        logger.info(
            "LLM call [%s] %.1fs prompt=%dchars out=%dchars max_tokens=%d truncated=%s",
            _current_stage[0] or "?", time.monotonic() - started, prompt_chars,
            self._generated_chars(result), self.max_tokens,
            self._hit_length_stop(result),
        )

    def _log_failure(self, started: float, prompt_chars: int, exc: BaseException) -> None:
        """One line for a call that gave up, so a stall has a witness.

        Only successful calls used to be logged, which is why 153 s spent on
        five requests that each sat on a dead connection until the timeout
        looked like the process doing nothing at all — no line named the calls
        that were hanging or how long each had been waiting.
        """
        logger.warning(
            "LLM call [%s] FAILED after %.1fs prompt=%dchars max_tokens=%d: %s",
            _current_stage[0] or "?", time.monotonic() - started, prompt_chars,
            self.max_tokens, f"{type(exc).__name__}: {exc}"[:200],
        )

    def _escalate(self, call, *args, **kwargs):
        started = time.monotonic()
        prompt_chars = self._size_of(args)
        # Restore the configured budget afterwards. The escalation raises it in
        # place, and this client is shared by the generator and by every
        # concurrent judge call, so one truncated generation used to leave all
        # later calls — including judge prompts of a few thousand characters —
        # asking for 32768, and a platform rejection then clamped the budget
        # down for everyone in turn.
        budget = self.max_tokens
        try:
            result = self._escalating(call, *args, **kwargs)
        except Exception as exc:
            self._log_failure(started, prompt_chars, exc)
            raise
        finally:
            self.max_tokens = budget
        self._log_call(started, prompt_chars, result)
        return result

    def _escalating(self, call, *args, **kwargs):
        budget = self.max_tokens or 4096
        result = call(*args, **kwargs)
        self._record(result)
        while self._hit_length_stop(result):
            nxt = self._next_budget(budget)
            if nxt is None:
                return result
            # Mutating the client field works for both providers (max_tokens is
            # a constructor arg, not a per-call one, for Anthropic).
            self.max_tokens = nxt
            try:
                result = call(*args, **kwargs)
            except Exception as e:
                clamped = self._clamped_budget(e, budget, nxt)
                if clamped is None:
                    return result
                self.max_tokens = clamped
                result = call(*args, **kwargs)
                self._record(result)
                return result
            self._record(result)
            budget = nxt
        return result

    async def _escalate_async(self, call, *args, **kwargs):
        started = time.monotonic()
        prompt_chars = self._size_of(args)
        budget = self.max_tokens  # see _escalate
        try:
            result = await self._escalating_async(call, *args, **kwargs)
        except Exception as exc:
            self._log_failure(started, prompt_chars, exc)
            raise
        finally:
            self.max_tokens = budget
        self._log_call(started, prompt_chars, result)
        return result

    async def _escalating_async(self, call, *args, **kwargs):
        budget = self.max_tokens or 4096
        result = await call(*args, **kwargs)
        self._record(result)
        while self._hit_length_stop(result):
            nxt = self._next_budget(budget)
            if nxt is None:
                return result
            self.max_tokens = nxt
            try:
                result = await call(*args, **kwargs)
            except Exception as e:
                clamped = self._clamped_budget(e, budget, nxt)
                if clamped is None:
                    return result
                self.max_tokens = clamped
                result = await call(*args, **kwargs)
                self._record(result)
                return result
            self._record(result)
            budget = nxt
        return result

    def generate(self, *a, **kw):
        return self._escalate(super().generate, *a, **kw)

    async def agenerate(self, *a, **kw):
        return await self._escalate_async(super().agenerate, *a, **kw)


# Log every HTTP round trip to the model provider. Doubles the log volume, and
# is meant to be turned off once the stalls are explained. `_log_call` reports
# how long a call took but not when the request left or what came back, so five
# requests that each sat on a dead connection for 153 s were indistinguishable
# from the provider being uniformly slow.
HTTP_TRACE = True


_TRACE_STARTED = "ragbench_trace_started"


class _WireTrace:
    """Log the moment a request actually reaches the transport.

    The `request` event hook fires *before* httpx hands anything to the
    transport, so an `HTTP →` line proves only that the client accepted the
    request — not that it went out. A stall needs those told apart: one that
    reached the wire and got no answer is the far side's; one that never
    reached it never left the client, and the 20 s was spent inside httpx.

    The evidence today is a hang that cost 20 s (the timeout) and then came
    back in 0.2 s on the retry. Both shapes produce exactly that, and nothing
    on record separates them.
    """

    def __init__(self, inner):
        self._inner = inner

    def _log_response(self, request, response) -> None:
        started = request.extensions.get(_TRACE_STARTED)
        elapsed = f"{time.monotonic() - started:.2f}s" if started else "?"
        logger.info("WIRE ← %s %s in %s", response.status_code, request.url, elapsed)

    def _log_failure(self, request, exc) -> None:
        started = request.extensions.get(_TRACE_STARTED)
        elapsed = f"{time.monotonic() - started:.2f}s" if started else "?"
        logger.warning("WIRE ✗ %s %s after %s: %s", request.method, request.url,
                       elapsed, type(exc).__name__)

    async def handle_async_request(self, request):
        logger.info("WIRE → %s", request.url)
        try:
            response = await self._inner.handle_async_request(request)
        except Exception as exc:
            self._log_failure(request, exc)
            raise
        self._log_response(request, response)
        return response

    def handle_request(self, request):
        logger.info("WIRE → %s", request.url)
        try:
            response = self._inner.handle_request(request)
        except Exception as exc:
            self._log_failure(request, exc)
            raise
        self._log_response(request, response)
        return response

    def close(self):
        self._inner.close()

    async def aclose(self):
        await self._inner.aclose()

    # httpx drives the transport as a context manager (`async with client`
    # enters it), so a wrapper that only forwards requests breaks that path.
    def __enter__(self):
        self._inner.__enter__()
        return self

    def __exit__(self, *exc):
        return self._inner.__exit__(*exc)

    async def __aenter__(self):
        await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc):
        return await self._inner.__aexit__(*exc)


def _attach_http_trace(llm) -> None:
    """Trace the client's requests on both the sync and the async path.

    The outbound line is when we sent it; the inbound line is from the same
    clock. Together they separate "the request went out late" (ours) from "the
    answer came back late" (theirs). A request with no inbound line is one
    nothing answered — the shape a stall leaves, and the one that was invisible
    until now. Error responses also log their body, which is where a provider
    explains itself.

    The timing is measured here rather than read off `response.elapsed`: a
    response event hook fires *before* the body is read, and httpx raises
    "'.elapsed' may only be accessed after the response has been read or
    closed" if asked at that point. That exception escapes through httpx and
    surfaces as an APIConnectionError, i.e. the tracer turns every call into a
    connection failure — measured, 0 of 20 calls survived it. So the inbound
    line times up to the response *headers*, which is what a stall holds up
    anyway.
    """
    def on_request(request) -> None:
        request.extensions[_TRACE_STARTED] = time.monotonic()
        logger.info("HTTP → %s %s", request.method, request.url)

    def on_response(response) -> None:
        started = response.request.extensions.get(_TRACE_STARTED)
        elapsed = f"{time.monotonic() - started:.2f}s" if started else "?"
        notes = "".join(
            f" {key}={response.headers[key]}"
            for key in ("retry-after", "x-ratelimit-remaining", "x-ratelimit-limit",
                        "anthropic-ratelimit-requests-remaining", "x-request-id")
            if key in response.headers
        )
        if response.status_code >= 400:
            try:
                notes += f" body={response.text[:300]!r}"
            except Exception:  # a stream that was never read has no .text
                notes += " body=<unreadable>"
        logger.info("HTTP ← %s in %s%s", response.status_code, elapsed, notes)

    # Separate hooks per client kind, because httpx calls them differently:
    # the sync client calls the hook and ignores what it returns, while the
    # async client does `await hook(request)`. A plain function on the async
    # client therefore awaits None — "'NoneType' object can't be awaited" —
    # and every request on that path dies as an APIConnectionError. Attaching
    # the sync pair to both is exactly what happened the first time; the sync
    # calls in testing passed and the pipeline, which is async throughout,
    # produced nothing at all.
    async def on_request_async(request) -> None:
        on_request(request)

    async def on_response_async(response) -> None:
        on_response(response)

    for attr, request_hook, response_hook in (
        ("_client", on_request, on_response),
        ("_async_client", on_request_async, on_response_async),
    ):
        try:
            httpx_client = getattr(getattr(llm, attr), "_client")
        except AttributeError as exc:
            logger.warning("HTTP trace not attached to %s: %s", attr, exc)
            continue
        httpx_client.event_hooks["request"].append(request_hook)
        httpx_client.event_hooks["response"].append(response_hook)
        httpx_client._transport = _WireTrace(httpx_client._transport)


def _build_llm(config: ModelConfig, max_tokens: int, cache_dir: Path | None,
               read_cache: bool = True):
    """LangchainLLMWrapper over an OpenAI- or Anthropic-format client, with
    optional disk cache. Returns (wrapper, raw, cache); cache is None when
    cache_dir is None, and read_cache=False keeps writing without reading."""
    from ragas.cache import DiskCacheBackend
    from ragas.llms import LangchainLLMWrapper

    class _CountedDiskCache(DiskCacheBackend):
        """Counts cache hits. read_enabled=False makes it write-only: the
        result is still stored, but never reused this run."""

        def __init__(self, *args, read_enabled: bool = True, **kwargs):
            super().__init__(*args, **kwargs)
            self.read_enabled = read_enabled
            self.hits = 0
            self.misses = 0

        def has_key(self, key):
            return self.read_enabled and super().has_key(key)

        def get(self, key):
            # Only a lookup that found something is a hit. Counting every call
            # is how a cold cache came to report "280 hits" — that number was
            # the number of lookups, and it read as though the cache were doing
            # something. _CountedStore, below, had it right.
            if not self.read_enabled:
                self.misses += 1
                return None
            value = super().get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    # Phases run in separate threads/event loops; a keep-alive connection bound
    # to a closed loop causes "Event loop is closed" when reused. Close each
    # connection after its response so nothing is shared across loops.
    common: dict = {
        "api_key": config.api_key or "none",
        "max_tokens": max_tokens,
        # Three retries, and this layer is still the only one that retries a
        # network failure (ragas' RunConfig is set to a single attempt). The
        # count buys recovery from a short outage, measured the hard way: eight
        # concurrent calls hit one ~10 s silent window together, the endpoint
        # was answering again 3 s after the first failure, and with a single
        # retry both attempts fell inside the same dead window — one escaped
        # ReadTimeout killed a whole generation run. The SDK's backoff
        # (exponential, jittered) walks later attempts past a blip like that.
        # What the count does not buy is patience: the timeout below is a
        # first-byte timeout, so an endpoint that never starts answering still
        # costs its 10 s per attempt, ~40 s before this gives up on it — the
        # old "one retry" kept that at ~20 s, and that trade is what changed
        # here, not the timeout. Billing and auth errors are unaffected either
        # way: the SDK never retries 401/402/403-class responses (its
        # _should_retry covers 408/409/429/5xx and connection errors only), so
        # a dead account still fails on the first attempt.
        "max_retries": 3,
        "timeout": LLM_TIMEOUT,
        # Stream, so that "first byte" means the model started answering rather
        # than that it finished. Unstreamed, the server buffers the whole reply
        # and the first byte is the last one — measured 24 s on a large prompt,
        # against this 10 s timeout, which is why every long call failed and the
        # gate never ran.
        "streaming": True,
        # Without this the streamed response carries no usage, and the token
        # counters shown to the user go blank.
        "stream_usage": True,
        "default_headers": {"Connection": "close"},
    }

    # Thinking is on by default for reasoning models and the judge only needs a
    # JSON verdict, so it is switched off wherever the protocol allows it.
    is_ark = "volces" in (config.base_url or "")
    # What the save-time probe found, sent as raw body fields. Empty for a config
    # that was never probed and for one whose endpoint would not be turned off.
    probed: dict = json.loads(config.thinking_param) if config.thinking_param else {}

    if config.api_format == "anthropic":
        from langchain_anthropic import ChatAnthropic

        class _MeteredAnthropic(_MeteredLLM, ChatAnthropic):
            totals: dict = {}

        # Official Anthropic parameter (ThinkingConfigParam includes
        # {"type": "disabled"}). Kept as the fallback for a config that was never
        # probed: Anthropic-format endpoints default to not thinking.
        common["thinking"] = probed.get("thinking") or {"type": "disabled"}
        raw = _MeteredAnthropic(
            model=config.model,
            base_url=_anthropic_sdk_base_url(host_gateway(config.base_url)),
            **common,
        )
    else:
        from langchain_openai import ChatOpenAI

        class _MeteredOpenAI(_MeteredLLM, ChatOpenAI):
            totals: dict = {}

        # The OpenAI protocol has no such parameter, so it goes in the body as a
        # provider-specific field. Volcano Ark's is "thinking"; anything else has
        # to come from the probe, because guessing was what left DeepSeek
        # thinking for 520 s while sending nothing at all.
        extra_body: dict = {"thinking": {"type": "disabled"}} if is_ark else {}
        extra_body.update(probed)
        if extra_body:
            common["extra_body"] = extra_body
        raw = _MeteredOpenAI(
            model=config.model, base_url=host_gateway(config.base_url), **common
        )

    cache = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = _CountedDiskCache(cache_dir=str(cache_dir), read_enabled=read_cache)
    if HTTP_TRACE:
        _attach_http_trace(raw)
    wrapper = LangchainLLMWrapper(raw, cache=cache)
    return wrapper, raw, cache


def _fill_skipped(
    results: list[list[float] | None], skipped: list[str], on_skip=None
) -> list[list[float]]:
    """Zero-fill skipped texts (shape must match input length for ragas).

    Per design doc §7: a text that still fails at batch size 1 is skipped and
    recorded, not allowed to interrupt the pipeline.
    """
    dim = next((len(v) for v in results if v), 0)
    if dim == 0:
        raise RuntimeError(f"All {len(results)} embedding calls failed")
    if skipped and on_skip:
        on_skip(len(skipped), skipped[0])
    return [v if v is not None else [0.0] * dim for v in results]


async def _embed_texts_async(embed_call, texts: list[str], max_concurrency: int, on_skip=None):
    """Embed texts; on batch failure split recursively; single-text failure is
    skipped (zero vector) and reported instead of interrupting the pipeline."""
    sem = asyncio.Semaphore(max_concurrency)
    results: list[list[float] | None] = [None] * len(texts)
    skipped: list[str] = []

    async def run(indexes: list[int]) -> None:
        batch = [texts[i] for i in indexes]
        try:
            async with sem:
                vecs = await embed_call(batch)
            for i, v in zip(indexes, vecs):
                results[i] = v
        except Exception as e:
            if len(indexes) == 1:
                skipped.append(texts[indexes[0]][:80])
                logger.warning("Embedding single text failed, skipping: %s", e)
            else:
                mid = len(indexes) // 2
                await run(indexes[:mid])
                await run(indexes[mid:])

    await run(list(range(len(texts))))
    return _fill_skipped(results, skipped, on_skip)


def _embed_texts_sync(embed_call, texts: list[str], on_skip=None) -> list[list[float]]:
    results: list[list[float] | None] = [None] * len(texts)
    skipped: list[str] = []

    def run(indexes: list[int]) -> None:
        batch = [texts[i] for i in indexes]
        try:
            vecs = embed_call(batch)
            for i, v in zip(indexes, vecs):
                results[i] = v
        except Exception as e:
            if len(indexes) == 1:
                skipped.append(texts[indexes[0]][:80])
                logger.warning("Embedding single text failed, skipping: %s", e)
            else:
                mid = len(indexes) // 2
                run(indexes[:mid])
                run(indexes[mid:])

    run(list(range(len(texts))))
    return _fill_skipped(results, skipped, on_skip)


def _embedding_cache_namespace(model: str) -> str:
    """The cache namespace for one embedding model: readable and store-safe.

    The name cannot be used raw. The store behind CacheBackedEmbeddings rejects
    keys holding ":" among others, and Ollama's model ids carry one —
    "bge-m3:latest" is what its own API returns — so an evaluation against
    Ollama died on every embedding job with InvalidKeyException.

    Sanitising alone is not enough either: "a:b" and "a?b" clean up to the same
    string, and a namespace is what separates one model's vectors from
    another's. Two models sharing one means the wrong vectors are served.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)[:40]
    return f"{safe}-{hashlib.sha256(model.encode()).hexdigest()[:12]}"


def _build_embeddings(config: ModelConfig, llm_concurrency: int, on_skip=None,
                      cache_dir: Path | None = None, read_cache: bool = True):
    """OpenAI-format embeddings with recursive batch splitting + skip-on-single-failure.
    With cache_dir, results persist via langchain CacheBackedEmbeddings
    (namespace = the model name, made store-safe — see the helper above).
    read_cache=False keeps writing results but never reads them.
    Returns (wrapper, store); store is None when caching is off."""
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    # Everything gets the caller's concurrency, local endpoints included. A
    # localhost check used to pin them to 4, which assumes the server is Ollama
    # — Ollama takes about four at a time and queues the rest, so the cap only
    # moved where the queue lives. Nothing here can know what a customer runs
    # behind that address, and a local vLLM or TEI would just be throttled. What
    # the callers pass is 16 by default, which a service that cannot take it
    # answers by queuing (Ollama) or by an error that the recursive batch
    # splitting below absorbs.
    max_concurrency = llm_concurrency

    class _RobustEmbeddings(OpenAIEmbeddings):
        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await _embed_texts_async(
                super().aembed_documents, texts, max_concurrency, on_skip
            )

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return _embed_texts_sync(super().embed_documents, texts, on_skip)

    raw = _RobustEmbeddings(
        base_url=host_gateway(config.base_url),
        api_key=config.api_key or "none",
        model=config.model,
        chunk_size=32,
        max_retries=5,
        # Same reasoning as LLM_TIMEOUT: a local or platform embedding endpoint
        # answers in well under a second, so a request still open after
        # LLM_TIMEOUT is stuck and should be retried rather than waited on.
        request_timeout=LLM_TIMEOUT,
        # Send raw text instead of tiktoken token arrays — OpenAI-compatible
        # endpoints like Ollama reject token-id input ("invalid input type").
        check_embedding_ctx_length=False,
        # See _build_llm: avoid keep-alive connections crossing event loops.
        default_headers={"Connection": "close"},
    )
    store = None
    if cache_dir is not None:
        from langchain_classic.embeddings import CacheBackedEmbeddings
        from langchain_classic.storage import LocalFileStore

        class _CountedStore(LocalFileStore):
            """Counts embedding cache hits. read_enabled=False makes it
            write-only: vectors are still stored, never reused this run."""

            def __init__(self, *args, read_enabled: bool = True, **kwargs):
                super().__init__(*args, **kwargs)
                self.read_enabled = read_enabled
                self.hits = 0
                self.misses = 0

            def mget(self, keys):
                if not self.read_enabled:
                    self.misses += len(keys)
                    return [None] * len(keys)
                results = super().mget(keys)
                self.hits += sum(1 for r in results if r is not None)
                self.misses += sum(1 for r in results if r is None)
                return results

        cache_dir.mkdir(parents=True, exist_ok=True)
        store = _CountedStore(str(cache_dir), read_enabled=read_cache)
        raw = CacheBackedEmbeddings.from_bytes_store(
            raw, store, namespace=_embedding_cache_namespace(config.model),
            # embed_query is uncached by default; ragas' Response Relevancy
            # calls it for the question vector, so enable it explicitly.
            query_embedding_cache=True,
            key_encoder="sha256")
    return LangchainEmbeddingsWrapper(raw), store


def _default_transforms(docs, llm, emb):
    from ragas.testset.transforms import default_transforms

    return default_transforms(documents=docs, llm=llm, embedding_model=emb)


def _apply_transforms(kg, transforms, run_config):
    from ragas.testset.transforms import apply_transforms

    apply_transforms(kg, transforms, run_config)


def _load_kg(kg_path: Path):
    from ragas.testset.graph import KnowledgeGraph

    return KnowledgeGraph.load(str(kg_path))


def _new_kg_with_docs(seed_paths: list[Path]):
    from langchain_core.documents import Document
    from ragas.testset.graph import KnowledgeGraph, Node, NodeType

    docs = [
        Document(
            page_content=p.read_text(encoding="utf-8", errors="ignore")[:MAX_DOC_CHARS],
            metadata={"source": p.name},
        )
        for p in seed_paths
    ]
    kg = KnowledgeGraph()
    for doc in docs:
        kg.nodes.append(
            Node(
                type=NodeType.DOCUMENT,
                properties={"page_content": doc.page_content, "document_metadata": doc.metadata},
            )
        )
    return kg, docs


def _generate_personas(kg, llm, language: str):
    """Both languages ship an asset: ragas' own prompt asks for "a unique name",
    which models read as "invent a person" — a run against a Chinese corpus came
    back with personas named after people in the documents, and the questions
    written from that viewpoint addressed them by name ("陆文龙啊，你作为…"). The
    example in ragas' own prompt is an identity ("Digital Marketing Specialist"),
    so an identity label is the intent — it just has to be said outright.

    How many of these the synthesizers actually use is ragas' call, not this
    one's: see NUM_PERSONAS."""
    from ragas.testset.persona import PersonaGenerationPrompt, generate_personas_from_kg

    lang_name = "chinese" if language == "zh" else "english"
    asset = PROMPTS_DIR / f"persona_generation_prompt_{lang_name}.json"
    prompt = (
        PersonaGenerationPrompt.load(str(asset)) if asset.exists() else None
    )
    kwargs = {"kg": kg, "llm": llm, "num_personas": NUM_PERSONAS}
    if prompt is not None:  # a missing asset must never stop generation
        kwargs["persona_generation_prompt"] = prompt
    personas = generate_personas_from_kg(**kwargs)
    # With fewer summary clusters than personas ragas pads the list by sampling
    # with replacement, which would hand the same viewpoint two turns. Keep the
    # first of each name; the point of the count is distinct viewpoints.
    seen: set[str] = set()
    unique = []
    for p in personas:
        if p.name not in seen:
            seen.add(p.name)
            unique.append(p)
    return unique


def _cached_personas(corpus_name: str, lang_name: str, fingerprint: dict):
    """Personas for this exact graph, if a previous run on this corpus saved them.

    A persona call is one LLM round-trip and its result depends only on the
    graph's document summaries and the prompt — not on which testset is being
    built. They used to be regenerated for every testset because the only cache
    in play (the LLM disk cache) is scoped to a testset directory. Keyed on the
    same fingerprint that decides KG reuse, so reusing a graph reuses its
    personas and rebuilding one regenerates them."""
    path = personas_path_for(corpus_name)
    if not path.exists():
        return None
    try:
        entry = json.loads(path.read_text()).get(lang_name)
    except (OSError, ValueError):
        return None
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint or entry.get("num_personas") != NUM_PERSONAS:
        return None
    try:
        from ragas.testset.persona import Persona

        return [Persona(name=p["name"], role_description=p["role_description"])
                for p in entry["personas"]]
    except (KeyError, TypeError, ValueError):
        return None


def _save_personas(corpus_name: str, lang_name: str, fingerprint: dict,
                   personas: list) -> None:
    path = personas_path_for(corpus_name)
    store: dict = {}
    if path.exists():
        try:
            store = json.loads(path.read_text())
        except (OSError, ValueError):
            store = {}
    store[lang_name] = {
        "fingerprint": fingerprint,
        "num_personas": NUM_PERSONAS,
        "personas": [{"name": p.name, "role_description": p.role_description}
                     for p in personas],
    }
    try:
        path.write_text(json.dumps(store, ensure_ascii=False, indent=2))
    except OSError as e:
        # A cache write must never break a run.
        logger.warning("Could not save personas cache (%s)", e)


def _make_synthesizers(llm, language: str):
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    synths = {
        "single": SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi_specific": MultiHopSpecificQuerySynthesizer(llm=llm),
        "multi_abstract": MultiHopAbstractQuerySynthesizer(llm=llm),
    }
    # load_prompts RETURNS the loaded prompts; it does not apply them. Without
    # the set_prompts call the shipped assets are silently ignored and ragas'
    # defaults are used instead. Both languages ship assets so the reference
    # answers stay question-focused either way.
    lang_name = "chinese" if language == "zh" else "english"
    for synth in synths.values():
        loaded = synth.load_prompts(str(PROMPTS_DIR), language=lang_name)
        synth.set_prompts(**loaded)
    return synths


def _make_generator(llm, emb, kg, personas):
    from ragas.testset import TestsetGenerator

    return TestsetGenerator(
        llm=llm, embedding_model=emb, knowledge_graph=kg, persona_list=personas
    )


def _generate_chunk(gen, synth, size: int, run_config):
    """Generate `size` QA samples in one call; returns a list of sample dicts.

    `num_personas` is passed explicitly because ragas otherwise slices the
    persona list we hand the generator down to its own default: it builds every
    scenario as `self.persona_list[:num_personas]`, so a count asked for at
    generation time but not repeated here is silently ignored. Passing the same
    number in both places is what makes NUM_PERSONAS mean anything.
    """
    result = gen.generate(
        testset_size=size,
        query_distribution=[(synth, 1.0)],
        run_config=run_config,
        num_personas=NUM_PERSONAS,
    )
    return result.to_list()


# ---------------------------------------------------------------------------
# Progress & log helpers
# ---------------------------------------------------------------------------


# Mirrors the stage currently being worked on, so an LLM call logged deep inside
# ragas can say which phase it belonged to. `_update` is the one place the stage
# changes; a list rather than a plain global so the assignment is local. One
# run at a time is the norm, and the mirror is only a label on a log line: two
# concurrent runs would label each other's calls, nothing more.
_current_stage: list[str] = [""]


def _update(testset_id: int, **fields) -> None:
    if "stage" in fields:
        _current_stage[0] = str(fields["stage"])
    db = get_session_factory()()
    try:
        ts = db.get(Testset, testset_id)
        if ts is not None:
            for k, v in fields.items():
                setattr(ts, k, v)
            db.commit()
    finally:
        db.close()


# Guards the read-modify-write of log_entries. The event loop alone would not
# need it — the sequence has no await — but _log is also handed to the
# embeddings client as its on_skip callback, and that runs inside the thread
# pool _apply_transforms uses, so two skips reported at once could each read the
# same list and the second write would drop the first entry.
_log_lock = threading.Lock()


def _log(testset_id: int, key: str, params: dict | None = None) -> None:
    db = get_session_factory()()
    try:
        with _log_lock:
            ts = db.get(Testset, testset_id)
            if ts is not None:
                entries = json.loads(ts.log_entries)
                entries.append(
                    {
                        "time": datetime.now(timezone.utc).isoformat(),
                        "key": key,
                        "params": params or {},
                    }
                )
                ts.log_entries = json.dumps(entries, ensure_ascii=False)
                db.commit()
    finally:
        db.close()


# ---------------------------------------------------------------------------
# KG build / expansion
# ---------------------------------------------------------------------------


async def _build_kg(kg_path, seed_paths, llm, emb, run_config, testset_id, w_kg_pct, stage,
                    progress_base=2.0):
    """Build a KG from seed docs; per-transform progress updates."""
    kg, docs = _new_kg_with_docs(seed_paths)
    # Off the event loop: this only constructs objects and counts document
    # tokens, but the token counting pulls in tiktoken's vocabulary, and the
    # first lookup of that is a synchronous network fetch. On the loop it took
    # 30 s and every LLM call in flight timed out behind it — the whole run
    # failed, not just this step.
    transforms = await asyncio.to_thread(_default_transforms, docs, llm, emb)
    total = max(len(transforms), 1)
    for i, t in enumerate(transforms):
        if i == 0:
            _update(testset_id, stage=stage, stage_done=0, stage_total=total)
        await asyncio.to_thread(_apply_transforms, kg, [t], run_config)
        # advance only AFTER each transform completes (no half-step on start)
        _update(
            testset_id,
            stage=stage,
            stage_done=i + 1,
            stage_total=total,
            progress=round(progress_base + w_kg_pct * ((i + 1) / total), 1),
        )
    await asyncio.to_thread(kg.save, str(kg_path))
    return kg


async def _merge_new_docs(kg, new_seed_paths, llm, emb, run_config, testset_id,
                          w_kg_pct, stage="expanding_kg", update_progress=True,
                          round_no: int | None = None, progress_base: float = 0.0):
    """Incremental expansion on an in-memory KG: extract the new docs in a
    sub-KG, then merge their nodes into the graph.

    The `builders` split below is deliberately empty. ragas wraps its
    relationship builders in `Parallel`, which is not a RelationshipBuilder,
    so they run as part of the sub-KG pass and the merged nodes end up linked
    to each other but not back to the existing graph. Rerunning them across
    the whole graph is not the fix that looks like: measured at 21 s and
    16,680 relationships added *on top of* the ones already present, and it
    repeats the same again on every expansion, on a graph that only grows.

    During fallback top-up (update_progress=False) only the stage text is
    updated — the progress bar must never regress. When round_no is set, the
    stage counter carries the fallback round instead of transform counts.
    """
    from ragas.testset.transforms.base import RelationshipBuilder

    # Timed step by step. One expansion took 108 s of which 84 s produced no
    # LLM call, no HTTP request and no log line at all, and froze the UI for
    # the same stretch — the phase total alone could not say which step it was
    # in. A handful of lines per expansion is cheap next to that.
    _t = time.monotonic()

    def _step(label: str) -> None:
        nonlocal _t
        now = time.monotonic()
        logger.info("KG merge [%s] %s %.2fs", stage, label, now - _t)
        _t = now

    sub, docs = _new_kg_with_docs(new_seed_paths)
    _step("build_sub_kg")
    # Same reason as in _build_kg: the token counting here must not run on the
    # event loop.
    transforms = await asyncio.to_thread(_default_transforms, docs, llm, emb)
    _step("default_transforms")
    extractors = [t for t in transforms if not isinstance(t, RelationshipBuilder)]
    builders = [t for t in transforms if isinstance(t, RelationshipBuilder)]
    total = max(len(extractors) + len(builders), 1)

    def _stage_update(done: int) -> None:
        fields = {
            "stage": stage,
            "stage_done": round_no if round_no is not None else done,
            "stage_total": 0 if round_no is not None else total,
        }
        if update_progress:
            fields["progress"] = round(progress_base + w_kg_pct * (done / total), 1)
        _update(testset_id, **fields)

    _stage_update(0)  # show the stage text at phase entry; bar stays put
    _step("stage_update_0")
    for i, t in enumerate(extractors):
        await asyncio.to_thread(_apply_transforms, sub, [t], run_config)
        _step(f"extractor:{type(t).__name__}")
        _stage_update(i + 1)
    existing_ids = {n.id for n in kg.nodes}
    kg.nodes.extend([n for n in sub.nodes if n.id not in existing_ids])
    _step("merge_nodes")

    for j, t in enumerate(builders):
        await asyncio.to_thread(_apply_transforms, kg, [t], run_config)
        _step(f"builder:{type(t).__name__}")
        _stage_update(len(extractors) + j + 1)
    _stage_update(total)
    _step("stage_update_total")
    return kg


async def _expand_kg(kg_path, new_seed_paths, llm, emb, run_config, testset_id, w_kg_pct,
                     progress_base=2.0):
    """Load the KG from disk, merge new seed docs into it, and save."""
    kg = await asyncio.to_thread(_load_kg, kg_path)
    kg = await _merge_new_docs(kg, new_seed_paths, llm, emb, run_config, testset_id, w_kg_pct,
                               progress_base=progress_base)
    await asyncio.to_thread(kg.save, str(kg_path))
    return kg


# ---------------------------------------------------------------------------
# Generation / dedup
# ---------------------------------------------------------------------------


def _topup_size(missing: int, kept: int, generated: int) -> int:
    """How many samples to generate to close a gap of `missing`.

    Uses the observed yield (kept / generated so far) so the oversampling fits
    the corpus at hand: a clean corpus asks for barely more than the gap, a
    noisy one asks for proportionally more. Falls back to assuming half
    survives when there is no measurement yet.
    """
    yield_rate = kept / generated if generated > 0 else 0.5
    yield_rate = min(max(yield_rate, MIN_YIELD_RATE), 1.0)
    return max(missing + 1, math.ceil(missing / yield_rate))


def _needs_single_call(synth) -> bool:
    """True for synthesizers whose material pool is walked from its start.

    All three of ragas' query synthesizers choose their source material by
    walking `get_node_clusters()` from the beginning and stopping as soon as
    they have `n` scenarios — the abstract one asks the graph for exactly `n`
    indirect clusters (find_n_indirect_clusters), and the single- and multi-hop
    ones take `ceil(n / len(nodes))` scenarios per node, in list order. That
    selection is deterministic, so chunked calls (always the same chunk size)
    redraw the same leading entries and repeat the same questions.
    """
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    # Measured: chunked multi-hop calls (n=2) kept producing questions about the
    # same one cluster, while a single call of 8 produced 8 distinct questions.
    # Single-hop had been assumed to enumerate the whole node set — it does not,
    # it breaks out at n — and chunking a batch of 12 into 6 calls of 2 produced
    # a whole batch about the first two nodes of the graph, which then collapsed
    # to a handful of facts and was dropped by dedup.
    return isinstance(synth, (MultiHopAbstractQuerySynthesizer,
                              MultiHopSpecificQuerySynthesizer,
                              SingleHopSpecificQuerySynthesizer))


def _widen_pool_after_a_barren_round(gained: int) -> bool:
    """Whether the next fallback round should merge new documents.

    Two levers close a shortfall and the cheap one goes first. Rotating the
    fallback window onto material the graph has not been asked about costs
    nothing; merging documents costs a sub-KG build, an LLM extraction pass and
    a full graph save — 108 s in one measured run.

    Whether the graph has more to give is not a question about its size. A run
    asks for a couple of dozen samples and never draws on more than a fraction
    of one, so comparing that against the graph's cluster count — 917 in one
    measured run — is always true, and merging becomes unreachable. The question
    is answered by whether the round produced anything: a round that gained
    means the material still has something to say and rotating further is the
    right move; a round that gained nothing, or lost ground to the reviewer,
    means it does not, and only different material can help.

    round 1 of one run is the case this is for. It produced 2 samples and the
    semantic reviewer dropped both as near-duplicates of what was already
    kept — generated, not missing, and repeating because the material was.
    """
    return gained <= 0


def _rotate_node_pool(synth, start: int) -> None:
    """Make the next generation call work from a different slice of the material.

    Generation starts from the top of its material pool on every call, so the
    fallback round that follows a short batch regenerates the same questions —
    measured: a single-hop fallback whose added documents sat at the end of the
    graph gained 0, and two rounds later the type was still one short. Rotating
    the start spreads successive calls over the material instead.

    Two shapes, because the synthesizers take their material differently:

    * Single-hop and multi-hop-specific take the graph alone and walk the nodes
      in order — rotate the list itself.
    * Multi-hop abstract takes the count as well (`get_node_clusters(graph, n)`)
      and asks the graph for exactly those n clusters, in a deterministic order
      seeded by a hash of the node set. Rotating that result changes nothing,
      because the set is already chosen: the same n every call. So it is asked
      for `n * _CLUSTER_WINDOW_FACTOR` clusters and handed a moving window — the
      first n of a larger answer are the same n as before, so the extra ones are
      what successive calls get to move onto.

    The wrapper forwards whatever arguments it is handed and normalises both
    shapes into one call signature; an earlier version took a single argument
    and emptied an entire multi-hop-abstract type (0/10, two fallback rounds,
    nothing generated) when ragas called it with two.
    """
    from ragas.testset.synthesizers import (
        MultiHopAbstractQuerySynthesizer,
        MultiHopSpecificQuerySynthesizer,
        SingleHopSpecificQuerySynthesizer,
    )

    if not isinstance(synth, (SingleHopSpecificQuerySynthesizer,
                              MultiHopSpecificQuerySynthesizer,
                              MultiHopAbstractQuerySynthesizer)):
        return
    if getattr(synth, "_node_pool_original", None) is None:
        synth._node_pool_original = synth.get_node_clusters
        takes_count = isinstance(synth, MultiHopAbstractQuerySynthesizer)

        def start_from(*args, **kwargs):
            if takes_count and len(args) >= 2:
                graph, n = args[0], args[1]
                clusters = list(synth._node_pool_original(
                    graph, max(n, n * _CLUSTER_WINDOW_FACTOR)))
                if not clusters:
                    return clusters
                m = len(clusters)
                k = synth._node_pool_start % m
                # Wraps, so a start near the end still yields a full window
                # instead of a short one (a short window would silently raise
                # ragas' scenarios-per-cluster count).
                return [clusters[(k + i) % m] for i in range(min(n, m))]
            nodes = list(synth._node_pool_original(*args, **kwargs))
            if not nodes:
                return nodes
            k = synth._node_pool_start % len(nodes)
            return nodes[k:] + nodes[:k]

        synth.get_node_clusters = start_from
    synth._node_pool_start = start


async def _generate_many(gen, synth, target, run_config, testset_id, stage,
                         progress_base, progress_weight, round_no: int | None = None):
    """Generate `target` samples in sequential chunks (per-chunk progress).

    Concurrent generate() calls on a shared TestsetGenerator hang inside ragas,
    so chunks run sequentially; each chunk parallelizes internally via
    run_config.max_workers. Chunk size keeps progress granularity at ~10 steps.
    When round_no is set (fallback top-up), the stage counter shows the round.
    The counter shows real produced counts (including the 1.1x headroom), kept
    consistent with the generation log.
    """
    if target <= 0:
        return []
    # Chunking keeps the progress bar smooth, but some synthesizers derive
    # their material pool from the requested count (ragas'
    # MultiHopAbstractQuerySynthesizer calls find_n_indirect_clusters(n)).
    # Asking those in chunks repeats the same n-sized pool every time, so every
    # chunk returns the same questions — measured: 18 chunked (n=2) gave 4
    # unique, while a single call of 8 gave 8 unique across 6 companies.
    if _needs_single_call(synth):
        chunk = target
    else:
        chunk = max(1, math.ceil(target / 10))
    results: list[dict] = []
    done = 0
    bypassed = False
    # Where this call starts in the node pool, carried across calls so a
    # fallback round continues past what the previous one already asked about.
    start = getattr(synth, "_node_pool_start", 0)
    # Set the stage before generating, not only after each chunk: with one chunk
    # per type (see _needs_single_call) the label was otherwise still the
    # previous phase's for the whole generation, which sent a diagnosis of a
    # slow run to the gate when the calls belonged to generation.
    _update(testset_id, stage=stage, stage_done=round_no if round_no is not None else 0,
            stage_total=0 if round_no is not None else target, progress=round(
                min(99.0, progress_base + progress_weight * (done / target)), 1))
    while done < target:
        size = min(chunk, target - done)
        _rotate_node_pool(synth, start)
        try:
            samples = await asyncio.to_thread(_generate_chunk, gen, synth, size, run_config)
        except Exception as e:
            if "Invalid n value" in str(e) and not bypassed:
                # Some endpoints reject n>1; switch to bypass_n and retry once.
                bypassed = True
                logger.warning("Endpoint rejects n>1; switching to bypass_n and retrying")
                try:
                    if hasattr(gen.llm, "bypass_n"):
                        gen.llm.bypass_n = True
                    samples = await asyncio.to_thread(_generate_chunk, gen, synth, size, run_config)
                except Exception as e2:
                    logger.warning("Chunk generation failed after bypass_n: %s", e2)
                    samples = []
            else:
                logger.warning("Chunk generation failed: %s", e)
                samples = []
        for s in samples:
            s["user_input"] = normalize_qa_text(s.get("user_input", ""))
            s["reference"] = normalize_qa_text(s.get("reference", ""), multiline=True)
            if s.get("reference_contexts"):
                s["reference_contexts"] = strip_hop_markers(s["reference_contexts"])
        results.extend(samples)
        done += size
        start += size
        # Progress is capped at 99 during generation; 100 is reserved for done.
        pct = min(99.0, progress_base + progress_weight * (done / target))
        _update(
            testset_id,
            stage=stage,
            stage_done=round_no if round_no is not None else len(results),
            stage_total=0 if round_no is not None else target,
            progress=round(pct, 1),
        )
    # Leave the pool rotated past this call's slice so the next round (fallback
    # top-up) draws from material this one did not already turn into questions.
    if getattr(synth, "_node_pool_original", None) is not None:
        synth._node_pool_start = start
    return results


def _persist_items(testset_id: int, testset_name: str, items: list[dict]) -> None:
    db = get_session_factory()()
    try:
        db.query(TestsetItem).filter(TestsetItem.testset_id == testset_id).delete()
        rows = []
        for seq, s in enumerate(items, start=1):
            rows.append(TestsetItem(
                testset_id=testset_id,
                seq=seq,
                user_input=s.get("user_input", "") or "",
                reference=s.get("reference", "") or "",
                reference_contexts=json.dumps(s.get("reference_contexts") or [], ensure_ascii=False),
                synthesizer_name=s.get("synthesizer_name", "") or "",
                persona_name=s.get("persona_name", "") or "",
            ))
        db.add_all(rows)
        db.commit()
    finally:
        db.close()
    out = testset_dir_for(testset_name) / "testset.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for s in items:
            f.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _cache_fingerprint_current(llm_cfg: ModelConfig, max_tokens: int) -> dict:
    return {"model": llm_cfg.model, "base_url": llm_cfg.base_url, "max_tokens": max_tokens}


async def run_generation(testset_id: int) -> None:
    """Background task: generate a testset. Always reaches a terminal state."""
    if testset_id in _running:
        return
    _running.add(testset_id)
    started = time.monotonic()
    db = get_session_factory()()
    try:
        ts = db.get(Testset, testset_id)
        if ts is None:
            return
        name = ts.name
        try:
            await _pipeline(ts, started)
            db.refresh(ts)
            ts.status = Testset.STATUS_COMPLETED
            ts.completed_at = datetime.now(timezone.utc)
            db.commit()
            _log(testset_id, "done", {"seconds": round(time.monotonic() - started, 1)})
        except Exception as e:
            logger.exception("Testset %s: generation failed", name)
            db.rollback()
            ts = db.get(Testset, testset_id)
            if ts is not None:
                detail = f"stage={ts.stage}: {_error_detail(e)}"
                ts.status = Testset.STATUS_FAILED
                ts.error = detail[:1000]
                ts.completed_at = datetime.now(timezone.utc)
                db.commit()
                _log(testset_id, "failed", {"error": detail[:500]})
    finally:
        db.close()
        _running.discard(testset_id)


def _error_detail(e: BaseException) -> str:
    """Exception type + message + cause chain (+ request URL when available)."""
    parts = []
    cur: BaseException | None = e
    while cur is not None and len(parts) < 4:
        text = f"{type(cur).__name__}: {cur}"
        req = getattr(cur, "request", None)
        if req is not None:
            text += f" [{req.method} {req.url}]"
        parts.append(text)
        cur = cur.__cause__ or cur.__context__
    return " <- ".join(parts)


async def _pipeline(ts: Testset, started: float) -> None:
    """The generation pipeline. `ts` fields are read before any awaits mutate."""
    testset_id = ts.id
    targets = {
        "single": ts.n_single,
        "multi_specific": ts.n_multi_specific,
        "multi_abstract": ts.n_multi_abstract,
    }
    n_total = sum(targets.values())
    corpus_name = ts.corpus_name
    testset_dir = testset_dir_for(ts.name)
    testset_dir.mkdir(parents=True, exist_ok=True)

    db = get_session_factory()()
    try:
        llm_cfg = db.get(ModelConfig, ts.llm_config_id)
        emb_cfg = db.get(ModelConfig, ts.embedding_config_id)
        if llm_cfg is None or emb_cfg is None:
            raise RuntimeError("Model configuration not found")
        llm_snapshot = ModelConfig(
            name=llm_cfg.name, type=llm_cfg.type, api_format=llm_cfg.api_format,
            base_url=llm_cfg.base_url, api_key=llm_cfg.api_key, model=llm_cfg.model,
        )
        emb_snapshot = ModelConfig(
            name=emb_cfg.name, type=emb_cfg.type, api_format=emb_cfg.api_format,
            base_url=emb_cfg.base_url, api_key=emb_cfg.api_key, model=emb_cfg.model,
        )
    finally:
        db.close()

    # --- connectivity precheck: fail fast with a clear reason before spending
    # any LLM/embedding calls
    from app.services import model_connect

    _update(testset_id, stage="checking_connectivity", progress=0)
    llm_ok, llm_msg, llm_ms = await model_connect.test_connectivity(
        llm_snapshot.type, llm_snapshot.api_format, llm_snapshot.base_url,
        llm_snapshot.api_key, llm_snapshot.model,
    )
    if not llm_ok:
        raise RuntimeError(f"LLM connectivity test failed ({llm_snapshot.base_url}): {llm_msg}")
    emb_ok, emb_msg, emb_ms = await model_connect.test_connectivity(
        emb_snapshot.type, emb_snapshot.api_format, emb_snapshot.base_url,
        emb_snapshot.api_key, emb_snapshot.model,
    )
    if not emb_ok:
        raise RuntimeError(
            f"Embedding connectivity test failed ({emb_snapshot.base_url}): {emb_msg}"
        )
    _log(testset_id, "connectivityOk",
         {"llmMs": llm_ms, "embMs": emb_ms,
          "llm": llm_snapshot.name, "emb": emb_snapshot.name})

    # --- cache fingerprint: wipe llm_cache when output-affecting params change
    cache_dir = testset_dir / "llm_cache"
    fp_path = cache_dir / ".fingerprint.json"
    current = _cache_fingerprint_current(llm_snapshot, ts.llm_max_tokens)
    cached_fp = None
    if fp_path.exists():
        try:
            cached_fp = json.loads(fp_path.read_text())
        except Exception:
            cached_fp = None
    if cached_fp is not None and cached_fp != current and cache_dir.exists():
        shutil.rmtree(cache_dir)
        _log(testset_id, "cacheCleared", {})
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp_path.write_text(json.dumps(current))

    llm, llm_raw, _llm_cache = _build_llm(llm_snapshot, ts.llm_max_tokens, cache_dir)
    emb, _emb_store = _build_embeddings(
        emb_snapshot,
        ts.llm_concurrency,
        # Count only: the skipped text is a 60-character fragment the customer
        # cannot act on, and these are usually transient endpoint failures.
        on_skip=lambda count, _preview: _log(
            testset_id, "embeddingSkipped", {"count": count}
        ),
    )

    from ragas.run_config import RunConfig


    # --- language detection (auto) or manual override — instant step,
    # invisible to the user (no stage text, no progress advance)
    detected = detect_language(corpus_name)
    language = detected if ts.prompt_language == "auto" else ts.prompt_language
    # Only the override is worth a row: auto-detection follows from the corpus
    # the customer uploaded, so telling them the language was "detected" says
    # nothing they do not already know.
    if language != detected:
        _log(testset_id, "langOverride", {"lang": language})

    # --- seeds (progress advances to 5% only AFTER selection completes;
    # clustering may take tens of seconds, the bar must not sit meanwhile)
    _update(testset_id, stage="selecting_seeds", progress=1)
    files = corpus_md_files(corpus_name)
    if not files:
        raise RuntimeError(f"No markdown files found in corpus {corpus_name}")
    m = estimate_seed_count(n_total, len(files), ratio=ts.amplify)
    seeds_path = seeds_path_for(corpus_name)
    kg_path = kg_path_for(corpus_name)
    # Guarded like the fingerprint read below: this file is rewritten on every
    # fallback round, so a concurrent run or a crash mid-write can leave it
    # truncated, and an unguarded read fails the whole run at selecting_seeds.
    old_seeds: list[str] = []
    if seeds_path.exists():
        try:
            old_seeds = json.loads(seeds_path.read_text())
        except (OSError, ValueError):
            logger.warning("seeds.json unreadable; treating the seed list as empty")

    reuse = bool(ts.reuse_kg) and kg_path.exists()
    # --- run seed: reusing a KG reproduces the same scenario sampling (old
    # questions may reappear); rebuilding or creating fresh gets a new seed.
    # A resume always uses the seed snapshotted on the testset record.
    fp_path = kg_fingerprint_path_for(corpus_name)
    stored_fp = None
    if fp_path.exists():
        try:
            stored_fp = json.loads(fp_path.read_text())
        except Exception:
            stored_fp = None
    stored_seed = stored_fp.pop("run_seed", None) if stored_fp else None
    if reuse:
        # KG is bound to a config fingerprint (models + chunk params); a
        # mismatch forces a full rebuild even when reuse was requested.
        if stored_fp != _kg_fingerprint(llm_snapshot, emb_snapshot):
            _log(testset_id, "kgFingerprintMismatch", {})
            reuse = False
    if ts.run_seed:
        run_seed = ts.run_seed  # resume: keep the original seed
    elif reuse and stored_seed is not None:
        run_seed = stored_seed  # KG reuse: same seed -> same scenarios
    else:
        run_seed = secrets.randbelow(2**31 - 1) + 1
    _update(testset_id, run_seed=run_seed)
    # max_retries=1: one attempt, no retry — the client retries a network failure
    # once, and ragas retrying every exception on top of that only ever repeated a
    # call that had already been given its second chance.
    run_config = RunConfig(max_workers=ts.llm_concurrency, timeout=180, max_retries=1,
                           seed=run_seed)
    expand = reuse and 0 < len(old_seeds) < m
    if reuse and not expand:
        seed_paths = [Path(p) for p in old_seeds]
    else:
        keep = [p for p in old_seeds if Path(p).exists()] if reuse else []
        need = m - len(keep)
        multi_share = (
            (ts.n_multi_specific + ts.n_multi_abstract) / n_total if n_total else 0.0
        )
        new = (
            await select_seeds_smart(files, need, set(keep), corpus_name, emb, testset_id,
                                     multi_share=multi_share, seed=run_seed)
            if need > 0
            else []
        )
        seed_paths = [Path(p) for p in keep] + new

    # --- progress weights
    #
    # The bar's ceiling is 99, not 100: the fallback cannot be sized in advance
    # — nothing knows how many rounds it will take — so it holds the last point
    # rather than guessing. That leaves the 94 points from 5 to 99 for the work
    # that *can* be sized, split 57/43 between generation and checking. The
    # ratio is the wall time the two phases actually took across the R1-R3 runs
    # (101 s generating, 76 s checking on the two reuse rounds, which share a
    # graph and a run seed). Two runs with identical settings still landed 15
    # points apart, so the decimal places do not matter and are not worth
    # chasing; what mattered is that generation used to take all 95 and
    # checking none, which parked the bar on 99 from the end of generation
    # through the whole of checking and the fallback.
    _update(testset_id, progress=5)  # seed selection finished
    build_kg = (not reuse) or expand

    # The bar's shape is fixed, not measured.
    #
    # The graph ends at 30 whether it was reused, expanded or rebuilt, and the
    # three question types split the 69 points that remain evenly. Both shares
    # used to be proportional — the graph to the seeds it processed, each type
    # to its item count — and the graph's came from timing six expansions on one
    # corpus. A bar that moves at a different rate per corpus is not telling the
    # reader how far along they are; it is telling them about that corpus.
    #
    # Even shares for the three types for the same reason: nothing knows in
    # advance how many check-and-replenish rounds a type will need, and its item
    # count does not predict it.
    KG_DONE = 30.0
    w_kg_pct = KG_DONE - 5.0
    # A type's 23 points split 12 for its generation and 11 for its
    # check-and-replenish loop. Fixed numbers, not measurements — the 57/43 this
    # replaces came from timing a few runs, the way the graph share did, and it
    # divided something the customer never sees divided: the loop carries one
    # label, numbered by round.
    #
    # Inside the 11, the first check takes ROUND/2 and the rest is spread evenly
    # over the rounds the loop may run, so the bar moves every round instead of
    # only at its end; a type needing fewer rounds than the cap is granted the
    # remainder at once, because that is what happened.
    GEN_POINTS = 12.0
    LOOP_POINTS = 11.0
    ROUND = LOOP_POINTS / FALLBACK_ROUNDS
    gen_weights = {k: GEN_POINTS for k in targets}
    check_weights = {k: ROUND / 2 for k in targets}
    # The two halves have to add up to LOOP_POINTS between them. The version
    # this replaces made the second half ROUND/2 * FALLBACK_ROUNDS, which is
    # 5.5 — leaving every type 4.4 short, so the bar finished at 85.8 and the
    # last 14.2 arrived in one jump when the testset completed.
    loop_weights = {k: LOOP_POINTS - check_weights[k] for k in targets}

    # A build with no seed documents cannot work: ragas divides by the document
    # count inside its default transforms and raises ZeroDivisionError, which
    # reads as a library failure rather than "this corpus has nothing usable".
    # Reachable whenever every .md is shorter than MIN_DOC_CHARS — short
    # announcements and tables converted by MarkItDown are routinely that small —
    # because the seed COUNT is derived from all files while selection keeps only
    # the long ones.
    if build_kg and not seed_paths:
        raise RuntimeError(
            f"No usable documents in corpus '{corpus_name}': every file is "
            f"shorter than {MIN_DOC_CHARS} characters, which is too little to "
            f"build questions from"
        )

    # --- knowledge graph
    kg_t0 = time.monotonic()
    if not build_kg:
        # KG load is instant: invisible to the user (no stage, no progress)
        kg = await asyncio.to_thread(_load_kg, kg_path)
        _log(testset_id, "kgReused", {})
    elif expand:
        kg = await _expand_kg(kg_path, seed_paths[len(old_seeds):], llm, emb, run_config,
                              testset_id, w_kg_pct, progress_base=5.0)
        _log(testset_id, "kgExpanded",
             {"added": len(seed_paths) - len(old_seeds), "seconds": round(time.monotonic() - kg_t0, 1)})
    else:
        kg = await _build_kg(kg_path, seed_paths, llm, emb, run_config, testset_id,
                             w_kg_pct, "building_kg", progress_base=5.0)
        _log(testset_id, "kgBuilt", {"seconds": round(time.monotonic() - kg_t0, 1)})

    seeds_path.write_text(json.dumps([str(p) for p in seed_paths], ensure_ascii=False))
    fp = _kg_fingerprint(llm_snapshot, emb_snapshot)
    fp["run_seed"] = run_seed
    kg_fingerprint_path_for(corpus_name).write_text(json.dumps(fp))

    # --- personas
    # Rounded like every other progress write: the KG transform above rounds to
    # one decimal, so writing the raw float here steps the bar back by 0.03.
    #
    # No stage of its own. Personas are cached against the corpus fingerprint,
    # so this is either a file read or three short calls — measured 0.9 s when
    # built and 0 s when reused, inside a run of 95-155 s. Naming a step that
    # fast leaves the bar flickering a label the customer cannot use, and
    # "reused" would only repeat what kgReused already says.
    _update(testset_id, progress=round(w_kg_pct + 5, 1))
    lang_name = "chinese" if language == "zh" else "english"
    personas = _cached_personas(corpus_name, lang_name, fp)
    if personas is None:
        personas = await asyncio.to_thread(_generate_personas, kg, llm, language)
        _save_personas(corpus_name, lang_name, fp, personas)

    gen = _make_generator(llm, emb, kg, personas)
    synths = _make_synthesizers(llm, language)

    # --- one type at a time: generate it, then check and replenish it
    #
    # Each type runs its whole loop before the next one starts, so the bar reads
    # as a line per type, and it is what lets a type be checked against the ones
    # already settled: the batch handed to the verdict is this type's questions
    # plus every question already final, so a pair repeating an earlier type's
    # fact is seen and dropped. Checking all three at the end instead would need
    # a separate cross-type pass, and a drop from that pass would send a finished
    # type back into its loop.
    # Raw sample counts per type (initial pass + each top-up). The fallback sizes
    # its next top-up from the yield this material has actually shown.
    generated: dict[str, int] = {}
    final_items: list[dict] = []
    actual: dict[str, int] = {}
    # Everything already final, for the next type to be compared against.
    settled: list[dict] = []
    base_progress = w_kg_pct + 5
    for type_key in QUESTION_TYPES:
        target = targets[type_key]
        if target == 0:
            actual[type_key] = 0
            continue
        want = math.ceil(ts.gen_amplify * target)
        # The type's whole runtime, for the closing line: generation, checking
        # and every top-up round, not just the first generation pass.
        type_t0 = time.monotonic()
        samples = await _generate_many(
            gen, synths[type_key], want, run_config, testset_id,
            stage=f"gen_{type_key}", progress_base=base_progress,
            progress_weight=gen_weights[type_key])
        base_progress += gen_weights[type_key]
        check_progress = base_progress
        # One label for the whole check-and-replenish loop, numbered by round:
        # the customer reads which type it is and how many passes it has taken,
        # not which of the two halves is running. The halves stay visible in the
        # log.
        _update(testset_id, stage=f"checking_{type_key}", stage_done=1, stage_total=1)
        # Three separate checks reject a sample before it reaches the pool —
        # wrong language, malformed or context-dependent question, reference
        # the material does not support — but to the customer they are one
        # thing: this batch was filtered. The breakdown goes to the server log
        # and only the outcome reaches the UI, on the genType line below.
        #
        # Language guardrail: drop samples whose question drifted from the
        # corpus language (models occasionally answer in the wrong language).
        ok_samples = [s for s in samples if language_matches(s.get("user_input", ""), language)]
        dropped_lang = len(samples) - len(ok_samples)
        # Quality guardrail: drop empty / context-dependent questions.
        clean_samples = [s for s in ok_samples if sample_quality_ok(s)]
        dropped_form = len(ok_samples) - len(clean_samples)
        # Reference faithfulness gate: a reference unsupported by its own
        # material makes correct system answers score low downstream.
        before_gate = len(clean_samples)
        clean_samples = await _filter_faithful(llm, clean_samples, testset_id, type_key,
                                               language)
        logger.info(
            "Filter [%s] generated=%d language=%d form=%d gate=%d passed=%d",
            type_key, len(samples), dropped_lang, dropped_form,
            before_gate - len(clean_samples), len(clean_samples),
        )
        raw_count = len(samples)             # before the guardrails
        generated[type_key] = raw_count
        # The first number the customer counts from. Which of the three filters
        # took what, and why, is on the server log above.
        _log(testset_id, "typeFirstGen", {"type": type_key, "count": raw_count})
        gate_dropped = raw_count - len(clean_samples)
        samples = clean_samples
        # What the customer counts on: every produced / topped-up line adds to it,
        # every check line subtracts, and it lands on the closing line's candidate count.
        pool_total = raw_count

        # Questions the semantic reviewer has rejected. The rejection has to
        # outlive the round that made it: `kept` is rebuilt from the whole pool
        # every round, so an item dropped here is offered again next round and
        # can be accepted on a second look — one did exactly that and shipped as
        # a duplicate. Only reviewer drops are banned; items merely trimmed away
        # (the pool usually holds more than the target) stay available.
        dup_rejected: set[int] = set()

        async def review_dupes(pool: list[dict],
                               round_no: int | None = None) -> tuple[list[dict], int, int]:
            """Embedding-dedup a pool, run the semantic reviewer, then trim.

            Trimming LAST is the whole point. It used to run first, and then any
            question the reviewer dropped left the type one short of target with
            no way back: the next round trimmed the enlarged pool to target
            again, the reviewer dropped one again, and the count sat at
            target-1 through every fallback round. Measured twice in a row
            (093009, 093010): multi-hop abstract held at 9/10 with two rounds
            that each reported "gained 0". Review first, cut to size after.
            """
            # No embedding pass in front of this any more. At cosine >= 0.95 it
            # dropped questions that share a long opening but ask about a
            # different attribute — measured over the pairs it dropped across
            # five runs, 13 of 15 were ones the reviewer also calls duplicates,
            # and the other 2 asked a different second attribute and should have
            # stayed. The two ranges overlap (real duplicates 0.959-0.999,
            # false kills 0.959 and 0.981), so no threshold separates them.
            candidates = [s for s in pool
                          if id(s) not in dup_rejected and s.get("user_input")]
            # Everything this round judges, taken before the exact-duplicate
            # pass so its drops still count toward the customer's check count.
            shown = {id(s) for s in candidates}
            candidates, exact_dropped = _exact_duplicates(candidates)
            if exact_dropped:
                logger.info("Exact dup [%s] dropped=%d: %s", type_key,
                            len(exact_dropped),
                            " | ".join(_excerpt(q) for q in exact_dropped[:REJECTED_SAMPLE_COUNT]))
            kept_now = await _semantic_dedup(llm, candidates, testset_id, type_key, language)
            dup_rejected.update(shown - {id(s) for s in kept_now})
            # Screen the batch, then check each suspect against its own answer and
            # material — see _screen_and_verify. Runs here, on what is still
            # standing, so the first pass and every fallback round both get it,
            # and a pair it rejects is banned like a reviewer rejection rather
            # than being offered again next round.
            kept_now = await _screen_and_verify(
                llm, kept_now, testset_id, type_key, language, round_no,
                others=[x["user_input"] for x in settled],
                testset_name=ts.name)
            dup_rejected.update(shown - {id(s) for s in kept_now})
            # What this round's reviewer took out of the pool, counted before
            # the trim below: the trim is "there is more than enough", not a
            # rejection, and the caller reports the two differently.
            dropped_here = len(shown) - len(kept_now)
            # How many survived the reviewer, before the trim to target. The
            # caller reports it: without it the per-type lines do not add up,
            # because the trim is neither generation nor a rejection.
            candidates = len(kept_now)
            rng = random.Random(run_seed)
            rng.shuffle(kept_now)
            return kept_now[:target], dropped_here, candidates

        kept, dropped_here, candidates = await review_dupes(samples)
        # One check line per round, covering everything between "generated" and
        # "kept": the three pre-pool filters, the dedup pass and the check. The
        # customer counts one number, not three mechanisms.
        if gate_dropped + dropped_here:
            _log(testset_id, "typeChecked", {
                "type": type_key, "dropped": gate_dropped + dropped_here,
                "remaining": pool_total - gate_dropped - dropped_here,
            })
        pool_total = candidates
        check_progress += check_weights[type_key]
        _update(testset_id, progress=round(check_progress, 1))
        # One round carries one slice of this type's segment. Whatever the type
        # does not spend is granted when it finishes, so the bar is never short
        # of its segment because a round turned out to be unnecessary.
        round_slice = loop_weights.get(type_key, 0.0) / max(1, FALLBACK_ROUNDS)
        rounds = 0
        # Rotate first, merge only once a round has come back empty — the
        # reasoning, and why size is the wrong measure, is on
        # _widen_pool_after_a_barren_round.
        merge_more = False
        while len(kept) < target and rounds < FALLBACK_ROUNDS:
            rounds += 1
            logger.info("Fallback [%s] round=%d starting", type_key, rounds)
            missing = target - len(kept)
            kept_before_round = len(kept)
            extra: list[Path] = []
            if merge_more:
                logger.info("Fallback [%s] round=%d widening the pool", type_key, rounds)
                # Seed count scales with the user's sampling amplification.
                extra = select_seeds(
                    files, max(3, math.ceil(ts.amplify * missing)),
                    exclude={str(p) for p in seed_paths},
                    seed=run_seed + rounds,  # vary per round so rounds don't repeat picks
                )
            _update(testset_id, stage=f"checking_{type_key}",
                    stage_done=rounds + 1, stage_total=0)
            if extra:
                ex_t0 = time.monotonic()
                # Keeps the loop's own label. Merging documents is a step inside
                # the round, not a stage of its own: a separate label showed the
                # customer "处理中…" (the fallback for a code with no copy) for
                # the whole merge, because the copy for the old label went when
                # the two halves of the loop were collapsed into one. round_no
                # is the loop's numbering, where round 1 is the first check.
                kg = await _merge_new_docs(kg, extra, llm, emb, run_config,
                                           testset_id, w_kg_pct,
                                           stage=f"checking_{type_key}",
                                           update_progress=False,
                                           round_no=rounds + 1)
                seed_paths = seed_paths + extra
                _s = time.monotonic()
                seeds_path.write_text(json.dumps([str(p) for p in seed_paths],
                                                 ensure_ascii=False))
                logger.info("KG merge [%s] write_seeds %.2fs",
                            f"fallback_expand_{type_key}", time.monotonic() - _s)
                _s = time.monotonic()
                await asyncio.to_thread(kg.save, str(kg_path))
                logger.info("KG merge [%s] save %.2fs",
                            f"fallback_expand_{type_key}", time.monotonic() - _s)
                # Server log only: to the customer this is part of the top-up
                # below, not a step of its own.
                logger.info("KG merge [%s] added=%d %.1fs", f"fallback_expand_{type_key}",
                            len(extra), time.monotonic() - ex_t0)
            topup = _topup_size(missing, len(kept), generated.get(type_key, 0))
            generated[type_key] = generated.get(type_key, 0) + topup
            # round_no is the round the customer sees, where round 1 is the
            # first check — one ahead of the fallback index. Passing the index
            # would step the bar back a round for the length of the top-up.
            more = await _generate_many(
                gen, synths[type_key], topup, run_config, testset_id,
                stage=f"fallback_gen_{type_key}", progress_base=check_progress,
                progress_weight=0, round_no=rounds + 1)
            batch_size = len(more)
            pool_total += batch_size
            _log(testset_id, "typeToppedUp", {"type": type_key, "count": batch_size,
                                              "total": pool_total})
            more = [s for s in more if language_matches(s.get("user_input", ""), language)]
            more = [s for s in more if sample_quality_ok(s)]
            more = await _filter_faithful(llm, more, testset_id, type_key, language)
            round_gate_dropped = batch_size - len(more)
            samples = samples + more
            kept, dropped_here, candidates = await review_dupes(samples, round_no=rounds)
            if round_gate_dropped + dropped_here:
                _log(testset_id, "typeChecked", {
                    "type": type_key, "dropped": round_gate_dropped + dropped_here,
                    "remaining": pool_total - round_gate_dropped - dropped_here,
                })
            pool_total = candidates
            check_progress += round_slice
            _update(testset_id, progress=round(check_progress, 1))
            gained = len(kept) - kept_before_round
            # The round's yield: how much new material went in versus how many
            # unique QA pairs came out. Reading that pair tells whether the
            # material was exhausted (needs more docs) or the model's output was
            # rejected by the gate (more docs will not help). Server log only —
            # the customer gets the top-up count and the merged check count.
            logger.info("Fallback [%s] round=%d topup=%d docs=%d gained=%d",
                        type_key, rounds, batch_size, len(extra), gained)
            # Nothing gained means the graph did not answer this round, so the
            # next one stops asking it and merges new documents instead.
            merge_more = _widen_pool_after_a_barren_round(gained)
        check_progress += loop_weights.get(type_key, 0.0) - rounds * round_slice
        _update(testset_id, progress=round(check_progress, 1))
        if len(kept) < target:
            _log(testset_id, "shortfall", {"type": type_key, "missing": target - len(kept)})
        actual[type_key] = len(kept)
        # Closes the arithmetic. Every check line above subtracts from a produced or
        # topped-up line, which leaves the reviewer's survivors; `candidates` is
        # that number and `count` is what the trim to target kept. Without the
        # pair the customer's own subtraction lands above the count in the
        # list, and the difference — items dropped only because the pool held
        # more than asked for — has nothing to explain it.
        # Two lines, because the trim only happened in one of them: saying
        # "kept at random" when the pool held exactly the target claims a choice
        # that was never made.
        _log(testset_id, "typeDone" if candidates > len(kept) else "typeDoneAll", {
            "type": type_key, "count": len(kept), "candidates": candidates,
            "seconds": round(time.monotonic() - type_t0, 1),
        })
        final_items.extend(kept)
        base_progress = check_progress
        settled.extend(kept)

    # --- persist
    # An empty result must fail, not complete. Generation errors are swallowed
    # per chunk (a chunk that raises yields zero samples and the loop moves on),
    # so a run whose LLM calls all failed — expired key, exhausted quota, wrong
    # model name — reached this line with nothing and was still marked
    # completed, progress 100: an empty testset that can be exported and
    # evaluated, with no error to explain it.
    if not final_items:
        raise RuntimeError(
            "No QA pairs were produced: every generation call failed or was "
            "rejected. Check the LLM endpoint and the log entries above"
        )
    # persist is instant: no progress write; completion sets 100
    _persist_items(testset_id, ts.name, final_items)
    totals = getattr(llm_raw, "totals", {"prompt": 0, "completion": 0, "calls": 0})
    _update(testset_id, token_usage=json.dumps(totals), progress=100, stage="done",
            actual_single=actual["single"], actual_multi_specific=actual["multi_specific"],
            actual_multi_abstract=actual["multi_abstract"])


async def resume_interrupted() -> None:
    """On startup, restart generation for testsets still marked as generating.

    The LLM disk cache makes already-finished calls free, so re-running is cheap.

    Only one per corpus is restarted. Two runs on one corpus interleave writes
    to the graph, seed list and persona cache under data/corpus/<name>/, so a
    pair left generating by a crash would resume into exactly the corruption
    the API guards against — and then keep doing it on every restart. The
    furthest-along run keeps the corpus (least work to redo); the rest are
    marked failed with the reason, which puts them back in the UI as resumable
    once that run is done.
    """
    db = get_session_factory()()
    try:
        rows = db.query(Testset).filter(Testset.status == Testset.STATUS_GENERATING).all()
        rows.sort(key=lambda r: (-(r.progress or 0), r.id))
        ids: list[int] = []
        seen: set[int] = set()
        for row in rows:
            if row.corpus_id in seen:
                row.status = Testset.STATUS_FAILED
                row.error = (
                    "Another testset on this corpus was still generating when the "
                    "server restarted. Resume this one once that run has finished."
                )
                continue
            seen.add(row.corpus_id)
            ids.append(row.id)
        db.commit()
    finally:
        db.close()
    for tid in ids:
        logger.info("Resuming generation for testset id=%s", tid)
        asyncio.create_task(run_generation(tid))
