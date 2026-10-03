"""Evaluation pipeline: run a testset against a target RAG system and score it.

Phases: connectivity precheck -> query the target system per QA item (each in a
fresh conversation) -> ragas metrics -> summary. Status machine, progress rules,
logging and resume mirror testset_gen (advance only after a task completes,
never regress, never hit 100% early).
"""

import asyncio
import functools
import hashlib
import json
import logging
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import EvalRun, EvalRunItem, ModelConfig, RagSystemConfig, TestsetItem
from app.db.session import get_session_factory
from app.services import model_connect, rag_client, testset_gen

# Precheck retries: the judge embedding may be a busy LOCAL service (e.g.
# Ollama saturated by other runs) — a slow answer is not a failure. The UI's
# manual "test" button stays single-shot/fast; only the pipeline retries.
_PRECHECK_ATTEMPTS = 3
_PRECHECK_BACKOFF = (3, 6)  # seconds between attempts


async def _precheck_model(model_type: str, cfg) -> tuple[bool, str, int | None]:
    """test_connectivity with bounded retries and linear backoff."""
    ok, msg, ms = False, "not attempted", None
    for i in range(_PRECHECK_ATTEMPTS):
        ok, msg, ms = await model_connect.test_connectivity(
            model_type, cfg.api_format, cfg.base_url, cfg.api_key, cfg.model)
        if ok:
            break
        if i < _PRECHECK_ATTEMPTS - 1:
            await asyncio.sleep(_PRECHECK_BACKOFF[i])
    return ok, msg, ms

logger = logging.getLogger(__name__)

_running: set[int] = set()

# Metric keys -> required data. Context metrics are only selectable when the
# RAG system config provides a contexts extraction path.
METRIC_KEYS = ("faithfulness", "response_relevancy", "context_precision",
               "context_recall", "answer_correctness")
CONTEXT_METRICS = ("faithfulness", "context_precision", "context_recall")

# ragas 0.4.x names result columns after the metric CLASSES, not our keys:
# ResponseRelevancy -> "answer_relevancy",
# LLMContextPrecisionWithReference -> "llm_context_precision_with_reference".
_RAGAS_COLUMN = {
    "response_relevancy": "answer_relevancy",
    "context_precision": "llm_context_precision_with_reference",
}

JUDGE_MAX_TOKENS = 8192
# A judge call that fails to parse yields NaN, which would silently drop that
# item from the metric's average. Retry just those items (measured: 0-1 items
# per run were dropped this way).
METRIC_RETRIES = 2
# Double-grading was tried here and removed: ~85% of items grade identically, so
# a second pass is pure cost, and the ~15% that swing do so because the item is
# ill-defined — re-grading cannot recover a value that does not exist. The
# remaining failure that DOES matter is a metric silently dropping an item, which
# METRIC_RETRIES above handles.
# Cache-partition key: bumping the budget must not reuse entries produced
# under a different one (a truncated result cached then would look valid).
_OUTPUT_BUDGET = JUDGE_MAX_TOKENS

# Progress layout: precheck is instant (invisible); querying 0->70, metrics ->99
QUERY_WEIGHT = 70.0
EVAL_WEIGHT = 29.0


def eval_path_for(name: str) -> Path:
    """Per-run artifact directory (no side effect).

    Refuses a name that resolves outside data/evaluation. The request schema
    validates names too, but this is the sink: the delete path rmtrees what this
    returns, so a row written before the schema check (or by any future caller)
    must not be able to point it at the logs or at data/ itself.
    """
    base = get_settings().data_dir / "evaluation"
    path = base / name
    if not path.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe evaluation name: {name!r}")
    return path


def eval_dir_for(name: str) -> Path:
    d = eval_path_for(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _judge_cache_dir(cfg: ModelConfig, variant: str = "") -> Path:
    """Shared judge cache, partitioned by model identity + variant.

    Two things the underlying cache keys do NOT capture, so the directory name
    must: the model (ragas keys omit it — verified against 0.4.3) and the
    output budget (max_tokens is a constructor arg, not a call arg, so a
    truncated entry cached under one budget would otherwise be reused under
    another). Same model+variant shares a directory, so re-runs hit the cache."""
    fp = hashlib.sha256(
        f"{cfg.base_url}|{cfg.model}|{variant}".encode()
    ).hexdigest()[:12]
    d = get_settings().data_dir / "evaluation" / "_judge_cache" / fp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _update(run_id: int, **fields) -> None:
    db = get_session_factory()()
    try:
        run = db.get(EvalRun, run_id)
        if run is not None:
            for k, v in fields.items():
                setattr(run, k, v)
            db.commit()
    finally:
        db.close()


def _log(run_id: int, key: str, params: dict | None = None) -> None:
    db = get_session_factory()()
    try:
        run = db.get(EvalRun, run_id)
        if run is not None:
            entries = json.loads(run.log_entries)
            entries.append(
                {
                    "time": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "params": params or {},
                }
            )
            run.log_entries = json.dumps(entries, ensure_ascii=False)
            db.commit()
    finally:
        db.close()


@functools.lru_cache(maxsize=1)
def _context_precision_with_verdicts():
    """ragas' context precision, keeping the per-chunk verdicts it discards.

    The metric asks the judge about every retrieved chunk and then keeps only
    the average precision. That 0/1 sequence is what every rank-based reading is
    a view of — Hit@k, the k curve, where the hit landed — and it costs nothing
    to keep: same prompt, same `_calculate_average_precision`, so the
    context_precision number is unchanged by construction.

    In a factory because the ragas import is lazy elsewhere (it is heavy), and
    cached so the class is built once.
    """
    from ragas.metrics import LLMContextPrecisionWithReference
    from ragas.metrics._context_precision import QAC, Verification
    from ragas.metrics.base import ensembler

    class _ContextPrecisionWithVerdicts(LLMContextPrecisionWithReference):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # (user_input, reference) -> verdict per retrieved chunk. The rows
            # and the score column are paired by position, but the sample object
            # is not handed back by evaluate(), so the pair is the key.
            self.verdicts: dict[tuple[str, str], list[int]] = {}

        async def _single_turn_ascore(self, sample, callbacks) -> float:
            answers = []
            for context in list(sample.retrieved_contexts or []):
                verdicts = await self.context_precision_prompt.generate_multiple(
                    data=QAC(question=sample.user_input, context=context,
                             answer=sample.reference),
                    llm=self.llm,
                    callbacks=callbacks,
                )
                aggregated = ensembler.from_discrete(
                    [[result.model_dump() for result in verdicts]], "verdict")
                answers.append(Verification(**aggregated[0]))
            self.verdicts[(sample.user_input, sample.reference)] = [
                1 if answer.verdict else 0 for answer in answers
            ]
            return self._calculate_average_precision(answers)

    return _ContextPrecisionWithVerdicts


def _build_metric_instances(metric_keys: list[str]):
    from ragas.metrics import (
        AnswerCorrectness,
        Faithfulness,
        LLMContextRecall,
        ResponseRelevancy,
    )

    mapping = {
        "faithfulness": Faithfulness,
        "context_precision": _context_precision_with_verdicts(),
        "context_recall": LLMContextRecall,
        # needs only question/answer/reference — works without a contexts path
        "answer_correctness": AnswerCorrectness,
    }

    def build(key: str):
        if key == "response_relevancy":
            # ragas draws this one `strictness` times and averages. The default
            # of 3 costs a different amount per protocol and it is not obvious
            # from the setting: an OpenAI-format judge takes n=3 as three
            # completions in ONE request, while an Anthropic-format client has
            # no `n` and ragas sends the same prompt three times (measured
            # against ragas 0.4.3). One draw removes the multiplier; the mean is
            # the same number either way, with more run-to-run noise.
            return ResponseRelevancy(strictness=1)
        return mapping[key]()

    supported = set(mapping) | {"response_relevancy"}
    return [build(k) for k in metric_keys if k in supported]


async def run_evaluation(run_id: int) -> None:
    """Background task: run an evaluation. Always reaches a terminal state."""
    if run_id in _running:
        return
    _running.add(run_id)
    started = time.monotonic()
    db = get_session_factory()()
    try:
        run = db.get(EvalRun, run_id)
        if run is None:
            return
        name = run.name
        try:
            await _pipeline(run, started)
        except Exception as e:
            detail = testset_gen._error_detail(e)
            logger.exception("Evaluation %s failed", run_id)
            _update(run_id, status=EvalRun.STATUS_FAILED, error=detail)
            _log(run_id, "failed", {"error": detail})
            return
        _update(
            run_id,
            status=EvalRun.STATUS_COMPLETED,
            progress=100,
            stage="done",
            completed_at=datetime.now(timezone.utc),
        )
        _log(run_id, "done", {"seconds": round(time.monotonic() - started, 1)})
    finally:
        db.close()
        _running.discard(run_id)


async def _query_target(run: EvalRun, cfg: RagSystemConfig, headers: dict,
                        items: list[TestsetItem], done_rows: dict) -> None:
    """Call the target RAG system for every item; rows are upserted per item.

    done_rows maps seq -> existing EvalRunItem row (resume path): answered rows
    are kept untouched, previously failed rows are updated in place. Per-item
    failures are recorded and never abort the run.
    """
    run_id = run.id
    total = run.stage_total or len(items)
    sem = asyncio.Semaphore(run.concurrency)
    done = len([r for r in done_rows.values() if r.answer is not None])
    out_path = eval_dir_for(run.name) / "responses.jsonl"

    async def one(item: TestsetItem) -> None:
        nonlocal done
        row = done_rows.get(item.seq)
        if row is not None and row.answer is not None:
            return  # resume: already answered
        result: dict
        try:
            async with sem:
                answer, contexts, elapsed = await rag_client.call_rag_system(
                    base_url=cfg.base_url,
                    api_key=cfg.api_key,
                    headers=headers,
                    body_template=cfg.body_template,
                    answer_path=cfg.answer_path,
                    contexts_path=cfg.contexts_path,
                    timeout=cfg.timeout,
                    question=item.user_input,
                )
            result = {"seq": item.seq, "user_input": item.user_input, "answer": answer,
                      "contexts": contexts, "seconds": round(elapsed, 2), "error": None}
            elapsed_s = round(elapsed, 2)
        except Exception as e:
            result = {"seq": item.seq, "user_input": item.user_input, "answer": None,
                      "contexts": [], "seconds": None,
                      "error": f"{type(e).__name__}: {e}"}
            elapsed_s = None
            _log(run_id, "itemFailed", {"seq": item.seq,
                                        "error": result["error"][:200]})
        db = get_session_factory()()
        try:
            if row is not None:
                db.merge(EvalRunItem(
                    id=row.id, run_id=run_id, seq=item.seq,
                    user_input=item.user_input, reference=item.reference,
                    answer=result["answer"],
                    contexts=json.dumps(result["contexts"], ensure_ascii=False),
                    error=result["error"], seconds=elapsed_s,
                ))
            else:
                db.add(EvalRunItem(
                    run_id=run_id, seq=item.seq, user_input=item.user_input,
                    reference=item.reference, answer=result["answer"],
                    contexts=json.dumps(result["contexts"], ensure_ascii=False),
                    error=result["error"], seconds=elapsed_s,
                ))
            db.commit()
        finally:
            db.close()
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        done += 1
        _update(run_id, stage_done=done,
                progress=round(min(QUERY_WEIGHT, QUERY_WEIGHT * done / total), 1))

    await asyncio.gather(*(one(i) for i in items))


async def _pipeline(run: EvalRun, started: float) -> None:
    run_id = run.id
    db = get_session_factory()()
    try:
        cfg = db.get(RagSystemConfig, run.rag_system_id)
        llm_cfg = db.get(ModelConfig, run.llm_config_id)
        emb_cfg = db.get(ModelConfig, run.embedding_config_id)
        items = db.execute(
            select(TestsetItem).where(TestsetItem.testset_id == run.testset_id)
            .order_by(TestsetItem.seq)
        ).scalars().all()
    finally:
        db.close()
    if cfg is None or llm_cfg is None or emb_cfg is None:
        raise RuntimeError("Referenced config no longer exists")
    # The adapter persists only templates/paths; endpoint and credential live
    # on the run (snapshot at creation).
    cfg.base_url = run.run_base_url
    cfg.api_key = run.run_api_key
    cfg.timeout = run.timeout
    if not items:
        raise RuntimeError("Testset has no items")

    metric_keys = json.loads(run.metrics)
    _update(run_id, stage="checking", stage_total=len(items), stage_done=0)

    # --- connectivity precheck: judge LLM, judge embedding, target RAG system
    ok_llm, msg_llm, ms_llm = await _precheck_model("llm", llm_cfg)
    ok_emb, msg_emb, ms_emb = await _precheck_model("embedding", emb_cfg)
    if not (ok_llm and ok_emb):
        raise RuntimeError(f"Judge model connectivity failed: LLM={msg_llm} Embedding={msg_emb}")
    headers = json.loads(cfg.headers or "{}")
    try:
        await rag_client.call_rag_system(
            base_url=cfg.base_url, api_key=cfg.api_key, headers=headers,
            body_template=cfg.body_template, answer_path=cfg.answer_path,
            contexts_path=cfg.contexts_path, timeout=cfg.timeout,
            question=items[0].user_input,
        )
    except Exception as e:
        raise RuntimeError(f"Target RAG system unreachable: {type(e).__name__}: {e}") from e
    _log(run_id, "connectivityOk",
         {"llmMs": ms_llm, "embMs": ms_emb, "llm": llm_cfg.name, "emb": emb_cfg.name})

    # --- query the target system (resume: answered rows are kept, failed
    # rows are retried in place)
    db = get_session_factory()()
    try:
        done_rows = {
            r.seq: r
            for r in db.execute(
                select(EvalRunItem).where(EvalRunItem.run_id == run_id)
            ).scalars().all()
        }
    finally:
        db.close()
    _update(run_id, stage="querying", stage_done=len(done_rows))
    q_t0 = time.monotonic()
    await _query_target(run, cfg, headers, items, done_rows)
    db = get_session_factory()()
    try:
        run_items = db.execute(
            select(EvalRunItem).where(EvalRunItem.run_id == run_id)
            .order_by(EvalRunItem.seq)
        ).scalars().all()
    finally:
        db.close()
    failed = [r for r in run_items if r.answer is None]
    answered = [r for r in run_items if r.answer is not None]
    _update(run_id, total_items=len(run_items), failed_items=len(failed),
            progress=QUERY_WEIGHT)
    _log(run_id, "queryDone", {"total": len(run_items), "failed": len(failed),
                               "seconds": round(time.monotonic() - q_t0, 1)})
    # Target-system latency (successful calls only). Mean for the overall
    # picture, median because one stuck item should not distort it, max to
    # surface the tail a user actually notices.
    latencies = [r.seconds for r in run_items if r.seconds is not None]
    if latencies:
        _log(run_id, "latency", {
            "avg": round(statistics.mean(latencies), 1),
            "median": round(statistics.median(latencies), 1),
            "max": round(max(latencies), 1),
        })
    if not answered:
        raise RuntimeError("All target-system calls failed; nothing to evaluate")

    # --- ragas metrics
    _update(run_id, stage="evaluating", stage_done=0, stage_total=len(metric_keys))
    e_t0 = time.monotonic()
    # The judge cache always WRITES; use_judge_cache only controls whether an
    # existing entry may be reused. Unchecking forces a full recompute, and the
    # fresh results replace the stored ones (self-healing on stale entries).
    use_cache = bool(getattr(run, "use_judge_cache", True))  # column is Integer
    llm, llm_raw, llm_cache = testset_gen._build_llm(
        llm_cfg, JUDGE_MAX_TOKENS,
        _judge_cache_dir(llm_cfg, f"llm{_OUTPUT_BUDGET}") / "llm",
        read_cache=use_cache)
    emb, emb_store = testset_gen._build_embeddings(
        emb_cfg, run.judge_concurrency,
        cache_dir=_judge_cache_dir(emb_cfg, "emb") / "emb", read_cache=use_cache)
    scores_by_seq, verdicts_by_seq = await asyncio.to_thread(
        _evaluate, run.id, answered, metric_keys, llm, emb, run.judge_concurrency)
    totals = getattr(llm_raw, "totals", {"prompt": 0, "completion": 0, "calls": 0})

    db = get_session_factory()()
    try:
        for r in db.execute(
            select(EvalRunItem).where(EvalRunItem.run_id == run_id)
        ).scalars().all():
            if r.seq in scores_by_seq:
                r.scores = json.dumps(_finite(scores_by_seq[r.seq]), ensure_ascii=False)
            if r.seq in verdicts_by_seq:
                r.context_verdicts = json.dumps(verdicts_by_seq[r.seq])
        db.commit()
    finally:
        db.close()

    summary = _summarize(scores_by_seq, metric_keys)
    # Say how much of the testset each mean actually covers. Silence here means
    # every metric scored every item.
    short = _coverage(scores_by_seq, metric_keys, len(answered))
    if short:
        listed = ", ".join(f"{k} {n}/{len(answered)}" for k, n in short)
        logger.warning("Evaluation %s: partial coverage — %s", run_id, listed)
    _update(run_id, summary=json.dumps(summary), token_usage=json.dumps(totals),
            progress=QUERY_WEIGHT + EVAL_WEIGHT, stage_done=len(metric_keys))
    _log(run_id, "evalDone", {"seconds": round(time.monotonic() - e_t0, 1),
                              **{k: summary.get(k) for k in metric_keys}})
    if short:
        _log(run_id, "metricCoverage", {
            "list": ", ".join(f"{k} {n}/{len(answered)}" for k, n in short),
            "items": len(answered),
        })
    _log(run_id, "judgeCache", {
        "readEnabled": 1 if use_cache else 0,
        "llmCalls": llm_raw.totals["calls"],
        "llmHits": llm_cache.hits if llm_cache else 0,
        "llmMisses": llm_cache.misses if llm_cache else 0,
        "embHits": emb_store.hits if emb_store else 0,
        "embMisses": emb_store.misses if emb_store else 0,
    })


def _coverage(scores_by_seq: dict[int, dict[str, float]],
              metric_keys: list[str], total: int) -> list[tuple[str, int]]:
    """The requested metrics that scored fewer than `total` items, with counts.

    Every mean in the summary is taken over the items that produced a score, so
    a metric covering 25 of 30 reads exactly like one covering all 30 — and one
    covering none drops out of the summary entirely, reading like a metric that
    was never requested. Both shapes hide the same thing, which is how a run
    finished with fifteen metric gaps across its items and reported success.
    """
    short = []
    for key in metric_keys:
        got = sum(1 for s in scores_by_seq.values()
                  if key in s and s[key] is not None and s[key] == s[key])
        if got < total:
            short.append((key, got))
    return short


def _finite(scores: dict) -> dict:
    """Scores as JSON can carry them.

    A metric that scored nothing is stored as NaN, and json.dumps writes that
    as a bare NaN — which is not JSON. One of them makes the browser's
    JSON.parse throw on the entire string, so parseJson falls back to {} and
    every per-item column reads "-", the metrics that scored fine included.
    The summary was never affected: _summarize leaves those keys out.
    """
    return {k: (v if isinstance(v, (int, float)) and math.isfinite(v) else None)
            for k, v in scores.items()}


def _summarize(scores_by_seq: dict[int, dict[str, float]],
               metric_keys: list[str]) -> dict[str, float]:
    """Mean of each metric over the items that produced a usable score.

    An item with no score — the judge call never returned, or came back
    unparsable — is left out of its metric's mean rather than counted as zero.
    """
    summary = {}
    for key in metric_keys:
        vals = [s[key] for s in scores_by_seq.values()
                if key in s and s[key] is not None and s[key] == s[key]]  # skip NaN
        if vals:
            summary[key] = round(sum(vals) / len(vals), 4)
    return summary


def _evaluate(run_id: int, run_items: list[EvalRunItem], metric_keys: list[str],
              llm, emb, concurrency: int, report_progress: bool = True) -> tuple[
                  dict[int, dict[str, float]], dict[int, list[int]]]:
    """Run ragas evaluate() synchronously (called in a worker thread), one
    metric per call so the progress bar can advance per metric (n/4).

    Returns the per-item scores and, when context_precision ran, the judge's
    verdict for each retrieved chunk of each item.
    """
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
    from ragas.run_config import RunConfig

    scores: dict[int, dict[str, float]] = {}
    verdicts: dict[int, list[int]] = {}

    def run_metric(rows: list[EvalRunItem], key: str) -> None:
        if not rows:
            return
        samples = [
            SingleTurnSample(
                user_input=r.user_input,
                response=r.answer,
                retrieved_contexts=json.loads(r.contexts),
                reference=r.reference,
            )
            for r in rows
        ]
        metrics = _build_metric_instances([key])
        result = evaluate(
            dataset=EvaluationDataset(samples=samples),
            metrics=metrics,
            llm=llm,
            embeddings=emb,
            run_config=RunConfig(max_workers=concurrency, timeout=180, max_retries=3),
            show_progress=False,
        )
        df = result.to_pandas()
        col = _RAGAS_COLUMN.get(key, key)
        for r, (_, row) in zip(rows, df.iterrows()):
            if col in row:
                scores.setdefault(r.seq, {})[key] = round(float(row[col]), 4)
        # Only context_precision produces a per-chunk verdict, and only that
        # verdict makes a rank-based reading possible.
        per_item = getattr(metrics[0], "verdicts", None)
        if per_item:
            for r in rows:
                found = per_item.get((r.user_input, r.reference))
                if found is not None:
                    verdicts[r.seq] = found

    def unscored(rows: list[EvalRunItem], key: str) -> list[EvalRunItem]:
        """Rows without a usable score: never produced, or NaN (judge parse failure)."""
        missing = []
        for r in rows:
            value = scores.get(r.seq, {}).get(key)
            if value is None or value != value:
                missing.append(r)
        return missing

    # Context metrics only on items that actually have contexts; answer metrics
    # on everything that got an answer. One metric per evaluate() call keeps
    # the progress bar moving (metrics share no LLM calls anyway).
    total = len(metric_keys)
    for i, key in enumerate(metric_keys):
        rows = ([r for r in run_items if json.loads(r.contexts)]
                if key in CONTEXT_METRICS else run_items)
        run_metric(rows, key)
        for _ in range(METRIC_RETRIES):
            pending = unscored(rows, key)
            if not pending:
                break
            logger.info("Metric %s: retrying %d unscored item(s)", key, len(pending))
            run_metric(pending, key)
        if report_progress:
            _update(run_id, stage="evaluating", stage_done=i + 1, stage_total=total,
                    progress=round(QUERY_WEIGHT + EVAL_WEIGHT * (i + 1) / total, 1))
    return scores, verdicts


async def recompute_item(run_id: int, seq: int) -> None:
    """Re-score one item of a finished run, and refresh the run's means.

    Exists for the case the run itself cannot fix: a judge call that came back
    unusable. Its raw reply is cached, so the retry inside `_evaluate` replays
    the same bad answer and the item keeps no score — measured at 4 of ~900
    judgements. Nothing else in the product could score that item again short of
    re-running the whole evaluation.

    The retry has to read around that cache: `read_cache=False` still writes, so
    the fresh result replaces the stored one and a later full re-run inherits
    the fix.
    """
    db = get_session_factory()()
    try:
        run = db.get(EvalRun, run_id)
        item = db.execute(
            select(EvalRunItem).where(EvalRunItem.run_id == run_id, EvalRunItem.seq == seq)
        ).scalar_one_or_none()
        if run is None or item is None:
            raise LookupError(f"run {run_id} has no item {seq}")
        llm_cfg = db.get(ModelConfig, run.llm_config_id)
        emb_cfg = db.get(ModelConfig, run.embedding_config_id)
        metric_keys = json.loads(run.metrics)
        # Only the metrics with no score: re-running the ones that already have
        # a value would replace a good number with a fresh sample of the judge's
        # noise, and move the run's mean for reasons nobody asked for.
        current = json.loads(item.scores or "{}")
        missing = [k for k in metric_keys
                   if current.get(k) is None or current.get(k) != current.get(k)]
        # The item goes in regardless; `_evaluate` applies the same per-metric
        # rule it applies in a full run (context metrics need retrieved chunks,
        # answer metrics need an answer), so the two paths cannot drift.
        llm, _raw, _cache = testset_gen._build_llm(
            llm_cfg, JUDGE_MAX_TOKENS,
            _judge_cache_dir(llm_cfg, f"llm{_OUTPUT_BUDGET}") / "llm",
            read_cache=False)
        emb, _store = testset_gen._build_embeddings(
            emb_cfg, run.judge_concurrency,
            cache_dir=_judge_cache_dir(emb_cfg, "emb") / "emb", read_cache=False)
        scores, verdicts = await asyncio.to_thread(
            _evaluate, run.id, [item], missing, llm, emb, run.judge_concurrency,
            False)
        if item.seq in scores:
            current.update(scores[item.seq])
            item.scores = json.dumps(_finite(current), ensure_ascii=False)
        if item.seq in verdicts:
            item.context_verdicts = json.dumps(verdicts[item.seq])
        # The item's score changed, so every mean it feeds has to be redone —
        # otherwise the summary would disagree with the rows under it.
        everything = {}
        for row in db.execute(
            select(EvalRunItem).where(EvalRunItem.run_id == run_id)
        ).scalars().all():
            everything[row.seq] = json.loads(row.scores or "{}")
        run.summary = json.dumps(_summarize(everything, metric_keys), ensure_ascii=False)
        db.commit()
    finally:
        db.close()
    _log(run_id, "itemRescored", {"seq": seq})


async def resume_interrupted() -> None:
    """Restart evaluations that were running when the server stopped."""
    db = get_session_factory()()
    try:
        runs = db.execute(
            select(EvalRun).where(EvalRun.status == EvalRun.STATUS_RUNNING)
        ).scalars().all()
        ids = [r.id for r in runs]
    finally:
        db.close()
    for run_id in ids:
        logger.info("Resuming interrupted evaluation %s", run_id)
        _log(run_id, "autoResumed")
        asyncio.create_task(run_evaluation(run_id))
