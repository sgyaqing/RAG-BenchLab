"""Every log entry and stage the backend emits must have a frontend i18n string.

A missing log key renders as the raw key in the UI (e.g. "evaluation.logTokens");
a missing stage key falls back to the generic "处理中…", which tells the customer
nothing about what is running. Both are invisible to backend tests — this
cross-checks the two sides.

The stage scan reads the literals the module writes. Stage values handed in
through _build_kg/_expand_kg are defaulted there, so they are covered by their
defaults; a value that only ever arrives as a call-site argument would slip
past — that is what this test would have caught on the day one did.
"""

import re
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
LOCALES = BACKEND.parent / "frontend" / "src" / "i18n" / "locales"

# backend module -> the i18n section its log keys live in
SOURCES = {
    "app/services/eval_run.py": "evaluation",
    "app/services/testset_gen.py": "testset",
    "app/services/adapter_assist.py": "ragSystem",
}


def _section_keys(section: str, locale: str) -> set[str]:
    text = (LOCALES / f"{locale}.ts").read_text(encoding="utf-8")
    start = text.index(f"  {section}: {{")
    end = text.index("\n  },", start)
    return set(re.findall(r"^\s*(\w+):", text[start:end], re.M))


def _emitted_keys(rel_path: str) -> set[str]:
    src = (BACKEND / rel_path).read_text(encoding="utf-8")
    return set(re.findall(r'_log\([^,]+,\s*"(\w+)"', src))


@pytest.mark.parametrize("module,section", SOURCES.items())
@pytest.mark.parametrize("locale", ["zh", "en"])
def test_every_emitted_log_key_has_copy(module, section, locale):
    keys = _section_keys(section, locale)
    missing = []
    for key in _emitted_keys(module):
        i18n_key = f"log{key[0].upper()}{key[1:]}"
        # failure entries resolve a code-specific variant first (logFailed_xxx)
        if i18n_key not in keys and f"{i18n_key}_" not in " ".join(keys):
            missing.append(i18n_key)
    assert not missing, f"{module} emits log keys with no {locale} copy: {sorted(missing)}"


# --- stages -----------------------------------------------------------------
#
# Collapsing the check-and-replenish loop into one label deleted the copy for
# fallback_expand_*, but the merge step still emitted it, so the bar read
# "处理中…" for the whole merge. Nothing failed: the backend wrote a stage, the
# frontend could not name it, and the fallback is silent by design.

def _emitted_stages(rel_path: str) -> set[str]:
    import app.services.testset_gen as tg

    src = (BACKEND / rel_path).read_text(encoding="utf-8")
    codes = set(re.findall(r'stage=(?:f)?"([a-z_]+)"', src))
    codes |= {m + "X" for m in re.findall(r'stage=f"([a-z_]+)\{', src)}
    out: set[str] = set()
    for code in codes:
        if code.endswith("X"):
            out |= {code[:-1] + t for t in tg.QUESTION_TYPES}
        else:
            out.add(code)
    return out


@pytest.mark.parametrize("locale", ["zh", "en"])
def test_every_emitted_stage_has_copy(locale):
    keys = _section_keys("testset", locale)
    missing = []
    for code in sorted(_emitted_stages("app/services/testset_gen.py")):
        i18n_key = "stage" + "".join(w.capitalize() for w in code.split("_"))
        if i18n_key not in keys:
            missing.append(f"{code} -> {i18n_key}")
    assert not missing, f"testset_gen emits stages with no {locale} copy: {missing}"
