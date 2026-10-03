"""Tests for QA text hygiene: normalize_qa_text / is_standalone_question."""

from app.services import testset_gen


def test_cjk_spaces_removed():
    assert (
        testset_gen.normalize_qa_text("香 港 联 合 交 易 所 有 限 公 司 的 网 站 是 什 么")
        == "香港联合交易所有限公司的网站是什么"
    )


def test_full_width_space_between_cjk_removed():
    assert testset_gen.normalize_qa_text("香　港　联 合") == "香港联合"


def test_spaces_around_latin_kept():
    # PDF-extraction style spaces around numbers/English are NOT between two
    # CJK chars and must survive.
    assert (
        testset_gen.normalize_qa_text("香港联合交易所有限公司网站（ www.hkex.com.hk）是官方渠道")
        == "香港联合交易所有限公司网站（ www.hkex.com.hk）是官方渠道"
    )
    # "6月 7日" keeps its space (next to digits); "日 的" is between two CJK
    # chars and is collapsed.
    assert testset_gen.normalize_qa_text("A股 6月 7日 的公告") == "A股 6月 7日的公告"


def test_zero_width_and_control_chars_removed():
    assert testset_gen.normalize_qa_text("什么是​可转债﻿？") == "什么是可转债？"
    assert testset_gen.normalize_qa_text("ab\x00c\x07") == "abc"


def test_question_prefix_stripped():
    assert testset_gen.normalize_qa_text("问题：什么是可转债？") == "什么是可转债？"
    assert testset_gen.normalize_qa_text("Question: What is convertible bond?") == "What is convertible bond?"
    assert testset_gen.normalize_qa_text("Q: 什么是可转债？") == "什么是可转债？"


def test_wrapping_quotes_stripped():
    assert testset_gen.normalize_qa_text("“什么是可转债？”") == "什么是可转债？"
    assert testset_gen.normalize_qa_text('"What is X?"') == "What is X?"
    # Quotes used inside the text are not a wrapping pair — keep them.
    assert testset_gen.normalize_qa_text('“他说"不涨"”意味着什么？') == '“他说"不涨"”意味着什么？'


def test_single_line_collapses_whitespace():
    assert testset_gen.normalize_qa_text("  什么是\n可转债？ \n") == "什么是 可转债？"


def test_multiline_keeps_paragraphs():
    assert testset_gen.normalize_qa_text("第一段。  \n\n\n\n第二段。", multiline=True) == "第一段。\n\n第二段。"


def test_empty_input():
    assert testset_gen.normalize_qa_text("") == ""
    assert testset_gen.normalize_qa_text(None) == ""


def test_standalone_question_filter():
    assert not testset_gen.is_standalone_question("本文提到的公司是谁？")
    assert not testset_gen.is_standalone_question("根据所提供的材料，触发条件是什么？")
    assert not testset_gen.is_standalone_question("这段话说明了什么？")
    assert not testset_gen.is_standalone_question("According to the text, what is X?")
    assert not testset_gen.is_standalone_question("What does the provided document say about X?")
    # Legit standalone questions must survive.
    assert testset_gen.is_standalone_question("这段时间公司业绩如何？")
    assert testset_gen.is_standalone_question("苹果公司2023年的营收是多少？")
    assert testset_gen.is_standalone_question("平银转债的赎回条款是什么？")


def test_strip_hop_markers():
    ctxs = ["<1-hop>\n\n第十三条 经依法登记……", "<2-hop>\n\n二、关联方基本情况", "无前缀的原文"]
    assert testset_gen.strip_hop_markers(ctxs) == [
        "第十三条 经依法登记……",
        "二、关联方基本情况",
        "无前缀的原文",
    ]


def test_sample_quality_ok():
    good = {"user_input": "什么是可转债？", "reference": "一种可转换为股票的债券。"}
    assert testset_gen.sample_quality_ok(good)
    assert not testset_gen.sample_quality_ok({"user_input": "", "reference": "答案"})
    assert not testset_gen.sample_quality_ok({"user_input": "什么是可转债？", "reference": "  "})
    assert not testset_gen.sample_quality_ok({"user_input": "本文说了什么？", "reference": "答案"})
