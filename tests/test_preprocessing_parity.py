from semantic_model.hashes import content_addressed_id
from semantic_model.preprocessing import (
    PreprocessingContract,
    build_model_input,
    build_model_inputs,
)


def contract():
    raw = {
        "contract_version": "test-preprocessing-v1",
        "status": "TEST_ONLY",
        "include_board_context": True,
        "board_marker": "[BOARD]",
        "text_marker": "[TEXT]",
        "separator": "\n",
        "normalize_unicode": "NFC",
        "strip_outer_whitespace": True,
    }
    raw["preprocessing_contract_id"] = content_addressed_id(
        raw, omit_keys={"preprocessing_contract_id"}
    )
    return PreprocessingContract.from_mapping(raw)


def test_prepare_train_and_infer_use_identical_model_input(canonical_input):
    preprocessing = contract()
    expected = "[BOARD] 601012 隆基绿能\n[TEXT] 我会继续持有，因为订单增长。"
    assert build_model_input(canonical_input, preprocessing) == expected
    assert build_model_inputs([canonical_input], preprocessing) == [expected]


def test_missing_explicit_board_context_uses_only_local_stock_fields(canonical_input, clone):
    record = clone(canonical_input)
    del record["board_context"]
    assert build_model_input(record, contract()).startswith("[BOARD] 601012 隆基绿能\n")


def test_native_v035_template_applies_no_hidden_normalization(canonical_input, clone):
    raw = {
        "schema_version": "semantic-preprocessing-contract-v0.3.5",
        "normalize_text": {
            "exact_template": "[股票]{stock_code} {stock_name} [帖子]{model_text}",
            "normalizations_applied": [],
            "lowercase": False,
            "unicode_normalization": None,
            "whitespace_collapse": False,
            "url_masking": False,
            "emoji_removal": False,
            "traditional_simplified_conversion": False,
            "truncation": None,
        },
        "feature_stack_order": ["char_tfidf", "word_tfidf"],
        "char_tfidf": {"analyzer": "char"},
        "word_tfidf": {"analyzer": "word"},
        "expected_fitted_feature_counts": {"char": 1, "word": 1, "total": 2},
    }
    record = clone(canonical_input)
    record["model_text"] = "  MiXeD  空格  "
    preprocessing = PreprocessingContract.from_mapping(raw)
    assert build_model_input(record, preprocessing) == (
        "[股票]601012 隆基绿能 [帖子]  MiXeD  空格  "
    )
