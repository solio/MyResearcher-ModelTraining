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

