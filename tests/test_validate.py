from src.validate import validate_sbar

RETRIEVED = {"S1", "S2", "S3"}


def good_sbar():
    return {
        "situation": "p1 at hour 12: HR 112, MAP 61 [S1].",
        "background": "Lactate 3.1 mmol/L; last cultures pending [S1].",
        "assessment": "Concerning for evolving sepsis [S1].",
        "recommendation": "Consider escalation to rapid response per bundle [S2].",
    }


def test_clean_sbar_is_valid():
    result = validate_sbar(good_sbar(), retrieved_ids=RETRIEVED)
    assert result.status == "valid"
    assert result.flags == []


def test_forged_citation_is_rejected():
    sbar = good_sbar()
    sbar["assessment"] = "Definite sepsis [S9]."
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "rejected"
    assert "citation_forgery:S9" in result.flags


def test_missing_section_is_rejected():
    sbar = good_sbar()
    del sbar["background"]
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "rejected"
    assert "missing_section:background" in result.flags


def test_treatment_order_is_flagged():
    sbar = good_sbar()
    sbar["recommendation"] = "Start antibiotics now [S1]."
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "needs_review"
    assert "treatment_order" in result.flags


def test_uncited_assessment_is_flagged():
    sbar = good_sbar()
    sbar["assessment"] = "Concerning for evolving sepsis."
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "needs_review"
    assert "uncited_claim:assessment" in result.flags


def test_finalization_language_is_flagged():
    sbar = good_sbar()
    sbar["assessment"] = "Sepsis is confirmed [S1]."
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "needs_review"
    assert "finalization_language" in result.flags


def test_citation_ids_are_case_insensitive():
    sbar = good_sbar()
    sbar["assessment"] = "Concerning for evolving sepsis [s1]."
    result = validate_sbar(sbar, retrieved_ids=RETRIEVED)
    assert result.status == "valid"
