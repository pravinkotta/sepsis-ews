import pytest

from src.generate import GenerationError, OpenAIClient, draft_sbar

CHUNKS = [
    {
        "citation_id": "S1",
        "doc": "ssc-2021",
        "title": "SSC 2021",
        "text": "Sepsis is life-threatening organ dysfunction; measure lactate.",
        "source": "test",
    },
    {
        "citation_id": "S2",
        "doc": "local-escalation-policy",
        "title": "Local policy",
        "text": "SBAR escalation to rapid response requires a human approval token.",
        "source": "test",
    },
]

PATIENT = {
    "patient_id": "p1",
    "as_of_hour": 12,
    "risk": 0.42,
    "threshold": 0.0062,
    "drivers": [{"feature": "Lactate_last", "value": 3.1}],
    "facts": {"HR": 112, "Lactate": 3.1, "MAP": 61},
}


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_extractive_draft_is_cited_and_valid():
    draft = draft_sbar(PATIENT, CHUNKS)
    assert draft.generator == "extractive"
    assert draft.validation.status == "valid"
    text = " ".join(draft.sbar.values())
    assert "[S1]" in text
    assert "[S9]" not in text


def test_llm_draft_with_valid_json():
    response = (
        '{"situation": "Risk is elevated.", '
        '"background": "Lactate 3.1 [S1].", '
        '"assessment": "Concerning for evolving sepsis [S1].", '
        '"recommendation": "Consider clinician review [S2]."}'
    )
    draft = draft_sbar(PATIENT, CHUNKS, client=FakeClient([response]))
    assert draft.generator == "llm"
    assert draft.validation.status == "valid"


def test_llm_forged_citation_is_rejected_then_raises():
    response = (
        '{"situation": "Risk is elevated.", '
        '"background": "Lactate 3.1 [S9].", '
        '"assessment": "Concerning for evolving sepsis [S9].", '
        '"recommendation": "Consider review [S9]."}'
    )
    with pytest.raises(GenerationError, match="could not produce a valid draft"):
        draft_sbar(PATIENT, CHUNKS, client=FakeClient([response, response]), max_repairs=1)


def test_llm_invalid_json_repairs_once_then_succeeds():
    invalid = "not json"
    valid = (
        '{"situation": "Risk is elevated.", '
        '"background": "Lactate 3.1 [S1].", '
        '"assessment": "Concerning for evolving sepsis [S1].", '
        '"recommendation": "Consider review [S2]."}'
    )
    client = FakeClient([invalid, valid])
    draft = draft_sbar(PATIENT, CHUNKS, client=client, max_repairs=1)
    assert draft.validation.status == "valid"
    assert "previous response was invalid" in client.prompts[1]


def test_no_key_fails_loudly(monkeypatch):
    # Do not let the developer's real `.env` satisfy this test.
    monkeypatch.setattr("src.generate.load_env", lambda: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(GenerationError, match="OPENAI_API_KEY"):
        OpenAIClient()
