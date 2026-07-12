import pytest

from llm_finetune.schema import QAExample, SchemaError, load_jsonl


def test_from_raw_valid():
    ex = QAExample.from_raw(
        {"id": " a1 ", "question": " q? ", "answer": " a ", "context": " c "}
    )
    assert ex.id == "a1"
    assert ex.question == "q?"
    assert ex.answer == "a"
    assert ex.context == "c"


def test_from_raw_rejects_empty_answer():
    with pytest.raises(SchemaError):
        QAExample.from_raw({"id": "a1", "question": "q", "answer": "  "})


def test_from_raw_rejects_missing_field():
    with pytest.raises(SchemaError):
        QAExample.from_raw({"id": "a1", "question": "q"})


def test_to_chat_includes_context_and_roles():
    ex = QAExample(id="a1", question="What?", answer="This.", context="Some ctx.")
    chat = ex.to_chat()
    roles = [m["role"] for m in chat]
    assert roles == ["system", "user", "assistant"]
    assert "Some ctx." in chat[1]["content"]
    assert chat[2]["content"] == "This."


def test_load_sample_dataset_is_valid_and_unique():
    examples = load_jsonl("data/sample/domain_qa.jsonl")
    assert len(examples) == 20
    assert len({e.id for e in examples}) == 20


def test_load_jsonl_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "dup.jsonl"
    path.write_text(
        '{"id": "x", "question": "q", "answer": "a"}\n'
        '{"id": "x", "question": "q2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    with pytest.raises(SchemaError):
        load_jsonl(str(path))
