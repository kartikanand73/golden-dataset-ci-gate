import pytest

from app.schemas import DatasetError, category_counts, load_dataset
from tests.conftest import GOLDEN, ROOT, case, write_jsonl


def test_golden_dataset_loads_and_validates():
    cases = load_dataset(GOLDEN)
    assert len(cases) == 60
    assert category_counts(cases) == {
        "PASSWORD_RESET": 15, "ACCESS_REQUEST": 15,
        "SECURITY_INCIDENT": 15, "GENERAL_SUPPORT": 15,
    }
    assert any("nuanced" in c.tags for c in cases)
    # Every security incident in the golden set must expect human review.
    assert all(c.expected.requires_human for c in cases
               if c.expected.category.value == "SECURITY_INCIDENT")


def test_development_dataset_is_separate_from_golden():
    dev = load_dataset(ROOT / "data" / "development_dataset.jsonl")
    golden_tickets = {c.ticket for c in load_dataset(GOLDEN)}
    assert 15 <= len(dev) <= 25
    assert not golden_tickets & {c.ticket for c in dev}


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = write_jsonl(tmp_path / "d.jsonl", [case("case-001"), case("case-001")])
    with pytest.raises(DatasetError, match="duplicate case_id 'case-001'"):
        load_dataset(path)


def test_invalid_expected_labels_are_rejected(tmp_path):
    bad = case("case-001")
    bad["expected"]["category"] = "BILLING"
    path = write_jsonl(tmp_path / "d.jsonl", [bad])
    with pytest.raises(DatasetError, match="expected.category"):
        load_dataset(path)


def test_non_boolean_requires_human_is_rejected(tmp_path):
    bad = case("case-001")
    bad["expected"]["requires_human"] = "yes"
    with pytest.raises(DatasetError):
        load_dataset(write_jsonl(tmp_path / "d.jsonl", [bad]))


def test_unknown_fields_are_rejected(tmp_path):
    bad = case("case-001")
    bad["expected"]["notes"] = "extra"
    with pytest.raises(DatasetError):
        load_dataset(write_jsonl(tmp_path / "d.jsonl", [bad]))
