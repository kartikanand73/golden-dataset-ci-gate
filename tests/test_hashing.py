from app.hashing import canonical_json, sha256_bytes, sha256_file, sha256_json
from app.schemas import load_model_config
from tests.conftest import CONFIG, GOLDEN, ROOT


def test_sha256_known_vector():
    assert sha256_bytes(b"abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_dataset_and_prompt_hashes_are_stable():
    assert sha256_file(GOLDEN) == sha256_file(GOLDEN)
    prompt = ROOT / "config" / "prompt.txt"
    assert sha256_file(prompt) == sha256_bytes(prompt.read_bytes())


def test_config_hash_ignores_key_order(tmp_path):
    reordered = tmp_path / "model.yaml"
    reordered.write_text("max_retries: 2\ntimeout_seconds: 30\nmodel: mock-v1-baseline\n"
                         "temperature: 0\nprovider: mock\n", encoding="utf-8", newline="\n")
    a = sha256_json(load_model_config(CONFIG).model_dump(mode="json"))
    b = sha256_json(load_model_config(reordered).model_dump(mode="json"))
    assert a == b


def test_config_hash_changes_when_model_changes():
    config = load_model_config(CONFIG)
    changed = config.model_copy(update={"model": "mock-v2-regression"})
    assert sha256_json(config.model_dump()) != sha256_json(changed.model_dump())


def test_canonical_json_is_order_independent():
    assert canonical_json({"b": 1, "a": [1, 2]}) == canonical_json({"a": [1, 2], "b": 1})


def test_dataset_hash_changes_when_one_label_changes(tmp_path):
    # Bytes, not text: text mode would translate newlines on Windows.
    copy = tmp_path / "golden.jsonl"
    data = GOLDEN.read_bytes()
    copy.write_bytes(data)
    assert sha256_file(copy) == sha256_file(GOLDEN)
    copy.write_bytes(data.replace(b'"requires_human": true', b'"requires_human": false', 1))
    assert sha256_file(copy) != sha256_file(GOLDEN)
