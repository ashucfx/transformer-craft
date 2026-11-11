import torch

from transformer_craft.data import ByteTokenizer, CodeDataset, MaskedDataset


def test_byte_tokenizer_round_trip() -> None:
    tokenizer = ByteTokenizer()
    text = "def greet():\n    return 'hello'\n"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_code_dataset_pads_last_window() -> None:
    dataset = CodeDataset([1, 2, 3], sequence_length=4)
    item = dataset[0]
    assert torch.equal(item["input_ids"], torch.tensor([1, 2, 3, 256]))
    assert torch.equal(item["attention_mask"], torch.tensor([True, True, True, False]))


def test_masked_dataset_is_deterministic_and_never_masks_padding() -> None:
    dataset = MaskedDataset([1, 2, 3], sequence_length=4, probability=1.0, seed=7)
    first, second = dataset[0], dataset[0]
    assert torch.equal(first["input_ids"], second["input_ids"])
    assert first["input_ids"][-1] == 256
    assert first["labels"][-1] == -100


def test_dataset_split_is_deterministic_and_non_empty() -> None:
    dataset = CodeDataset(list(range(20)), sequence_length=2)
    training, validation = dataset.split(0.25, seed=11)
    repeat_training, repeat_validation = dataset.split(0.25, seed=11)
    assert len(training) == 8
    assert len(validation) == 2
    assert training.windows == repeat_training.windows
    assert validation.windows == repeat_validation.windows


def test_masked_dataset_split_preserves_masking_configuration() -> None:
    dataset = MaskedDataset(list(range(20)), sequence_length=2, probability=0.4, seed=9)
    training, validation = dataset.split(0.25)
    assert training.probability == validation.probability == 0.4
    assert training.seed == validation.seed == 9
