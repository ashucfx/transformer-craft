import torch

from transformer_craft.data import ByteTokenizer, SequencePairDataset


def test_sequence_pair_dataset_builds_teacher_forcing_tensors() -> None:
    tokenizer = ByteTokenizer()
    dataset = SequencePairDataset.from_text_pairs(
        [("input", "output")], tokenizer, sequence_length=8
    )
    item = dataset[0]
    assert item["encoder_input_ids"].shape == (8,)
    assert item["decoder_input_ids"][0] == tokenizer.special.bos
    assert item["labels"][0] == tokenizer.encode("output")[0]
    assert item["labels"][-1] == -100
    assert item["encoder_attention_mask"].dtype == torch.bool


def test_sequence_pair_dataset_masks_target_padding() -> None:
    dataset = SequencePairDataset([([1, 2], [3])], sequence_length=4)
    item = dataset[0]
    assert item["decoder_input_ids"].tolist() == [258, 3, 256, 256]
    assert item["labels"].tolist() == [3, 259, -100, -100]
    assert item["decoder_attention_mask"].tolist() == [True, True, False, False]
