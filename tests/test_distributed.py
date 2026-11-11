from transformer_craft.training import distributed_world_size, is_main_process


def test_distributed_helpers_default_to_single_process() -> None:
    assert distributed_world_size() == 1
    assert is_main_process()


def test_distributed_world_size_reads_environment(monkeypatch: object) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    assert distributed_world_size() == 2
    assert not is_main_process()
