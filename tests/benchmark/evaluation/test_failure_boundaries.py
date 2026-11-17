"""Allocator ValueErrors survive the model and environment measurement boundaries."""

from types import SimpleNamespace

import pytest

from ir2_benchmark.config import HostConfig
from ir2_benchmark.evaluation import episode, throughput
from ir2_benchmark.evaluation.errors import ModelExecutionError


@pytest.mark.parametrize("oom", [True, False])
def test_model_boundary_preserves_allocator_error(
    monkeypatch: pytest.MonkeyPatch,
    oom: bool,  # noqa: FBT001 - pytest supplies this parameter
) -> None:
    error = ValueError("RESOURCE_EXHAUSTED: device allocation" if oom else "bad model shape")
    lane = SimpleNamespace(prepared=SimpleNamespace(group="same", model=object()))

    def fail(_prepared: object) -> None:
        raise error

    monkeypatch.setattr(episode, "evaluation_kernel", fail)
    with pytest.raises(ValueError if oom else ModelExecutionError) as caught:
        episode.execute_batch((lane,))
    if oom:
        assert caught.value is error
    else:
        assert caught.value.__cause__ is error


@pytest.mark.parametrize("oom", [True, False])
def test_throughput_boundary_classifies_only_allocator_valueerror(
    monkeypatch: pytest.MonkeyPatch,
    oom: bool,  # noqa: FBT001 - pytest supplies this parameter
) -> None:
    error = ValueError("RESOURCE_EXHAUSTED: device allocation" if oom else "bad input shape")
    data = SimpleNamespace(occupancy=SimpleNamespace(shape=(8, 8)))

    def fail(*_args: object, **_kwargs: object) -> None:
        raise error

    monkeypatch.setattr(throughput, "measure", fail)
    if oom:
        result = throughput.benchmark_environment(data, HostConfig(), batches=(1,))
        assert result[0].status == "oom"
        assert result[0].error == str(error)
    else:
        with pytest.raises(ValueError, match="bad input shape"):
            throughput.benchmark_environment(data, HostConfig(), batches=(1,))
