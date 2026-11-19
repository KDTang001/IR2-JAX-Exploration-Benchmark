"""Host-side scenario preparation shared by CLI smoke and training entry points."""

from pathlib import Path

from ir2_benchmark.config import HostConfig
from ir2_benchmark.env.reset import from_prepared_map
from ir2_benchmark.maps.generate import generate_map
from ir2_benchmark.maps.prepare import load_ir2_map, prepare_map
from ir2_benchmark.types import MapInput


def load_scenario(config: HostConfig, seed: int, map_path: Path | None = None) -> MapInput:
    """Load an explicit IR2 map, or construct the documented64² development scenario."""
    if map_path is None:
        data = generate_map(
            "rooms",
            seed,
            size=64,
            n_agents=config.n_agents,
            width_range=(2, 4),
            room_scale_range=(12, 24),
        ).data
    else:
        data = load_ir2_map(map_path)
    return from_prepared_map(prepare_map(data, buckets=(64, 512, 768, 1024)))
