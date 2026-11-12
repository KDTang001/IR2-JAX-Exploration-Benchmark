"""Batched parity against the unmodified original IR2 sensor implementation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location("original_ir2_sensor", ROOT / "sensor.py")
assert SPEC is not None
assert SPEC.loader is not None
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)


def _oracle(
    truth: np.ndarray, local: np.ndarray, position: np.ndarray, radius: float, shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the unchanged original sensor with its native cell encoding."""
    height, width = shape
    legacy_truth = np.where(truth[:height, :width] == 0, 255, 1)
    legacy_local = np.asarray([255, 1, 127])[local[:height, :width]]
    updated = ORIGINAL.sensor_work(position, radius, legacy_local.copy(), legacy_truth)
    footprint = (
        ORIGINAL.sensor_work(position, radius, np.full_like(legacy_truth, 127), legacy_truth) != 127  # noqa: PLR2004 - original unknown-cell encoding
    )
    expected = local.copy()
    expected[:height, :width] = np.where(updated == 255, 0, np.where(updated == 1, 1, 2))  # noqa: PLR2004 - original map encoding or fixture density
    visible = np.zeros_like(truth, dtype=bool)
    visible[:height, :width] = footprint
    return expected, visible, visible & (truth == 0)


def _check_batch(
    truths: np.ndarray, positions: np.ndarray, radius: float, shapes: np.ndarray
) -> None:
    """Compare all result fields under a compiled batch transformation."""
    jax = pytest.importorskip("jax")
    from ir2_benchmark.sensing.ir2_sensor import sense_ir2  # noqa: PLC0415

    rng = np.random.default_rng(187)
    # Existing knowledge must survive outside the current physical footprint.
    locals_ = np.where(rng.random(truths.shape) < 0.2, truths, 2).astype(np.uint8)  # noqa: PLR2004 - original map encoding or fixture density
    expected = [
        _oracle(world, local, pose, radius, shape)
        for world, local, pose, shape in zip(truths, locals_, positions, shapes, strict=True)
    ]
    before = locals_.copy()

    def sense(
        world: jax.Array, local: jax.Array, pose: jax.Array, shape: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        return sense_ir2(world, local, pose, radius, shape)

    actual = jax.jit(jax.vmap(sense))(truths, locals_, positions, shapes)
    for index, field in enumerate(actual):
        np.testing.assert_array_equal(field, np.stack([value[index] for value in expected]))
    np.testing.assert_array_equal(locals_, before)


@pytest.mark.parametrize("radius", [0, 1, 3, 17, 40])
def test_batched_empty_walls_corners_diagonals_and_borders(radius: int) -> None:
    truths = np.zeros((8, 39, 47), dtype=np.uint8)
    truths[1, :, 20] = 1  # Thin wall: the free space behind it stays unknown.
    truths[2, :, 20:34] = 1  # Thick wall: stop before the tenth occupied cell.
    truths[3, 19:, 20] = 1
    truths[3, 19, 20:] = 1
    np.fill_diagonal(truths[4], 1)
    truths[5, 0, :] = 1
    truths[6, :, -1] = 1
    truths[7] = (np.random.default_rng(52).random((39, 47)) < 0.15).astype(np.uint8)  # noqa: PLR2004 - original map encoding or fixture density
    positions = np.array(
        [[23, 19], [15, 19], [15, 19], [19, 18], [18, 19], [0, 0], [46, 38], [2, 3]],
        dtype=np.int32,
    )
    _check_batch(truths, positions, radius, np.tile([39, 47], (8, 1)))


def test_batched_dynamic_valid_shapes_stop_at_unpadded_borders() -> None:
    truths = np.ones((4, 41, 49), dtype=np.uint8)
    shapes = np.array([[17, 23], [31, 39], [41, 49], [2, 3]], dtype=np.int32)
    positions = np.array([[0, 0], [38, 30], [24, 20], [2, 1]], dtype=np.int32)
    for world, (height, width) in zip(truths, shapes, strict=True):
        world[:height, :width] = 0
        world[height // 2, : width // 2] = 1
    _check_batch(truths, positions, 80, shapes)


def test_batched_native_map_radius_80() -> None:
    from PIL import Image  # noqa: PLC0415

    paths = sorted((ROOT / "DungeonMaps" / "test" / "corridor").glob("*.png"))
    if not paths:
        pytest.skip("Native DungeonMaps fixture is unavailable")
    with Image.open(paths[0]) as source:
        truth = (np.asarray(source.convert("L")) <= 150).astype(np.uint8)  # noqa: PLR2004 - original map encoding or fixture density
    free = np.argwhere(truth == 0)
    assert len(free) > 0
    positions = free[np.linspace(0, len(free) - 1, 4, dtype=int)][:, ::-1].astype(np.int32)
    _check_batch(np.repeat(truth[None], 4, axis=0), positions, 80, np.tile(truth.shape, (4, 1)))
