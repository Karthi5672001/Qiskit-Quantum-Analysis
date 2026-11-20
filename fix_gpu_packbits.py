# Replace CuPy's unsupported axis-based packbits call.
#
# CuPy 14.2 does not support cupy.packbits(..., axis=1). This script
# replaces gpu_pack_detector_bits with an explicit row-wise CUDA
# implementation using padding, reshaping, bit weights and reduction.


from __future__ import annotations

import re
from pathlib import Path


PROJECT_ROOT = Path(
    r"C:\Users\Karthi\Qauntum"
).resolve()

GPU_MODULE_PATH = (
    PROJECT_ROOT
    / "src"
    / "quantum_qec"
    / "gpu.py"
)


REPLACEMENT_FUNCTION = '''def gpu_pack_detector_bits(
    detection_events: np.ndarray,
) -> np.ndarray:
    """Pack detector bits along each shot using CuPy.

    CuPy 14.2 does not support the ``axis`` argument of
    ``cupy.packbits``. This implementation therefore packs each
    detector row explicitly on the GPU.

    Detectors are padded with zeros to a multiple of eight. Each
    consecutive group of eight detector values is converted into one
    byte using little-endian bit order, matching:

        numpy.packbits(events, axis=1, bitorder="little")
    """

    cupy_module = require_cupy()

    events = validate_detector_events(
        detection_events
    )

    gpu_events = cupy_module.asarray(
        events,
        dtype=cupy_module.uint8,
    )

    shot_count = int(
        gpu_events.shape[0]
    )

    detector_count = int(
        gpu_events.shape[1]
    )

    padding_bits = (
        -detector_count
    ) % 8

    if padding_bits:
        gpu_events = cupy_module.pad(
            gpu_events,
            pad_width=(
                (0, 0),
                (0, padding_bits),
            ),
            mode="constant",
            constant_values=0,
        )

    packed_width = int(
        gpu_events.shape[1] // 8
    )

    grouped_bits = gpu_events.reshape(
        shot_count,
        packed_width,
        8,
    )

    bit_weights = cupy_module.asarray(
        [
            1,
            2,
            4,
            8,
            16,
            32,
            64,
            128,
        ],
        dtype=cupy_module.uint16,
    )

    gpu_packed = (
        grouped_bits.astype(
            cupy_module.uint16,
            copy=False,
        )
        * bit_weights
    ).sum(
        axis=2,
        dtype=cupy_module.uint16,
    ).astype(
        cupy_module.uint8,
        copy=False,
    )

    return cupy_module.asnumpy(
        gpu_packed
    )
'''

# Replace the function and preserve the rest of gpu.py
def main() -> None:


    if not GPU_MODULE_PATH.exists():
        raise FileNotFoundError(
            f"GPU module not found: {GPU_MODULE_PATH}"
        )

    source = GPU_MODULE_PATH.read_text(
        encoding="utf-8-sig"
    )

    pattern = re.compile(
        r"def gpu_pack_detector_bits\("
        r".*?"
        r"(?=\ndef _median_elapsed_seconds\()",
        flags=re.DOTALL,
    )

    match = pattern.search(
        source
    )

    if match is None:
        raise RuntimeError(
            "Could not locate gpu_pack_detector_bits() "
            "before _median_elapsed_seconds()."
        )

    updated_source = pattern.sub(
        REPLACEMENT_FUNCTION.rstrip(),
        source,
        count=1,
    )

    GPU_MODULE_PATH.write_text(
        updated_source,
        encoding="utf-8",
        newline="\n",
    )

    print(
        f"Updated: {GPU_MODULE_PATH}"
    )

    print(
        "Replaced cupy.packbits(axis=1) with "
        "explicit little-endian row packing."
    )


if __name__ == "__main__":
    main()