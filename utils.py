from collections import Counter

import numpy as np
from PIL import Image


def detect_scale_from_signal_parametric(
    signal: np.ndarray,
    std_multiplier: float = 1.5,
    min_peak_distance: int = 2,
    median_tolerance: int = 2,
    median_confidence_ratio: float = 0.7,
) -> int:

    if len(signal) < 3:
        return 1
    mean_val, std_val = np.mean(signal), np.std(signal)
    if std_val == 0:
        return 1

    threshold = mean_val + std_multiplier * std_val
    peaks: list[int] = []
    for i in range(1, len(signal) - 1):
        if (
            signal[i] > threshold
            and signal[i] > signal[i - 1]
            and signal[i] > signal[i + 1]
        ):
            if not peaks or i - peaks[-1] > min_peak_distance:
                peaks.append(i)

    if len(peaks) <= 2:
        return 1

    spacings = [peaks[i + 1] - peaks[i] for i in range(len(peaks) - 1)]
    if not spacings:
        return 1
    median_spacing = int(np.median(spacings))

    close_spacings = [
        s for s in spacings if abs(s - median_spacing) <= median_tolerance
    ]

    if len(close_spacings) / len(spacings) > median_confidence_ratio:
        return max(1, median_spacing)

    return max(1, Counter(spacings).most_common(1)[0][0])


def runs_based_detect_multipass(image: np.ndarray) -> int:

    h, w, c = image.shape
    image_rgb = image[:, :, :3] if c == 4 else image

    diff_x = np.diff(image_rgb.astype(np.int32), axis=1)
    signal_x = np.sum(np.any(diff_x != 0, axis=2), axis=0)
    diff_y = np.diff(image_rgb.astype(np.int32), axis=0)
    signal_y = np.sum(np.any(diff_y != 0, axis=2), axis=1)

    param_sets = [
        {"name": "Default", "std_multiplier": 1.5, "median_tolerance": 2},
        {"name": "Tolerant", "std_multiplier": 0.8, "median_tolerance": 3},
    ]

    for params in param_sets:
        print(f"Trying to use “{params['name']}”...")
        scale_x = detect_scale_from_signal_parametric(
            signal_x, **{k: v for k, v in params.items() if k != "name"}
        )
        scale_y = detect_scale_from_signal_parametric(
            signal_y, **{k: v for k, v in params.items() if k != "name"}
        )

        valid_scales = [s for s in (scale_x, scale_y) if s > 1]
        if valid_scales:
            return min(valid_scales)

    return 1


def downscale_by_dominant_color(image: np.ndarray, scale: int) -> np.ndarray:

    h, w, c = image.shape
    new_h, new_w = h // scale, w // scale
    downscaled_image = np.zeros((new_h, new_w, c), dtype=np.uint8)
    for y in range(new_h):
        for x in range(new_w):
            block = image[y * scale : (y + 1) * scale, x * scale : (x + 1) * scale]
            pixels_in_block = block.reshape(-1, c)
            if c == 4:
                opaque_pixels = pixels_in_block[pixels_in_block[:, 3] > 128]
                if len(opaque_pixels) == 0:
                    downscaled_image[y, x] = [0, 0, 0, 0]  # Block is fully transparent
                    continue
                # Find dominant color among opaque pixels
                colors, counts = np.unique(
                    opaque_pixels[:, :3], axis=0, return_counts=True
                )
                dominant_color = colors[counts.argmax()]
                downscaled_image[y, x] = [
                    *dominant_color,
                    255,
                ]  # Set new pixel as fully opaque
            else:  # RGB image
                colors, counts = np.unique(pixels_in_block, axis=0, return_counts=True)
                dominant_color = colors[counts.argmax()]
                downscaled_image[y, x] = dominant_color
    return downscaled_image


def force_detect_scale(image: np.ndarray, max_test_scale: int = 16) -> int:
    # (Code identical to previous version, omitted for brevity)
    h, w, c = image.shape
    image_rgb = image[:, :, :3] if c == 4 else image
    best_scale, min_error = 1, float("inf")
    print("Starting fallback: brute-force testing...")
    for scale in range(2, max_test_scale + 1):
        if h % scale != 0 or w % scale != 0:
            continue
        downscaled = downscale_by_dominant_color(image_rgb, scale)
        reconstructed_arr = np.array(
            Image.fromarray(downscaled).resize((w, h), Image.NEAREST)
        )
        error = np.sum(
            (image_rgb.astype("float") - reconstructed_arr.astype("float")) ** 2
        ) / float(h * w)
        print(f"  Testing scale={scale}, error={error:.2f}")
        if error < min_error:
            min_error, best_scale = error, scale
    return best_scale


def unfake_scaledown_detect_pipeline(
    image: np.ndarray, force_fallback_threshold: int = 256
) -> tuple[np.ndarray, int]:
    """
    Three-stage detection pipeline with full RGBA support.
    """
    detected_scale = runs_based_detect_multipass(image)

    h, w, _ = image.shape
    if detected_scale == 1 and (
        h > force_fallback_threshold or w > force_fallback_threshold
    ):
        detected_scale = force_detect_scale(image)

    if detected_scale > 1:
        downscaled_image = downscale_by_dominant_color(image, detected_scale)
    else:
        downscaled_image = image.copy()

    return downscaled_image, detected_scale
