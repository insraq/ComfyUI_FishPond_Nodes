import cv2
import numpy as np
import torch
from PIL import Image


## DataType Conversion Functions
def tensor2pil(image):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def tensor2ndarray(image):
    return np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)


def ndarray2tensor(image):
    return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)


# --------------------
# Global Parameters
# --------------------
BORDER_WIDTH = 3
MIN_BG_RATIO = 0.01  # Allow very little background (subject fills the frame)
MIN_FG_RATIO = 0.03  # Allow very small foreground (tiny objects/particles)


def bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    """BGR → Lab"""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)


def estimate_bg_color_from_border(
    img_lab: np.ndarray, border_width: int = BORDER_WIDTH
):

    h, w = img_lab.shape[:2]
    top = img_lab[0:border_width, :, :]
    bottom = img_lab[h - border_width : h, :, :]
    left = img_lab[:, 0:border_width, :]
    right = img_lab[:, w - border_width : w, :]

    border_pixels = np.concatenate(
        [
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)

    # Background color: median is more robust to noise than mean
    bg_color = np.median(border_pixels, axis=0)

    # Calculate the distance of all border pixels to this color
    diff = border_pixels - bg_color[None, :]
    dist = np.linalg.norm(diff, axis=1)

    inlier_thresh = 15.0
    inliers = dist < inlier_thresh
    inlier_ratio = np.mean(inliers)

    print(
        f"[estimate_bg_color] inlier_ratio={inlier_ratio:.3f} (thresh={inlier_thresh})"
    )

    if inlier_ratio < 0.42:  # At least 42% of the border should be background
        print("[estimate_bg_color] inlier ratio too low, reject.")
        return bg_color.astype(np.float32), False, 10.0

    # 3. Adaptive threshold based only on inliers statistics
    valid_dist = dist[inliers]
    max_dist = float(valid_dist.max())
    p95 = float(np.percentile(valid_dist, 95))

    print(f"[estimate_bg_color] inliers stats: max={max_dist:.3f}, p95={p95:.3f}")

    suggested_thresh = max(p95 * 1.2, max_dist + 1.0)
    suggested_thresh = float(np.clip(suggested_thresh, 3.0, 20.0))
    is_uniform = True

    return bg_color.astype(np.float32), is_uniform, suggested_thresh


def flood_background_from_edges(
    img_lab: np.ndarray,
    bg_color: np.ndarray,
    color_thresh: float = 10.0,
    hole_ratio_threshold: float = 0.18,
) -> np.ndarray:

    h, w = img_lab.shape[:2]
    diff = np.linalg.norm(img_lab - bg_color, axis=2)
    binary_mask = (diff <= color_thresh).astype(np.uint8)
    padded_mask = np.pad(binary_mask, pad_width=1, mode="constant", constant_values=1)
    cv2.floodFill(padded_mask, None, seedPoint=(0, 0), newVal=2)
    outer_bg_mask = (padded_mask[1:-1, 1:-1] == 2).astype(np.uint8)
    confirmed_bg_diffs = diff[outer_bg_mask == 1]

    if confirmed_bg_diffs.size > 0:
        bg_mean = np.mean(confirmed_bg_diffs)
        bg_std = np.std(confirmed_bg_diffs)
        adaptive_thresh = bg_mean + 2.0 * bg_std
        strict_thresh = float(np.clip(adaptive_thresh, 1.0, color_thresh))

        print(
            f"[flood_background] Adaptive Stats - Mean: {bg_mean:.2f}, Std: {bg_std:.2f} -> Strict Thresh: {strict_thresh:.2f}"
        )
    else:
        strict_thresh = color_thresh * 0.8
        print(
            f"[flood_background] Flood failed, using fallback thresh: {strict_thresh:.2f}"
        )

    internal_holes_mask = (diff <= strict_thresh) & (outer_bg_mask == 0)

    num_holes = np.sum(internal_holes_mask)
    num_potential_fg = np.sum(outer_bg_mask == 0)

    if num_potential_fg > 0:
        hole_ratio = num_holes / num_potential_fg
        print(
            f"[flood_background] Hole ratio: {hole_ratio:.3f} (Threshold: {hole_ratio_threshold})"
        )

        if hole_ratio < hole_ratio_threshold:
            print(
                f"[flood_background] Hole ratio {hole_ratio:.3f} < {hole_ratio_threshold}, ignoring internal holes."
            )
            internal_holes_mask[:] = False

    final_bg_mask = outer_bg_mask | internal_holes_mask.astype(np.uint8)

    print(
        f"[flood_background] Final - Outer: {int(outer_bg_mask.sum())}, Holes: {int(internal_holes_mask.sum())}"
    )
    return final_bg_mask


def quality_check_masks(
    bg_mask: np.ndarray,
    min_bg_ratio: float = MIN_BG_RATIO,
    min_fg_ratio: float = MIN_FG_RATIO,
    return_reason: bool = False,
):

    h, w = bg_mask.shape
    total = h * w
    bg_area = int(bg_mask.sum())
    fg_area = total - bg_area

    bg_ratio = bg_area / total
    fg_ratio = fg_area / total

    print(f"[quality_check] bg_ratio={bg_ratio:.3f}, fg_ratio={fg_ratio:.3f}")

    if bg_ratio < min_bg_ratio:
        print("[quality_check] Background ratio too small, check failed")
        if return_reason:
            return False, "bg_too_small"
        return False

    if fg_ratio < min_fg_ratio:
        print("[quality_check] Foreground ratio too small, check failed")
        if return_reason:
            return False, "fg_too_small"
        return False

    # Check if the bounding box of the foreground region touches the edges
    fg_mask = (bg_mask == 0).astype(np.uint8)
    ys, xs = np.where(fg_mask > 0)
    if len(xs) == 0:
        print("[quality_check] No foreground pixels detected")
        if return_reason:
            return False, "no_fg"
        return False

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    if return_reason:
        return True, "ok"
    return True


def flood_with_adaptive_thresh(
    img_lab: np.ndarray,
    bg_color: np.ndarray,
    base_thresh: float,
    max_attempts: int = 3,
    hole_ratio_threshold: float = 0.18,
):

    thresh = float(base_thresh)
    last_bg_mask = None

    for i in range(max_attempts):
        print(f"[adaptive_flood] attempt {i + 1}, thresh={thresh:.3f}")
        bg_mask = flood_background_from_edges(
            img_lab,
            bg_color,
            color_thresh=thresh,
            hole_ratio_threshold=hole_ratio_threshold,
        )
        last_bg_mask = bg_mask

        ok, reason = quality_check_masks(
            bg_mask,
            min_bg_ratio=MIN_BG_RATIO,
            min_fg_ratio=MIN_FG_RATIO,
            return_reason=True,
        )

        if ok:
            print(f"[adaptive_flood] success on attempt {i + 1}")
            return True, bg_mask, thresh

        if reason == "bg_too_small":
            thresh *= 1.3
        elif reason in ("fg_too_small", "no_fg"):
            thresh *= 0.7
        else:
            break

    print("[adaptive_flood] all attempts failed")
    return False, last_bg_mask, thresh


def create_foreground_rgba(img_bgr: np.ndarray, alpha_mask: np.ndarray) -> np.ndarray:

    h, w = alpha_mask.shape
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0:3] = img_rgb
    rgba[..., 3] = alpha_mask
    return rgba


def remove_floating_small_components(
    alpha_mask: np.ndarray,
    ratio_threshold: float = 0.08,
    connectivity: int = 4,
    min_pixels: int = 4,
):

    if alpha_mask is None or alpha_mask.size == 0:
        return alpha_mask

    fg = (alpha_mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return alpha_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        fg, connectivity=connectivity
    )

    if num_labels <= 2:
        return alpha_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return alpha_mask

    amax = int(areas.max())
    if amax <= 0:
        return alpha_mask

    cutoff = max(int(amax * float(ratio_threshold)), int(min_pixels))

    remove = np.zeros_like(fg, dtype=np.uint8)
    for comp_label in range(1, num_labels):
        area = int(stats[comp_label, cv2.CC_STAT_AREA])
        if area < cutoff:
            remove[labels == comp_label] = 1

    if remove.sum() == 0:
        return alpha_mask

    cleaned = alpha_mask.copy()
    cleaned[remove == 1] = 0
    return cleaned


class MOD_RMBG_NODE:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_hole_ratio": (
                    "FLOAT",
                    {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    FUNCTION = "process_image"
    CATEGORY = "Billbum/PixelTools"

    def process_image(self, image, min_hole_ratio=0.18):

        out_rgba_list = []
        out_fg_list = []
        out_bg_list = []

        for img_tensor in image:
            img_np = np.clip(255.0 * img_tensor.cpu().numpy(), 0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            h, w = img_bgr.shape[:2]
            print(f"[process_image] size: {w} x {h}")

            img_lab = bgr_to_lab(img_bgr)

            def get_fallback_return_data():

                if img_np.shape[2] == 4:
                    rgba = img_np
                else:
                    rgba = np.dstack((img_np, np.full((h, w), 255, dtype=np.uint8)))

                fg_mask = np.full((h, w), 255, dtype=np.uint8)
                bg_mask_vis = np.zeros((h, w), dtype=np.uint8)
                return (
                    ndarray2tensor(rgba),
                    ndarray2tensor(fg_mask),
                    ndarray2tensor(bg_mask_vis),
                )

            bg_color, is_uniform, suggested_thresh = estimate_bg_color_from_border(
                img_lab, border_width=BORDER_WIDTH
            )

            should_fallback = False
            if not is_uniform:
                print("[process_image] Edge colors not uniform, skipping.")
                should_fallback = True
            else:
                ok, bg_mask, final_thresh = flood_with_adaptive_thresh(
                    img_lab,
                    bg_color,
                    base_thresh=suggested_thresh,
                    max_attempts=3,
                    hole_ratio_threshold=min_hole_ratio,
                )
                if not ok or bg_mask is None:
                    print("[process_image] Adaptive flood failed, skipping.")
                    should_fallback = True

            if should_fallback:
                f_rgba, f_fg, f_bg = get_fallback_return_data()
                out_rgba_list.append(f_rgba)
                out_fg_list.append(f_fg)
                out_bg_list.append(f_bg)
            else:
                print(f"[process_image] final used thresh={final_thresh:.3f}")

                final_alpha = (1 - bg_mask).astype(np.uint8) * 255
                final_alpha = remove_floating_small_components(final_alpha)
                rgba_np = create_foreground_rgba(img_bgr, final_alpha)

                fg_mask_np = final_alpha
                bg_mask_vis_np = 255 - final_alpha

                out_rgba_list.append(ndarray2tensor(rgba_np))
                out_fg_list.append(ndarray2tensor(fg_mask_np))
                out_bg_list.append(ndarray2tensor(bg_mask_vis_np))

        return (
            torch.cat(out_rgba_list, dim=0),
            torch.cat(out_fg_list, dim=0),
            torch.cat(out_bg_list, dim=0),
        )
