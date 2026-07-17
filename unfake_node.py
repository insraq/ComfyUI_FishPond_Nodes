import math

import nest_asyncio
import numpy as np
import torch
import unfake
from PIL import Image

from comfy.utils import common_upscale

from .utils import unfake_scaledown_detect_pipeline

nest_asyncio.apply()


## ========Common Functions======== ##
def pillow_scale_up_nearest(img, target_resolution=1024):

    R_MIN = int(target_resolution * 0.8)
    R_MAX = int(target_resolution * 1.6)

    w, h = img.size
    base = max(w, h)
    if base >= R_MAX:
        return img

    k_min = max(1, math.ceil(R_MIN / base))
    k_max = math.floor(R_MAX / base)
    if k_min > k_max:
        scale_factor = target_resolution / base
        new_size = (int(w * scale_factor), int(h * scale_factor))
        return img.resize(new_size, resample=Image.Resampling.NEAREST)

    best_k = min(
        range(k_min, k_max + 1), key=lambda k: abs(base * k - target_resolution)
    )
    new_size = (w * best_k, h * best_k)

    return img.resize(new_size, resample=Image.Resampling.NEAREST)


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


def tensor_longest_side(image):
    height = image.shape[1]
    width = image.shape[2]
    return max(height, width)


## =========Node Classes========= ##
class CustomUnfake:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "max_colors": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 256,
                        "step": 1,
                        "display": "number",
                    },
                ),
                "detect_method": (["auto", "runs", "edge"], {"default": "auto"}),
                "downscale_method": (
                    ["dominant", "median", "mode", "mean", "content-adaptive"],
                    {"default": "dominant"},
                ),
                "cleanup_morph": ("BOOLEAN", {"default": True}),
                "cleanup_jaggies": ("BOOLEAN", {"default": False}),
                "transparent_background": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("align_img", "align_info")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "unfaked_image"
    CATEGORY = "BillBum/PixelTools"

    def unfaked_image(
        self,
        image,
        max_colors,
        detect_method,
        downscale_method,
        cleanup_morph,
        cleanup_jaggies,
        transparent_background,
    ):

        output_tensors = []
        info_list = []

        for img_tensor in image:
            pil_img = tensor2pil(img_tensor.unsqueeze(0))

            if max_colors == 0:
                result = unfake.process_image_sync(
                    pil_img,
                    detect_method=detect_method,
                    downscale_method=downscale_method,
                    cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                    snap_grid=True,
                    transparent_background=transparent_background,
                    auto_color_detect=True,
                )
            else:
                result = unfake.process_image_sync(
                    pil_img,
                    max_colors=max_colors,
                    detect_method=detect_method,
                    downscale_method=downscale_method,
                    cleanup={"morph": cleanup_morph, "jaggy": cleanup_jaggies},
                    snap_grid=True,
                    transparent_background=transparent_background,
                )
            manifest = result["manifest"]
            width, height = manifest.final_size
            final_size = f"{width}x{height}"
            final_colors = manifest.processing_steps["color_quantization"][
                "final_colors"
            ]
            info_list.append(f"{final_size}px_c{final_colors}")
            output_tensors.append(pil2tensor(result["image"]))

        # Each frame may have a different resolution, so emit a list of
        # (1, H, W, C) tensors instead of concatenating into one batch.
        return (output_tensors, info_list)


class ForceDetectScale:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):

        return {
            "required": {
                "image": ("IMAGE",),
                "bypass_resolution": (
                    "INT",
                    {"default": 256, "min": 8, "max": 4096, "step": 8},
                ),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "FLOAT",
    )
    RETURN_NAMES = ("down_img", "detected_scale")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "force_detect_scale"
    CATEGORY = "BillBum/PixelTools"

    def force_detect_scale(self, image, bypass_resolution):

        output_tensors = []
        scales = []

        for img_tensor in image:
            single = img_tensor.unsqueeze(0)

            if tensor_longest_side(single) > bypass_resolution:
                np_img = tensor2ndarray(single)
                _, scale = unfake_scaledown_detect_pipeline(np_img, bypass_resolution)
                pil_img = Image.fromarray(np_img)
                original_width, original_height = pil_img.size
                new_width = int(original_width // scale)
                new_height = int(original_height // scale)
                upscaled_pil = pil_img.resize(
                    (new_width, new_height), Image.Resampling.NEAREST
                )
                upscaled_np = np.array(upscaled_pil)
                output_tensors.append(ndarray2tensor(upscaled_np))
                scales.append(float(scale))

            else:
                output_tensors.append(single)
                scales.append(1.0)

        # Frames can be downscaled by different factors, so emit lists rather
        # than concatenating tensors of mismatched sizes.
        return (output_tensors, scales)


class ImageScaleDownByWH:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                    },
                ),
                "height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 0,
                        "max": 4096,
                        "step": 1,
                    },
                ),
            },
            "optional": {
                "crop": (["disabled", "center"],),
            },
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "out_img",
        "width",
        "height",
    )
    FUNCTION = "down_by_scale"
    CATEGORY = "BillBum/PixelTools"

    def down_by_scale(self, image, width, height, crop="disabled"):

        upscale_method = "nearest-exact"

        image = image.movedim(-1, 1)
        image = common_upscale(image, width, height, upscale_method, crop)
        image = image.movedim(1, -1)

        return (
            image,
            image.shape[2],
            image.shape[1],
        )


class ImageScaleDownByFactor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "out_img",
        "width",
        "height",
    )
    FUNCTION = "down_by_scale"
    CATEGORY = "BillBum/PixelTools"

    def down_by_scale(self, image, scale_factor):
        if scale_factor == 0:
            _, h, w, _ = image.shape
            return (image, w, h)

        output_images = []

        for img_tensor in image:
            pil_img = tensor2pil(img_tensor.unsqueeze(0))
            original_width, original_height = pil_img.size

            new_width, new_height = (
                int(original_width // scale_factor),
                int(original_height // scale_factor),
            )

            if new_width < 1 or new_height < 1:
                new_width = w
                new_height = h

            resized_pil = pil_img.resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )
            output_images.append(pil2tensor(resized_pil))

        tensor_out = torch.cat(output_images, dim=0)
        final_width, final_height = tensor_out.shape[3], tensor_out.shape[2]

        return (tensor_out, final_width, final_height)


class ImageUpscaleByInt:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_int": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "out_img",
        "width",
        "height",
    )
    FUNCTION = "up_by_int"
    CATEGORY = "BillBum/PixelTools"

    def up_by_int(self, image, scale_int):
        if scale_int <= 1:
            _, h, w, _ = image.shape
            return (image, w, h)

        output_images = []

        for img_tensor in image:
            pil_img = tensor2pil(img_tensor.unsqueeze(0))
            original_width, original_height = pil_img.size

            new_width, new_height = (
                int(original_width * scale_int),
                int(original_height * scale_int),
            )

            resized_pil = pil_img.resize(
                (new_width, new_height), Image.Resampling.NEAREST
            )
            output_images.append(pil2tensor(resized_pil))

        tensor_out = torch.cat(output_images, dim=0)
        final_width, final_height = tensor_out.shape[3], tensor_out.shape[2]

        return (tensor_out, final_width, final_height)


class PixelUpscale2Target:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_resolution": ("INT", {"default": 1024}),
                "crop_and_pad": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "IMAGE",
    )
    RETURN_NAMES = (
        "upscaled_img",
        "cropped_origin_img",
    )
    FUNCTION = "process_gif"
    CATEGORY = "BillBum/PixelTools"

    def process_gif(self, image, target_resolution, crop_and_pad):

        pil_images = [tensor2pil(image[i : i + 1]) for i in range(image.shape[0])]

        cropped_pil_images = pil_images

        if crop_and_pad:
            is_transparent_gif = False
            first_frame = pil_images[0]

            if first_frame.mode == "RGBA":
                w, h = first_frame.size
                if w > 0 and h > 0:
                    try:
                        corners = [
                            first_frame.getpixel((0, 0)),
                            first_frame.getpixel((w - 1, 0)),
                            first_frame.getpixel((0, h - 1)),
                            first_frame.getpixel((w - 1, h - 1)),
                        ]
                        if all(c[3] < 192 for c in corners):
                            is_transparent_gif = True

                    except IndexError:
                        print(
                            f"IndexError when accessing pixels of the image with size ({w}, {h})"
                        )
                        pass

            if is_transparent_gif:
                global_bbox = None
                for pil_img in pil_images:
                    img_rgba = (
                        pil_img if pil_img.mode == "RGBA" else pil_img.convert("RGBA")
                    )
                    bbox = img_rgba.getbbox()
                    if bbox:
                        if global_bbox is None:
                            global_bbox = list(bbox)
                        else:
                            global_bbox[0] = min(global_bbox[0], bbox[0])
                            global_bbox[1] = min(global_bbox[1], bbox[1])
                            global_bbox[2] = max(global_bbox[2], bbox[2])
                            global_bbox[3] = max(global_bbox[3], bbox[3])

                if global_bbox:
                    temp_cropped_list = []
                    crop_w, crop_h = (
                        global_bbox[2] - global_bbox[0],
                        global_bbox[3] - global_bbox[1],
                    )
                    max_side = max(crop_w, crop_h)
                    pad = int(max_side * 0.15)
                    final_size = max_side + pad * 2

                    for pil_img in pil_images:
                        final_img = Image.new(
                            "RGBA", (final_size, final_size), (0, 0, 0, 0)
                        )
                        cropped_part = pil_img.crop(global_bbox)
                        paste_pos = (
                            (final_size - crop_w) // 2,
                            (final_size - crop_h) // 2,
                        )
                        final_img.paste(cropped_part, paste_pos, cropped_part)
                        temp_cropped_list.append(final_img)

                    if temp_cropped_list:
                        cropped_pil_images = temp_cropped_list
            else:
                w, h = pil_images[0].size
                min_side = min(w, h)

                half_side = min_side // 2
                center_x, center_y = w // 2, h // 2

                box = (
                    center_x - half_side,
                    center_y - half_side,
                    center_x + half_side,
                    center_y + half_side,
                )

                temp_cropped_list = []
                for pil_img in pil_images:
                    temp_cropped_list.append(pil_img.crop(box))

                cropped_pil_images = temp_cropped_list

        cropped_tensors = [pil2tensor(img) for img in cropped_pil_images]
        image_batch_after_crop = torch.cat(cropped_tensors, dim=0)

        upscaled_pil_images = [
            pillow_scale_up_nearest(pil_img, target_resolution)
            for pil_img in cropped_pil_images
        ]

        final_tensors = [pil2tensor(img) for img in upscaled_pil_images]
        batch_output = torch.cat(final_tensors, dim=0)

        return (
            batch_output,
            image_batch_after_crop,
        )


class Images_To_RGB:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "image_to_rgb"

    CATEGORY = "BillBum/Tools"

    def image_to_rgb(self, images):

        if len(images) > 1:
            tensors = []
            for image in images:
                tensors.append(pil2tensor(tensor2pil(image).convert("RGB")))
            tensors = torch.cat(tensors, dim=0)
            return (tensors,)
        else:
            return (pil2tensor(tensor2pil(images).convert("RGB")),)
