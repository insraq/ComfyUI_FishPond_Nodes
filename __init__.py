from .rmbg_tools import MOD_RMBG_NODE
from .unfake_node import *

## Meneger Mapping
NODE_CLASS_MAPPINGS = {
    "Unfake_PixelateTools": CustomUnfake,
    "ForceDetectPixelateScale": ForceDetectScale,
    "NearestImageScaleDown": ImageScaleDownByWH,
    "ImageScaleDownByFactor": ImageScaleDownByFactor,
    "ImageUpscaleByInt": ImageUpscaleByInt,
    "PixelUpscale2Target": PixelUpscale2Target,
    "RemoveBackgroundPixel": MOD_RMBG_NODE,
    "ImagesToRGB": Images_To_RGB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Unfake_PixelateTools": "Unfake Pixelate Tools",
    "ForceDetectPixelateScale": "Force Detect Pixelate Scale",
    "NearestImageScaleDown": "Image Downscale By W&H",
    "ImageScaleDownByFactor": "Image Downscale By Factor",
    "PixelUpscale2Target": "Pixel IMG Upscale To Target",
    "ImageUpscaleByInt": "Image Upscale By Integer",
    "RemoveBackgroundPixel": "Remove Background (FloodFill)",
    "ImagesToRGB": "Images Convert RGB",
}
