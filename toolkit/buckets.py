import math
from typing import TypedDict


class BucketResolution(TypedDict):
    width: int
    height: int


def get_resolution(width, height):
    num_pixels = width * height
    # determine same number of pixels for square image
    square_resolution = int(num_pixels**0.5)
    return square_resolution


def get_bucket_for_image_size(
    width: int, height: int, resolution: int = 512, divisibility: int = 8
) -> BucketResolution:
    total_pixels = width * height
    max_pixels = resolution * resolution

    target_pixels = min(total_pixels, max_pixels)

    scaler = (target_pixels / total_pixels) ** 0.5
    w_raw = (width * scaler) / divisibility
    h_raw = (height * scaler) / divisibility

    candidates = [
        (math.floor(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.floor(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.floor(h_raw) * divisibility),
        (math.ceil(w_raw) * divisibility, math.ceil(h_raw) * divisibility),
    ]
    capped = [(w, h) for w, h in candidates if w > 0 and h > 0 and w * h <= max_pixels]
    if not capped:
        capped = [
            (
                max(divisibility, math.floor(w_raw) * divisibility),
                max(divisibility, math.floor(h_raw) * divisibility),
            )
        ]

    new_width, new_height = min(
        capped, key=lambda wh: abs(wh[0] * wh[1] - target_pixels)
    )

    return {"width": new_width, "height": new_height}


def get_native_resolution_for_image_size(
    width: int, height: int, resolution: int = 512, divisibility: int = 16
) -> BucketResolution:
    """Calculates target dimensions for a native resolution model (NR-MMDiT).
    Preserves native aspect ratio without aspect ratio quantization or cropping,
    scaling down only if total pixels exceed resolution * resolution, and
    snapping both dimensions to divisibility.
    """
    total_pixels = width * height
    max_pixels = resolution * resolution
    if total_pixels > max_pixels:
        scale = math.sqrt(max_pixels / total_pixels)
        target_w = max(divisibility, round((width * scale) / divisibility) * divisibility)
        target_h = max(divisibility, round((height * scale) / divisibility) * divisibility)
    else:
        target_w = max(divisibility, round(width / divisibility) * divisibility)
        target_h = max(divisibility, round(height / divisibility) * divisibility)
    return {"width": target_w, "height": target_h}

