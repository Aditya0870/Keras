import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class CropParams:
    top: int
    bottom: int
    left: int
    right: int

def get_border_params(rgb_image, tolerance=0.1, cut_off=20, value=0, level_diff_threshold=5, channel_axis=-1, min_border=5) -> CropParams:
    gray_image = np.mean(rgb_image, axis=channel_axis)
    h, w = gray_image.shape

    def num_value_pixels(arr):
        return np.sum(np.abs(arr - value) < level_diff_threshold)

    def is_above_tolerance(arr, total_pixels):
        return (num_value_pixels(arr) / total_pixels) > tolerance

    top = min_border
    while is_above_tolerance(gray_image[top, :], w) and top < h-1:
        top += 1
        if top > cut_off:
            break

    bottom = h - min_border
    while is_above_tolerance(gray_image[bottom, :], w) and bottom > 0:
        bottom -= 1
        if h - bottom > cut_off:
            break

    left = min_border
    while is_above_tolerance(gray_image[:, left], h) and left < w-1:
        left += 1
        if left > cut_off:
            break

    right = w - min_border
    while is_above_tolerance(gray_image[:, right], h) and right > 0:
        right -= 1
        if w - right > cut_off:
            break

    return CropParams(top, bottom, left, right)

def get_white_border(rgb_image, value=255, **kwargs) -> CropParams:
    assert np.max(rgb_image) <= 255 and np.min(rgb_image) >= 0, "RGB image values are not in range [0, 255]."
    return get_border_params(rgb_image, value=value, **kwargs)

def get_black_border(rgb_image, **kwargs) -> CropParams:
    return get_border_params(rgb_image, value=0, **kwargs)

def crop_image(image: np.ndarray, crop_params: CropParams) -> np.ndarray:
    return image[crop_params.top:crop_params.bottom, crop_params.left:crop_params.right]

def crop_images(*images: np.ndarray, crop_params: CropParams) -> Tuple[np.ndarray]:
    return tuple(crop_image(image, crop_params) for image in images)

def crop_black_or_white_border(rgb_image, *other_images: np.ndarray, tolerance=0.1, cut_off=20, level_diff_threshold=5) -> Tuple[np.ndarray]:
    crop_params = get_black_border(rgb_image, tolerance=tolerance, cut_off=cut_off, level_diff_threshold=level_diff_threshold)
    cropped_images = crop_images(rgb_image, *other_images, crop_params=crop_params)
    crop_params = get_white_border(cropped_images[0], tolerance=tolerance, cut_off=cut_off, level_diff_threshold=level_diff_threshold)
    cropped_images = crop_images(*cropped_images, crop_params=crop_params)
    return cropped_images

