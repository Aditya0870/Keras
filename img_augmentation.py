import math
import random
import cv2
import numpy as np

class RandomFliplr(object):
    def __init__(self, probability=0.5):
        self.__probability = probability

    def __call__(self, sample):
        prob = random.random()

        if prob < self.__probability:
            for k, v in sample.items():
                if len(v.shape) >= 2:
                    sample[k] = np.fliplr(v).copy()

        return sample

def apply_min_size(sample, size, image_interpolation_method=cv2.INTER_AREA):
    shape = list(sample["disparity"].shape)

    if shape[0] >= size[0] and shape[1] >= size[1]:
        return sample

    scale = [0, 0]
    scale[0] = size[0] / shape[0]
    scale[1] = size[1] / shape[1]

    scale = max(scale)

    shape[0] = math.ceil(scale * shape[0])
    shape[1] = math.ceil(scale * shape[1])

    sample["image"] = cv2.resize(sample["image"], tuple(shape[::-1]), interpolation=image_interpolation_method)
    sample["disparity"] = cv2.resize(sample["disparity"], tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), tuple(shape[::-1]), interpolation=cv2.INTER_NEAREST)
    sample["mask"] = sample["mask"].astype(bool)

    return tuple(shape)

class RandomCrop(object):
    def __init__(self, width, height, resize_if_needed=False, image_interpolation_method=cv2.INTER_AREA):
        self.__size = (height, width)
        self.__resize_if_needed = resize_if_needed
        self.__image_interpolation_method = image_interpolation_method

    def __call__(self, sample):
        shape = sample["disparity"].shape

        if self.__size[0] > shape[0] or self.__size[1] > shape[1]:
            if self.__resize_if_needed:
                shape = apply_min_size(sample, self.__size, self.__image_interpolation_method)
            else:
                raise Exception("Output size {} bigger than input size {}.".format(self.__size, shape))

        offset = (
            np.random.randint(shape[0] - self.__size[0] + 1),
            np.random.randint(shape[1] - self.__size[1] + 1),
        )

        for k, v in sample.items():
            if k == "code" or k == "basis":
                continue

            if len(sample[k].shape) >= 2:
                sample[k] = v[offset[0]: offset[0] + self.__size[0], offset[1]: offset[1] + self.__size[1]]

        return sample

class Resize(object):
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=1, resize_method="lower_bound", image_interpolation_method=cv2.INTER_AREA, letter_box=False):
        self.__width = width
        self.__height = height
        self.__resize_target = resize_target
        self.__keep_aspect_ratio = keep_aspect_ratio
        self.__multiple_of = ensure_multiple_of
        self.__resize_method = resize_method
        self.__image_interpolation_method = image_interpolation_method
        self.__letter_box = letter_box

    def constrain_to_multiple_of(self, x, min_val=0, max_val=None):
        y = (np.round(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if max_val is not None and y > max_val:
            y = (np.floor(x / self.__multiple_of) * self.__multiple_of).astype(int)

        if y < min_val:
            y = (np.ceil(x / self.__multiple_of) * self.__multiple_of).astype(int)

        return y

    def get_size(self, width, height):
        scale_height = self.__height / height
        scale_width = self.__width / width

        if self.__keep_aspect_ratio:
            if self.__resize_method == "lower_bound":
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "upper_bound":
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.__resize_method == "minimal":
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f"resize_method {self.__resize_method} not implemented")

        if self.__resize_method == "lower_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.__width)
        elif self.__resize_method == "upper_bound":
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.__height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.__width)
        elif self.__resize_method == "minimal":
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f"resize_method {self.__resize_method} not implemented")

        return (new_width, new_height)

    def make_letter_box(self, sample):
        top = bottom = (self.__height - sample.shape[0]) // 2
        left = right = (self.__width - sample.shape[1]) // 2
        sample = cv2.copyMakeBorder(sample, top, bottom, left, right, cv2.BORDER_CONSTANT, None, 0)
        return sample

    def __call__(self, sample):
        width, height = self.get_size(sample["image"].shape[1], sample["image"].shape[0])

        sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=self.__image_interpolation_method)

        if self.__letter_box:
            sample["image"] = self.make_letter_box(sample["image"])

        if self.__resize_target:
            if "disparity" in sample:
                sample["disparity"] = cv2.resize(sample["disparity"], (width, height), interpolation=cv2.INTER_NEAREST)

                if self.__letter_box:
                    sample["disparity"] = self.make_letter_box(sample["disparity"])

            if "depth" in sample:
                sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

                if self.__letter_box:
                    sample["depth"] = self.make_letter_box(sample["depth"])

            sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

            if self.__letter_box:
                sample["mask"] = self.make_letter_box(sample["mask"])

            sample["mask"] = sample["mask"].astype(bool)

        return sample

class ResizeFixed(object):
    def __init__(self, size):
        self.__size = size

    def __call__(self, sample):
        sample["image"] = cv2.resize(sample["image"], self.__size[::-1], interpolation=cv2.INTER_LINEAR)
        sample["disparity"] = cv2.resize(sample["disparity"], self.__size[::-1], interpolation=cv2.INTER_NEAREST)
        sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), self.__size[::-1], interpolation=cv2.INTER_NEAREST)
        sample["mask"] = sample["mask"].astype(bool)

        return sample

class Rescale(object):
    def __init__(self, max_val=1.0, use_mask=True):
        self.__max_val = max_val
        self.__use_mask = use_mask

    def __call__(self, sample):
        disp = sample["disparity"]

        if self.__use_mask:
            mask = sample["mask"]
        else:
            mask = np.ones_like(disp, dtype=np.bool)

        if np.sum(mask) == 0:
            return sample

        min_val = np.min(disp[mask])
        max_val = np.max(disp[mask])

        if max_val > min_val:
            sample["disparity"][mask] = (disp[mask] - min_val) / (max_val - min_val) * self.__max_val
        else:
            sample["disparity"][mask] = np.ones_like(disp[mask]) * self.__max_val / 2.0

        return sample

