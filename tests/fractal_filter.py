import cv2
import numpy as np


def fractalize(img: np.ndarray, scaling_factors):
    # assert scaling_factors[0] > 0 and scaling_factors[1] > 0, "scaling factors must be greater that zero."
    if scaling_factors[0] == scaling_factors[1] or not(scaling_factors[0] > 0 and scaling_factors[1] > 0):
        return img
    img = img / 255
    # assert scaling_factors[0] != scaling_factors[1], "scaling factors must be different."
    # eps = np.finfo(np.float64).eps
    # scaling_factors.sort()
    scaled_imgs = ([np.clip(cv2.filter2D(img.copy(), -1, np.ones(shape=(s, s))), a_min=1, a_max=np.inf) for s in sorted(scaling_factors)])
    # log(N_0 / N_1) = dim * log(S_0 / S_1)
    # return np.log((scaled_imgs[0] / (scaled_imgs[1] + eps)) + eps) / np.log(scaling_factors[0] / scaling_factors[1])
    scaling_factors_ratio = scaling_factors[0] / scaling_factors[1]
    # scaled_imgs_ratio = np.clip(scaled_imgs[0] / scaled_imgs[1], a_min=scaling_factors_ratio, a_max=1.0)
    scaled_imgs_ratio = scaled_imgs[0] / scaled_imgs[1]
    return np.log(scaled_imgs_ratio) / np.log(scaling_factors_ratio)


def image_fractal(fractaled_img):
    return (255 * fractaled_img / fractaled_img.max()).astype(np.uint8)

