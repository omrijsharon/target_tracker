import cv2
import numpy as np


# def fractalize(img: np.ndarray, scaling_factors):
#     # assert scaling_factors[0] > 0 and scaling_factors[1] > 0, "scaling factors must be greater that zero."
#     if scaling_factors[0] == scaling_factors[1] or not(scaling_factors[0] > 0 and scaling_factors[1] > 0):
#         return img
#     img = img / 255
#     # assert scaling_factors[0] != scaling_factors[1], "scaling factors must be different."
#     eps = np.finfo(np.float64).eps
#     scaling_factors.sort(reverse=True)
#     scaled_imgs = ([cv2.filter2D(img.copy(), -1, np.ones(shape=(s, s))) for s in scaling_factors])
#     # log(N_0 / N_1) = dim * log(S_0 / S_1)
#     # return np.log((scaled_imgs[0] / (scaled_imgs[1] + eps)) + eps) / np.log(scaling_factors[0] / scaling_factors[1])
#     return np.log((scaled_imgs[1] / (scaled_imgs[0] + eps)) + eps) / np.log(scaling_factors[1] / scaling_factors[0])
#
#
# def image_fractal(fractaled_img):
#     return (255 * fractaled_img / 2).astype(np.uint8)



