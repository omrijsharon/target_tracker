import cv2
import numpy as np

from utils.helper_functions import mask_color_from_HSV


def process_frame(frame, lower, upper):
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = mask_color_from_HSV(hsv, lower, upper)
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)
    idx = np.argwhere(mask)
    if len(idx) > 0:
        idx_x_mean = int(idx[:, 0].mean())
        idx_y_mean = int(idx[:, 1].mean())
        idx_x_std = int(idx[:, 0].std())
        idx_y_std = int(idx[:, 1].std())
        if not (np.isnan(idx_x_mean) + np.isnan(idx_y_mean) + np.isnan(idx_x_std) + np.isnan(idx_y_std)):
            cv2.rectangle(masked_img, (int(idx_y_mean - idx_y_std - 1), int(idx_x_mean - idx_x_std - 1)),
                          (int(idx_y_mean + idx_y_std + 1), int(idx_x_mean + idx_x_std + 1)), (0, 255, 0), 2)
    return masked_img