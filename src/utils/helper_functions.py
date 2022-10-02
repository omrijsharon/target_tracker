import cv2
import numpy as np


def mask_color_from_HSV(hsv, lower, upper):
    if lower[0] < upper[0]:
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    else:
        lower1 = (lower[0], lower[1], lower[2])
        upper1 = (179, upper[1], upper[2])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        lower2 = (0, lower[1], lower[2])
        upper2 = (upper[0], upper[1], upper[2])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
        return mask


def standardize_hsv(hue, sat, val):
    return int(hue) % 180, int(np.clip(sat, 0, 255)), int(np.clip(val, 0, 255))


# optimize using heap
def top_n_regions(regions, n=3):
    try:
        idx = np.argpartition([region.area for region in regions], max(-n, -len(regions)))[-n:]
    except:
        pass
    return list(map(regions.__getitem__, idx)) if len(idx) > 0 else []
    # top_n_regions = regions[0:n]
    # top_n_regions = sorted(top_n_regions, key=lambda x: x.area, reverse=False)
    # for region in regions[n:]:
    #     if region.area > top_n_regions[0].area:
    #         for
    #         top_n_regions[0] = region
    # return top_n_regions
