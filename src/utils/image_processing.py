import cv2
import numpy as np
from copy import deepcopy
from utils.helper_functions import mask_color_from_HSV, top_n_regions
from skimage import measure
from skimage.color import label2rgb


def process_frame(frame, lower, upper, kernel_size=11, n_bbox=3):
    frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = mask_color_from_HSV(hsv, lower, upper)
    masked_img = cv2.bitwise_and(frame, frame, mask=mask)
    # idx = np.argwhere(mask)
    # if len(idx) > 0:
    #     idx_x_mean = int(idx[:, 0].mean())
    #     idx_y_mean = int(idx[:, 1].mean())
    #     idx_x_std = int(idx[:, 0].std())
    #     idx_y_std = int(idx[:, 1].std())
    #     if not (np.isnan(idx_x_mean) + np.isnan(idx_y_mean) + np.isnan(idx_x_std) + np.isnan(idx_y_std)):
    #         cv2.rectangle(masked_img, (int(idx_y_mean - idx_y_std - 1), int(idx_x_mean - idx_x_std - 1)),
    #                       (int(idx_y_mean + idx_y_std + 1), int(idx_x_mean + idx_x_std + 1)), (0, 255, 0), 2)
    labels = measure.label(mask, return_num=True, connectivity=2, background=0)
    regions = top_n_regions(measure.regionprops(labels[0]), n_bbox)
    bbox = [region.bbox for region in regions]
    for box in bbox:
        cv2.rectangle(masked_img, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    return masked_img


def count_pixels_with_dilation(region_box, kernel_size):
    return cv2.dilate(region_box, np.ones((kernel_size, kernel_size)), iterations=1).sum()


def calc_region_dimension(region, kernel_sizes:tuple):
    max_kernel_size = max(kernel_sizes)
    min_kernel_size = min(kernel_sizes)
    width = region.bbox[3] - region.bbox[1]
    height = region.bbox[2] - region.bbox[0]
    region_box = np.zeros(shape=(height + (max_kernel_size - 1),
                                 width + (max_kernel_size - 1)), dtype=np.uint8)
    idx = region.coords - np.array(region.bbox[:2]) + int((max_kernel_size - 1) / 2)
    region_box[tuple(idx.T)] = 1
    n_ratio = count_pixels_with_dilation(region_box, max_kernel_size) / count_pixels_with_dilation(region_box, min_kernel_size)
    s_ratio = max_kernel_size / min_kernel_size
    dim = np.log(n_ratio) / np.log(s_ratio)
    return dim

def regions_process(mask, frame, n_bbox=3, kernel_size=3):
    regions_mask = np.zeros(shape=frame.shape[:2], dtype=np.uint8)
    labels = measure.label(mask, return_num=True, connectivity=2, background=0)
    # image_label_overlay = (255 * label2rgb(labels[0], image=frame, bg_label=0)).astype(np.uint8)
    regions = measure.regionprops(labels[0])
    max_area_regions = top_n_regions(regions, n_bbox)
    # bbox = [region.bbox for region in max_area_regions]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    kernel_sizes = (3, 5)
    dimImg = np.zeros_like(mask, dtype=np.uint8)
    for region in max_area_regions:
        regions_mask[tuple(region.coords.T)] = 1
        dimImg[tuple(region.coords.T)] = 255 * calc_region_dimension(region, kernel_sizes) / 2
        # regions_mask[np.where(labels[0] == region.label)] = 1
    frame = cv2.bitwise_and(frame, frame, mask=regions_mask)
    # for box in bbox:
    #     cv2.rectangle(frame, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)
    return frame, dimImg
