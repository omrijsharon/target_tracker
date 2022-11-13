from functools import partial

import cv2
import numpy as np
import mss
import mss.tools
import time

from contoursDedection import get_custom_kernel, stackImages, empty
from core.screen_capture import cannyThreshold
from fractal_filter import image_fractal, fractalize
from utils.helper_functions import np_to_cv, top_n_regions
from utils.image_processing import regions_process
from skimage import measure

# top_left_corner = []
# bottom_right_corner = []
# custom_kernel = np.ones(shape=(1, 1), dtype=np.float64)
# custom_kernel_img = np.ones(shape=(1, 1), dtype=np.uint8)
# is_moused = False
# imgGray = np.ones(shape=(1, 1), dtype=np.uint8)

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
# cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 23, 255, empty)
# cv2.createTrackbar("Area", "Parameters", 1000, 1000, empty)
cv2.createTrackbar("Blur Kernel", "Parameters", 5, 20, empty)
cv2.createTrackbar("Low Threshold", "Parameters", 50, 100, empty)
cv2.createTrackbar("Ratio", "Parameters", 3, 10, empty)
cv2.createTrackbar("Canny Kernel", "Parameters", 3, 100, empty)
cv2.createTrackbar("Sigma", "Parameters", 0, 30, empty)
cv2.createTrackbar("Children Points", "Parameters", 1, 30, empty)
cv2.createTrackbar("Child Kernel", "Parameters", 3, 30, empty)


# cv2.createTrackbar("Scaling_1", "Parameters", 3, 51, empty)
# cv2.createTrackbar("Scaling_2", "Parameters", 5, 51, empty)

def pixelize_data(p):
    return list(set(tuple(map(tuple, p.astype(int)))))


def fromScreen(monitor_number=0):
    # Get information of monitor 2
    # mon = sct.monitors[monitor_number]
    # The screen part to capture
    monitor = {
        "top": 200,  # 100px from the top
        "left": 991,  # 100px from the left
        "width": 1868 - 991,
        "height": 693 - 200,
        "mon": monitor_number,
    }
    # counter = 0
    # t0 = perf_counter()
    img_byte = sct.grab(monitor)
    frame = np.frombuffer(img_byte.rgb, np.uint8).reshape(monitor["height"], monitor["width"], 3)[:, :, ::-1]
    return frame
    # t_diff = perf_counter() - t0


def canny_filter(frame):
    lowThreshold = cv2.getTrackbarPos("Low Threshold", "Parameters")
    ratio = cv2.getTrackbarPos("Ratio", "Parameters")
    kernel_size = cv2.getTrackbarPos("Canny Kernel", "Parameters")
    return cannyThreshold(frame, lowThreshold, ratio, kernel_size)


def mouse_kernel(imgGray, mouse_output, custom_kernel, action, x, y, flags, *userdata):
    # Referencing global variables
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        mouse_output[0] = top_left_corner[0][0], top_left_corner[0][1]
        # is_moused = False
    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        mouse_output[1] = bottom_right_corner[0][0], bottom_right_corner[0][1]
        # diff_y = bottom_right_corner[0][0] - top_left_corner[0][0]
        # diff_x = bottom_right_corner[0][1] - top_left_corner[0][1]
        custom_kernel = kernel_from_rectangle(mouse_output[0], mouse_output[1], imgGray)
        mouse_output[2] = True  # is_moused


def kernel_from_rectangle(top_left_corner, bottom_right_corner, imgGray):
    # custom_kernel_img = np.zeros_like(imgGray)
    kernel = imgGray[top_left_corner[1]:bottom_right_corner[1],
             top_left_corner[0]:bottom_right_corner[0]].astype(np.float64)
    # custom_kernel_img[top_left_corner[0][1]:bottom_right_corner[0][1],
    # top_left_corner[0][0]:bottom_right_corner[0][0]] = custom_kernel
    kernel = kernel / np.linalg.norm(kernel)
    return kernel


def convolute_image(imgGray, top_left_corner, bottom_right_corner):
    # if top_left_corner[1] == bottom_right_corner[1] or top_left_corner[0] == bottom_right_corner[0]:
    #     print(top_left_corner, bottom_right_corner)
    kernel = imgGray[top_left_corner[1]:bottom_right_corner[1],
             top_left_corner[0]:bottom_right_corner[0]].astype(np.float64)
    if np.linalg.norm(kernel) <= 0:
        print(top_left_corner, bottom_right_corner)
        print(kernel)
    kernel_norm = np.linalg.norm(kernel)
    if kernel_norm != 0:
        kernel = kernel / kernel_norm
    integral_kernel = np.ones_like(kernel, dtype=np.float64)
    integralImg = np.sqrt(cv2.filter2D(imgGray.astype(np.float64) ** 2, cv2.CV_64F, integral_kernel).astype(np.float64))
    convImg = cv2.filter2D(imgGray.astype(np.float64), cv2.CV_64F, kernel).astype(np.float64) / (integralImg + 1e-8)
    return convImg


def update_custom_kernel(canny_frame, frame, idx, n_bbox=3):
    # intersect circle with most adjacent bbox of regions
    conv_point_x = 472
    conv_point_y = 297
    for i in idx:
        conv_point_x = i[0]
        conv_point_y = i[1]
    labels = measure.label(canny_frame, return_num=True, connectivity=2, background=0)
    regions = measure.regionprops(labels[0])
    regions_containing_point = []
    is_bbox = False
    for region in regions:
        min_row, min_col, max_row, max_col = region.bbox
        if (min_row <= conv_point_x <= max_row) and (min_col <= conv_point_y <= max_col):
            regions_containing_point.append(region)
            is_bbox = True
    if not is_bbox:
        min_row, min_col, max_row, max_col = 0, 0, 0, 0
        return min_row, min_col, max_row, max_col
    min_row, min_col, max_row, max_col = min(regions_containing_point, key=lambda x: x.area_bbox).bbox
    top_left_corner = (min_col, min_row)
    bottom_right_corner = (max_col, max_row)
    # Lin - what to do if no intersection is found??
    kernel_from_rectangle(top_left_corner, bottom_right_corner)
    return min_row, min_col, max_row, max_col


def process_image(top_left_corner, bottom_right_corner):
    mon, img, canny_frame, imgGray, kernel_size, imgBlur = pre_conv_process_image()
    convImg = convolute_image(imgGray, top_left_corner, bottom_right_corner)
    # convImg = (convImg / ((1+q)*(integralImg+q))) ** p
    idx = np.argwhere(convImg == np.nanmax(convImg))
    return mon, img, canny_frame, imgGray, kernel_size, imgBlur, convImg, idx


def image_with_circles(circle_indices):
    img_with_circles = img.copy()
    for i in circle_indices:
        cv2.circle(img_with_circles, tuple(i[::-1]), 5, (0, 255, 0), 2)
    return img_with_circles


def wait_for_mouse(mouse_output):
    while True:
        if mouse_output[2]:  # is_moused
            return
            # mon, img, canny_frame, imgGray, kernel_size, imgBlur, convImg, idx, img_with_circles = process_image()
            # return mon, img, canny_frame, imgGray, kernel_size, imgBlur, convImg, idx, img_with_circles
        else:
            img = np_to_cv(fromScreen())
            imgStack = stackImages(1.0, ([img], [img]))
            cv2.imshow("Stack", imgStack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def pre_conv_process_image():
    mon = sct.monitors[0]
    img = np_to_cv(fromScreen())
    canny_frame = canny_filter(img)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel_size = 2 * cv2.getTrackbarPos("Blur Kernel", "Parameters") + 1
    imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 1)
    return mon, img, canny_frame, imgGray, kernel_size, imgBlur


if __name__ == '__main__':
    with mss.mss() as sct:
        mon, img, canny_frame, imgGray, kernel_size, imgBlur = pre_conv_process_image()
        custom_kernel = np.ones(shape=(1, 1), dtype=np.float64)
        mouse_output = [0, 0, False]  # [top_left_corner, bottom_right_corner, is_moused]
        partial_mouse_kernel = partial(mouse_kernel, imgGray, mouse_output, custom_kernel)
        cv2.namedWindow("Stack")
        cv2.setMouseCallback("Stack", partial_mouse_kernel)
        wait_for_mouse(mouse_output)
        top_left_corner = mouse_output[0]
        bottom_right_corner = mouse_output[1]
        mon, img, canny_frame, imgGray, kernel_size, imgBlur, convImg, idx = process_image(
            top_left_corner, bottom_right_corner)
        vertical_max = imgGray.shape[0]
        horizontal_max = imgGray.shape[1]
        sigma = cv2.getTrackbarPos("Sigma", "Parameters")
        num_of_children = cv2.getTrackbarPos("Children Points", "Parameters")
        idx_children = pixelize_data(idx[0] + sigma * np.random.randn(num_of_children, 2))
        # while True:
        #     img_with_circles = image_with_circles(idx)
        #     imgStack = stackImages(1.0, ([img], [img_with_circles]))
        #     cv2.imshow("Stack", imgStack)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        while True:
            shift = cv2.getTrackbarPos("Child Kernel", "Parameters")
            idx_children_new = []
            mon, img, canny_frame, imgGray, kernel_size, imgBlur = pre_conv_process_image()
            for idx in idx_children:
                # LIN: make sure idx does not deviate
                top_left_corner = np.clip(idx[1] - shift, 0, horizontal_max - 1), np.clip(idx[0] - shift, 0,
                                                                                          vertical_max - 1)
                bottom_right_corner = np.clip(idx[1] + shift, 0, horizontal_max - 1), np.clip(idx[0] + shift, 0,
                                                                                              vertical_max - 1)
                if top_left_corner[0] >= bottom_right_corner[0] or top_left_corner[1] >= bottom_right_corner[1]:
                    continue
                convImg = convolute_image(imgGray, top_left_corner, bottom_right_corner)
                temp = np.argwhere(convImg == np.nanmax(convImg))
                # print(temp)
                new_idx = temp[np.random.randint(0, temp.__len__())]
                idx_children_new.append(new_idx)
            img_with_circles = image_with_circles(idx_children_new)
            # imgStack = stackImages(1.0, ([img], [img_with_circles]))
            convImg_show = convImg * (255 / convImg.max())
            cv2.circle(convImg_show, tuple(idx_children_new[0][::-1]), 5, (0, 255, 0), 2)
            imgStack = stackImages(1.0, ([img], [np_to_cv(convImg_show)]))
            cv2.imshow("Stack", imgStack)
            idx_children = idx_children_new
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
