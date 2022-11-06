import cv2
import numpy as np
import mss
import mss.tools

from contoursDedection import get_custom_kernel, stackImages, empty
from core.screen_capture import cannyThreshold
from fractal_filter import image_fractal, fractalize
from utils.helper_functions import np_to_cv, top_n_regions
from utils.image_processing import regions_process
from skimage import measure

top_left_corner = []
bottom_right_corner = []
custom_kernel = np.ones(shape=(1, 1), dtype=np.float64)
custom_kernel_img = np.ones(shape=(1, 1), dtype=np.uint8)
is_moused = False

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
cv2.createTrackbar("Blur Kernel Size", "Parameters", 5, 20, empty)
cv2.createTrackbar("Low Threshold", "Parameters", 50, 100, empty)
cv2.createTrackbar("Ratio", "Parameters", 3, 10, empty)
cv2.createTrackbar("Canny Kernel Size", "Parameters", 3, 100, empty)


# cv2.createTrackbar("Scaling_1", "Parameters", 3, 51, empty)
# cv2.createTrackbar("Scaling_2", "Parameters", 5, 51, empty)

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
    # print(counter / t_diff)


def canny_filter(frame):
    lowThreshold = cv2.getTrackbarPos("Low Threshold", "Parameters")
    ratio = cv2.getTrackbarPos("Ratio", "Parameters")
    kernel_size = cv2.getTrackbarPos("Canny Kernel Size", "Parameters")
    return cannyThreshold(frame, lowThreshold, ratio, kernel_size)


def mouse_kernel(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, is_moused
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        is_moused = False
        # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        # diff_y = bottom_right_corner[0][0] - top_left_corner[0][0]
        # diff_x = bottom_right_corner[0][1] - top_left_corner[0][1]
        kernel_from_rectangle(top_left_corner, bottom_right_corner)
        is_moused = True


def kernel_from_rectangle(top_left_corner, bottom_right_corner):
    global custom_kernel, custom_kernel_img
    # print("!!! kernel_from_rectangle")
    # print(f"top_left_corner: {top_left_corner}", )
    # print(f"bottom_right_corner: {bottom_right_corner}")
    # print(f"imgGray_size: {imgGray.shape}")
    custom_kernel_img = np.zeros_like(imgGray)
    custom_kernel = imgGray[top_left_corner[0][1]:bottom_right_corner[0][1],
                    top_left_corner[0][0]:bottom_right_corner[0][0]].astype(np.float64)
    custom_kernel_img[top_left_corner[0][1]:bottom_right_corner[0][1],
    top_left_corner[0][0]:bottom_right_corner[0][0]] = custom_kernel
    custom_kernel = custom_kernel / np.linalg.norm(custom_kernel)


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
    print(f"is bbox: {is_bbox}")
    if not is_bbox:
        min_row, min_col, max_row, max_col = 0, 0, 0, 0
        return min_row, min_col, max_row, max_col
    min_row, min_col, max_row, max_col = min(regions_containing_point, key=lambda x: x.area_bbox).bbox
    top_left_corner = [(min_col, min_row)]
    bottom_right_corner = [(max_col, max_row)]
    # Lin - what to do if no intersection is found??
    if is_moused:
        kernel_from_rectangle(top_left_corner, bottom_right_corner)
    # print(min_row, min_col, max_row, max_col)
    return min_row, min_col, max_row, max_col


if __name__ == '__main__':
    with mss.mss() as sct:
        # img = fromScreen()
        cv2.namedWindow("Stack")
        cv2.setMouseCallback("Stack", mouse_kernel)
        idx = [[258, 365]]
        while True:
            # success, img = cap.read()
            mon = sct.monitors[0]
            img = np_to_cv(fromScreen())
            canny_frame = canny_filter(img)
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(f"idx: {idx}")
            kernel_size = 2 * cv2.getTrackbarPos("Blur Kernel Size", "Parameters") + 1
            imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 1)
            integral_kernel = np.ones_like(custom_kernel, dtype=np.float64)
            integralImg = np.sqrt(
                cv2.filter2D(imgGray.astype(np.float64) ** 2, cv2.CV_64F, integral_kernel).astype(np.float64))
            convImg = cv2.filter2D(imgGray.astype(np.float64), cv2.CV_64F, custom_kernel).astype(np.float64) / (
                integralImg)
            convImg = (convImg / ((1+q)*(integralImg+q))) ** p
            idx = np.argwhere(convImg == convImg.max())
            # print(idx)
            img_with_circles = img.copy()
            for i in idx:
                cv2.circle(img_with_circles, tuple(i[::-1]), 5, (0, 255, 0), 2)
            labels = measure.label(canny_frame, return_num=True, connectivity=2, background=0)
            regions = measure.regionprops(labels[0])
            max_area_regions = top_n_regions(regions, 50)
            bboxes = [region.bbox for region in max_area_regions]
            for box in bboxes:
                cv2.rectangle(img_with_circles, (box[1], box[0]), (box[3], box[2]), (0, 255, 0), 2)  # LIN - NOTE
            if is_moused:
                min_row, min_col, max_row, max_col = update_custom_kernel(canny_frame, img, idx)
                cv2.rectangle(img_with_circles, (min_col, min_row), (max_col, max_row), (0, 0, 255), 2)  # LIN - NOTE
            convImg *= 255 / convImg.max()
            convImg = np_to_cv(convImg)
            canny_show, dimImg = regions_process(canny_frame, img, n_bbox=3)
            # imgStack = stackImages(1.0, ([img_with_circles],
            #                              [np_to_cv(canny_show)]))
            imgStack = stackImages(1.0, ([img], [img_with_circles]))
            # [np_to_cv(255 * integralImg / integralImg.max())]))
            # imgStack = stackImages(0.8, ([img, imgContour, imgDil]))
            cv2.imshow("Stack", imgStack)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
