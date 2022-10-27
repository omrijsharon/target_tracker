import cv2
import numpy as np
import mss
import mss.tools

from contoursDedection import get_custom_kernel, stackImages, empty
from fractal_filter import image_fractal, fractalize
from utils.helper_functions import np_to_cv

top_left_corner = []
bottom_right_corner = []
custom_kernel = np.ones(shape=(1, 1), dtype=np.float64)
custom_kernel_img = np.ones(shape=(1, 1), dtype=np.uint8)

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 50)
# cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
# cv2.createTrackbar("Threshold2", "Parameters", 23, 255, empty)
# cv2.createTrackbar("Area", "Parameters", 1000, 1000, empty)
cv2.createTrackbar("KernelSize", "Parameters", 0, 20, empty)


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


def kernel_from_rectangle(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, custom_kernel, custom_kernel_img
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        # diff_y = bottom_right_corner[0][0] - top_left_corner[0][0]
        # diff_x = bottom_right_corner[0][1] - top_left_corner[0][1]
        custom_kernel_img = np.zeros_like(imgGray)
        custom_kernel = imgGray[top_left_corner[0][1]:bottom_right_corner[0][1],
                        top_left_corner[0][0]:bottom_right_corner[0][0]].astype(np.float64)
        custom_kernel_img[top_left_corner[0][1]:bottom_right_corner[0][1],
        top_left_corner[0][0]:bottom_right_corner[0][0]] = custom_kernel
        custom_kernel = custom_kernel / np.linalg.norm(custom_kernel)


if __name__ == '__main__':
    with mss.mss() as sct:
        # img = fromScreen()
        cv2.namedWindow("Stack")
        cv2.setMouseCallback("Stack", kernel_from_rectangle)
        while True:
            # success, img = cap.read()
            mon = sct.monitors[0]
            img = np_to_cv(fromScreen())
            imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kernel_size = 2 * cv2.getTrackbarPos("KernelSize", "Parameters") + 1
            imgBlur = cv2.GaussianBlur(imgGray, (kernel_size, kernel_size), 1)
            integral_kernel = np.ones_like(custom_kernel, dtype=np.float64)
            integralImg = np.sqrt(cv2.filter2D(imgGray.astype(np.float64) ** 2, cv2.CV_64F, integral_kernel).astype(np.float64))
            convImg = cv2.filter2D(imgGray.astype(np.float64), cv2.CV_64F, custom_kernel).astype(np.float64)/(integralImg)
            idx = np.argwhere(convImg == convImg.max())
            img_with_circles = img.copy()
            for i in idx:
                cv2.circle(img_with_circles, tuple(i[::-1]), 5, (0, 255, 0), 2)
            convImg *= 255 / convImg.max()
            convImg = np_to_cv(convImg)
            imgStack = stackImages(1.0, ([img_with_circles],
                                         [convImg]))
                                         # [np_to_cv(255 * integralImg / integralImg.max())]))
            # imgStack = stackImages(0.8, ([img, imgContour, imgDil]))
            cv2.imshow("Stack", imgStack)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
