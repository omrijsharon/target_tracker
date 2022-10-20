import cv2
import numpy as np
import mss
import mss.tools

from fractal_filter import image_fractal, fractalize
from utils.helper_functions import np_array_to_cv_array

top_left_corner = []
bottom_right_corner = []
custom_kernel = np.ones(shape=(1, 1), dtype=np.uint8)
custom_kernel_img = np.ones(shape=(1, 1), dtype=np.uint8)

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Threshold1", "Parameters", 23, 255, empty)
cv2.createTrackbar("Threshold2", "Parameters", 23, 255, empty)
cv2.createTrackbar("Area", "Parameters", 1000, 1000, empty)
cv2.createTrackbar("KernelSize", "Parameters", 5, 20, empty)
cv2.createTrackbar("Scaling_1", "Parameters", 3, 51, empty)
cv2.createTrackbar("Scaling_2", "Parameters", 5, 51, empty)



def get_custom_kernel(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner, custom_kernel,custom_kernel_img
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        # Draw the rectangle
        print("I LOVE MICE")
        print(top_left_corner[0])
        print(bottom_right_corner[0])
        diff_y = bottom_right_corner[0][0] - top_left_corner[0][0]
        diff_x = bottom_right_corner[0][1] - top_left_corner[0][1]
        custom_kernel_img = np.zeros_like(imgGray)
        custom_kernel = imgDil[top_left_corner[0][1]:bottom_right_corner[0][1],
                        top_left_corner[0][0]:bottom_right_corner[0][0]]
        custom_kernel_img[top_left_corner[0][1]:bottom_right_corner[0][1],
                        top_left_corner[0][0]:bottom_right_corner[0][0]] = custom_kernel
        custom_kernel = custom_kernel / (diff_x * diff_y)
        # cv2.rectangle(np_array_to_cv_array(img), top_left_corner[0], bottom_right_corner[0], (0, 255, 0), 2, 8)
        # cv2.imshow("Window", custom_kernel)


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


def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    areaMin = cv2.getTrackbarPos("Area", "Parameters")
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # print(area)
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            # print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            # print(len(approx))
            objCor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if objCor == 3:
                objectType = "Tri"
            elif objCor == 4:
                aspRatio = w / float(h)
                if aspRatio > 0.98 and aspRatio < 1.03:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif objCor > 4:
                objectType = "Circles"
            else:
                objectType = "None"

            cv2.rectangle(imgContour, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.putText(imgContour, objectType,
            #             (x + (w // 2) - 10, y + (h // 2) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7,
            #             (0, 0, 0), 2)


with mss.mss() as sct:
    # img = fromScreen()
    cv2.namedWindow("Stack")
    # cv2.setMouseCallback("Stack", get_custom_kernel)
    while True:
        # success, img = cap.read()
        mon = sct.monitors[0]
        img = fromScreen()
        imgContour = img.copy()
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imgBlur = cv2.GaussianBlur(imgGray, (7, 7), 1)
        threshold1 = cv2.getTrackbarPos("Threshold1", "Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2", "Parameters")
        imgCanny = cv2.Canny(imgBlur, threshold1, threshold2)
        kernel_size = cv2.getTrackbarPos("KernelSize", "Parameters")
        kernel = np.ones((kernel_size, kernel_size))
        imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
        s_1 = cv2.getTrackbarPos("Scaling_1", "Parameters")
        s_2 = cv2.getTrackbarPos("Scaling_2", "Parameters")
        imgFractled = image_fractal(fractalize(imgCanny,[s_1,s_2]))
        imgFiltered = cv2.filter2D(imgDil,cv2.CV_64F,custom_kernel)
        imgFiltered /= imgFiltered.max()
        imgFiltered *= 255
        imgFiltered = imgFiltered.astype(np.uint8)
        getContours(imgDil)

        imgBlank = np.zeros_like(img)
        # imgStack = stackImages(0.5, ([img, custom_kernel_img, imgCanny],
        #                              [imgFractled, imgContour, imgFiltered]))
        imgStack = stackImages(0.5, ([img, imgFractled, imgDil]))
        cv2.imshow("Stack", imgStack)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
