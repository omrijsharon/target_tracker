import numpy as np
from time import perf_counter
import mss
import cv2
import mss.tools
from functools import partial
# from __future__ import print_function
import argparse

max_lowThreshold = 100


def process_frame(frame):
    cv2.imshow('Edge Map', CannyThreshold(frame))
    # cv2.imshow('frame', frame)
    return cv2.waitKey(1) & 0xFF == ord('q')


def main(process_func, monitor_number=0):
    with mss.mss() as sct:
        # Get information of monitor 2
        mon = sct.monitors[monitor_number]
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
        while True:
            img_byte = sct.grab(monitor)
            frame = np.frombuffer(img_byte.rgb, np.uint8).reshape(monitor["height"], monitor["width"], 3)[:, :, ::-1]
            # counter += 1
            if process_func(frame):
                break
        # t_diff = perf_counter() - t0
        # print(counter / t_diff)


def CannyThreshold(src, lowThreshold=100, ratio=3, kernel_size=3):
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.blur(src_gray, (3, 3))
    detected_edges = cv2.Canny(img_blur, lowThreshold, lowThreshold * ratio, kernel_size)
    mask = detected_edges != 0
    dst = src * (mask[:, :, None].astype(src.dtype))
    return dst

# parser = argparse.ArgumentParser(description='Code for Canny Edge Detector tutorial.')
# parser.add_argument('--input', help='Path to input image.', default='fruits.jpg')
# args = parser.parse_args()

if __name__ == '__main__':
    main(process_frame)
