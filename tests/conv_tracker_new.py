from functools import partial

import cv2
import mss

from utils.helper_functions import get_initial_pixels, show_to_user, image_with_circles, track_pixels, np_to_cv, \
    condense, gray_scale_image, mouse_kernel, read_image_from_monitor, uncondense


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Std Deviation", "Parameters", 0, 30, empty)
cv2.createTrackbar("Num of Pixels", "Parameters", 1, 30, empty)
cv2.createTrackbar("Kernel Size", "Parameters", 20, 60, empty)

if __name__ == '__main__':
    with mss.mss() as sct:
        num_of_pixels, sigma = cv2.getTrackbarPos("Num of Pixels", "Parameters"), cv2.getTrackbarPos("Std Deviation",
                                                                                                     "Parameters")
        mouse_output = [(0, 0), (0, 0), False]  # [top_left_corner, bottom_right_corner, is_moused]
        partial_mouse_kernel = partial(mouse_kernel, mouse_output)
        cv2.namedWindow("Stack")
        cv2.setMouseCallback("Stack", partial_mouse_kernel)
        while True:
            prev_frame = read_image_from_monitor(sct)
            show_to_user(np_to_cv(prev_frame), np_to_cv(prev_frame))
            if cv2.waitKey(1) & 0xFF == ord('q') or mouse_output[2]:  # is_moused
                break
        pixels = get_initial_pixels(mouse_output[0], mouse_output[1], num_of_pixels, sigma)
        condensed_prev_frame = condense(gray_scale_image(prev_frame))
        while True:
            show_to_user(prev_frame, image_with_circles(prev_frame, pixels))
            kernel_size = cv2.getTrackbarPos("Kernel Size", "Parameters")
            pixels, current_frame, condensed_current_frame, convImg, kernels = track_pixels(sct, condensed_prev_frame, pixels,
                                                                          kernel_size)
            condensed_prev_frame = condensed_current_frame
            prev_frame = current_frame
            # kernel_to_show = uncondense(kernels[0])
            # kernel_to_show /= kernel_to_show.max()
            # kernel_to_show = 255 * (kernel_to_show)
            # convImg_to_show = uncondense(convImg)
            # convImg_to_show /= convImg_to_show.max()
            # convImg_to_show = 255 * (convImg_to_show)
            # show_to_user(prev_frame, np_to_cv(convImg_to_show))
            # cv2.imshow("kernel", np_to_cv(kernel_to_show))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
