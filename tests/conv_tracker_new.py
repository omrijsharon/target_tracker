import cv2
import mss

from utils.helper_functions import get_initial_pixels, show_to_user, image_with_circles, track_pixels


def empty(a):
    pass


cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 240)
cv2.createTrackbar("Std Deviation", "Parameters", 0, 30, empty)
cv2.createTrackbar("Num of Pixels", "Parameters", 1, 30, empty)
cv2.createTrackbar("Kernel Size", "Parameters", 3, 30, empty)

if __name__ == '__main__':
    with mss.mss() as sct:
        num_of_pixels, sigma = cv2.getTrackbarPos("Num of Pixels", "Parameters") , cv2.getTrackbarPos("Std Deviation", "Parameters")
        condensed_prev_frame, pixels = get_initial_pixels(num_of_pixels, sigma)
        while True:
            show_to_user(condensed_prev_frame, image_with_circles(condensed_prev_frame, pixels))
            kernel_size = cv2.getTrackbarPos("Kernel Size", "Parameters")
            condensed_current_frame, pixels = track_pixels(condensed_prev_frame, pixels, kernel_size)
            condensed_prev_frame = condensed_current_frame
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
