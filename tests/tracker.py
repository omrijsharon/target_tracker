import cv2
import numpy as np
from time import sleep, time, perf_counter


def get_image_from_webcam():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    return frame


def process_frame(frame, **kwags):
    hue = kwags.get('hue')
    hue_range = kwags.get('hue_range')
    frame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bound = np.array((hue - hue_range, hue + hue_range)) % 180
    # bound = np.clip(bound, 0, 179)
    mask = mask_color_from_HSV(hsv, *bound)
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


def get_and_show_stream_from_webcam():
    cap = cv2.VideoCapture(0)
    hue_range = 4
    t0 = perf_counter()
    i = 0
    hue = 1
    while True:
        ret, frame = cap.read()
        masked_img = process_frame(frame, hue=hue, hue_range=hue_range)
        cv2.imshow('masked_img', masked_img)
        if cv2.waitKey(1) == ord('q'):
            break
        # sleep(0.1)
        i += 1
        if i > 180:
            break
        #     i = 0
    # print(i)
    print(1000 * (perf_counter() - t0) / (i - 1))
    # print((i-1) / (perf_counter() - t0))
    cap.release()
    cv2.destroyAllWindows()


def mask_color_from_HSV(hsv, lower, upper):
    if lower < upper:
        lower = (int(lower), 20, 5)
        upper = (int(upper), 255, 255)
        mask = cv2.inRange(hsv, lower, upper)
        return mask
    else:
        lower1 = (int(lower), 20, 5)
        upper1 = (179, 255, 255)
        mask1 = cv2.inRange(hsv, lower1, upper1)
        lower2 = (0, 20, 5)
        upper2 = (int(upper), 255, 255)
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = mask1 + mask2
        return mask


if __name__ == '__main__':
    get_and_show_stream_from_webcam()
