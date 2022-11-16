from functools import partial

import cv2
import numpy as np
import yaml

from mouse_detection import sct


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


def yaml_reader(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    return data


def np_to_cv(np_array):
    return np_array.astype(np.uint8)


def get_monitor(top_left, bottom_right, monitor_number=0):
    """
    :param top_left:
    :param bottom_right:
    :param monitor_number:
    :return: mss monitor dict
    """
    return {
        "top": top_left[0],  # 100px from the top
        "left": top_left[1],  # 100px from the left
        "width": bottom_right[1] - top_left[1],
        "height": bottom_right[0] - top_left[0],
        "mon": monitor_number,
    }


def read_image_from_monitor(top_left=[200, 991], bottom_right=[693, 1868], monitor_number=0):
    """
    :param top_left: [row,col]
    :param bottom_right: [row,col]
    :param monitor_number:
    :return: current frame sliced by top_left and bottom_right
    """
    monitor = get_monitor(top_left, bottom_right, monitor_number)
    img_byte = sct.grab(monitor)
    img = np.frombuffer(img_byte.rgb, np.uint8).reshape(monitor["height"], monitor["width"], 3)[:, :, ::-1]
    return img


def mouse_kernel(mouse_output, action, x, y, flags, *userdata):
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        mouse_output[0] = top_left_corner[0][0], top_left_corner[0][1]
    # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        mouse_output[1] = bottom_right_corner[0][0], bottom_right_corner[0][1]
        mouse_output[2] = True  # is_moused


def wait_for_mouse(mouse_output):
    """
    waits for user to create rectangle, sets mouse_output accordingly
    :param mouse_output: list of [top_left_corner, bottom_right_corner, is_moused]
    :return: current frame
    """
    partial_mouse_kernel = partial(mouse_kernel, mouse_output)
    cv2.namedWindow("Stack")
    cv2.setMouseCallback("Stack", partial_mouse_kernel)
    while True:
        frame = read_image_from_monitor()
        show_to_user(np_to_cv(frame), np_to_cv(frame))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if mouse_output[2]:  # is_moused
            return frame


def gray_scale_image(img):
    """
    :param img: colored image
    :return: gray scaled image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def condense(imgGray):
    """
    :param imgGray: values in img are in [0,255]
    :return: condensed_img s.t. condensed_img[i][j] = 2*(imgGray[i][j]/255) - 1
             values in [-1,1]
    """
    return 2 * (imgGray / 255) - 1


def uncondense(condensed_img):
    """
    :param condensed_img: values in [-1,1]
    :return: uint8 imgGray s.t. imgGray[i][j] = ((condensed_img[i][j] + 1)/2)*255
    """
    return ((condensed_img + 1) / 2) * 255


def generate_normalized_kernel(imgGray, pixel: list, kernel_size: int):
    """
    creates kernel of size kernel_size centered in pixel and cropped from imgGray
    :param imgGray: gray scaled image, shape: w x h, condensed
    :param pixel: [row,col]
    :param kernel_size: int which creates a kernel sized (2*kernel_size+1) x (2*kernel_size+1)
    :return: normalized cropped kernel form imgGray, None if no proper kernel
    """
    horizontal_max = imgGray.shape[1]
    vertical_max = imgGray.shape[0]
    top_left_corner = np.clip(pixel[1] - kernel_size, 0, horizontal_max - 1), np.clip(pixel[0] - kernel_size, 0,
                                                                                      vertical_max - 1)
    bottom_right_corner = np.clip(pixel[1] + kernel_size, 0, horizontal_max - 1), np.clip(pixel[0] + kernel_size, 0,
                                                                                          vertical_max - 1)
    # pixel is on margins
    if top_left_corner[0] >= bottom_right_corner[0] or top_left_corner[1] >= bottom_right_corner[1]:
        return None
    condensed_imgGray = condense(imgGray)
    kernel = condensed_imgGray[top_left_corner[1]:bottom_right_corner[1],
             top_left_corner[0]:bottom_right_corner[0]].astype(np.float64)
    return kernel / np.linalg.norm(kernel)


def cosim_convolution(current_imgGray, prev_frame_kernel, p=1, q=0):
    """
    convolutes current_imgGray with sharpened cosine similarity
    :param current_imgGray: gray scaled image, shape: w x h, condensed
    :param prev_frame_kernel: kernel generated from previous frame, condensed
    :param p,q: cosine similarity params
    :return: convoluted image using prev_frame_kernel generated by pixel and kernel_size. values in [-1,1]
    """
    return None


def get_brightest_pixel(conv_imgGray):
    """
    :param conv_imgGray: condensed conv_imgGray
    :return: brightest pixel coord
    """
    np.argwhere(convImg == np.nanmax(convImg))
    return None


def image_with_circles(img, pixels):
    """
    :param img: non-condensed
    :param pixels:
    :return: img with circles around pixels
    """
    img_with_circles = img.copy()
    for i in pixels:
        cv2.circle(img_with_circles, tuple(i[::-1]), 5, (0, 255, 0), 2)
    return img_with_circles


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


def show_to_user(img1, img2):
    """
    shows img1,img2 on screen
    :param img1:
    :param img2:
    :return: None
    """
    imgStack = stackImages(1.0, ([img1], [img2]))
    cv2.imshow("Stack", imgStack)
    return None


def get_initial_pixels(num_of_pixels=1, sigma=0):
    """
    1. reads and shows current frame
    2. gets rectangle from user (if no rectangle, go back to 1), and takes its center pixel
    3. returns num_of_pixels pixels around center_pixel with normal distribution (with standard deviation: sigma)
    :param num_of_pixels:
    :param sigma:
    :return: num_of_pixels pixels around center_pixel, current_frame
    """
    mouse_output = [0, 0, False]  # [top_left_corner, bottom_right_corner, is_moused]
    current_frame = wait_for_mouse(mouse_output)
    top_left_corner, bottom_right_corner = mouse_output[0], mouse_output[1]
    center_pixel = [np.floor(((top_left_corner[0] + bottom_right_corner[0]) / 2)),
                    np.floor(((top_left_corner[1] + bottom_right_corner[1]) / 2))]
    random_pixels = center_pixel + sigma * np.random.randn(num_of_pixels, 2)
    # condensed_current_frame = condense(gray_scale_image(frame))
    return random_pixels, current_frame


def track_pixels(condensed_prev_frame, pixels, kernel_size):
    """
    1. reads current frame and gray scale it
    2. creates kernels from condensed_prev_frame given kernel_size and pixels
    3. convolutes current frame with kernels and get the brightest pixel for each kernel
    4. returns brightest pixels, current_frame, condensed_current_frame
    :param condensed_prev_frame: condensed
    :param pixels:
    :param kernel_size:
    :return: brightest pixels, condensed_current_frame
    """


if __name__ == '__main__':
    data = yaml_reader(r'C:\Users\linta\PycharmProjects\target_tracker\src\config\ui_canny_edge_detection.yaml')
    print(data)
