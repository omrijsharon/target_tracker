# Import packages
import cv2

# Lists to store the bounding box coordinates
import mss
import numpy as np

from utils.helper_functions import np_array_to_cv_array

top_left_corner = []
bottom_right_corner = []

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

# function which will be called on mouse input
def drawRectangle(action, x, y, flags, *userdata):
    # Referencing global variables
    global top_left_corner, bottom_right_corner
    # Mark the top left corner when left mouse button is pressed
    if action == cv2.EVENT_LBUTTONDOWN:
        top_left_corner = [(x, y)]
        # When left mouse button is released, mark bottom right corner
    elif action == cv2.EVENT_LBUTTONUP:
        bottom_right_corner = [(x, y)]
        # Draw the rectangle
        cv2.rectangle(image, top_left_corner[0], bottom_right_corner[0], (0, 255, 0), 2, 8)
        cv2.imshow("Window", image)


# Read Images
with mss.mss() as sct:
    image = np_array_to_cv_array(fromScreen())
# Make a temporary image, will be useful to clear the drawing
temp = image.copy()
# Create a named window
cv2.namedWindow("Window")
# highgui function called when mouse events occur
cv2.setMouseCallback("Window", drawRectangle)

k = 0
# Close the window when key q is pressed
while k != 113:
    # Display the image
    cv2.imshow("Window", image)
    k = cv2.waitKey(0)
    # If c is pressed, clear the window, using the dummy image
    if (k == 99):
        image = temp.copy()
        cv2.imshow("Window", image)

cv2.destroyAllWindows()