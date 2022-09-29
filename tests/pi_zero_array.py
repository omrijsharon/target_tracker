from io import BytesIO
import numpy as np
import cv2 as cv
import picamera
import picamera.array
from PIL import Image


with picamera.PiCamera() as camera:
    camera.resolution = (640, 480)
    camera.framerate = 90
    stream = BytesIO()
    while True:
        camera.capture(stream, format='jpeg')
        stream.seek(0)
        image = Image.open(stream)
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
        cv.imshow('frame', image)
        if cv.waitKey(1) == ord('q'):
            break
cv.destroyAllWindows()