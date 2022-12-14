from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame
from PIL import Image, ImageTk
import cv2
import numpy as np

from utils.helper_functions import standardize_hsv
from utils.image_processing import process_frame

root = Tk()
root.geometry("840x780")
root.title("HSV Color Picker")

row = 2
hue_label = Label(root, text="Hue")
hue_label.grid(row=row, column=0, pady=4, padx=4)
hue_scl = Scale(root, from_=0, to=179, orient=HORIZONTAL, length=300)
hue_scl.set(170)
hue_scl.grid(row=row, column=1, pady=4, padx=4)

hue_range_label = Label(root, text="Hue Range")
hue_range_label.grid(row=row+1, column=0, pady=4, padx=4)
hue_range_scl = Scale(root, from_=0, to=179, orient=HORIZONTAL, length=300)
hue_range_scl.set(10)
hue_range_scl.grid(row=row+1, column=1, pady=4, padx=4)

saturation_label = Label(root, text="Saturation")
saturation_label.grid(row=row+3, column=0, pady=4, padx=4)
saturation_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
saturation_scl.set(127)
saturation_scl.grid(row=row+3, column=1, pady=4, padx=4)

saturation_range_label = Label(root, text="Saturation Range")
saturation_range_label.grid(row=row+4, column=0, pady=4, padx=4)
saturation_range_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
saturation_range_scl.set(127)
saturation_range_scl.grid(row=row+4, column=1, pady=4, padx=4)

value_label = Label(root, text="Value")
value_label.grid(row=row+6, column=0, pady=4, padx=4)
value_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
value_scl.set(127)
value_scl.grid(row=row+6, column=1, pady=4, padx=4)

value_range_label = Label(root, text="Value Range")
value_range_label.grid(row=row+7, column=0, pady=4, padx=4)
value_range_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
value_range_scl.set(127)
value_range_scl.grid(row=row+7, column=1, pady=4, padx=4)

mainFrame = Frame(root)
mainFrame.place(x=20, y=20)

#Capture video frames
lmain = Label(root)
lmain.grid(row=row+8, column=1, pady=4, padx=4)

cap = cv2.VideoCapture(0)


def show_frame():
    ret, frame = cap.read()
    hue = hue_scl.get()
    hue_range = hue_range_scl.get()
    saturation = saturation_scl.get()
    saturation_range = saturation_range_scl.get()
    value = value_scl.get()
    value_range = value_range_scl.get()
    lower = standardize_hsv(hue - hue_range, saturation - saturation_range, value - value_range)
    upper = standardize_hsv(hue + hue_range, saturation + saturation_range, value + value_range)
    masked_img = process_frame(frame, lower, upper, kernel_size=21, n_bbox=5)

    img = Image.fromarray(masked_img[:, :, ::-1]).resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


show_frame()  #Display
root.mainloop()
