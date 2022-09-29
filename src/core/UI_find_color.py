from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame
from PIL import Image, ImageTk
import cv2

root = Tk()
root.geometry("640x780")
root.title("HSV Color Picker")

row = 2
hue_label = Label(root, text="Hue")
hue_label.grid(row=row, column=0, pady=4, padx=4)
hue_scl = Scale(root, from_=0, to=179, orient=HORIZONTAL, length=300)
hue_scl.set(0)
hue_scl.grid(row=row, column=1, pady=4, padx=4)

hue_range_label = Label(root, text="Hue Range")
hue_range_label.grid(row=row+1, column=0, pady=4, padx=4)
hue_range_scl = Scale(root, from_=0, to=179, orient=HORIZONTAL, length=300)
hue_range_scl.set(0)
hue_range_scl.grid(row=row+1, column=1, pady=4, padx=4)

saturation_label = Label(root, text="Saturation")
saturation_label.grid(row=row+3, column=0, pady=4, padx=4)
saturation_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
saturation_scl.set(0)
saturation_scl.grid(row=row+3, column=1, pady=4, padx=4)

saturation_range_label = Label(root, text="Saturation Range")
saturation_range_label.grid(row=row+4, column=0, pady=4, padx=4)
saturation_range_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
saturation_range_scl.set(0)
saturation_range_scl.grid(row=row+4, column=1, pady=4, padx=4)

value_label = Label(root, text="Value")
value_label.grid(row=row+6, column=0, pady=4, padx=4)
value_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
value_scl.set(0)
value_scl.grid(row=row+6, column=1, pady=4, padx=4)

value_range_label = Label(root, text="Value Range")
value_range_label.grid(row=row+7, column=0, pady=4, padx=4)
value_range_scl = Scale(root, from_=0, to=255, orient=HORIZONTAL, length=300)
value_range_scl.set(0)
value_range_scl.grid(row=row+7, column=1, pady=4, padx=4)


root.mainloop()
