import os
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame
from PIL import Image, ImageTk
import cv2
import numpy as np
import argparse

from utils.helper_functions import yaml_reader


def dir_path(string):
    if os.path.exists(string):
        return string
    else:
        raise FileExistsError(string)


def source_change(string):
    if string == "webcam" or string==0:
        return 0
    else:
        return dir_path(string)


parser = argparse.ArgumentParser(description='Auto generates UI from config file.')
parser.add_argument('--param_config_path', type=dir_path, help='Path to config params.', default='ui.yaml')
parser.add_argument('--source', type=source_change, help='Source of video.', default='webcam')
args = parser.parse_args()

params = yaml_reader(args.param_config_path)

root = Tk()
root.geometry("740x680")
root.title(os.path.split(args.param_config_path)[-1].split(".")[0].replace("_"," "))

for i, (k, v) in enumerate(params.items()):
    row = i + 2
    v.update({
        "label": Label(root, text=v["name"]),
        "scale": Scale(root, from_=v["from"], to=v["to"], orient=HORIZONTAL, length=300)
    })
    v["label"].grid(row=row, column=0, pady=0, padx=0)
    v["scale"].set(v["default"])
    v["scale"].grid(row=row, column=1, pady=0, padx=0)

lmain = Label(root)
lmain.grid(row=i+3, column=1, pady=0, padx=0)

cap = cv2.VideoCapture(args.source)


def show_frame():
    ret, frame = cap.read()

    img = Image.fromarray(frame[:, :, ::-1]).resize((640, 480))
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


show_frame()  #Display
root.mainloop()