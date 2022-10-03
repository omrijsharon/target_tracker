import os
from tkinter import *
from tkinter import ttk
from tkinter.ttk import Frame

import mss
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
    if string == "webcam" or string == 0:
        return 0
    elif string == "screen" or string == "monitor":
        return 1
    else:
        return dir_path(string)

parser = argparse.ArgumentParser(description='Auto generates UI from config file.')
parser.add_argument('--param_config_path', type=dir_path, help='Path to config params.', default='ui.yaml')
parser.add_argument('--source', type=source_change, help='Source of video.', default='webcam')
args = parser.parse_args()

params = yaml_reader(args.param_config_path)

root = Tk()
root.geometry("980x680")
root.title(os.path.split(args.param_config_path)[-1].split(".")[0].replace("_", " "))

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
lmain.grid(row=i + 3, column=1, pady=0, padx=0)

if args.source != 1:  # not monitor
    cap = cv2.VideoCapture(args.source)
else:
    monitor = {
        "top": 200,  # 100px from the top
        "left": 991,  # 100px from the left
        "width": 1868 - 991,
        "height": 693 - 200,
        "mon": 0,
    }


def show_frame():
    if args.source != 1:
        ret, frame = cap.read()
    else:
        img_byte = sct.grab(monitor)
        frame = np.frombuffer(img_byte.rgb, np.uint8).reshape(monitor["height"], monitor["width"], 3)[:, :, ::-1]
    img = Image.fromarray(frame[:, :, ::-1]).resize(frame.shape[:-1][::-1])
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)


if args.source != 1:
    show_frame()  # Display
    root.mainloop()
else:
    with mss.mss() as sct:
        show_frame()  # Display
        root.mainloop()
