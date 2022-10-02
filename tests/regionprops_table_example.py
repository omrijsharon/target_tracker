import numpy as np
from skimage import measure
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    resolution = 480, 640
    img_original = np.random.rand(*resolution)
    region_area_threshold = 64
    for threshold in np.linspace(0.3, 1, 101):
        plt.clf()
        # threshold = 0.7
        img = img_original.copy()
        img[img < (1-threshold)] = 0
        img[img >= (1-threshold)] = 1
        labels = measure.label(img, return_num=True, connectivity=2, background=0)
        regions = [region for region in measure.regionprops(labels[0]) if region.area >= region_area_threshold]
        bbox = [region.bbox for region in regions]
        img = cv2.merge([img, img, img])
        print(len(bbox))
        for box in bbox:
            cv2.rectangle(img, (box[1], box[0]), (box[3], box[2]), (0, 1, 0), 1)
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break
        # plt.imshow(labels[0], cmap=mpl.colormaps['jet'])
        # plt.title(f'threshold={threshold}')
        # plt.pause(0.5)
    # plt.show()
    # props = measure.regionprops_table(labels, properties=['centroid'])
    # regions = measure.regionprops(labels)
    # while True:
    #     cv2.imshow('img', labels)
    #     if cv2.waitKey(1) == ord('q'):
    #         break
