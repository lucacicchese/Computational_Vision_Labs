import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import cv2

from hm_utils import insert_shape


def dispVanishingPoints(I, VPX, VPY, borders, colorIds=None):
    #DISPVANISHINGPOINTS 
    if colorIds is None:
        colorIds = [1] * len(VPX)
    colors = ['yellow', 'blue', 'red', 'green', 'black']
    s = I.shape
    pad_info = [0, 0, 0, 0]  # x- x+ y- y+
    Ip = I.copy()
    background = 255 * 4 // 5
    if max(VPX) > s[1]:
        Ip = np.pad(Ip, ((0, 0), (0, int(max(VPX) - s[1])), (0, 0)), constant_values=background)
        pad_info[1] = int(max(VPX) - s[1])
    if min(VPX) < 0:
        Ip = np.pad(Ip, ((0, 0), (int(-min(VPX)), 0), (0, 0)), constant_values=background)
        pad_info[0] = int(-min(VPX))
    if max(VPY) > s[0]:
        Ip = np.pad(Ip, ((0, int(max(VPY) - s[0])), (0, 0), (0, 0)), constant_values=background)
        pad_info[3] = int(max(VPY) - s[0])
    if min(VPY) < 0:
        Ip = np.pad(Ip, ((int(-min(VPY)), 0), (0, 0), (0, 0)), constant_values=background)
        pad_info[2] = int(-min(VPY))
    for k in range(len(VPX)):
        Ip = insert_shape(Ip, 'FilledCircle', [VPX[k] + pad_info[0], VPY[k] + pad_info[2], 5], color=colors[colorIds[k]], LineWidth=2)
        Ip = insert_shape(Ip, 'Line', [borders[k][0][0] + pad_info[0], borders[k][0][1] + pad_info[2], borders[k][0][2] + pad_info[0], borders[k][0][3] + pad_info[2]], color=colors[colorIds[k]], LineWidth=2)
        Ip = insert_shape(Ip, 'Line', [borders[k][1][0] + pad_info[0], borders[k][1][1] + pad_info[2], borders[k][1][2] + pad_info[0], borders[k][1][3] + pad_info[2]], color=colors[colorIds[k]], LineWidth=2)
    Ip = insert_shape(Ip, 'Line', [VPX[0] + pad_info[0], VPY[0] + pad_info[2], VPX[1] + pad_info[0], VPY[1] + pad_info[2]], color='white', LineWidth=2)
    plt.close('all')
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(Ip, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    input('pause')
