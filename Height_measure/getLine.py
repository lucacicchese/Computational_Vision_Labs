import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt

from hm_utils import insert_shape, imshow


def getLine(I, colorId=1):
    #GETLINE given an image, takes 2 graphical inputs from the user to draw and return the
    #line through two points; returns the image with the new line drawn and
    #line parameters m and q
    if colorId is None:
        colorId = 1
    s = I.shape
    colors = ['yellow', 'blue', 'red', 'green', 'black']
    x = [0, 0]
    y = [0, 0]
    plt.close('all')
    plt.figure(figsize=(10, 8))
    imshow(I)
    plt.title('Select first point')
    plt.draw()
    plt.pause(0.001)
    pts = plt.ginput(1)
    if pts:
        x[0], y[0] = pts[0]
    I = insert_shape(I, 'FilledCircle', [x[0], y[0], 3], color=colors[colorId], LineWidth=2)
    plt.close('all')
    plt.figure(figsize=(10, 8))
    imshow(I)
    plt.title('Select second point')
    plt.draw()
    plt.pause(0.001)
    pts = plt.ginput(1)
    if pts:
        x[1], y[1] = pts[0]
    I = insert_shape(I, 'FilledCircle', [x[1], y[1], 3], color=colors[colorId], LineWidth=2)
    if x[1] != x[0]:
        m = (y[1] - y[0]) / (x[1] - x[0])
        q = y[0] - m * x[0]
        X = []
        if 0 <= m * 0 + q <= s[0]:
            X.append(0)
        if 0 <= m * s[1] + q <= s[0]:
            X.append(s[1])
        if 0 < (0 - q) / m < s[1]:
            X.append(round((0 - q) / m))
        if 0 < (s[0] - q) / m < s[1]:
            X.append(round((s[0] - q) / m))
        I = insert_shape(I, 'Line', [round(X[0]), round(m * X[0] + q), round(X[1]), round(m * X[1] + q)], color=colors[colorId], LineWidth=2)
        imshow(I)
        borders = [round(X[0]), round(m * X[0] + q), round(X[1]), round(m * X[1] + q)]
    else:
        X = [x[0], x[1]]
        m = np.inf
        q = x[0]
        I = insert_shape(I, 'Line', [round(X[0]), 0, round(X[1]), s[0]], color=colors[colorId], LineWidth=2)
        imshow(I)
        borders = [round(X[0]), 0, round(X[1]), s[0]]
    inputs = [x[0], y[0], x[1], y[1]]
    return I, m, q, borders, inputs
