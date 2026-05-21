import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

# make sure imports work when modules are executed from the repo root
sys.path.append(os.path.dirname(__file__))

COLOR_MAP = {
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'black': (0, 0, 0),
    'white': (255, 255, 255),
}


def insert_shape(I, shape, params, color='green', LineWidth=2):
    """Insert a simple shape into an image, mimicking MATLAB insertShape."""
    output = I.copy()
    bgr = COLOR_MAP.get(color, COLOR_MAP['green'])
    if shape == 'FilledCircle':
        x, y, radius = params
        x = int(round(x))
        y = int(round(y))
        radius = int(round(radius))
        cv2.circle(output, (x, y), radius, bgr, thickness=-1)
    elif shape == 'Line':
        x1, y1, x2, y2 = params
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))
        cv2.line(output, (x1, y1), (x2, y2), bgr, thickness=LineWidth)
    else:
        raise ValueError(f"Unsupported shape: {shape}")
    return output


def imshow(I, title=None):
    """Show an image using matplotlib to mimic MATLAB imshow.

    Reuse an existing figure when present to avoid creating duplicate
    (blank) windows when callers create their own figure before calling
    this helper (e.g. `plt.figure()` in `getLine`).
    """
    # Create a new figure only if none exists
    if not plt.get_fignums():
        plt.figure(figsize=(10, 8))

    ax = plt.gca()
    ax.clear()
    if I.ndim == 2:
        ax.imshow(I, cmap='gray')
    else:
        ax.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))
    ax.axis('off')
    if title:
        ax.set_title(title)
    plt.draw()
    plt.pause(0.001)
