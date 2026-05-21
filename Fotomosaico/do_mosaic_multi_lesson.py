import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fotomosaic import (
    show_image,
    sift_features,
    match_features,
    resize_for_sift,
    ransac_homography,
    transform_img,
)


def press_enter_to_continue(message='Press Enter to continue...'):
    input(message)


def main():
    # load the image sequence for incremental mosaicing
    image_files = ['keble_a.jpg', 'keble_b.jpg', 'keble_c.jpg']
    imgs = []
    for fname in image_files:
        im = cv2.imread(fname)
        if im is None:
            raise FileNotFoundError(f"Could not load required image: {fname}")
        im, scale = resize_for_sift(im, max_size=1000)
        imgs.append(im)
    merge_order = [0, 1, 2]

    im1 = imgs[merge_order[0]]
    mosaic = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)

    for idx in merge_order[1:]:
        im_i = imgs[idx]
        i1 = mosaic.astype(np.float32)
        f1, d1 = sift_features(i1)

        i2 = cv2.cvtColor(im_i, cv2.COLOR_BGR2GRAY).astype(np.float32)
        f2, d2 = sift_features(i2)

        if d1.size == 0 or d2.size == 0:
            print(f"No descriptors found for image {idx}; stopping mosaic.")
            break

        mm = match_features(d1, d2, ratio_thresh=0.75)
        mm = mm[mm[:, 2] < 0.95, :]
        if mm.shape[0] < 4:
            print(f"Not enough strong matches for image {idx}; stopping mosaic.")
            break

        Hmi, _ = ransac_homography(f1, f2, mm)
        Him = np.linalg.inv(Hmi)

        sz_o = mosaic.shape
        b_o = np.array([[0.0, 0.0, 1.0], [sz_o[1], 0.0, 1.0], [0.0, sz_o[0], 1.0], [sz_o[1], sz_o[0], 1.0]], dtype=np.float64)

        i2n, i2n_mask, b_ = transform_img(i2, Him, b_o)
        i1n, i1n_mask, _ = transform_img(mosaic, np.eye(3, dtype=np.float64), b_)

        mosaic = i1n.copy()
        i12n_mask = i1n_mask & i2n_mask
        mosaic[i12n_mask] = (i1n[i12n_mask] + i2n[i12n_mask]) * 0.5
        i2n_only_mask = ~i1n_mask & i2n_mask
        mosaic[i2n_only_mask] = i2n[i2n_only_mask]

        show_image(mosaic.astype(np.uint8), title='Updated mosaic')
        press_enter_to_continue()

    plt.close('all')


if __name__ == '__main__':
    main()
