import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from fotomosaic import (
    show_image,
    sift_features,
    match_features,
    resize_for_sift,
    compute_homography,
    find_homography_ransac,
    ransac_homography,
    transform_img,
    reinard,
)


def press_enter_to_continue(message='Press Enter to continue...'):
    input(message)


def main():
    # load the two input images and convert them to grayscale
    im1 = cv2.imread('im1_.png')
    im2 = cv2.imread('im2_.png')
    if im1 is None or im2 is None:
        raise FileNotFoundError('Could not load im1_.png or im2_.png')

    im1, scale1 = resize_for_sift(im1, max_size=1000)
    im2, scale2 = resize_for_sift(im2, max_size=1000)
    i1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # remove the white top border, as in the MATLAB version
    i1 = i1[10:, :]
    i2 = i2[10:, :]

    # show both input images side by side
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(i1.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title('Image 1')
    plt.subplot(1, 2, 2)
    plt.imshow(i2.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title('Image 2')
    plt.show(block=False)
    press_enter_to_continue()

    # detect SIFT keypoints and descriptors in both images
    f1, d1 = sift_features(i1)
    f2, d2 = sift_features(i2)

    # match descriptors using ratio test instead of a full distance matrix
    mm = match_features(d1, d2, ratio_thresh=0.75)
    mm = mm[mm[:, 2] < 0.95, :]

    # visualize matches as red-green overlay of the two grayscale images
    im12 = np.zeros((i1.shape[0], i1.shape[1], 3), dtype=np.float32)
    im12[:, :, 0] = i1
    im12[:, :, 1] = i2
    show_image(im12 / 255.0, title='Found matches')
    plt.plot([f1[mm[:, 0].astype(int), 0], f2[mm[:, 1].astype(int), 0]],
             [f1[mm[:, 0].astype(int), 1], f2[mm[:, 1].astype(int), 1]], '-', color='yellow')
    plt.draw()
    plt.pause(0.001)
    press_enter_to_continue()

    # compute the homography from image 1 to image 2 using all matches
    pt1 = np.column_stack((f1[:, :2], np.ones((f1.shape[0], 1), dtype=np.float64)))
    pt2 = np.column_stack((f2[:, :2], np.ones((f2.shape[0], 1), dtype=np.float64)))
    if mm.shape[0] < 4:
        raise RuntimeError('Not enough matches to compute initial homography.')
    H, _ = compute_homography(pt1[mm[:, 0].astype(int), :].T, pt2[mm[:, 1].astype(int), :].T)

    # verify the homography on one example match
    idx_check = min(32, mm.shape[0] - 1)
    pt1_test = pt1[mm[idx_check, 0].astype(int), :].T
    pt2_test = pt2[mm[idx_check, 1].astype(int), :].T
    pt1_test_ = H @ pt1_test
    pt1_test_ /= pt1_test_[2]
    print('Test correspondence:')
    print(np.column_stack((pt1_test_, pt2_test)))
    press_enter_to_continue()

    # transform image 1 into the frame of image 2
    sz_o = i2.shape
    b_o = np.array([[0.0, 0.0, 1.0], [sz_o[1], 0.0, 1.0], [0.0, sz_o[0], 1.0], [sz_o[1], sz_o[0], 1.0]], dtype=np.float64)
    i1n, i1n_mask, b_ = transform_img(i1, H, b_o)
    i2n, i2n_mask, _ = transform_img(i2, np.eye(3, dtype=np.float64), b_)

    # build a first mosaic without outlier rejection
    mosaic = i1n.copy()
    i12n_mask = i1n_mask & i2n_mask
    mosaic[i12n_mask] = (i1n[i12n_mask] + i2n[i12n_mask]) * 0.5
    i2n_only_mask = ~i1n_mask & i2n_mask
    mosaic[i2n_only_mask] = i2n[i2n_only_mask]

    show_image(mosaic.astype(np.uint8), title='Wrong mosaic')
    press_enter_to_continue()

    # run OpenCV RANSAC to find a robust homography using only inliers
    matched1 = f1[mm[:, 0].astype(int), :2]
    matched2 = f2[mm[:, 1].astype(int), :2]
    H, inlier_mask = find_homography_ransac(matched1, matched2, reprojThreshold=4.0, maxIters=3000)
    if inlier_mask.sum() < 4:
        raise RuntimeError('RANSAC failed to find enough inliers.')
    mm_inliers = mm[inlier_mask.astype(bool), :]

    idx_check = min(32, mm_inliers.shape[0] - 1)
    pt1_test = pt1[mm_inliers[idx_check, 0].astype(int), :].T
    pt2_test = pt2[mm_inliers[idx_check, 1].astype(int), :].T
    pt1_test_ = H @ pt1_test
    pt1_test_ /= pt1_test_[2]
    print('Test correspondence after RANSAC:')
    print(np.column_stack((pt1_test_, pt2_test)))
    press_enter_to_continue()

    show_image(im12 / 255.0, title='Inlier matches')
    plt.plot([f1[mm_inliers[:, 0].astype(int), 0], f2[mm_inliers[:, 1].astype(int), 0]],
             [f1[mm_inliers[:, 0].astype(int), 1], f2[mm_inliers[:, 1].astype(int), 1]], '-', color='yellow')
    plt.draw()
    plt.pause(0.001)
    press_enter_to_continue()

    i1n, i1n_mask, b_ = transform_img(i1, H, b_o)
    i2n, i2n_mask, _ = transform_img(i2, np.eye(3, dtype=np.float64), b_)

    mosaic = i1n.copy()
    i12n_mask = i1n_mask & i2n_mask
    mosaic[i12n_mask] = (i1n[i12n_mask] + i2n[i12n_mask]) * 0.5
    i2n_only_mask = ~i1n_mask & i2n_mask
    mosaic[i2n_only_mask] = i2n[i2n_only_mask]

    show_image(mosaic.astype(np.uint8), title='Good mosaic')
    press_enter_to_continue()

    i1n_cc = reinard(i1n.copy(), i1n_mask, i2n, i2n_mask)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(i1n_cc.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title('I1 with intensities registered w.r.t. I2')
    plt.subplot(1, 2, 2)
    plt.imshow(i2n.astype(np.uint8), cmap='gray')
    plt.axis('off')
    plt.title('I2')
    plt.show(block=False)
    press_enter_to_continue()

    mosaic = i1n_cc.copy()
    i12n_mask = i1n_mask & i2n_mask
    mosaic[i12n_mask] = (i1n_cc[i12n_mask] + i2n[i12n_mask]) * 0.5
    i2n_only_mask = ~i1n_mask & i2n_mask
    mosaic[i2n_only_mask] = i2n[i2n_only_mask]

    show_image(mosaic.astype(np.uint8), title='Good mosaic with color correction')
    press_enter_to_continue()


if __name__ == '__main__':
    main()
