import cv2
import numpy as np


def do_mosaic(img1_path, img2_path, ratio_thresh=0.95,
              ransac_iters=2000, ransac_thresh=5.0):
    """
    Create a grayscale image mosaic from two images.

    Parameters
    ----------
    img1_path : str
        Path to first image.
    img2_path : str
        Path to second image.
    ratio_thresh : float
        Match quality threshold.
    ransac_iters : int
        Number of RANSAC iterations.
    ransac_thresh : float
        Inlier reprojection threshold.

    Returns
    -------
    mosaic : np.ndarray
        Final stitched grayscale mosaic.
    H : np.ndarray
        Estimated homography matrix.
    matches : np.ndarray
        Initial matches.
    inlier_matches : np.ndarray
        RANSAC inlier matches.
    """

    # ---------------------------------------------------------
    # Load images
    # ---------------------------------------------------------
    im1 = cv2.imread(img1_path)
    im2 = cv2.imread(img2_path)

    if im1 is None or im2 is None:
        raise ValueError("Could not load one or both images.")

    i1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # ---------------------------------------------------------
    # SIFT
    # ---------------------------------------------------------
    sift = cv2.SIFT_create()

    kp1, d1 = sift.detectAndCompute(i1.astype(np.uint8), None)
    kp2, d2 = sift.detectAndCompute(i2.astype(np.uint8), None)

    f1 = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp1])
    f2 = np.array([[k.pt[0], k.pt[1], k.size, k.angle] for k in kp2])

    # ---------------------------------------------------------
    # Distance matrix
    # ---------------------------------------------------------
    m = pairwise_distances(d1, d2)

    # ---------------------------------------------------------
    # Matching
    # ---------------------------------------------------------
    mm = get_matches(m)

    mm = mm[mm[:, 2] < ratio_thresh]

    # ---------------------------------------------------------
    # RANSAC Homography
    # ---------------------------------------------------------
    H, mm_inliers = ransac_homography(
        f1, f2, mm,
        max_iter=ransac_iters,
        th=ransac_thresh
    )

    # ---------------------------------------------------------
    # Warp images
    # ---------------------------------------------------------
    sz_o = i2.shape

    b_o = np.array([
        [0, 0, 1],
        [sz_o[1], 0, 1],
        [0, sz_o[0], 1],
        [sz_o[1], sz_o[0], 1]
    ])

    i1n, i1n_mask, b_ = transform_img(i1, H, b_o)
    i2n, i2n_mask, _ = transform_img(i2, np.eye(3), b_)

    # ---------------------------------------------------------
    # Reinhard intensity normalization
    # ---------------------------------------------------------
    mosaic = reinhard(i1n.copy(), i1n_mask, i2n, i2n_mask)

    # ---------------------------------------------------------
    # Blend
    # ---------------------------------------------------------
    overlap = i1n_mask & i2n_mask
    mosaic[overlap] = (i1n[overlap] + i2n[overlap]) * 0.5

    i2_only = (~i1n_mask) & i2n_mask
    mosaic[i2_only] = i2n[i2_only]

    mosaic = np.clip(mosaic, 0, 255).astype(np.uint8)

    return mosaic, H, mm, mm_inliers


# =====================================================================
# Utilities
# =====================================================================

def pairwise_distances(a, b):
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    ab = a @ b.T
    return np.sqrt(np.maximum(aa + bb - 2 * ab, 0))


def get_matches(m):
    r = np.zeros(m.shape[0], dtype=bool)
    c = np.zeros(m.shape[1], dtype=bool)

    l = min(m.shape)
    mm = []

    flat_idx = np.argsort(m, axis=None)
    i, j = np.unravel_index(flat_idx, m.shape)

    for ii, jj in zip(i, j):
        if not r[ii] and not c[jj]:
            r[ii] = True
            c[jj] = True
            mm.append([ii, jj])

        if len(mm) >= l:
            break

    mm = np.array(mm)

    scores = []

    for k in range(len(mm)):
        ii, jj = mm[k]

        v = m[ii, jj]

        aux_r = m[ii].copy()
        aux_r[aux_r < v] = np.inf
        aux_r[jj] = np.inf
        vr = np.min(aux_r)

        aux_c = m[:, jj].copy()
        aux_c[aux_c < v] = np.inf
        aux_c[ii] = np.inf
        vc = np.min(aux_c)

        scores.append(2 * v / (vr + vc))

    scores = np.array(scores)

    mm = np.column_stack([mm, scores])

    mm = mm[np.argsort(mm[:, 2])]

    return mm


def data_normalize(pts):
    c = np.mean(pts[:2], axis=1)

    s = np.sqrt(2) / np.mean(
        np.sqrt((pts[0] - c[0]) ** 2 + (pts[1] - c[1]) ** 2)
    )

    T = np.array([
        [s, 0, -c[0] * s],
        [0, s, -c[1] * s],
        [0, 0, 1]
    ])

    return T


def compute_homography(pts1, pts2):
    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = T1 @ pts1
    npts2 = T2 @ pts2

    l = npts1.shape[1]

    A = []

    for k in range(l):
        x1, y1, w1 = npts1[:, k]
        x2, y2, w2 = npts2[:, k]

        A.append([0, 0, 0,
                  -w2 * x1, -w2 * y1, -w2 * w1,
                  y2 * x1, y2 * y1, y2 * w1])

        A.append([w2 * x1, w2 * y1, w2 * w1,
                  0, 0, 0,
                  -x2 * x1, -x2 * y1, -x2 * w1])

    A = np.array(A)

    _, D, Vt = np.linalg.svd(A)

    H = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T2) @ H @ T1

    return H / H[2, 2], D


def get_inliers(pt1, pt2, H, th, idx):

    idx = idx.astype(int)

    p1 = pt1[:, idx[:, 0]]
    p2 = pt2[:, idx[:, 1]]

    p2_ = H @ p1
    p2_ = p2_[:2] / p2_[2]

    err2 = np.sum((p2_ - p2[:2]) ** 2, axis=0)

    Hinv = np.linalg.inv(H)

    p1_ = Hinv @ p2
    p1_ = p1_[:2] / p1_[2]

    err1 = np.sum((p1_ - p1[:2]) ** 2, axis=0)

    err = err1 + err2

    return np.where(err < th)[0]


def steps(pps, inl, p):
    e = 1 - inl
    return np.log(1 - p) / np.log(1 - (1 - e) ** pps)


def ransac_homography(pts1, pts2, mm,
                      max_iter=2000,
                      th=5,
                      p=0.99,
                      pps=4):

    pt1 = np.column_stack([
        pts1[:, :2],
        np.ones(len(pts1))
    ]).T

    pt2 = np.column_stack([
        pts2[:, :2],
        np.ones(len(pts2))
    ]).T

    midx = []
    Nc = max_iter

    for c in range(max_iter):

        sidx = np.random.choice(len(mm), pps, replace=False)

        H, D = compute_homography(
            pt1[:, mm[sidx, 0].astype(int)],
            pt2[:, mm[sidx, 1].astype(int)]
        )

        if D[-2] < 0.1:
            continue

        nidx = get_inliers(pt1, pt2, H, th, mm[:, :2])

        if len(nidx) > len(midx):
            midx = nidx

            inl_ratio = len(midx) / len(mm)

            Nc = steps(pps, inl_ratio, p)

        if c > Nc:
            break

    mm_inliers = mm[midx]

    H, _ = compute_homography(
        pt1[:, mm_inliers[:, 0].astype(int)],
        pt2[:, mm_inliers[:, 1].astype(int)]
    )

    return H, mm_inliers


def transform_img(im, H, b_o):

    sz = im.shape

    b = np.array([
        [0, 0, 1],
        [sz[1], 0, 1],
        [0, sz[0], 1],
        [sz[1], sz[0], 1]
    ]).T

    b_ = (H @ b).T
    b_ = b_ / b_[:, [2]]

    c = np.array([
        np.floor(np.min(np.vstack([b_, b_o]), axis=0)),
        np.ceil(np.max(np.vstack([b_, b_o]), axis=0))
    ])

    sz_ = (
        int(c[1, 1] - c[0, 1]),
        int(c[1, 0] - c[0, 0])
    )

    T = np.array([
        [1, 0, -c[0, 0]],
        [0, 1, -c[0, 1]],
        [0, 0, 1]
    ])

    H_ = T @ H
    H_inv = np.linalg.inv(H_)

    im_ = np.zeros(sz_, dtype=np.float32)
    mask_ = np.zeros(sz_, dtype=bool)

    for x_ in range(sz_[1]):
        for y_ in range(sz_[0]):

            pt = H_inv @ np.array([x_, y_, 1])

            x = pt[0] / pt[2]
            y = pt[1] / pt[2]

            xf = int(np.floor(x))
            yf = int(np.floor(y))

            xc = xf + 1
            yc = yf + 1

            if (xf < 0 or xc >= sz[1] or
                    yf < 0 or yc >= sz[0]):
                continue

            mask_[y_, x_] = True

            im_[y_, x_] = (
                im[yf, xf] * (xc - x) * (yc - y) +
                im[yc, xc] * (x - xf) * (y - yf) +
                im[yf, xc] * (x - xf) * (yc - y) +
                im[yc, xf] * (xc - x) * (y - yf)
            )

    return im_, mask_, b_


def reinhard(i1n, i1n_mask, i2n, i2n_mask):

    mu1 = np.mean(i1n[i1n_mask])
    sigma1 = np.std(i1n[i1n_mask])

    mu2 = np.mean(i2n[i2n_mask])
    sigma2 = np.std(i2n[i2n_mask])

    aux = i1n[i1n_mask]
    aux = (aux - mu1) / sigma1 * sigma2 + mu2

    i1n[i1n_mask] = aux

    return i1n


if __name__ == "__main__":
    mosaic, H, matches, inliers = do_mosaic(
        'Picture1.jpg',
        'Picture2.jpg'
    )

    

    # Save result
    cv2.imwrite("mosaic_result_picture.jpg", mosaic)

    # Optional: display with matplotlib
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 8))
    plt.imshow(mosaic, cmap='gray')
    plt.axis('off')
    plt.show()