import numpy as np
import cv2
import matplotlib.pyplot as plt


def show_image(image, title=None, cmap=None):
    """Show an image with matplotlib and keep the figure open."""
    plt.figure(figsize=(10, 8))
    if cmap is not None:
        plt.imshow(image, cmap=cmap)
    elif image.ndim == 2:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    if title is not None:
        plt.title(title)
    plt.draw()
    plt.pause(0.001)


def resize_for_sift(image, max_size=1000):
    """Resize the image if its largest dimension exceeds max_size."""
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image, 1.0
    scale = max_size / float(max(h, w))
    new_w = int(np.round(w * scale))
    new_h = int(np.round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def sift_features(image, nfeatures=800, contrastThreshold=0.04, edgeThreshold=10):
    """Extract SIFT keypoints and descriptors from a grayscale image."""
    image_uint8 = np.asarray(image, dtype=np.uint8)
    sift = cv2.SIFT_create(
        nfeatures=nfeatures,
        contrastThreshold=contrastThreshold,
        edgeThreshold=edgeThreshold,
    )
    keypoints, descriptors = sift.detectAndCompute(image_uint8, None)
    if descriptors is None:
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0, 128), dtype=np.float32)

    frames = np.zeros((len(keypoints), 4), dtype=np.float32)
    for i, kp in enumerate(keypoints):
        frames[i, 0] = kp.pt[0]
        frames[i, 1] = kp.pt[1]
        frames[i, 2] = kp.size
        frames[i, 3] = kp.angle
    return frames, descriptors.astype(np.float32)


def match_features(d1, d2, ratio_thresh=0.75):
    """Match descriptors with k-NN and ratio test to avoid full distance matrices."""
    if d1.size == 0 or d2.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    raw_matches = matcher.knnMatch(d1, d2, k=2)
    matches = []
    for pair in raw_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio_thresh * n.distance:
            ratio = m.distance / max(n.distance, 1e-9)
            matches.append((m.queryIdx, m.trainIdx, ratio))
    if len(matches) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    mm = np.array(matches, dtype=np.float32)
    sort_idx = np.argsort(mm[:, 2])
    return mm[sort_idx]


def get_matches(m):
    """Compute unique matches from a descriptor distance matrix."""
    if m.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    r = np.zeros(m.shape[0], dtype=bool)
    c = np.zeros(m.shape[1], dtype=bool)
    mnum = min(m.shape[0], m.shape[1])
    mm = np.zeros((mnum, 3), dtype=np.float32)

    idx = np.argsort(m.ravel())
    i_idxs, j_idxs = np.unravel_index(idx, m.shape)

    kc = 0
    for k in range(len(idx)):
        if not r[i_idxs[k]] and not c[j_idxs[k]]:
            r[i_idxs[k]] = True
            c[j_idxs[k]] = True
            mm[kc, 0] = i_idxs[k]
            mm[kc, 1] = j_idxs[k]
            kc += 1
        if kc >= mnum:
            break

    mm = mm[:kc, :]
    if mm.size == 0:
        return mm

    for k in range(mm.shape[0]):
        v = m[int(mm[k, 0]), int(mm[k, 1])]
        aux_r = np.copy(m[int(mm[k, 0]), :])
        aux_r[aux_r < v] = np.inf
        aux_r[int(mm[k, 1])] = np.inf
        vr = np.min(aux_r)

        aux_c = np.copy(m[:, int(mm[k, 1])])
        aux_c[aux_c < v] = np.inf
        aux_c[int(mm[k, 0])] = np.inf
        vc = np.min(aux_c)

        mm[k, 2] = 2.0 * v / (vr + vc)

    sort_idx = np.argsort(mm[:, 2])
    mm = mm[sort_idx]
    return mm


def pdist2(a, b):
    """Compute the Euclidean distance matrix between two descriptor sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a[:, None, :] - b[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2)).astype(np.float32)


def data_normalize(pts):
    """Compute a normalization matrix for homogeneous points."""
    c = np.mean(pts[:2, :], axis=1)
    distances = np.sqrt((pts[0, :] - c[0]) ** 2 + (pts[1, :] - c[1]) ** 2)
    s = np.sqrt(2) / np.mean(distances)
    T = np.array([[s, 0.0, -c[0] * s], [0.0, s, -c[1] * s], [0.0, 0.0, 1.0]], dtype=np.float64)
    return T


def compute_homography(pts1, pts2):
    """Estimate a homography mapping pts1 to pts2 using normalized DLT."""
    T1 = data_normalize(pts1)
    T2 = data_normalize(pts2)

    npts1 = T1 @ pts1
    npts2 = T2 @ pts2
    l = npts1.shape[1]

    A = np.zeros((3 * l, 9), dtype=np.float64)
    A[:l, 3:6] = -np.repeat(npts2[2, :].reshape(l, 1), 3, axis=1) * npts1.T
    A[:l, 6:9] = np.repeat(npts2[1, :].reshape(l, 1), 3, axis=1) * npts1.T
    A[l:2 * l, 0:3] = np.repeat(npts2[2, :].reshape(l, 1), 3, axis=1) * npts1.T
    A[l:2 * l, 6:9] = -np.repeat(npts2[0, :].reshape(l, 1), 3, axis=1) * npts1.T
    A[2 * l:3 * l, 0:3] = -np.repeat(npts2[1, :].reshape(l, 1), 3, axis=1) * npts1.T
    A[2 * l:3 * l, 3:6] = np.repeat(npts2[0, :].reshape(l, 1), 3, axis=1) * npts1.T

    _, D, vh = np.linalg.svd(A)
    H = vh[-1, :].reshape((3, 3)).T
    H = np.linalg.inv(T2) @ H @ T1
    return H, D


def find_homography_ransac(pts1, pts2, reprojThreshold=4.0, maxIters=3000):
    """Estimate a robust homography using OpenCV RANSAC."""
    if pts1.shape[0] < 4 or pts2.shape[0] < 4:
        return np.eye(3, dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    H, mask = cv2.findHomography(
        pts1.astype(np.float32),
        pts2.astype(np.float32),
        cv2.RANSAC,
        reprojThreshold,
        maxIters=maxIters,
    )
    if H is None:
        return np.eye(3, dtype=np.float64), np.zeros((0,), dtype=np.uint8)
    return H.astype(np.float64), mask.ravel().astype(np.uint8)


def get_inliers(pt1, pt2, H, th, idx):
    """Find inliers based on symmetric transfer error."""
    pt2_ = H @ pt1[:, idx[:, 0].astype(int)]
    tmp2_ = pt2_[:2, :] / pt2_[2, :] - pt2[:2, idx[:, 1].astype(int)]
    err1 = np.sum(tmp2_ * tmp2_, axis=0)

    pt1_ = np.linalg.solve(H, pt2[:, idx[:, 1].astype(int)])
    tmp1_ = pt1_[:2, :] / pt1_[2, :] - pt1[:2, idx[:, 0].astype(int)]
    err2 = np.sum(tmp1_ * tmp1_, axis=0)

    err = err1 + err2
    aux = np.sqrt(err) / 2.0 < th
    return np.nonzero(aux)[0]


def steps(pps, inl, p):
    """Compute the remaining number of RANSAC steps."""
    e = 1.0 - inl
    return np.log(1.0 - p) / np.log(1.0 - (1.0 - e) ** pps)


def ransac_homography(pts1, pts2, mm, max_iter=2000, th=5, p=0.99, pps=4):
    """Estimate a robust homography via RANSAC."""
    if mm.shape[0] < 4:
        return np.eye(3, dtype=np.float64), np.zeros((0, 3), dtype=int)

    pt1 = np.column_stack((pts1[:, :2], np.ones((pts1.shape[0], 1), dtype=np.float64)))
    pt2 = np.column_stack((pts2[:, :2], np.ones((pts2.shape[0], 1), dtype=np.float64)))
    pt1 = pt1.T
    pt2 = pt2.T

    midx = np.array([], dtype=int)
    Nc = max_iter
    for c in range(max_iter):
        sidx = np.random.randint(0, mm.shape[0], size=pps)
        H, D = compute_homography(pt1[:, mm[sidx, 0].astype(int)], pt2[:, mm[sidx, 1].astype(int)])
        if D[-2] < 0.1:
            continue

        nidx = get_inliers(pt1, pt2, H, th, mm)
        if nidx.shape[0] > midx.shape[0]:
            midx = nidx
            Nc = int(np.ceil(steps(pps, midx.shape[0] / mm.shape[0], p)))
        if c > Nc:
            break

    if midx.size == 0:
        return np.eye(3, dtype=np.float64), np.zeros((0, 3), dtype=int)

    midx = mm[midx, :].astype(int)
    H, _ = compute_homography(pt1[:, midx[:, 0]], pt2[:, midx[:, 1]])
    return H, midx


def reinard(i1n, i1n_mask, i2n, i2n_mask):
    """Adjust the intensity statistics of one image to match another."""
    i12n_mask = i1n_mask & i2n_mask
    mu1 = np.mean(i1n[i12n_mask])
    sigma1 = np.std(i1n[i12n_mask])
    mu2 = np.mean(i2n[i12n_mask])
    sigma2 = np.std(i2n[i12n_mask])
    aux = i1n[i1n_mask]
    aux = (aux - mu1) / sigma1 * sigma2 + mu2
    i1n[i1n_mask] = aux
    return i1n


def transform_img(im, H, b_o):
    """Warp an image using a homography and return the resampled image and mask."""
    sz = im.shape
    b = np.array([[0, 0, 1], [sz[1], 0, 1], [0, sz[0], 1], [sz[1], sz[0], 1]], dtype=np.float64).T
    b_ = H @ b
    b_ = (b_ / b_[2, :]).T

    combined = np.vstack((b_, b_o))
    min_xy = np.floor(np.min(combined, axis=0))
    max_xy = np.ceil(np.max(combined, axis=0))
    width = max(1, int(max_xy[0] - min_xy[0]))
    height = max(1, int(max_xy[1] - min_xy[1]))

    T = np.array([[1.0, 0.0, -min_xy[0]], [0.0, 1.0, -min_xy[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_ = T @ H

    im_ = cv2.warpPerspective(im, H_, (width, height), flags=cv2.INTER_LINEAR)
    mask_src = np.ones((sz[0], sz[1]), dtype=np.uint8)
    mask_ = cv2.warpPerspective(mask_src, H_, (width, height), flags=cv2.INTER_NEAREST).astype(bool)
    return im_.astype(np.float32), mask_, b_


def prepare_homogeneous_corners(im):
    """Prepare the four image corners in homogeneous coordinates."""
    sz = im.shape
    return np.array([[0.0, 0.0, 1.0], [sz[1], 0.0, 1.0], [0.0, sz[0], 1.0], [sz[1], sz[0], 1.0]], dtype=np.float64)
