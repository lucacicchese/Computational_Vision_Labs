import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np
import cv2

from getVanishingPoint import getVanishingPoint
from dispVanishingPoints import dispVanishingPoints
from getVanishingLine import getVanishingLine
from setReference import setReference
from setTarget import setTarget
from hm_utils import insert_shape, imshow


def main():
    I = cv2.imread('./tennis2.jpg')
    startI = I.copy()
    s = I.shape
    vpx = [0, 0]
    vpy = [0, 0]
    borders = [None, None]
    for u in range(2):
        I, vpx[u], vpy[u], borders[u] = getVanishingPoint(I, u + 1)
    dispVanishingPoints(I, vpx, vpy, borders, [1, 2])
    m_vl, q_vl = getVanishingLine(vpx[0], vpy[0], vpx[1], vpy[1])
    I, m_ref, q_ref, h, A_x, A_y, B_x, B_y = setReference(startI)
    I, m_tar, q_tar, a_x, a_y, c_x, c_y = setTarget(I)
    if m_tar != np.inf and m_ref != np.inf:
        D = np.cross([m_tar, -1, q_tar], [m_ref, -1, q_ref])
        D = D / D[2]
    elif m_tar == np.inf and m_ref != np.inf:
        D = np.cross([1 / q_tar, 0, 1], [m_ref, -1, q_ref])
        D = D / D[2]
    elif m_tar != np.inf and m_ref == np.inf:
        D = np.cross([m_tar, -1, q_tar], [1 / q_ref, 0, 1])
        D = D / D[2]
    else:
        D = np.cross([1 / q_tar, 0, 1], [1 / q_ref, 0, 1])
    D_x = D[0]
    D_y = D[1]
    Aa_line = np.cross([a_x, a_y, 1], [A_x, A_y, 1])
    I = insert_shape(I, 'Line', [0, -Aa_line[2] / Aa_line[1], s[1], -(Aa_line[0] * s[1] + Aa_line[2]) / Aa_line[1]], color='green', LineWidth=2)
    if m_vl != np.inf:
        parallel_vp = np.cross(Aa_line, [m_vl, -1, q_vl])
        parallel_vp = parallel_vp / parallel_vp[2]
    else:
        parallel_vp = np.cross(Aa_line, [1 / q_vl, 0, 1])
    height_line = np.cross(parallel_vp, [c_x, c_y, 1])
    I = insert_shape(I, 'Line', [0, -height_line[2] / height_line[1], s[1], -(height_line[0] * s[1] + height_line[2]) / height_line[1]], color='white', LineWidth=2)
    imshow(I)
    if m_ref != np.inf:
        C = np.cross([m_ref, -1, q_ref], height_line)
    else:
        C = np.cross([1 / q_ref, 0, 1], height_line)
    C = C / C[2]
    C_x = C[0]
    C_y = C[1]
    AC = np.linalg.norm(np.array([A_x, A_y]) - np.array([C_x, C_y]))
    BD = np.linalg.norm(np.array([B_x, B_y]) - np.array([D_x, D_y]))
    CD = np.linalg.norm(np.array([C_x, C_y]) - np.array([D_x, D_y]))
    AB = np.linalg.norm(np.array([A_x, A_y]) - np.array([B_x, B_y]))
    cross_ratio = AC * BD / (AB * CD)
    AB_real = h
    AC_real = AB_real * cross_ratio
    print('heigth = ' + str(AC_real))


if __name__ == '__main__':
    main()
