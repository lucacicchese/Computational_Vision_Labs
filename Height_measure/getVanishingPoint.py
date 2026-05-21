import os
import sys
sys.path.append(os.path.dirname(__file__))

from getLine import getLine
from getIntersection import getIntersection


def getVanishingPoint(I, colorId=1):
    #GETVANISHINGPOINT compute one vanishing point on the image
    if colorId is None:
        colorId = 1
    [I, m1, q1, borders1, _] = getLine(I, colorId)
    [I, m2, q2, borders2, _] = getLine(I, colorId)
    borders = [borders1, borders2]
    p_x, p_y = getIntersection(m1, q1, m2, q2)
    return I, p_x, p_y, borders
