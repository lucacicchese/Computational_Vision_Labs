import os
import sys
sys.path.append(os.path.dirname(__file__))

from getLine import getLine
from hm_utils import imshow


def setTarget(I):
    #SETTARGET set the segment to measure; the first point selected to define
    #the segment is the one on the ground
    imshow(I)
    [I, m, q, _, ins] = getLine(I, 2)
    a_x = ins[0]
    a_y = ins[1]
    c_x = ins[2]
    c_y = ins[3]
    return I, m, q, a_x, a_y, c_x, c_y
