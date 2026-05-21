import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np

#UNTITLED2 compute intersection of 2 lines y=m1*x+q1 and y=m2*x+q2

def getIntersection(m1, q1, m2, q2):
    if (m1 < np.inf) and (m2 < np.inf):
        p_x = round((q1 - q2) / (m2 - m1))
        p_y = round(m1 * p_x + q1)
    elif m1 == np.inf and (m2 < np.inf):
        p_x = q1
        p_y = round(m2 * p_x + q2)
    elif m2 == np.inf and (m1 < np.inf):
        p_x = q2
        p_y = round(m1 * p_x + q1)
    else:
        raise ValueError('vanishing point at infinity')
    if m1 == m2:
        raise ValueError('vanishing point at infinity')
    return p_x, p_y
