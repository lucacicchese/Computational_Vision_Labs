import os
import sys
sys.path.append(os.path.dirname(__file__))

import numpy as np


def getVanishingLine(vpx1, vpy1, vpx2, vpy2):
    if vpx2 != vpx1:
        m = (vpy2 - vpy1) / (vpx2 - vpx1)
        q = vpy1 - m * vpx1
    else:
        m = np.inf
        q = vpx1
    return m, q
