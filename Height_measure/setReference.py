import os
import sys
sys.path.append(os.path.dirname(__file__))

from getLine import getLine
from hm_utils import imshow


def setReference(I):
    #fissa un segmento perpendicolare a e poggiato su il terreno, di cui si conosca la lunghezza 
    #il primo punto selezionato per definire il segmento è quello che poggia
    #a terra
    imshow(I)
    [I, m, q, _, ins] = getLine(I, 2)
    A_x = ins[0]
    A_y = ins[1]
    B_x = ins[2]
    B_y = ins[3]
    h = float(input('insert segment length in real world\n'))
    return I, m, q, h, A_x, A_y, B_x, B_y
