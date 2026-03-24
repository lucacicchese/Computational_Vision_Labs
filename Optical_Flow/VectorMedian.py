# ----------------------------------------------------------
#
# VectorMedian: calcola il vettore mediano in una finestra
# di vicinato attorno al pixel (x, y).
#
# Il vettore mediano e' definito come il vettore nella finestra
# che minimizza la somma delle distanze euclidee dagli altri
# vettori (Astola et al., 1990). E' robusto agli outlier
# rispetto alla media vettoriale.
#
# Input:
#   x, y    - coordinata del pixel centrale (1-based, come MATLAB)
#   Vx, Vy  - matrici del flusso ottico (shape: [height, width])
#   r       - raggio della finestra in unita' di campionamento s
#             (la finestra copre 2r+1 campioni per lato)
#   s       - passo di campionamento all'interno della finestra
#
# Output:
#   med     - array [vx_med, vy_med]: il vettore mediano
#
# ----------------------------------------------------------

import numpy as np


def VectorMedian(x, y, Vx, Vy, r, s):

    # ----------------------------------------------------------
    # Calcolo degli indici di inizio del vicinato lungo x e y.
    # Si parte da x - s*r e si avanza di s finche' si rientra
    # nei limiti dell'immagine (indici 1-based come in MATLAB).
    # ----------------------------------------------------------
    xstart = x - s * r
    while xstart < 1:
        xstart += s

    ystart = y - s * r
    while ystart < 1:
        ystart += s

    # Indici 1-based del vicinato (come range(xstart, x+s*r+1, s) in MATLAB)
    xneigh = np.arange(xstart, min(Vx.shape[1], x + s * r) + 1, s)  # colonne
    yneigh = np.arange(ystart, min(Vx.shape[0], y + s * r) + 1, s)  # righe

    # Conversione in indici 0-based per NumPy
    xneigh_0 = xneigh - 1
    yneigh_0 = yneigh - 1

    # Estrazione del sotto-campo di flusso nella finestra e appiattimento
    Vsx = Vx[np.ix_(yneigh_0, xneigh_0)].ravel()
    Vsy = Vy[np.ix_(yneigh_0, xneigh_0)].ravel()

    n = len(Vsx)

    # ----------------------------------------------------------
    # Calcolo della matrice delle distanze euclidee al quadrato
    # tra tutti i vettori della finestra.
    # dstV[h,k] = ||v_h - v_k||^2
    # La matrice e' simmetrica: si calcola solo il triangolo
    # superiore e si copia per simmetria (come nel codice MATLAB).
    # ----------------------------------------------------------
    dstV = np.zeros((n, n))
    for h in range(n - 1):
        for k in range(h + 1, n):
            d = (Vsx[k] - Vsx[h])**2 + (Vsy[k] - Vsy[h])**2
            dstV[h, k] = d
            dstV[k, h] = d  # matrice simmetrica

    # Il vettore mediano e' quello con la somma minima delle distanze
    imin = np.argmin(dstV.sum(axis=0))
    med = np.array([Vsx[imin], Vsy[imin]])

    return med
