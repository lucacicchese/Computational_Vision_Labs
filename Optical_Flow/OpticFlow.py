# ----------------------------------------------------------
#
# Programma Python per la stima del flusso ottico
# con il metodo di Lucas-Kanade (locale, basato su derivate
# spaziali e temporali) e regolarizzazione con Vector Median.
#
# Pipeline:
#   1. Lettura di una sequenza di nframes immagini
#   2. Pre-filtraggio temporale e calcolo delle derivate
#      spaziali (fx, fy) e temporale (ft) con filtri separabili
#   3. Stima del flusso (Vx, Vy) per ogni pixel risolvendo
#      il sistema di Lucas-Kanade: G * v = -b,
#      dove G e' la matrice di struttura locale e b il vettore
#      dei prodotti misti spazio-temporali
#   4. Affidabilita' stimata come il minore autovalore di G
#   5. Regolarizzazione con Vector Median su finestra (2r+1)^2
#   6. Visualizzazione del flusso grezzo, regolarizzato
#      e della mappa di affidabilita'
#
# ----------------------------------------------------------

import numpy as np
from scipy.signal import convolve2d
from skimage import io
import matplotlib.pyplot as plt

from VectorMedian import VectorMedian   # funzione ausiliaria (file separato)


# =========================================================================
# Parametri
# =========================================================================

s1     = 1      # passo di sottocampionamento per la stima del flusso
thrRel = 0.01   # soglia sull'affidabilita' (minimo autovalore di G)

# Parametri per la regolarizzazione con Vector Median
r  = 3   # raggio in unita' di campionamento: finestra (2r+1) x (2r+1) campioni
s3 = 8   # passo di campionamento all'interno della finestra del Vector Median

# Parametri per la visualizzazione del flusso
s2    = 8   # passo di sottocampionamento per la visualizzazione
scale = 2   # fattore di scala delle frecce quiver

# =========================================================================
# Filtri
# =========================================================================

# Filtro di pre-filtraggio temporale/spaziale (smoothing gaussiano 1D)
prefilt  = np.array([0.223755, 0.552490, 0.223755])

# Filtro derivativo antisimmetrico (derivata prima discreta)
derivfilt = np.array([0.453014, 0.0, -0.453010])

# Filtro di smoothing spaziale per la stima delle correlazioni
# (media pesata che privilegia il pixel centrale)
blur = np.array([1, 6, 15, 20, 15, 6, 1], dtype=float)
blur = blur / blur.sum()

# Numero di frame della sequenza (deve corrispondere alla lunghezza di prefilt/derivfilt)
nframes = 3

# =========================================================================
# Lettura della sequenza di immagini
# =========================================================================

# Legge la dimensione dell'immagine di riferimento
ref_img = io.imread('yos.10.pnm')
dimy, dimx = ref_img.shape[:2]

# Lettura dei nframes frame successivi
seq = []
for i in range(1, nframes + 1):
    filename = f'yos.{9 + i}.pnm'
    # filename = f'lab{2+i}gl.bmp'
    # filename = f'img{i}.bmp'
    print(f'Reading image {filename}')
    img = io.imread(filename).astype(np.float32)
    seq.append(img)

# Visualizzazione della sequenza come animazione (equivalente di movie())
fig_seq, ax_seq = plt.subplots()
for i in range(nframes):
    ax_seq.cla()
    ax_seq.imshow(seq[i], cmap='gray')
    ax_seq.set_title(f'Frame {i + 1}')
    plt.pause(0.3)
plt.draw()

# =========================================================================
# DERIVATE SPAZIALI E TEMPORALI
# =========================================================================

# ----------------------------------------------------------
# Pre-filtraggio temporale: combinazione lineare dei frame
# pesata con i coefficienti di prefilt.
# f e' l'immagine "media" smussata nel tempo.
# ----------------------------------------------------------
f = np.zeros((dimy, dimx), dtype=np.float32)
for i in range(nframes):
    f = f + prefilt[i] * seq[i]

# ----------------------------------------------------------
# Derivate spaziali fx e fy:
# Si applica prima il filtro di pre-filtraggio nella direzione
# ortogonale, poi il filtro derivativo nella direzione d'interesse.
# conv2d con 'same' mantiene le dimensioni dell'ingresso.
# prefilt[:, None] = filtro colonna (direzione y)
# prefilt[None, :] = filtro riga   (direzione x)
# ----------------------------------------------------------

# Derivata in x: pre-filtra in y, poi deriva in x
fx = convolve2d(convolve2d(f, prefilt[:, None], mode='same'),
                derivfilt[None, :], mode='same')

# Derivata in y: pre-filtra in x, poi deriva in y
fy = convolve2d(convolve2d(f, prefilt[None, :], mode='same'),
                derivfilt[:, None], mode='same')

# ----------------------------------------------------------
# Derivata temporale ft:
# Combinazione lineare dei frame pesata con i coefficienti
# di derivfilt (con segno negativo, come in MATLAB).
# ----------------------------------------------------------
ft = np.zeros((dimy, dimx), dtype=np.float32)
for i in range(nframes):
    ft = ft - derivfilt[i] * seq[i]

# Pre-filtraggio spaziale di ft (in y poi in x)
ft = convolve2d(convolve2d(ft, prefilt[:, None], mode='same'),
                prefilt[None, :], mode='same')

# =========================================================================
# STIMA DELLE CORRELAZIONI LOCALI (prodotti delle derivate, smussati con blur)
# =========================================================================
# I prodotti vengono prima formati pixel per pixel, poi smussati con il
# filtro blur separabilmente in y e in x.
# Questi sono gli elementi della matrice di struttura G e del vettore b
# di Lucas-Kanade, integrati su una finestra locale.

# blur[:, None] = filtro colonna; blur[None, :] = filtro riga
fx2 = convolve2d(convolve2d(fx * fx, blur[:, None], mode='same'), blur[None, :], mode='same')
fy2 = convolve2d(convolve2d(fy * fy, blur[:, None], mode='same'), blur[None, :], mode='same')
fxy = convolve2d(convolve2d(fx * fy, blur[:, None], mode='same'), blur[None, :], mode='same')
fxt = convolve2d(convolve2d(fx * ft, blur[:, None], mode='same'), blur[None, :], mode='same')
fyt = convolve2d(convolve2d(fy * ft, blur[:, None], mode='same'), blur[None, :], mode='same')

# =========================================================================
# STIMA DEL FLUSSO E DELL'AFFIDABILITA'
# =========================================================================
# Per ogni pixel (y, x), si risolve il sistema di Lucas-Kanade:
#
#   G * v = -b    con   G = [[fx2, fxy], [fxy, fy2]]
#                        b = [fxt, fyt]
#
# Il flusso e' valido solo se il minimo autovalore di G e' > thrRel
# (indica che entrambe le direzioni hanno variazione di gradiente
# sufficiente, evitando l'aperture problem).
# =========================================================================

ydim, xdim = fx.shape

# Dimensioni dell'output: stesse di fx quando s1=1
Vx     = np.zeros((ydim // s1, int(np.ceil(xdim / s1))))
Vy     = np.zeros((ydim // s1, int(np.ceil(xdim / s1))))
reliab = np.zeros((ydim, xdim))

cx = 0  # indice di colonna nell'output (0-based)
for x in range(0, xdim, s1):           # x: colonna nell'immagine (0-based)
    cy = 0                              # indice di riga nell'output (0-based)
    for y in range(0, ydim, s1):       # y: riga nell'immagine (0-based)

        # Matrice di struttura locale G (2x2, simmetrica)
        G = np.array([[fx2[y, x], fxy[y, x]],
                      [fxy[y, x], fy2[y, x]]])

        # Vettore dei prodotti spazio-temporali
        b = np.array([fxt[y, x], fyt[y, x]])

        # Autovalori di G (numpy li restituisce in ordine crescente)
        autoval = np.linalg.eigvalsh(G)
        reliab[y, x] = autoval[0]   # autovalore minimo = affidabilita'

        if reliab[y, x] < thrRel:
            # Affidabilita' insufficiente: flusso nullo
            Vx[cy, cx] = 0.0
            Vy[cy, cx] = 0.0
        else:
            # Soluzione del sistema G * v = -b (equivalente di -G\b in MATLAB)
            v = np.linalg.solve(G, -b)
            Vx[cy, cx] = v[0]
            Vy[cy, cx] = v[1]

        cy += 1
    cx += 1

# =========================================================================
# REGOLARIZZAZIONE CON VECTOR MEDIAN
# =========================================================================
# Per ogni posizione campionata (passo s2), si sostituisce il vettore
# di flusso con il vettore mediano calcolato su una finestra locale
# di raggio r con passo s3.
# VectorMedian usa indici 1-based (come MATLAB).

Vxn = np.zeros_like(Vx)
Vyn = np.zeros_like(Vy)

for rw in range(s2, ydim + 1, s2):          # rw: riga 1-based
    for cl in range(s2, xdim + 1, s2):      # cl: colonna 1-based
        vmed = VectorMedian(cl, rw, Vx, Vy, r, s3)
        Vxn[rw - 1, cl - 1] = vmed[0]       # conversione a indice 0-based
        Vyn[rw - 1, cl - 1] = vmed[1]

# =========================================================================
# VISUALIZZAZIONE
# =========================================================================

# Griglia di punti per quiver (coordinate di pixel, 1-based come MATLAB)
xramp = np.arange(s2, xdim + 1, s2)   # colonne campionate
yramp = np.arange(s2, ydim + 1, s2)   # righe campionate
xgrid, ygrid = np.meshgrid(xramp, yramp)

# Estrazione del flusso nelle posizioni campionate
# Conversione da indici 1-based a 0-based per l'accesso agli array
Vxn_sub = Vxn[s2 - 1:ydim:s2, s2 - 1:xdim:s2]
Vyn_sub = Vyn[s2 - 1:ydim:s2, s2 - 1:xdim:s2]
Vx_sub  = Vx[s2 - 1:ydim:s2,  s2 - 1:xdim:s2]
Vy_sub  = Vy[s2 - 1:ydim:s2,  s2 - 1:xdim:s2]

# ----------------------------------------------------------
# Figura 100: flusso regolarizzato (Vector Median), frecce rosse
# N.B. In MATLAB la figura mostra Vxn/Vyn come "noisy flow"
#      e Vx/Vy come "regularized flow": si mantiene lo stesso ordine.
# ----------------------------------------------------------
fig100, ax100 = plt.subplots(num=100)
ax100.imshow(seq[0], cmap='gray', origin='upper')
ax100.quiver(xgrid, ygrid, Vxn_sub, Vyn_sub,
             scale=1.0 / scale, scale_units='xy', angles='xy', color='red')
ax100.set_aspect('equal')
ax100.set_title('Noisy flow (Vector Median regularized)')

# ----------------------------------------------------------
# Figura 200: flusso grezzo (Lucas-Kanade), frecce blu
# ----------------------------------------------------------
fig200, ax200 = plt.subplots(num=200)
ax200.imshow(seq[0], cmap='gray', origin='upper')
ax200.quiver(xgrid, ygrid, Vx_sub, Vy_sub,
             scale=1.0 / scale, scale_units='xy', angles='xy', color='blue')
ax200.set_aspect('equal')
ax200.set_title('Regularized flow (Lucas-Kanade)')

# ----------------------------------------------------------
# Figura 300: mappa di affidabilita' (minimo autovalore di G)
# ----------------------------------------------------------
fig300, ax300 = plt.subplots(num=300)
ax300.imshow(reliab, cmap='gray')
ax300.set_aspect('equal')
ax300.set_title('Reliability map (minimum eigenvalue of G)')

plt.show()
