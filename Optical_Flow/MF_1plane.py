# ----------------------------------------------------------
#
# Programma Python per la simulazione del motion field
# indotto dal moto rigido (T, w) della camera rispetto
# a un piano di equazione Z(X,Y) = pX + qY + d, dove
# (p,q) e' il gradiente di Z, che puo' essere espresso
# in termini di angoli di slant (0,90) e tilt (-90,+90)
# come (p,q) = -tan(sigma)(cos(tau), sin(tau)). Slant e tilt
# fissano l'orientazione del piano, mentre d = Z(0,0) fissa
# la distanza dalla camera. La focale della camera e' f.
# (X,Y,Z) e' un punto nello spazio espresso in coordinate
# di camera (origine nel centro ottico, Z = asse ottico).
# La legge di proiezione e' (x,y) = f * (X/Z,Y/Z) (pinhole camera).
#
# ----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

# -------------------------------------------------------
# Moto di telecamera: velocita' di traslazione
# -------------------------------------------------------
# T = np.zeros(3)
T = np.array([0.0, 0.0, 300.0])
# T = np.array([50.0, 0.0,   0.0])
# T = np.array([ 0.0, 100.0, 100.0])
# T = np.array([50.0, 100.0,   0.0])
# T = np.array([ 0.0, 100.0, 300.0])
# T = np.array([ 0.0,   0.0,   0.0])

# -------------------------------------------------------
# Moto di telecamera: velocita' di rotazione
# -------------------------------------------------------
# w = np.zeros(3)
w = np.array([0.0, -5.0, 0.0]) * np.pi / 180
# w = np.array([0.0, -3.0,  5.0]) * np.pi / 180
# w = np.array([0.0,  0.0,  5.0]) * np.pi / 180
# w = np.array([0.0,  0.0,  0.0]) * np.pi / 180

# -------------------------------------------------------
# Posizione del piano rispetto alla camera
# Il piano ha equazione Z(X,Y) = p*X + q*Y + d
# Gli angoli sigma (slant) e tau (tilt) ne fissano l'orientazione
# -------------------------------------------------------
sigma = 45 * np.pi / 180   # angolo di slant
# sigma = 30 * np.pi / 180
# sigma = 0  * np.pi / 180
tau = 0 * np.pi / 180       # angolo di tilt

# Gradiente del piano
p = -np.tan(sigma) * np.cos(tau)
q = -np.tan(sigma) * np.sin(tau)
d = 4000  # distanza del piano dall'origine della camera (Z(0,0))

# -------------------------------------------------------
# Focale della telecamera (in pixel)
# -------------------------------------------------------
f = 512

# -------------------------------------------------------
# Offset del punto principale (centro dell'immagine)
# -------------------------------------------------------
xp = 320
yp = 240

# Intervallo di sottocampionamento per la visualizzazione dei vettori MF
s = 40

# Griglia di coordinate immagine centrate nell'origine (sistema ottico)
x = np.arange(-xp, xp + 1, dtype=float)  # equivalente di -xp:xp in MATLAB
y = np.arange(-yp, yp + 1, dtype=float)
xg, yg = np.meshgrid(x, y)               # griglia 2D (come meshgrid in MATLAB)

# -------------------------------------------------------
# Calcolo della profondita' Z per ogni punto della griglia
# usando la formula di retroproiezione del piano:
#   Z(x,y) = d / (1 - (p/f)*x - (q/f)*y)
# e dei corrispondenti punti 3D (X,Y,Z) in coordinate di camera
# -------------------------------------------------------
c = d
zg = c * np.ones(xg.shape) / (1 - (p / f) * xg - (q / f) * yg)
Xg = (zg * xg) / f
Yg = (zg * yg) / f

# -------------------------------------------------------
# Calcolo del motion field
# Il MF e' la derivata temporale del punto immagine (x,y)
# e viene decomposto in:
#   - componente polare/traslazionale (t): dipende da T
#   - componente assiale/rotazionale (w): dipende da w
# -------------------------------------------------------

# Componente traslazionale del MF
ugt = -(1.0 / zg * (f * T[0] - T[2] * xg))
vgt = -(1.0 / zg * (f * T[1] - T[2] * yg))

# Componente rotazionale del MF
ugw = -(  -xg * yg / f * w[0]  +  (f**2 + xg**2) * (w[1] / f)  -  yg * w[2])
vgw = -(-(f**2 + yg**2) / f * w[0]  +  xg * yg * (w[1] / f)    +  xg * w[2])

# MF totale (somma dei campi traslazionale e rotazionale)
ug = ugt + ugw
vg = vgt + vgw

# -------------------------------------------------------
# Display della scena 3D: piano + camera + vettori T e w
# -------------------------------------------------------
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')

# Superficie del piano (sottocampionata per efficienza)
ax3d.plot_surface(
    Xg[s-1::s, s-1::s],
    Yg[s-1::s, s-1::s],
    zg[s-1::s, s-1::s],
    alpha=0.6
)

# Asse ottico della camera (verde), da Z=0 a Z=f
ax3d.plot([0, 0], [0, 0], [0, f], 'g.-')

# Vettore di traslazione T (blu), se non nullo
if np.linalg.norm(T) > 0:
    ax3d.plot(
        [0, f / 2 * T[0] / np.linalg.norm(T)],
        [0, f / 2 * T[1] / np.linalg.norm(T)],
        [0, f / 2 * T[2] / np.linalg.norm(T)],
        'b.-'
    )

# Vettore di rotazione w (rosso), se non nullo
if np.linalg.norm(w) > 0:
    ax3d.plot(
        [0, f / 2 * w[0] / np.linalg.norm(w)],
        [0, f / 2 * w[1] / np.linalg.norm(w)],
        [0, f / 2 * w[2] / np.linalg.norm(w)],
        'r.-'
    )

ax3d.set_xlabel('X')
ax3d.set_ylabel('Y')
ax3d.set_zlabel('Z')

# Rettangolo del piano immagine (verde), a distanza focale f
# Vertici in ordine: [xmin,xmin,xmax,xmax,xmin] x [ymin,ymax,ymax,ymin,ymin]
rect_x = [x[0],  x[0],  x[-1], x[-1], x[0]]
rect_y = [y[0],  y[-1], y[-1], y[0],  y[0]]
ax3d.plot(rect_x, rect_y, f * np.ones(5), 'g-')

# -------------------------------------------------------
# Calcolo del FOE (Focus of Expansion) del MF traslazionale
# Il FOE e' il punto immagine dove tutti i flussi ottici
# traslazionali convergono; corrisponde alla direzione di T
# -------------------------------------------------------
if T[2] != 0:
    # Caso generale: FOE finito (T ha componente Z != 0)
    foe = np.array([[f / T[2] * T[0] + xp - 1,
                     f / T[2] * T[1] + yp - 1]])
else:
    # Caso degenere: T(3)=0, il FOE e' all'infinito
    # Si visualizza una direzione (vettore dal centro verso il FOE)
    denom = (T[0]**2 + T[1]**2)**0.5
    foe = np.array([
        [xp - 1, yp - 1],
        [T[0] * 25 / denom + xp - 1, T[1] * 25 / denom + yp - 1]
    ])

# -------------------------------------------------------
# Coordinate immagine sottocampionate per il display del MF
# (si aggiunge l'offset del punto principale per passare
#  dal sistema ottico alle coordinate pixel dell'immagine)
# -------------------------------------------------------
xc = xg[s-1::s, s-1::s] + xp
yc = yg[s-1::s, s-1::s] + yp

# -------------------------------------------------------
# Figura 2: MF totale
# -------------------------------------------------------
fig_total, ax_total = plt.subplots()
ax_total.set_aspect('equal')
ax_total.set_title('MF totale')

# Vettori del MF totale (verde)
ax_total.quiver(
    xc, yc,
    ug[s-1::s, s-1::s],
    vg[s-1::s, s-1::s],
    scale=1, scale_units='xy', angles='xy', color='green'
)
ax_total.plot(xc.ravel(), yc.ravel(), 'g.')

# FOE (giallo)
ax_total.plot(foe[0, 0], foe[0, 1], 'oy')
ax_total.plot(foe[:, 0], foe[:, 1], '-y')

# -------------------------------------------------------
# Figura 3: MF polare (traslazionale)
# -------------------------------------------------------
fig_polar, ax_polar = plt.subplots()
ax_polar.set_aspect('equal')
ax_polar.set_title('MF polare')

# Vettori del MF traslazionale (blu)
ax_polar.quiver(
    xc, yc,
    ugt[s-1::s, s-1::s],
    vgt[s-1::s, s-1::s],
    scale=1, scale_units='xy', angles='xy', color='blue'
)
ax_polar.plot(xc.ravel(), yc.ravel(), 'b.')

# FOE (violetto e ciano)
ax_polar.plot(foe[0, 0], foe[0, 1], 'o', color='#7E2F8E')  # violetto
ax_polar.plot(foe[:, 0], foe[:, 1], '-c')

# -------------------------------------------------------
# Figura 4: MF assiale (rotazionale)
# -------------------------------------------------------
fig_axial, ax_axial = plt.subplots()
ax_axial.set_aspect('equal')
ax_axial.set_title('MF assiale')

# Vettori del MF rotazionale (rosso)
ax_axial.quiver(
    xc, yc,
    ugw[s-1::s, s-1::s],
    vgw[s-1::s, s-1::s],
    scale=1, scale_units='xy', angles='xy', color='red'
)
ax_axial.plot(xc.ravel(), yc.ravel(), 'r.')

# -------------------------------------------------------
# Stima del FOE con e senza rumore (esperimento del 08-04-2016)
# Si usano n punti scelti a caso sull'immagine per stimare
# il FOE risolvendo un sistema lineare ai minimi quadrati
# -------------------------------------------------------

n = 100  # numero di punti usati per la stima

# Matrice dei dati: [x, y, u_traslazionale, v_traslazionale]
t_data = np.column_stack([xg.ravel(), yg.ravel(), ugt.ravel(), vgt.ravel()])

# Selezione casuale di n punti (equivalente di randperm in MATLAB)
idx = np.random.permutation(t_data.shape[0])[:n]
t_data = t_data[idx, :]

# Sistema lineare per la stima del FOE (senza rumore):
#   A * r = b  con  A = [-v, u]  e  b = -v*x + u*y
A = np.column_stack([-t_data[:, 3], t_data[:, 2]])
b = -t_data[:, 3] * t_data[:, 0] + t_data[:, 2] * t_data[:, 1]
r, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # FOE stimato senza rumore

# Aggiornamento del FOE ground truth (si rimuove l'offset del punto principale)
foe = foe - np.array([xp, yp]) + 1

print("FOE ground truth:")
print(foe)
print("FOE stimato (senza rumore):")
print(r.T)

# Visualizzazione del FOE sul grafico del MF totale
plt.figure(fig_total.number)
ax_total.plot(foe[0, 0] + xp - 1, foe[0, 1] + yp - 1, 'o', color='#7E2F8E')  # violetto
ax_total.plot(r[0] + xp - 1, r[1] + yp - 1, '+m')  # crocetta magenta: FOE stimato senza rumore

# -------------------------------------------------------
# Aggiunta di rumore gaussiano alle coordinate dei punti
# e nuova stima del FOE con rumore
# -------------------------------------------------------
noise_std = 60  # deviazione standard del rumore (in pixel); alternativa: 30
t_data[:, :2] = t_data[:, :2] + np.random.normal(0, noise_std, (t_data.shape[0], 2))

A = np.column_stack([-t_data[:, 3], t_data[:, 2]])
b = -t_data[:, 3] * t_data[:, 0] + t_data[:, 2] * t_data[:, 1]
r_noise, _, _, _ = np.linalg.lstsq(A, b, rcond=None)  # FOE stimato con rumore

print("FOE stimato (con rumore):")
print(r_noise.T)

# Cerchietto nero: FOE stimato con rumore
ax_total.plot(r_noise[0] + xp - 1, r_noise[1] + yp - 1, 'ok')

plt.show()