# #########################################################################
#
#   Structure from Motion basata sull'algoritmo di Longuet-Higgins
#
#   0. Data la focale f e il movimento della camera (T, Omega)
#   1. Costruzione di una scena sintetica basata su 7 piani intersecantisi
#   2. Generazione del motion field
#   3. Stima della parallasse con il metodo del "triangolo d'aria" di Cipolla
#      a. creazione di un triangolo equilatero con vertici random
#      b. interpolazione di un flusso lineare che sfrutta i 3 vettori
#         di motion field nei vertici
#      c. parallasse = m.f. reale - interpolato nel centro del triangolo
#   4. Stima del FOE che usa N vettori di parallasse
#      a. soluzione LS
#      b. soluzione RANSAC
#   5. Stima di Omega basata sul FOE
#   6. Stima della componente assiale del motion field
#   7. Stima della componente polare del motion field
#   8. Stima della profondita' Z(X,Y) in ogni pixel (x,y)
#
#   (C) Computational Vision Group @ DINFO Universita' di Firenze
#                       last modified: 30 March 2021
#
# #########################################################################

import numpy as np
import matplotlib
matplotlib.use('TkAgg')   # change to 'Qt5Agg' or 'Agg' if TkAgg is unavailable
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D       # noqa: F401
from matplotlib.path import Path              # per inpolygon


# =========================================================================
# get_flow: calcola motion field e invarianti differenziali del I ordine
#           per un singolo piano di parametri (sigma, tau, c)
#
# Input:
#   T, Om   - vettori traslazione/rotazione camera
#   sigma   - slant del piano (rad)
#   tau     - tilt del piano (rad)
#   f       - focale (pixel)
#   xp, yp  - half-size immagine (offset punto principale)
#   c       - intercetta (Z(0,0) = c)
#
# Output: dizionario con tutti i campi del MF e metadati
# =========================================================================
def get_flow(T, Om, sigma, tau, f, xp, yp, c):

    # Gradiente di profondita' del piano
    p = -np.tan(sigma) * np.cos(tau)
    q = -np.tan(sigma) * np.sin(tau)

    # Griglia di coordinate immagine centrate nell'origine ottica
    x = np.arange(-xp, xp + 1, dtype=float)
    y = np.arange(-yp, yp + 1, dtype=float)
    xg, yg = np.meshgrid(x, y)

    # Profondita' e coordinate 3D di ogni punto del piano
    zg = c * np.ones(xg.shape) / (1 - (p / f) * xg - (q / f) * yg)
    Xg = (zg * xg) / f
    Yg = (zg * yg) / f

    # Componente traslazionale (polare) del MF
    ugt = -(1.0 / zg * (f * T[0] - T[2] * xg))
    vgt = -(1.0 / zg * (f * T[1] - T[2] * yg))

    # Componente rotazionale (assiale) del MF
    ugw = -(  -xg * yg / f * Om[0]  +  (f**2 + xg**2) * (Om[1] / f)  -  yg * Om[2])
    vgw = -(-(f**2 + yg**2) / f * Om[0]  +  xg * yg * (Om[1] / f)    +  xg * Om[2])

    # MF totale
    ug = ugt + ugw
    vg = vgt + vgw

    # Invarianti differenziali del I ordine
    divergence = -(
        (-p * T[0] - q * T[1] + (-2 + 3 * p / f * xg + 3 * q / f * yg) * T[2]) / c
        + (-3) * yg * Om[0] / f + 3 * xg * Om[1] / f
    )
    rotational = -(
        (q * T[0] - p * T[1] + ((-1) * q / f * xg + p / f * yg) * T[2]) / c
        + xg * Om[0] / f + yg * Om[1] / f + 2 * Om[2]
    )
    shearx = -(
        (-p * T[0] + q * T[1] + (p / f * xg - q / f * yg) * T[2]) / c
        + yg * Om[0] / f + xg * Om[1] / f
    )
    sheary = -(
        (-q * T[0] - p * T[1] + (q / f * xg + p / f * yg) * T[2]) / c
        + (-1) * xg * Om[0] / f + yg * Om[1] / f
    )

    shear   = np.sqrt(shearx**2 + sheary**2)
    cimmup  = (divergence + shear) / 2
    cimmlo  = (divergence - shear) / 2

    # FOE di ground truth (coordinate pixel, origine all'angolo dell'immagine)
    if T[2] != 0:
        foe = np.array([[f / T[2] * T[0] + xp + 1,
                         f / T[2] * T[1] + yp + 1]])
    else:
        denom = (T[0]**2 + T[1]**2)**0.5
        foe = np.array([
            [xp + 1, yp + 1],
            [T[0] * 25 / denom + xp + 1,
             T[1] * 25 / denom + yp + 1]
        ])

    return {
        'xg': xg, 'yg': yg, 'zg': zg, 'Xg': Xg, 'Yg': Yg,
        'ugt': ugt, 'vgt': vgt, 'ugw': ugw, 'vgw': vgw,
        'ug': ug, 'vg': vg,
        'foe': foe,
        'xp': xp, 'yp': yp,
        'T': T, 'Om': Om,
        'sigma': sigma, 'tau': tau, 'f': f, 'c': c,
        'x': x, 'y': y,
        'cimmup': cimmup, 'cimmlo': cimmlo,
    }


# =========================================================================
# show_flow: visualizza fino a 4 figure per un dato campo r
# =========================================================================
def show_flow(r, s, ss, to_plot):

    xg  = r['xg'];  yg  = r['yg'];  zg  = r['zg']
    Xg  = r['Xg'];  Yg  = r['Yg']
    ugt = r['ugt']; vgt = r['vgt']
    ugw = r['ugw']; vgw = r['vgw']
    ug  = r['ug'];  vg  = r['vg']
    foe = r['foe']
    xp  = r['xp'];  yp  = r['yp']
    T   = r['T'];   Om  = r['Om'];  f = r['f']
    x   = r['x'];   y   = r['y']

    xc = xg[s-1::s, s-1::s] + xp
    yc = yg[s-1::s, s-1::s] + yp
    handles = [None] * 7

    # Scena 3D
    if to_plot[0]:
        fig = plt.figure()
        handles[0] = fig
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(Xg[s-1::s, s-1::s], Yg[s-1::s, s-1::s],
                        zg[s-1::s, s-1::s], alpha=0.6)
        ax.plot([0, 0], [0, 0], [0, f], 'g.-')
        if np.linalg.norm(T) > 0:
            ax.plot([0, f/2*T[0]/np.linalg.norm(T)],
                    [0, f/2*T[1]/np.linalg.norm(T)],
                    [0, f/2*T[2]/np.linalg.norm(T)], 'b.-')
        if np.linalg.norm(Om) > 0:
            ax.plot([0, f/2*Om[0]/np.linalg.norm(Om)],
                    [0, f/2*Om[1]/np.linalg.norm(Om)],
                    [0, f/2*Om[2]/np.linalg.norm(Om)], 'r.-')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        rect_x = [x[0],  x[0],  x[-1], x[-1], x[0]]
        rect_y = [y[0],  y[-1], y[-1], y[0],  y[0]]
        ax.plot(rect_x, rect_y, f * np.ones(5), 'g-')
        ax.set_title(ss)

    # MF totale
    if to_plot[1]:
        fig, ax = plt.subplots(); handles[1] = fig
        ax.set_aspect('equal'); ax.set_title(ss + 'complete motion field')
        ax.quiver(xc, yc, ug[s-1::s, s-1::s], vg[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='green')
        ax.plot(xc.ravel(), yc.ravel(), 'g.')
        ax.plot(foe[0, 0], foe[0, 1], 'or')
        ax.plot(foe[:, 0], foe[:, 1], '-r')

    # MF polare
    if to_plot[2]:
        fig, ax = plt.subplots(); handles[2] = fig
        ax.set_aspect('equal'); ax.set_title(ss + 'polar (T) component')
        ax.quiver(xc, yc, ugt[s-1::s, s-1::s], vgt[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='blue')
        ax.plot(xc.ravel(), yc.ravel(), 'b.')
        ax.plot(foe[0, 0], foe[0, 1], 'or')
        ax.plot(foe[:, 0], foe[:, 1], '-r')

    # MF assiale
    if to_plot[3]:
        fig, ax = plt.subplots(); handles[3] = fig
        ax.set_aspect('equal'); ax.set_title(ss + 'axial (w) component')
        ax.quiver(xc, yc, ugw[s-1::s, s-1::s], vgw[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='red')
        ax.plot(xc.ravel(), yc.ravel(), 'r.')

    return handles


# =========================================================================
# not_aligned: verifica che i 3 punti non siano (quasi) allineati
#   Restituisce True se la distanza minima tra un vertice e il lato
#   opposto e' > 50 pixel
# =========================================================================
def not_aligned(pts):
    # aux: per ogni coppia di vertici (lato), controlla la distanza del terzo
    aux = [(0, 1, 2), (1, 2, 0), (0, 2, 1)]
    min_dist = np.inf
    for (i, j, k) in aux:
        # retta omogenea passante per pts[i] e pts[j]
        line = np.cross(np.append(pts[i], 1), np.append(pts[j], 1)).astype(float)
        norm_line = np.sqrt(line[0]**2 + line[1]**2)
        line = line / norm_line
        dist = abs(np.dot(line, np.append(pts[k], 1))) / np.sqrt(line[0]**2 + line[1]**2)
        if dist < min_dist:
            min_dist = dist
    return min_dist > 50


# =========================================================================
# get_angle: calcola l'angolo (in gradi) tra ogni coppia di righe
#            di due matrici Nx2
# =========================================================================
def get_angle(tmp1, tmp2):
    angles = np.zeros(tmp1.shape[0])
    for i in range(tmp1.shape[0]):
        cos_a = np.dot(tmp1[i], tmp2[i]) / (np.linalg.norm(tmp1[i]) * np.linalg.norm(tmp2[i]))
        angles[i] = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))
    return angles


# =========================================================================
# get_pt_into_radius: restituisce la lista di punti interi sul cerchio
#                     di centro v e raggio radius
# =========================================================================
def get_pt_into_radius(v, radius):
    cx, cy = v[0], v[1]
    pt_list = []
    for xi in range(int(cx - radius), int(cx + radius) + 1):
        yi = np.sqrt(max(radius**2 - (xi - cx)**2, 0))
        pt_list.append([xi, cy + yi])
        pt_list.append([xi, cy - yi])
    pt_list = np.unique(np.round(pt_list).astype(int), axis=0)
    return pt_list


# =========================================================================
# inpolygon: equivalente di inpolygon di MATLAB
#   Restituisce un array booleano: True per i punti dentro il poligono
# =========================================================================
def inpolygon(px, py, vx, vy):
    path = Path(np.column_stack([vx, vy]))
    points = np.column_stack([px, py])
    return path.contains_points(points)


# =========================================================================
# MAIN
# =========================================================================

# -------------------------------------------------------------------------
# Inizializzazione
# -------------------------------------------------------------------------
xp = 320
yp = 320
offset = np.array([xp + 1, yp + 1])  # [(size(flow,2)+1)/2, (size(flow,1)+1)/2]
f  = 320.0
T  = np.array([100.0, 100.0, 1000.0])        # velocita' 3D di traslazione
Om = np.array([1.0, 2.0, -5.0]) * np.pi / 180  # velocita' 3D di rotazione

# Superficie cilindrica: tau e d costanti
d   = 6500    # distanza tra piano e origine sistema di riferimento camera
tau = 0.0 * np.pi / 180  # tilt

# -------------------------------------------------------------------------
# Calcolo del motion field per ciascuno dei 7 piani
# c = d/cos(sigma) e' l'intercetta del piano (Z(0,0))
# -------------------------------------------------------------------------
sigma_vals = np.array([-45, -30, -15, 0, 15, 30, 45]) * np.pi / 180
results = []
for sigma in sigma_vals:
    c_k = d / np.cos(sigma)
    results.append(get_flow(T, Om, sigma, tau, f, xp, yp, c_k))

r1, r2, r3, r4, r5, r6, r7 = results  # alias per compatibilita' con il codice MATLAB

# -------------------------------------------------------------------------
# Stack delle coordinate 3D e dei motion fields dei 7 piani
# X, Y, Z, ug, vg, ugt, vgt, ugw, vgw hanno shape (H, W, 7)
# -------------------------------------------------------------------------
nplanes = len(results)
H = results[0]['Xg'].shape[0]
W = results[0]['Xg'].shape[1]

X   = np.stack([r['Xg'] for r in results], axis=2)
Y   = np.stack([r['Yg'] for r in results], axis=2)
Z   = np.stack([r['zg'] for r in results], axis=2)
ug  = np.stack([r['ug']  for r in results], axis=2)
vg  = np.stack([r['vg']  for r in results], axis=2)
ugt = np.stack([r['ugt'] for r in results], axis=2)
vgt = np.stack([r['vgt'] for r in results], axis=2)
ugw = np.stack([r['ugw'] for r in results], axis=2)
vgw = np.stack([r['vgw'] for r in results], axis=2)

# Parametri p, q, c di ciascun piano
p_arr = np.array([-np.tan(r['sigma']) * np.cos(r['tau']) for r in results])
q_arr = np.array([-np.tan(r['sigma']) * np.sin(r['tau']) for r in results])
c_arr = np.array([r['c'] for r in results])

# -------------------------------------------------------------------------
# Plot 3D della camera e dei 7 piani (sottocampionati 1:20)
# -------------------------------------------------------------------------
clrs = ['r', 'g', 'b', 'm', 'c', 'y', 'k']
fig3d = plt.figure()
ax3d = fig3d.add_subplot(111, projection='3d')


Z_MAX = 3 * d
for i in range(nplanes):
    xx = X[::20, ::20, i]
    yy = Y[::20, ::20, i]
    zz = Z[::20, ::20, i]
    valid = np.isfinite(zz) & (zz > 0) & (zz < Z_MAX)
    ax3d.scatter(xx[valid].ravel(), yy[valid].ravel(), zz[valid].ravel(), s=0.5, c=clrs[i])

ax3d.plot([0, 0], [0, 0], [0, r1['f']], 'g.-')
rect_x = [r1['x'][0], r1['x'][0],  r1['x'][-1], r1['x'][-1], r1['x'][0]]
rect_y = [r1['y'][0], r1['y'][-1], r1['y'][-1], r1['y'][0],  r1['y'][0]]
ax3d.plot(rect_x, rect_y, r1['f'] * np.ones(5), 'g-')
ax3d.set_zlim([0, 32000])
ax3d.legend(['1','2','3','4','5','6','7','cc','image plane'])
ax3d.view_init(elev=0, azim=0)
ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

# -------------------------------------------------------------------------
# Costruzione del motion field della scena completa:
# per ogni pixel (y,x) si sceglie il piano con Z minima (piu' vicino)
# -------------------------------------------------------------------------
flow   = np.zeros((H, W, 2))
flow_t = np.zeros((H, W, 2))
flow_w = np.zeros((H, W, 2))
indexes = np.zeros((H, W), dtype=int)
Xfull   = np.zeros((H, W))
Yfull   = np.zeros((H, W))
Zfull   = np.zeros((H, W))

# Calcolo vettorizzato di Z per ogni piano e ogni pixel
# xpix, ypix sono coordinate pixel 1-indexed (come MATLAB)
xpix_grid = np.arange(1, W + 1)
ypix_grid = np.arange(1, H + 1)
xpix_2d, ypix_2d = np.meshgrid(xpix_grid, ypix_grid)  # shape (H, W)

# Z_all[i,y,x] = profondita' del piano i nel pixel (x,y)
Z_all = np.zeros((nplanes, H, W))
for i in range(nplanes):
    Z_all[i] = (c_arr[i] * f) / (
        f - p_arr[i] * (xpix_2d - offset[0]) - q_arr[i] * (ypix_2d - offset[1])
    )

# Per ogni pixel si seleziona il piano con Z minima
idx_min = np.argmin(Z_all, axis=0)  # shape (H, W)
zmin    = Z_all[idx_min, np.arange(H)[:, None], np.arange(W)[None, :]]

for i in range(nplanes):
    mask = (idx_min == i)
    flow[mask, 0]   = ug[mask, i]
    flow[mask, 1]   = vg[mask, i]
    flow_t[mask, 0] = ugt[mask, i]
    flow_t[mask, 1] = vgt[mask, i]
    flow_w[mask, 0] = ugw[mask, i]
    flow_w[mask, 1] = vgw[mask, i]

# AFTER (fixed):
indexes = idx_min
Zfull = np.clip(zmin, 0, Z_MAX)   # Z_MAX already defined above
valid_mask = np.isfinite(zmin) & (zmin > 0) & (zmin < Z_MAX)
Xfull = np.where(valid_mask, (xpix_2d - offset[0]) * Zfull / f, 0.0)
Yfull = np.where(valid_mask, (ypix_2d - offset[1]) * Zfull / f, 0.0)

# -------------------------------------------------------------------------
# Coordinate di tutti i pixel dell'immagine (1-indexed, come MATLAB)
# pt2_  = coordinate pixel assolute [x, y]
# pt2   = coordinate centrate in (0,0)
# -------------------------------------------------------------------------
xgrid_px, ygrid_px = np.meshgrid(np.arange(1, W + 1), np.arange(1, H + 1))
pt2_  = np.column_stack([xgrid_px.ravel(), ygrid_px.ravel()])   # coordinate assolute
pt2   = pt2_ - offset                                             # coordinate centrate

# -------------------------------------------------------------------------
# Plot iniziali
# -------------------------------------------------------------------------

# Mappa dei piani (quale piano e' visibile in ogni pixel)
fig, ax = plt.subplots()
ax.imshow(indexes, cmap='gray', vmin=0, vmax=nplanes - 1)
ax.set_title('Plane index map')

# Angoli del rettangolo immagine (coordinate pixel assolute)
pt_tl = [pt2_[:, 0].min(), pt2_[:, 1].min()]
pt_tr = [pt2_[:, 0].max(), pt2_[:, 1].min()]
pt_br = [pt2_[:, 0].max(), pt2_[:, 1].max()]
pt_bl = [pt2_[:, 0].min(), pt2_[:, 1].max()]
rect_pts = [pt_tl, pt_tr, pt_br, pt_bl, pt_tl]
rect_xs  = [p[0] for p in rect_pts]
rect_ys  = [p[1] for p in rect_pts]

# MF totale (sottocampionato 1 ogni 1000 punti)
fig, ax = plt.subplots()
ax.set_title('trasl. + rot.')
for k in range(0, pt2_.shape[0], 1000):
    xi, yi = int(pt2_[k, 0]), int(pt2_[k, 1])
    ax.quiver(xi, yi, flow[yi - 1, xi - 1, 0], flow[yi - 1, xi - 1, 1],
              scale=1, scale_units='xy', angles='xy', color='green')
ax.plot(rect_xs, rect_ys, '-k')
ax.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xm')

# MF traslazionale
fig, ax = plt.subplots()
ax.set_title('trasl. comp.')
for k in range(0, pt2_.shape[0], 1000):
    xi, yi = int(pt2_[k, 0]), int(pt2_[k, 1])
    ax.quiver(xi, yi, flow_t[yi - 1, xi - 1, 0], flow_t[yi - 1, xi - 1, 1],
              scale=1, scale_units='xy', angles='xy', color='red')
ax.plot(rect_xs, rect_ys, '-k')
ax.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xm')

# MF rotazionale
fig, ax = plt.subplots()
ax.set_title('rot. comp.')
for k in range(0, pt2_.shape[0], 1000):
    xi, yi = int(pt2_[k, 0]), int(pt2_[k, 1])
    ax.quiver(xi, yi, flow_w[yi - 1, xi - 1, 0], flow_w[yi - 1, xi - 1, 1],
              scale=1, scale_units='xy', angles='xy', color='blue')
ax.plot(rect_xs, rect_ys, '-k')
ax.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xm')

plt.show()
input("press any key to continue...")

# =========================================================================
#
#          PARALLASSE: metodo del TRIANGOLO D'ARIA
#
# =========================================================================
show = False  # se True, abilita i plot dentro il ciclo principale
# =========================================================================
# Calcolo vettorizzato dei vettori di parallasse
# =========================================================================

N = 1000

# --- Pre-calcolo lookup: mappa (x,y) centrato -> indice in pt2 ---
# Costruiamo un dizionario per trovare rapidamente l'indice di un punto
pt2_dict = {(int(pt2[k, 0]), int(pt2[k, 1])): k for k in range(len(pt2))}

# --- Pre-estrazione dei flow e coordinate 3D come array 2D indicizzabili ---
# Evita accessi ripetuti con indici (ry, rx) dentro i loop
flow_u  = flow[:, :, 0]    # shape (H, W)
flow_v  = flow[:, :, 1]
ug_full = np.stack([r['ug'] for r in results], axis=2)   # già calcolato come ug
vg_full = np.stack([r['vg'] for r in results], axis=2)

delta_v            = np.zeros((N, 2))
pt_delta_v         = np.zeros((N, 2))
mean_parallax_norm = np.zeros(N)
allinaplane        = np.ones(N, dtype=bool)

# Angoli di rotazione per costruire il triangolo equilatero (60 e -60 gradi)
theta60 = np.radians(60)
R60     = np.array([[np.cos(theta60), -np.sin(theta60)],
                    [np.sin(theta60),  np.cos(theta60)]])

done = 0
while done < N:
    i = done

    # ------------------------------------------------------------------
    # Selezione dei 3 vertici del triangolo equilatero
    # ------------------------------------------------------------------
    radius = 3

    # Punto A: pixel casuale lontano dal bordo
    tmpidx = np.where(
        (pt2[:, 0] > -(xp - radius)) & (pt2[:, 0] < (xp - radius)) &
        (pt2[:, 1] > -(yp - radius)) & (pt2[:, 1] < (yp - radius))
    )[0]
    if len(tmpidx) == 0:
        continue
    idx_A = tmpidx[np.random.randint(len(tmpidx))]
    A = pt2[idx_A]

    # Punto B: pixel sul cerchio di raggio=radius centrato in A
    # Genera tutti i candidati sul cerchio e filtra quelli nell'immagine
    B_candidates = get_pt_into_radius(A, radius)
    # Ricerca vettorizzata: confronto batch invece di set per ogni punto
    valid_B = [pt2_dict[tuple(b)] for b in B_candidates
               if tuple(b) in pt2_dict]
    if len(valid_B) == 0:
        continue
    idx_B = valid_B[np.random.randint(len(valid_B))]
    B = pt2[idx_B]

    # Punto C: rotazione di 60° di AB attorno ad A (triangolo equilatero)
    AB = B - A
    C  = np.round(A + AB @ R60.T).astype(int)
    idx_C = pt2_dict.get(tuple(C), None)
    if idx_C is None:
        continue

    idxs = [idx_A, idx_B, idx_C]
    pts  = pt2[idxs, :]   # (3, 2) vertici in coordinate centrate

    # ------------------------------------------------------------------
    # Indici raster 0-based dei 3 vertici
    # ------------------------------------------------------------------
    ry_v = (pts[:, 1] + offset[1]).astype(int) - 1   # shape (3,)
    rx_v = (pts[:, 0] + offset[0]).astype(int) - 1

    # ------------------------------------------------------------------
    # Parametri 3D del piano passante per i 3 vertici
    # Risolve [X Y 1] * [p q c]' = Z  (sistema 3x3)
    # ------------------------------------------------------------------
    pt3    = np.column_stack([Xfull[ry_v, rx_v],
                              Yfull[ry_v, rx_v],
                              Zfull[ry_v, rx_v]])  # (3, 3)
    A_mat  = np.column_stack([pt3[:, 0], pt3[:, 1], np.ones(3)])
    b_vec  = pt3[:, 2]
    try:
        x_sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
    except np.linalg.LinAlgError:
        continue
    p_tri, q_tri, c_tri = x_sol

    tau_tri   = np.arctan2(q_tri, p_tri)
    sigma_tri = np.arctan(np.sqrt(p_tri**2 + q_tri**2))
    r_airtria = get_flow(T, Om, sigma_tri, tau_tri, f, xp, yp, c_tri)

    # ------------------------------------------------------------------
    # Piano di appartenenza di ciascun vertice (debug)
    # ------------------------------------------------------------------
    idx_p = indexes[ry_v, rx_v]    # vettorizzato: 3 lookup in una riga
    allinaplane[i] = (len(set(idx_p)) == 1)

    # ------------------------------------------------------------------
    # Stima trasformazione affine u = M*x + w sui 3 vertici
    # Sfrutta la struttura a blocchi: risolve separatamente per u e v
    # ------------------------------------------------------------------
    us = flow_u[ry_v, rx_v]    # (3,) - vettorizzato
    vs = flow_v[ry_v, rx_v]    # (3,)

    # Matrice comune 3x3: [x, y, 1] per ciascun vertice
    A3 = np.column_stack([pts[:, 0], pts[:, 1], np.ones(3)])   # (3, 3)
    # Risolve A3 * [m1, m2, w]' = u  e  A3 * [m3, m4, w2]' = v
    xu, _, _, _ = np.linalg.lstsq(A3, us, rcond=None)   # [m11, m12, w1]
    xv, _, _, _ = np.linalg.lstsq(A3, vs, rcond=None)   # [m21, m22, w2]

    M_mat = np.array([[xu[0], xu[1]],
                      [xv[0], xv[1]]])
    w_vec = np.array([xu[2], xv[2]])

    # ------------------------------------------------------------------
    # Punti interni al triangolo: tutto vettorizzato
    # ------------------------------------------------------------------
    in_mask = inpolygon(pt2[:, 0], pt2[:, 1], pts[:, 0], pts[:, 1])
    idx_in  = np.where(in_mask)[0]
    if len(idx_in) == 0:
        continue

    pts_sel = pt2[idx_in]                       # (K, 2)
    ry_in   = (pts_sel[:, 1] + offset[1]).astype(int) - 1
    rx_in   = (pts_sel[:, 0] + offset[0]).astype(int) - 1

    # Estrazione vettorizzata dei flow (niente loop su k)
    real_flow = np.column_stack([flow_u[ry_in, rx_in],
                                 flow_v[ry_in, rx_in]])          # (K, 2)
    intr_flow = pts_sel @ M_mat.T + w_vec                        # (K, 2) broadcasting
    quadratic_flow = np.column_stack([r_airtria['ug'][ry_in, rx_in],
                                      r_airtria['vg'][ry_in, rx_in]])  # (K, 2)

    # Parallax flow = differenza tra flusso reale e interpolato
    parallax_flow = real_flow - intr_flow                         # (K, 2)

    # ------------------------------------------------------------------
    # Punto piu' vicino al centroide del triangolo
    # ------------------------------------------------------------------
    cc               = pts.mean(axis=0)
    dists            = np.linalg.norm(pts_sel - cc, axis=1)      # vettorizzato
    selected_pt_idx  = np.argmin(dists)
    pt0              = pts_sel[selected_pt_idx]

    pt_delta_v[i] = pt0
    delta_v[i]    = parallax_flow[selected_pt_idx]

    # Bias quadratico-lineare nel centroide
    QLnorm = np.linalg.norm(quadratic_flow[selected_pt_idx] - intr_flow[selected_pt_idx])
    mean_parallax_norm[i] = np.mean(np.linalg.norm(parallax_flow, axis=1))

    done += 1
    print(f'[{i+1}] Parallasse trovata, norma {np.linalg.norm(delta_v[i]):.4f} '
          f'(stesso piano = {int(allinaplane[i])}) bias quadratico = {QLnorm:.4f}')

    # ------------------------------------------------------------------
    # Visualizzazione (solo se show=True e triangolo su piu' piani)
    # ------------------------------------------------------------------
    if show and not allinaplane[i]:
        polr_flow = np.column_stack([r1['ugt'][ry_in, rx_in],
                                     r1['vgt'][ry_in, rx_in]])

        fig_tri, ax_tri = plt.subplots()
        ax_tri.imshow(indexes, cmap='gray')
        plot_pts = np.vstack([pts, pts[0]])
        ax_tri.plot(plot_pts[:, 0] + offset[0], plot_pts[:, 1] + offset[1], '+-r')
        plt.pause(2)

        fig_3d2 = plt.figure()
        ax_3d2  = fig_3d2.add_subplot(111, projection='3d')
        ax_3d2.scatter(Xfull[::20, ::20], Yfull[::20, ::20], Zfull[::20, ::20], s=0.5, c='b')
        ax_3d2.plot([0, 0], [0, 0], [0, r1['f']], 'g.-')
        ax_3d2.plot(rect_x, rect_y, r1['f'] * np.ones(5), 'g-')
        plot_pt3 = np.vstack([pt3, pt3[0]])
        ax_3d2.plot(plot_pt3[:, 0], plot_pt3[:, 1], plot_pt3[:, 2], 'xr-')
        ax_3d2.set_aspect('auto')

        step = max(1, len(pts_sel) // 50)
        pts_s  = pts_sel[::step];  rf_ = real_flow[::step]
        if_ = intr_flow[::step];   pf_ = parallax_flow[::step]
        qf_ = quadratic_flow[::step]; pol_ = polr_flow[::step]
        ox  = offset[0];  oy = offset[1]

        fig_f1, ax_f1 = plt.subplots()
        ax_f1.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, qf_[:, 0], qf_[:, 1],
                     scale=1, scale_units='xy', angles='xy', color='blue')
        ax_f1.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, if_[:, 0], if_[:, 1],
                     scale=1, scale_units='xy', angles='xy', color='red')
        ax_f1.quiver(pts[:, 0]+ox, pts[:, 1]+oy, us, vs,
                     scale=1, scale_units='xy', angles='xy', color='magenta')
        ax_f1.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xc')
        plot_tri = np.vstack([pts, pts[0]])
        ax_f1.plot(plot_tri[:, 0]+ox, plot_tri[:, 1]+oy, 'k-')
        ax_f1.set_aspect('equal'); ax_f1.set_title('flusso quadratico e lineare')
        plt.pause(1)

        fig_f2, ax_f2 = plt.subplots()
        ax_f2.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, rf_[:, 0], rf_[:, 1],
                     scale=1, scale_units='xy', angles='xy', color='green')
        ax_f2.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, if_[:, 0], if_[:, 1],
                     scale=1, scale_units='xy', angles='xy', color='red')
        ax_f2.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xc')
        ax_f2.plot(pt0[0]+ox, pt0[1]+oy, 'ob')
        ax_f2.quiver(pt0[0]+ox, pt0[1]+oy, delta_v[i, 0], delta_v[i, 1],
                     scale=1, scale_units='xy', angles='xy', color='blue')
        ax_f2.plot(plot_tri[:, 0]+ox, plot_tri[:, 1]+oy, 'k-')
        ax_f2.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, pf_[:, 0]*100, pf_[:, 1]*100,
                     scale=1, scale_units='xy', angles='xy', color='magenta')
        ax_f2.set_aspect('equal'); ax_f2.set_title('flusso reale e interpolato')
        plt.pause(1)

        fig_f4, ax_f4 = plt.subplots()
        ax_f4.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, pf_[:, 0]*100, pf_[:, 1]*100,
                     scale=1, scale_units='xy', angles='xy', color='cyan')
        ax_f4.quiver(pts_s[:, 0]+ox, pts_s[:, 1]+oy, pol_[:, 0], pol_[:, 1],
                     scale=1, scale_units='xy', angles='xy', color='magenta')
        ax_f4.plot(r1['foe'][0, 0], r1['foe'][0, 1], 'xc')
        ax_f4.plot(pt0[0]+ox, pt0[1]+oy, 'ob')
        ax_f4.quiver(pt0[0]+ox, pt0[1]+oy, delta_v[i, 0], delta_v[i, 1],
                     scale=1, scale_units='xy', angles='xy', color='blue')
        ax_f4.plot(plot_tri[:, 0]+ox, plot_tri[:, 1]+oy, 'k-')
        ax_f4.set_aspect('equal'); ax_f4.set_title('flusso polare e parallasse')
        plt.pause(1)

        input('Premi un tasto per continuare...')
        plt.close(fig_f1); plt.close(fig_f2)
        plt.close(fig_f4); plt.close(fig_3d2); plt.close(fig_tri)

# -------------------------------------------------------------------------
# Norma di ciascun vettore di parallasse
# -------------------------------------------------------------------------
delta_mag = np.linalg.norm(delta_v, axis=1)

# -------------------------------------------------------------------------
# Soglia: si tengono solo i migliori 25% in base alla norma del parallasse
# -------------------------------------------------------------------------
sidx = np.argsort(delta_mag)[::-1]   # ordine decrescente
xperc = round(0.25 * len(delta_mag))
delta_threshold = delta_mag[sidx[xperc]]

# -------------------------------------------------------------------------
# Summary STATS
# -------------------------------------------------------------------------
sumstats = np.column_stack([allinaplane.astype(float), mean_parallax_norm, delta_mag])
np.save('STATS.npy', sumstats)  # equivalente di save('STATS.mat',...)

sidx2 = np.argsort(sumstats[:, 2])[::-1]
maxval = max(sumstats[:, 1].max(), sumstats[:, 2].max())
sumstats_sort = sumstats[sidx2]
delta_mag_dev = np.diff(sumstats_sort[:, 2])
delta_mag_dev = np.append(delta_mag_dev, 0)
tmp = sumstats_sort[:, 0].copy()
tmp[tmp == 1] = maxval + 0.1 * maxval

fig, ax = plt.subplots()
nn = np.arange(1, len(sumstats_sort) + 1)
ax.plot(nn, tmp,                      'r-', label='AllInAPlane')
ax.plot(nn, sumstats_sort[:, 1],      'g-', label='MeanParallaxNorm')
ax.plot(nn, sumstats_sort[:, 2],      'b-', label='CentroidParallaxNorm')
ax.plot(nn, delta_mag_dev * 10,        'm-', label='CentroidPrlxDev')
ax.axhline(delta_threshold,            color='k', linestyle='--', label='Threshold')
# Linea verticale nel punto dove delta_mag_dev diventa zero
zero_idx = np.where(delta_mag_dev == 0)[0]
if len(zero_idx) > 0:
    ax.axvline(zero_idx[0], color='c', linestyle='--')
ax.legend(); ax.set_title('AllInAPlane vs ParallaxNorm')

# -------------------------------------------------------------------------
# Filtraggio: si tengono solo i vettori con norma > soglia
# -------------------------------------------------------------------------
tmp_idx   = delta_mag > delta_threshold
print(f"numero di vettori di parallasse con soglia superiore a {delta_threshold:.4f} "
      f"= {tmp_idx.sum()} su {N} = {tmp_idx.mean()*100:.1f}%")

delta_v    = delta_v[tmp_idx]
pt_delta_v = pt_delta_v[tmp_idx]

# =========================================================================
#                     Calcolo del FOE
# =========================================================================

# -------------------------------------------------------------------------
# FOE con LEAST SQUARES (LS)
# -------------------------------------------------------------------------
Afoe = np.column_stack([-delta_v[:, 1], delta_v[:, 0]])
Bfoe = -delta_v[:, 1] * pt_delta_v[:, 0] + delta_v[:, 0] * pt_delta_v[:, 1]
foe_LS, _, _, _ = np.linalg.lstsq(Afoe, Bfoe, rcond=None)

# -------------------------------------------------------------------------
# FOE con RANDOM SAMPLE CONSENSUS (RANSAC)
# -------------------------------------------------------------------------
# Calcolo delle rette omogenee (normalizzate) passanti per pt e pt+delta_v
lines = np.cross(
    np.column_stack([pt_delta_v, np.ones(len(pt_delta_v))]),
    np.column_stack([pt_delta_v + delta_v, np.ones(len(delta_v))])
)
norms_lines = np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
lines = lines / norms_lines[:, None]

max_consensus = -np.inf
inlier_idxs   = None
th_dist       = 10.0

for _ in range(2000):
    ridx = np.random.randint(0, len(pt_delta_v), 2)
    l1 = lines[ridx[0]]
    l2 = lines[ridx[1]]
    foe_test = np.cross(l1, l2)
    if abs(foe_test[2]) < 1e-12:
        continue
    foe_test = foe_test / foe_test[2]

    # Distanza di ogni retta dal FOE candidato
    dist = np.abs(lines @ foe_test) / np.sqrt(lines[:, 0]**2 + lines[:, 1]**2)
    consensus = (dist < th_dist).sum()
    if consensus > max_consensus:
        max_consensus = consensus
        inlier_idxs   = dist < th_dist

if inlier_idxs is not None and inlier_idxs.any():
    pt_dv_valid = pt_delta_v[inlier_idxs]
    dv_valid    = delta_v[inlier_idxs]
    Afoe_r = np.column_stack([-dv_valid[:, 1], dv_valid[:, 0]])
    Bfoe_r = -dv_valid[:, 1] * pt_dv_valid[:, 0] + dv_valid[:, 0] * pt_dv_valid[:, 1]
    foe_r, _, _, _ = np.linalg.lstsq(Afoe_r, Bfoe_r, rcond=None)
else:
    foe_r = np.array([np.nan, np.nan])

# Stampa dei risultati
print(f"FOE [ground truth]..........[{r1['foe'][0,0]:6.2f} {r1['foe'][0,1]:6.2f}]")
print(f"FOE [LeastSquares] found in [{foe_LS[0]+offset[0]:6.2f} {foe_LS[1]+offset[1]:6.2f}]")
print(f"FOE [RANSAC]       found in [{foe_r[0]+offset[0]:6.2f} {foe_r[1]+offset[1]:6.2f}] "
      f"with consensus {max_consensus}/{len(delta_v)}")


# =========================================================================
#      SfM con FOE LS: stima di Omega e ricostruzione 3D
# =========================================================================
foeLS = np.round(foe_LS).astype(int)

# Stima di Omega col metodo generale
Aomega = np.zeros((N, 3))
bomega = np.zeros(N)
for i in range(N):
    idx = np.random.randint(0, pt2.shape[0])
    pt  = pt2[idx]
    x_c, y_c = pt[0], pt[1]
    ry = int(y_c + offset[1]) - 1
    rx = int(x_c + offset[0]) - 1
    u_c = flow[ry, rx, 0]
    v_c = flow[ry, rx, 1]
    x_foe, y_foe = foeLS[0], foeLS[1]
    Aomega[i] = [
        -f * (x_c - x_foe) - (y_c / f) * (x_c * y_foe - y_c * x_foe),
        -f * (y_c - y_foe) + (x_c / f) * (x_c * y_foe - y_c * x_foe),
         x_c * (x_c - x_foe) + y_c * (y_c - y_foe)
    ]
    bomega[i] = -v_c * (x_c - x_foe) + u_c * (y_c - y_foe)

omega_LS, _, _, _ = np.linalg.lstsq(Aomega, bomega, rcond=None)
print(f"GROUND TRUTH OMEGA........[{Om[0]:6.3f} {Om[1]:6.3f} {Om[2]:6.3f}]")
print(f"ESTIMATED OMEGA (LS)..... [{omega_LS[0]:6.3f} {omega_LS[1]:6.3f} {omega_LS[2]:6.3f}]")

# Ricostruzione 3D con FOE LS
p3d_ls    = []
idx_3d    = []
idx_3d_   = []
for i in range(0, pt2.shape[0], 50):
    x_c, y_c = pt2[i, 0], pt2[i, 1]
    ry = int(y_c + offset[1]) - 1
    rx = int(x_c + offset[0]) - 1

    # Componente rotazionale stimata
    v_o = np.array([
        [x_c * y_c / f, -(f + x_c**2 / f), y_c],
        [(f + y_c**2 / f), -x_c * y_c / f, -x_c]
    ]) @ omega_LS

    v_t = np.array([flow[ry, rx, 0] - v_o[0],
                    flow[ry, rx, 1] - v_o[1]])

    idx_3d.append(i)
    if np.linalg.norm(v_t) > 1.5:
        # Profondita' scalata (usa il FOE LS stimato, non la variabile di loop)
        Z_est = (v_t @ np.array([x_c - foeLS[0], y_c - foeLS[1]])) / (v_t @ v_t)
        X_est = x_c * Z_est / f
        Y_est = y_c * Z_est / f
        p3d_ls.append([X_est, Y_est, Z_est])
        idx_3d_.append(i)

p3d_ls  = np.array(p3d_ls)
idx_3d  = np.array(idx_3d)
idx_3d_ = np.array(idx_3d_)

# Xfull/Yfull/Zfull sono array (H,W) indicizzati da [riga=y-1, col=x-1].
# pt2[i] = [x_centered, y_centered], con x=col, y=row (coordinate centrate).
# L'indice flat nel senso row-major e': (y + offset[1] - 1) * W + (x + offset[0] - 1)
def pt2_to_flat(pt2_rows, offset, W):
    """Converte righe di pt2 (coordinate centrate [x,y]) in indici flat row-major."""
    xs = pt2_rows[:, 0].astype(int) + int(offset[0]) - 1  # colonna 0-based
    ys = pt2_rows[:, 1].astype(int) + int(offset[1]) - 1  # riga 0-based
    return ys * W + xs

Xgt    = np.column_stack([Xfull.ravel(), Yfull.ravel(), Zfull.ravel()])

# Calcola gli indici flat corrispondenti ai punti usati nella ricostruzione
flat_idx_   = pt2_to_flat(pt2[idx_3d_],  offset, W)
flat_idx    = pt2_to_flat(pt2[idx_3d],   offset, W)

Xgt_   = Xgt[flat_idx_]
scale  = np.mean(np.linalg.norm(p3d_ls, axis=1) / np.linalg.norm(Xgt_, axis=1))

# Errore 3D medio (con rimozione del fattore di scala 1/Tz)
merr = np.mean(np.linalg.norm(Xgt_ - (1 / scale) * p3d_ls, axis=1))
print(f"3D RMSE (LS)............. [{merr:6.3f}]")

Xgt_plot = Xgt[flat_idx]
fig_ls = plt.figure()
ax_ls  = fig_ls.add_subplot(111, projection='3d')
ax_ls.scatter((1/scale)*p3d_ls[:, 0], (1/scale)*p3d_ls[:, 1], (1/scale)*p3d_ls[:, 2],
              s=0.5, c='r', label='Estimated 3D')
ax_ls.scatter(Xgt_plot[:, 0], Xgt_plot[:, 1], Xgt_plot[:, 2],
              s=0.5, c='g', label='Ground truth 3D')
ax_ls.plot([0, 0], [0, 0], [0, r1['f']], 'b.-')
ax_ls.plot(rect_x, rect_y, r1['f'] * np.ones(5), 'b-')
ax_ls.set_title('LS FOE Reconstructed 3D')
ax_ls.legend(); ax_ls.set_aspect('auto'); ax_ls.grid(True)


# =========================================================================
#      SfM con FOE RANSAC: stima di Omega e ricostruzione 3D
# =========================================================================
foeRAN = np.round(foe_r).astype(int)

Aomega2 = np.zeros((N, 3))
bomega2  = np.zeros(N)
for i in range(N):
    idx = np.random.randint(0, pt2.shape[0])
    pt  = pt2[idx]
    x_c, y_c = pt[0], pt[1]
    ry = int(y_c + offset[1]) - 1
    rx = int(x_c + offset[0]) - 1
    u_c = flow[ry, rx, 0]
    v_c = flow[ry, rx, 1]
    x_foe2, y_foe2 = foeRAN[0], foeRAN[1]
    Aomega2[i] = [
        -f * (x_c - x_foe2) - (y_c / f) * (x_c * y_foe2 - y_c * x_foe2),
        -f * (y_c - y_foe2) + (x_c / f) * (x_c * y_foe2 - y_c * x_foe2),
         x_c * (x_c - x_foe2) + y_c * (y_c - y_foe2)
    ]
    bomega2[i] = -v_c * (x_c - x_foe2) + u_c * (y_c - y_foe2)

omega_RANSAC, _, _, _ = np.linalg.lstsq(Aomega2, bomega2, rcond=None)
print(f"ESTIMATED OMEGA (RANSAC)..[{omega_RANSAC[0]:6.3f} {omega_RANSAC[1]:6.3f} {omega_RANSAC[2]:6.3f}]")

# Ricostruzione 3D con FOE RANSAC
p3d_ran = []
idx_3d_r  = []
idx_3d_r_ = []
radura = 10.0  # raggio della zona attorno al FOE dove la profondita' non e' valutata

for i in range(0, pt2.shape[0], 50):
    x_c, y_c = pt2[i, 0], pt2[i, 1]
    ry = int(y_c + offset[1]) - 1
    rx = int(x_c + offset[0]) - 1

    v_o = np.array([
        [x_c * y_c / f, -(f + x_c**2 / f), y_c],
        [(f + y_c**2 / f), -x_c * y_c / f, -x_c]
    ]) @ omega_RANSAC

    v_t = np.array([flow[ry, rx, 0] - v_o[0],
                    flow[ry, rx, 1] - v_o[1]])

    idx_3d_r.append(i)
    if np.linalg.norm([x_c - foeRAN[0], y_c - foeRAN[1]]) > radura:
        Z_est = (v_t @ np.array([x_c - foeRAN[0], y_c - foeRAN[1]])) / (v_t @ v_t)
        X_est = x_c * Z_est / f
        Y_est = y_c * Z_est / f
        p3d_ran.append([X_est, Y_est, Z_est])
        idx_3d_r_.append(i)

p3d_ran  = np.array(p3d_ran)
idx_3d_r = np.array(idx_3d_r)
idx_3d_r_= np.array(idx_3d_r_)

flat_idx_r_  = pt2_to_flat(pt2[idx_3d_r_], offset, W)
flat_idx_r   = pt2_to_flat(pt2[idx_3d_r],  offset, W)

Xgt_r  = Xgt[flat_idx_r_]
scale_r = np.mean(np.linalg.norm(p3d_ran, axis=1) / np.linalg.norm(Xgt_r, axis=1))

merr_r = np.mean(np.linalg.norm(Xgt_r - (1 / scale_r) * p3d_ran, axis=1))
print(f"3D RMSE (RANSAC)............. [{merr_r:6.3f}]")

Xgt_plot_r = Xgt[flat_idx_r]
fig_ran = plt.figure()
ax_ran  = fig_ran.add_subplot(111, projection='3d')
ax_ran.scatter((1/scale_r)*p3d_ran[:, 0], (1/scale_r)*p3d_ran[:, 1], (1/scale_r)*p3d_ran[:, 2],
               s=0.5, c='b', label='Estimated 3D')
ax_ran.scatter(Xgt_plot_r[:, 0], Xgt_plot_r[:, 1], Xgt_plot_r[:, 2],
               s=0.5, c='g', label='Ground truth 3D')
ax_ran.plot([0, 0], [0, 0], [0, r1['f']], 'r.-')
ax_ran.plot(rect_x, rect_y, r1['f'] * np.ones(5), 'r-')
ax_ran.set_title('RANSAC FOE Reconstructed 3D')
ax_ran.legend(); ax_ran.set_aspect('auto'); ax_ran.grid(True)

plt.show()