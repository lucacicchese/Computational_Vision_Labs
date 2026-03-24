# ----------------------------------------------------------
#
# Programma Python per la simulazione del motion field
# indotto dal moto rigido (T, w) della camera rispetto
# a uno spigolo formato da DUE PIANI. La struttura 3D
# della scena viene calcolata usando la parallasse, e viene
# anche approssimata usando la divergenza.
# Ciascun piano ha equazione Z(X,Y) = pX + qY + d, dove
# (p,q) e' il gradiente di Z, che puo' essere espresso
# in termini di angoli di slant (0,90) e tilt (-90,+90)
# come (p,q) = -tan(sigma)(cos(tau),sin(tau)). Slant e tilt
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


# =============================================================
# get_flow: calcola il motion field e gli invarianti differenziali
#           del I ordine per un piano di parametri (sigma, tau, d)
#
# Input:
#   T     - vettore di traslazione della camera (3,)
#   w     - vettore di rotazione della camera (3,)
#   sigma - angolo di slant del piano (rad)
#   tau   - angolo di tilt del piano (rad)
#   f     - focale della telecamera (pixel)
#   xp    - offset x del punto principale (pixel)
#   yp    - offset y del punto principale (pixel)
#   d     - distanza del piano (Z(0,0))
#
# Output: dizionario con tutti i campi del motion field e metadati
# =============================================================
def get_flow(T, w, sigma, tau, f, xp, yp, d):

    # Gradiente di profondita' del piano
    p = -np.tan(sigma) * np.cos(tau)
    q = -np.tan(sigma) * np.sin(tau)

    # Griglia di coordinate immagine centrate nell'origine ottica
    x = np.arange(-xp, xp + 1, dtype=float)
    y = np.arange(-yp, yp + 1, dtype=float)
    xg, yg = np.meshgrid(x, y)

    # Coordinate 3D di ogni punto del piano in funzione di (xg,yg):
    #   Z(x,y) = d / (1 - (p/f)*x - (q/f)*y)
    c = d
    zg = c * np.ones(xg.shape) / (1 - (p / f) * xg - (q / f) * yg)
    Xg = (zg * xg) / f
    Yg = (zg * yg) / f

    # Componente traslazionale (polare) del MF
    ugt = -(1.0 / zg * (f * T[0] - T[2] * xg))
    vgt = -(1.0 / zg * (f * T[1] - T[2] * yg))

    # Componente rotazionale (assiale) del MF
    ugw = -(  -xg * yg / f * w[0]  +  (f**2 + xg**2) * (w[1] / f)  -  yg * w[2])
    vgw = -(-(f**2 + yg**2) / f * w[0]  +  xg * yg * (w[1] / f)    +  xg * w[2])

    # MF totale
    ug = ugt + ugw
    vg = vgt + vgw

    # ------------------------------------------------------------------
    # Invarianti differenziali del I ordine (divergenza, rotazionale, shear)
    # Calcolati analiticamente per un piano affine
    # ------------------------------------------------------------------
    divergence = -(
        (-p * T[0] - q * T[1] + (-2 + 3 * p / f * xg + 3 * q / f * yg) * T[2]) / c
        + (-3) * yg * w[0] / f + 3 * xg * w[1] / f
    )
    rotational = -(
        (q * T[0] - p * T[1] + ((-1) * q / f * xg + p / f * yg) * T[2]) / c
        + xg * w[0] / f + yg * w[1] / f + 2 * w[2]
    )
    shearx = -(
        (-p * T[0] + q * T[1] + (p / f * xg - q / f * yg) * T[2]) / c
        + yg * w[0] / f + xg * w[1] / f
    )
    sheary = -(
        (-q * T[0] - p * T[1] + (q / f * xg + p / f * yg) * T[2]) / c
        + (-1) * xg * w[0] / f + yg * w[1] / f
    )

    shear = np.sqrt(shearx**2 + sheary**2)

    # Limiti superiore e inferiore della collision immediacy
    cimmup = (divergence + shear) / 2
    cimmlo = (divergence - shear) / 2

    # FOE di ground truth (origine all'angolo dell'immagine, coordinate pixel)
    if T[2] != 0:
        foe = np.array([[f / T[2] * T[0] + xp - 1,
                         f / T[2] * T[1] + yp - 1]])
    else:
        denom = (T[0]**2 + T[1]**2)**0.5
        foe = np.array([
            [xp - 1, yp - 1],
            [T[0] * 25 / denom + xp - 1, T[1] * 25 / denom + yp - 1]
        ])

    # Restituzione di tutti i risultati come dizionario (equivalente della struct MATLAB)
    return {
        'xg': xg, 'yg': yg, 'zg': zg, 'Xg': Xg, 'Yg': Yg,
        'ugt': ugt, 'vgt': vgt, 'ugw': ugw, 'vgw': vgw,
        'ug': ug, 'vg': vg,
        'foe': foe,
        'xp': xp, 'yp': yp,
        'T': T, 'w': w,
        'sigma': sigma, 'tau': tau, 'f': f, 'd': d,
        'x': x, 'y': y,
        'cimmup': cimmup, 'cimmlo': cimmlo,
    }


# =============================================================
# show_flow: visualizza fino a 4 figure per un dato campo r
#
# Input:
#   r        - dizionario restituito da get_flow
#   s        - passo di sottocampionamento per le frecce quiver
#   ss       - stringa prefisso per i titoli delle figure
#   to_plot  - lista/array di 4 bool: [scena3D, MFtotale, MFpolare, MFassiale]
#
# Output: lista di handle alle figure create (None se non create)
# =============================================================
def show_flow(r, s, ss, to_plot):

    # Estrazione dei campi dal dizionario
    xg  = r['xg'];  yg  = r['yg'];  zg  = r['zg']
    Xg  = r['Xg'];  Yg  = r['Yg']
    ugt = r['ugt']; vgt = r['vgt']
    ugw = r['ugw']; vgw = r['vgw']
    ug  = r['ug'];  vg  = r['vg']
    foe = r['foe']
    xp  = r['xp'];  yp  = r['yp']
    T   = r['T'];   w   = r['w'];   f = r['f']
    x   = r['x'];   y   = r['y']

    # Coordinate immagine sottocampionate + offset del punto principale
    xc = xg[s-1::s, s-1::s] + xp
    yc = yg[s-1::s, s-1::s] + yp

    handles = [None] * 7  # equivalente di h=zeros(7,1)

    # ------------------------------------------------------------------
    # Plot 1: scena 3D (piano + camera + vettori T e w)
    # ------------------------------------------------------------------
    if to_plot[0]:
        fig = plt.figure()
        handles[0] = fig
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(
            Xg[s-1::s, s-1::s], Yg[s-1::s, s-1::s], zg[s-1::s, s-1::s],
            alpha=0.6
        )
        ax.plot([0, 0], [0, 0], [0, f], 'g.-')  # asse ottico (verde)
        if np.linalg.norm(T) > 0:  # vettore traslazione (blu)
            ax.plot(
                [0, f/2*T[0]/np.linalg.norm(T)],
                [0, f/2*T[1]/np.linalg.norm(T)],
                [0, f/2*T[2]/np.linalg.norm(T)], 'b.-'
            )
        if np.linalg.norm(w) > 0:  # vettore rotazione (rosso)
            ax.plot(
                [0, f/2*w[0]/np.linalg.norm(w)],
                [0, f/2*w[1]/np.linalg.norm(w)],
                [0, f/2*w[2]/np.linalg.norm(w)], 'r.-'
            )
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        # Rettangolo del piano immagine (verde) a Z=f
        rect_x = [x[0],  x[0],  x[-1], x[-1], x[0]]
        rect_y = [y[0],  y[-1], y[-1], y[0],  y[0]]
        ax.plot(rect_x, rect_y, f * np.ones(5), 'g-')
        ax.set_title(ss)

    # ------------------------------------------------------------------
    # Plot 2: MF totale
    # ------------------------------------------------------------------
    if to_plot[1]:
        fig, ax = plt.subplots()
        handles[1] = fig
        ax.set_aspect('equal')
        ax.set_title(ss + 'complete motion field')
        ax.quiver(xc, yc,
                  ug[s-1::s, s-1::s], vg[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='green')
        ax.plot(xc.ravel(), yc.ravel(), 'g.')
        ax.plot(foe[0, 0], foe[0, 1], 'oy')
        ax.plot(foe[:, 0], foe[:, 1], '-y')

    # ------------------------------------------------------------------
    # Plot 3: MF polare (componente traslazionale)
    # ------------------------------------------------------------------
    if to_plot[2]:
        fig, ax = plt.subplots()
        handles[2] = fig
        ax.set_aspect('equal')
        ax.set_title(ss + 'polar (T) component')
        ax.quiver(xc, yc,
                  ugt[s-1::s, s-1::s], vgt[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='blue')
        ax.plot(xc.ravel(), yc.ravel(), 'b.')
        ax.plot(foe[0, 0], foe[0, 1], 'oy')
        ax.plot(foe[:, 0], foe[:, 1], '-y')

    # ------------------------------------------------------------------
    # Plot 4: MF assiale (componente rotazionale)
    # ------------------------------------------------------------------
    if to_plot[3]:
        fig, ax = plt.subplots()
        handles[3] = fig
        ax.set_aspect('equal')
        ax.set_title(ss + 'axial (w) component')
        ax.quiver(xc, yc,
                  ugw[s-1::s, s-1::s], vgw[s-1::s, s-1::s],
                  scale=1, scale_units='xy', angles='xy', color='red')
        ax.plot(xc.ravel(), yc.ravel(), 'r.')

    return handles


# =============================================================
# structure_from_motion: funzione principale
# =============================================================
def structure_from_motion():

    # Parametri della camera e del moto
    xp = 320
    yp = 320
    f  = 512
    T  = np.array([0.0, 0.0, 300.0])
    w  = np.array([0.0, 0.0, 10.0]) * np.pi / 180

    # ------------------------------------------------------------------
    # Piano 1: slant negativo (inclinato verso la camera da sinistra)
    # ------------------------------------------------------------------
    sigma1 = -45 * np.pi / 180  # slant
    tau1   =   0 * np.pi / 180  # tilt
    d1     = 4000               # distanza

    r1 = get_flow(T, w, sigma1, tau1, f, xp, yp, d1)

    # ------------------------------------------------------------------
    # Piano 2: slant positivo (inclinato verso la camera da destra)
    # ------------------------------------------------------------------
    sigma2 =  45 * np.pi / 180
    tau2   =   0 * np.pi / 180
    d2     = 4000

    r2 = get_flow(T, w, sigma2, tau2, f, xp, yp, d2)

    # Visualizzazione dei due piani separatamente
    # N.B. [1,0,1,0] => mostra solo scena 3D e MF polare (come nell'originale)
    s = 40  # passo di sottocampionamento per il display
    h1 = show_flow(r1, s, 'Plane #1 - ', [1, 0, 0, 0])
    h2 = show_flow(r2, s, 'Plane #2 - ', [1, 0, 0, 0])

    # ------------------------------------------------------------------
    # Costruzione della scena 3D composta:
    # - nella meta' sinistra  (xg < 0) si usa il piano 1
    # - nella meta' destra    (xg >= 0) si usa il piano 2
    # ------------------------------------------------------------------
    idx12 = r1['xg'] < 0  # maschera booleana: True dove appartiene al piano 1

    # Copia di r1 come base del campo composito
    r12 = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in r1.items()}

    # Sovrascrittura della meta' destra con i valori del piano 2
    for key in ['xg', 'yg', 'ug', 'vg', 'ugt', 'vgt', 'ugw', 'vgw', 'cimmup', 'cimmlo']:
        r12[key][~idx12] = r2[key][~idx12]

    # Visualizzazione della scena 3D composta
    # [0,1,1,1] => mostra MF totale, polare e assiale (omette scena 3D)
    h3 = show_flow(r12, s, 'flow for the 3D scene - ', [1, 0, 0, 0])

    # ------------------------------------------------------------------
    # Stima del FOE tramite parallasse di moto:
    # il campo di differenza t = v1 - v2 e' composto da vettori di parallasse
    # ------------------------------------------------------------------
    n = 100  # numero di punti per i minimi quadrati

    # Dati per la stima: [x, y, du, dv] con du=u1-u2, dv=v1-v2
    t_data = np.column_stack([
        r1['xg'].ravel(), r1['yg'].ravel(),
        (r1['ug'] - r2['ug']).ravel(),
        (r1['vg'] - r2['vg']).ravel()
    ])
    # Componente polare nei due piani (non usata nella stima, ma utile per debug)
    t1 = np.column_stack([r1['xg'].ravel(), r1['yg'].ravel(), r1['ugt'].ravel(), r1['vgt'].ravel()])
    t2 = np.column_stack([r1['xg'].ravel(), r1['yg'].ravel(), r2['ugt'].ravel(), r2['vgt'].ravel()])

    # Selezione casuale di n punti (equivalente di randperm in MATLAB)
    idx = np.random.permutation(t_data.shape[0])[:n]
    t_data = t_data[idx, :]

    # Sistema lineare per la stima del FOE ai minimi quadrati:
    #   A * r = b  con  A = [-v, u]  e  b = -v*x + u*y
    A = np.column_stack([-t_data[:, 3], t_data[:, 2]])
    b = -t_data[:, 3] * t_data[:, 0] + t_data[:, 2] * t_data[:, 1]
    rf, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # FOE di ground truth (origine nel centro immagine)
    foe = r1['foe'] - np.array([xp, yp]) + 1

    # ------------------------------------------------------------------
    # Stima della profondita' Z(x,y) tramite FOE di ground truth
    # Z = (u_t * (x - foe_x) + v_t * (y - foe_y)) / (u_t^2 + v_t^2)
    # ------------------------------------------------------------------
    Z_num = (r12['ugt'] * (r12['xg'] - foe[0, 0])) + (r12['vgt'] * (r12['yg'] - foe[0, 1]))
    Z_den = r12['ugt']**2 + r12['vgt']**2
    Z = Z_num / Z_den

    # Display della profondita' stimata come immagine (normalizzata)
    fig, ax = plt.subplots()
    ax.imshow(Z, cmap='gray', vmin=np.nanmin(Z), vmax=np.nanmax(Z))
    ax.set_title('3D scene: estimated depth')

    # Coordinate 3D per il plot: X = Z/f * x,  Y = Z/f * y
    # (Importante: visualizzare Z(X,Y) e non Z(x,y))
    X = Z / f * r12['xg']
    Y = Z / f * r12['yg']

    fig3d = plt.figure()
    ax3d = fig3d.add_subplot(111, projection='3d')
    ax3d.scatter(X.ravel(), Y.ravel(), Z.ravel(), s=0.5)
    ax3d.set_title('3D scene: estimated depth as a function of (X,Y)')
    ax3d.set_xlabel('X'); ax3d.set_ylabel('Y'); ax3d.set_zlabel('Z')

    # ------------------------------------------------------------------
    # Time-To-Collision (TTC):
    # La media tra i due bound della collision immediacy e' pari alla divergenza
    # TTC = 1 / cimmave
    # ------------------------------------------------------------------
    cimmave = (r12['cimmlo'] + r12['cimmup']) / 2  # = divergenza
    TTC = 1.0 / cimmave

    # Display del TTC come immagine (normalizzata)
    fig, ax = plt.subplots()
    ax.imshow(TTC, cmap='gray', vmin=np.nanmin(TTC), vmax=np.nanmax(TTC))
    ax.set_title('3D scene: inverse average bound (divergence) on TTC')

    # Coordinate 3D per il plot del TTC
    X_ttc = TTC / f * r12['xg']
    Y_ttc = TTC / f * r12['yg']

    fig3d2 = plt.figure()
    ax3d2 = fig3d2.add_subplot(111, projection='3d')
    ax3d2.scatter(X_ttc.ravel(), Y_ttc.ravel(), TTC.ravel(), s=0.5)
    ax3d2.set_title('3D scene: estimated TTC as a function of (X,Y)')
    ax3d2.set_xlabel('X'); ax3d2.set_ylabel('Y'); ax3d2.set_zlabel('TTC')

    # ------------------------------------------------------------------
    # Stampa dei FOE (tutti calcolati senza rumore: devono coincidere)
    # ------------------------------------------------------------------
    print('FOE estimated from 3D data (ground truth):')
    print(foe)
    print('FOE estimated from motion parallax field:')
    print(rf.T)

    plt.show()


# Punto di ingresso del programma
if __name__ == '__main__':
    structure_from_motion()