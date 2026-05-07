"""
find_corr_gui.py

Python conversion of the MATLAB GUIDE application `find_corr_gui`.

Features:
- Load left/right images
- Manually select correspondences
- Support occluded points
- Estimate and draw epipolar lines
- Estimate homography projection
- Save/load workspace
- Point selection/editing

Dependencies:
    pip install numpy opencv-python matplotlib

Run:
    python find_corr_gui.py
"""

import pickle
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.lines import Line2D


class FindCorrGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Find Correspondences")

        # Images
        self.im1 = None
        self.im2 = None

        # Current points
        self.p1 = None
        self.p2 = None

        # Corner points
        self.ic1 = None
        self.ic2 = None

        # Correspondence table
        self.table = []

        # Selected row
        self.selrow = None

        # Modes
        self.occlusion_mode = tk.BooleanVar(value=False)
        self.show_epi = tk.BooleanVar(value=False)
        self.show_homography = tk.BooleanVar(value=False)

        self.last_clicked = 1

        self._build_ui()

    # ==========================================================
    # UI
    # ==========================================================

    def _build_ui(self):

        top = tk.Frame(self.root)
        top.pack(side=tk.TOP, fill=tk.X)

        tk.Button(top, text="Load Left", command=self.load_left).pack(side=tk.LEFT)
        tk.Button(top, text="Load Right", command=self.load_right).pack(side=tk.LEFT)

        tk.Button(top, text="Save Workspace", command=self.save_workspace).pack(side=tk.LEFT)
        tk.Button(top, text="Load Workspace", command=self.load_workspace).pack(side=tk.LEFT)

        tk.Checkbutton(
            top,
            text="Occlusion",
            variable=self.occlusion_mode
        ).pack(side=tk.LEFT)

        tk.Checkbutton(
            top,
            text="Epipolar",
            variable=self.show_epi,
            command=self.update_drawings
        ).pack(side=tk.LEFT)

        tk.Checkbutton(
            top,
            text="Homography",
            variable=self.show_homography,
            command=self.update_drawings
        ).pack(side=tk.LEFT)

        tk.Button(top, text="Help", command=self.show_help).pack(side=tk.RIGHT)

        # Matplotlib figure
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 6))

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.onclick)

        # Table
        self.listbox = tk.Listbox(self.root, height=10)
        self.listbox.pack(fill=tk.X)

        self.listbox.bind("<<ListboxSelect>>", self.on_select)
        self.root.bind("<Delete>", self.delete_selected)

    # ==========================================================
    # Load Images
    # ==========================================================

    def load_left(self):

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp")]
        )

        if not path:
            return

        self.im1 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        self.ax1.clear()
        self.ax1.imshow(self.im1)
        self.ax1.set_title("Left Image")

        self.canvas.draw()

    def load_right(self):

        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.bmp")]
        )

        if not path:
            return

        self.im2 = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        self.ax2.clear()
        self.ax2.imshow(self.im2)
        self.ax2.set_title("Right Image")

        self.canvas.draw()

    # ==========================================================
    # Mouse Interaction
    # ==========================================================

    def onclick(self, event):

        if event.inaxes is None:
            return

        x = event.xdata
        y = event.ydata

        if event.inaxes == self.ax1:
            self.last_clicked = 1

            if event.button == 1:
                self.p1 = [x, y]

            elif event.button == 3:
                self.confirm_point()

        elif event.inaxes == self.ax2:
            self.last_clicked = 2

            if event.button == 1:
                self.p2 = [x, y]

            elif event.button == 3:
                self.confirm_point()

        self.update_drawings()

    # ==========================================================
    # Confirm Match
    # ==========================================================

    def confirm_point(self):

        if self.occlusion_mode.get():

            if self.p1 is not None:
                row = [self.p1[0], self.p1[1], np.nan, np.nan]
                self.table.append(row)

            elif self.p2 is not None:
                row = [np.nan, np.nan, self.p2[0], self.p2[1]]
                self.table.append(row)

        else:

            if self.p1 is not None and self.p2 is not None:
                row = [
                    self.p1[0], self.p1[1],
                    self.p2[0], self.p2[1]
                ]
                self.table.append(row)

        self.p1 = None
        self.p2 = None

        self.refresh_table()

    # ==========================================================
    # Table
    # ==========================================================

    def refresh_table(self):

        self.listbox.delete(0, tk.END)

        for i, row in enumerate(self.table):

            txt = (
                f"{i}: "
                f"L=({row[0]:.1f},{row[1]:.1f}) "
                f"R=({row[2]:.1f},{row[3]:.1f})"
            )

            self.listbox.insert(tk.END, txt)

        self.update_drawings()

    def on_select(self, event):

        selection = self.listbox.curselection()

        if selection:
            self.selrow = selection[0]
        else:
            self.selrow = None

        self.update_drawings()

    def delete_selected(self, event):

        if self.selrow is None:
            return

        del self.table[self.selrow]
        self.selrow = None

        self.refresh_table()

    # ==========================================================
    # Drawing
    # ==========================================================

    def update_drawings(self):

        self.redraw_images()

        self.draw_matches()

        if self.show_epi.get():
            self.draw_epipolar()

        if self.show_homography.get():
            self.draw_homography()

        self.canvas.draw()

    def redraw_images(self):

        self.ax1.clear()
        self.ax2.clear()

        if self.im1 is not None:
            self.ax1.imshow(self.im1)

        if self.im2 is not None:
            self.ax2.imshow(self.im2)

        self.ax1.set_title("Left")
        self.ax2.set_title("Right")

    def draw_matches(self):

        for i, row in enumerate(self.table):

            x1, y1, x2, y2 = row

            selected = (i == self.selrow)

            color = "b" if selected else "g"

            if np.isfinite(x1):
                self.ax1.plot(x1, y1, marker="+", color=color, markersize=12)

            if np.isfinite(x2):
                self.ax2.plot(x2, y2, marker="+", color=color, markersize=12)

        # Current points
        if self.p1 is not None:
            self.ax1.plot(self.p1[0], self.p1[1], "r+", markersize=15)

        if self.p2 is not None:
            self.ax2.plot(self.p2[0], self.p2[1], "r+", markersize=15)

    # ==========================================================
    # Fundamental Matrix
    # ==========================================================

    def compute_fundamental(self):

        pts = np.array(self.table)

        mask = (
            np.isfinite(pts[:, 0]) &
            np.isfinite(pts[:, 2])
        )

        pts = pts[mask]

        if len(pts) < 8:
            return None

        x1 = pts[:, :2]
        x2 = pts[:, 2:]

        F, _ = cv2.findFundamentalMat(
            x1,
            x2,
            cv2.FM_8POINT
        )

        return F

    def draw_epipolar(self):

        F = self.compute_fundamental()

        if F is None:
            return

        if self.last_clicked == 1 and self.p1 is not None:

            p = np.array([self.p1[0], self.p1[1], 1.0])

            line = F @ p

            self.draw_line(self.ax2, line)

        elif self.last_clicked == 2 and self.p2 is not None:

            p = np.array([self.p2[0], self.p2[1], 1.0])

            line = F.T @ p

            self.draw_line(self.ax1, line)

    def draw_line(self, ax, line):

        a, b, c = line

        xlim = ax.get_xlim()

        x = np.array(xlim)

        y = -(a * x + c) / b

        ax.plot(x, y, "c-")

    # ==========================================================
    # Homography
    # ==========================================================

    def draw_homography(self):

        pts = np.array(self.table)

        mask = (
            np.isfinite(pts[:, 0]) &
            np.isfinite(pts[:, 2])
        )

        pts = pts[mask]

        if len(pts) < 4:
            return

        x1 = pts[:, :2]
        x2 = pts[:, 2:]

        H, _ = cv2.findHomography(x1, x2)

        if H is None:
            return

        if self.last_clicked == 1 and self.p1 is not None:

            p = np.array([self.p1[0], self.p1[1], 1.0])

            q = H @ p
            q /= q[2]

            self.ax2.plot(q[0], q[1], "c+", markersize=15)

        elif self.last_clicked == 2 and self.p2 is not None:

            Hinv = np.linalg.inv(H)

            p = np.array([self.p2[0], self.p2[1], 1.0])

            q = Hinv @ p
            q /= q[2]

            self.ax1.plot(q[0], q[1], "c+", markersize=15)

    # ==========================================================
    # Save / Load
    # ==========================================================

    def save_workspace(self):

        path = filedialog.asksaveasfilename(
            defaultextension=".pkl"
        )

        if not path:
            return

        data = {
            "table": self.table,
            "im1": self.im1,
            "im2": self.im2,
            "ic1": self.ic1,
            "ic2": self.ic2
        }

        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load_workspace(self):

        path = filedialog.askopenfilename(
            filetypes=[("Pickle", "*.pkl")]
        )

        if not path:
            return

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.table = data["table"]
        self.im1 = data["im1"]
        self.im2 = data["im2"]
        self.ic1 = data["ic1"]
        self.ic2 = data["ic2"]

        self.refresh_table()

    # ==========================================================
    # Help
    # ==========================================================

    def show_help(self):

        txt = (
            "Instructions:\n\n"
            "Left click = select point\n"
            "Right click = confirm match\n"
            "Delete = remove selected match\n\n"
            "Workflow:\n"
            "1. Select point in left image\n"
            "2. Select point in right image\n"
            "3. Right click to confirm\n"
        )

        messagebox.showinfo("Help", txt)


# ==============================================================
# Main
# ==============================================================

def main():

    root = tk.Tk()

    app = FindCorrGUI(root)

    root.mainloop()


if __name__ == "__main__":
    main()