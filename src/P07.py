"""P07 - Lego-Sortierer: Kamera erkennt Legosteine, UR3e + Robotiq sortiert sie in Kisten

Start:  python src/P07.py
Benötigt: PyQt6, ur_rtde, opencv-python, numpy (+ P06.py und robotiq_gripper.py im selben Ordner)

Ablauf (Reiter rechts):
  1. Kalibrierung: rote Kalibrierspitze einspannen (Knopf 2x drücken). Roboter im Freedrive
     ("senkrecht" = nur X/Y/Z + Drehung um Z frei) mit der Spitze an >= 3 (besser 5-6) weit
     verteilte Stellen auf den Tisch stellen. Jeweils "Pose übernehmen" -> Standbild ->
     Spitze im Bild anklicken (Mausrad = Zoom, rechte Maustaste = verschieben).
  2. Farben: HSV-Bereich je Steinfarbe einstellen (Regler, Pipette im Bild, Farbwähler).
  3. Ablagen: im Freedrive je Farbe die Pose über der Kiste speichern, dazu eine
     Beobachtungsposition, in der der Roboter nicht im Kamerabild ist.
  4. Sortieren: erkennt alle Steine, greift sie nacheinander und legt sie in die Kisten.

Greifer-Offset: wird per Pivot-Kalibrierung gemessen. Die Spitze bleibt auf EINEM Punkt
stehen, der Roboter wird mehrmals verschieden geneigt. Aus R_i * t + p_i = q (für alle Posen)
folgt per Least Squares der Spitzen-Offset t im TCP-KS. Damit ist die Spitzenposition für jede
Pose bekannt -> die Kamera-Kalibrierpunkte dürfen schräg aufgenommen werden (Kamera sieht die
Spitze). Die Fingerspitzen liegen tip_len (30 mm) hinter der Spitze auf der Werkzeug-Z-Achse.
Ohne Pivot (t = 0) funktioniert es weiterhin, dann aber nur mit senkrechten Kalibrierposen.
Der TCP am Teach Pendant ist egal, er muss nur von Pivot bis Sortieren gleich bleiben.

ACHTUNG: Die Software bewegt einen echten Roboter. Not-Halt in Reichweite halten!
"""
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QIcon, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QColorDialog, QComboBox, QDoubleSpinBox, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

try:
    from dashboard_client import DashboardClient    # Teil von ur_rtde
except ImportError:
    DashboardClient = None

from P06 import RTDEControlInterface, RTDEReceiveInterface, RobotiqGripper, Worker, apply_dark, fmt_pose

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "p07_config.json")
GRIPPER_PORT = 63352

DEFAULT_CFG = {
    "robot_ip": "192.168.0.100",
    "camera_index": 0,
    "calib": {"points": [], "H": None, "plane": None, "method": "", "img_size": None},
    "tip_len_mm": 30.0,      # Überstand der Kalibrierspitze über die geschlossenen Finger
    "tip_offset": [0.0, 0.0, 0.0],   # Spitze im TCP-KS [m] (Pivot-Kalibrierung)
    "pivot": {"poses": [], "rms_mm": None, "manual": False},
    "force_limit_N": 15.0,   # Kraftgrenze beim Absenken (0 = aus)
    "contact_detect": True,  # UR-eigene Kontakterkennung (tool_contact) beim Absenken
    "grip_h_mm": 4.0,        # Höhe der Fingerspitzen über dem Tisch beim Greifen
    "approach_mm": 30.0,     # ab hier langsam auf Greifhöhe fahren
    "travel_mm": 120.0,      # Transferhöhe der Fingerspitzen über dem Tisch
    "corr_mm": [0.0, 0.0],   # Feinkorrektur X/Y für Greifpunkte (z. B. Parallaxe)
    # pre_pos 205 = Kalibrierspitze passt gerade so zwischen die Finger (am Roboter getestet)
    # open_pos = max. Öffnung beim Greifen (Robotiq 0 = ganz offen .. 255 = zu), in der GUI als %
    "gripper": {"pre_pos": 205, "open_pos": 0, "speed": 150, "force": 60, "angle_off_deg": 0.0},
    "speed": {"v": 0.15, "a": 0.8, "v_slow": 0.03, "vj": 0.8, "aj": 1.0},
    "detect": {"min_area": 150, "max_area": 15000, "morph": 1},
    "roi": [],               # Arbeitsbereich als Polygon im Bild (Pixel)
    "home_q": None,
    "home_pose": None,
    "colors": [
        {"name": "Rot", "lo": [170, 120, 70], "hi": [8, 255, 255], "enabled": True, "drop": None},
        {"name": "Gelb", "lo": [18, 120, 100], "hi": [35, 255, 255], "enabled": True, "drop": None},
        {"name": "Grün", "lo": [40, 80, 50], "hi": [85, 255, 255], "enabled": True, "drop": None},
        {"name": "Blau", "lo": [95, 120, 50], "hi": [125, 255, 255], "enabled": True, "drop": None},
    ],
}


# ====================================================================== Konfiguration
def _merge(default, data):
    if isinstance(default, dict) and isinstance(data, dict):
        out = json.loads(json.dumps(default))
        for k, v in data.items():
            out[k] = _merge(out[k], v) if k in out else v
        return out
    return data


def load_cfg():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return _merge(DEFAULT_CFG, json.load(f))
    except FileNotFoundError:
        return json.loads(json.dumps(DEFAULT_CFG))


def save_cfg(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ====================================================================== Geometrie
def rotvec_down(theta):
    """Drehvektor für senkrecht nach unten zeigendes Werkzeug, Tool-X in Richtung theta (Basis-KS).
    R = Rz(theta) * Rx(pi) ist eine 180°-Drehung um die Achse (cos(theta/2), sin(theta/2), 0)."""
    return [math.pi * math.cos(theta / 2), math.pi * math.sin(theta / 2), 0.0]


def _rot(pose):
    return cv2.Rodrigues(np.array(pose[3:6], dtype=float))[0]


def yaw_of(pose):
    R = _rot(pose)
    return math.atan2(R[1, 0], R[0, 0])


def tilt_of(pose):
    """Winkel zwischen Werkzeug-Z und der Senkrechten nach unten [rad]."""
    return math.acos(float(np.clip(-_rot(pose)[2, 2], -1.0, 1.0)))


def tool_point(pose, t):
    """Basis-Koordinaten eines Punkts, der im TCP-KS gegeben ist."""
    return np.array(pose[:3], float) + _rot(pose) @ np.asarray(t, float)


def pose_for_point(target, rotvec, t):
    """TCP-Pose, bei der der Werkzeugpunkt t (TCP-KS) mit Orientierung rotvec auf target liegt."""
    R = cv2.Rodrigues(np.array(rotvec, float))[0]
    return [*(np.asarray(target, float) - R @ np.asarray(t, float)), *rotvec]


def pivot_calibrate(poses):
    """Spitze an festem Punkt q, verschiedene Orientierungen: R_i t - q = -p_i.
    Gibt (t im TCP-KS, q in Basis, Residuen [m]) zurück."""
    A = np.vstack([np.hstack([_rot(p), -np.eye(3)]) for p in poses])
    b = -np.concatenate([np.array(p[:3], float) for p in poses])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    t, q = sol[:3], sol[3:]
    res = np.array([np.linalg.norm(tool_point(p, t) - q) for p in poses])
    return t, q, res


def fit_mapping(px, xy):
    """Pixel -> Basis-XY. >= 4 Punkte: Homographie (auch schräge Kamera), 3 Punkte: affin."""
    px, xy = np.asarray(px, np.float64), np.asarray(xy, np.float64)
    if len(px) >= 4:
        H, _ = cv2.findHomography(px, xy, 0)
        if H is not None:
            return H, "Homographie"
    M = np.hstack([px, np.ones((len(px), 1))])
    sol, *_ = np.linalg.lstsq(M, xy, rcond=None)
    return np.vstack([sol.T, [0, 0, 1]]), "Affin"


def map_pts(H, px):
    p = np.hstack([np.asarray(px, np.float64), np.ones((len(px), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


class Calib:
    def __init__(self, H, plane, img_size, tip):
        self.H, self.plane, self.img_size = H, plane, img_size
        self.tip = np.asarray(tip, float)      # Spitzen-Offset, mit dem kalibriert wurde

    @staticmethod
    def from_cfg(cfg):
        c = cfg["calib"]
        if not c.get("H") or not c.get("plane"):
            return None
        return Calib(np.array(c["H"], float), c["plane"], c.get("img_size"), c.get("tip_offset", [0, 0, 0]))

    def to_base(self, u, v):
        p = self.H @ np.array([u, v, 1.0])
        return float(p[0] / p[2]), float(p[1] / p[2])

    def table_z(self, x, y):
        """Tischhöhe (Basis-Z der Spitze beim Berühren) an (x, y). Mit t = 0 ist das die TCP-Z."""
        a, b, c = self.plane
        return a * x + b * y + c


# ====================================================================== Bildverarbeitung
def in_hue(h, lo, hi):
    return ((h >= lo) & (h <= hi)) if lo <= hi else ((h >= lo) | (h <= hi))


def color_mask(hsv, lo, hi, morph=1):
    """HSV-Maske; H min > H max bedeutet Bereich über 0 hinweg (z. B. Rot 170..8)."""
    lo, hi = np.array(lo, np.uint8), np.array(hi, np.uint8)
    if lo[0] <= hi[0]:
        m = cv2.inRange(hsv, lo, hi)
    else:
        m = (cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8))
             | cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi))
    if morph > 0:
        k = np.ones((3, 3), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=morph)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=morph)
    return m


def hue_mid(lo, hi):
    return (lo + hi) / 2 if lo <= hi else ((lo + hi + 180) / 2) % 180


def color_bgr(col):
    lo, hi = col["lo"], col["hi"]
    hsv = np.uint8([[[round(hue_mid(lo[0], hi[0])) % 180, (lo[1] + hi[1]) // 2, (lo[2] + hi[2]) // 2]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def hue_range(hues, tol):
    """Kleinster Kreisbogen (H 0..179), der alle Farbtöne enthält, plus Toleranz."""
    hs = sorted({int(round(h)) % 180 for h in hues})
    n = len(hs)
    if n == 1:
        lo = hi = hs[0]
    else:
        gaps = [hs[i + 1] - hs[i] for i in range(n - 1)] + [hs[0] + 180 - hs[-1]]
        i = int(np.argmax(gaps))
        lo, hi = hs[(i + 1) % n], hs[i]
    if (hi - lo) % 180 + 2 * tol >= 179:
        return 0, 179
    return (lo - tol) % 180, (hi + tol) % 180


def detect_bricks(frame, cfg, calib=None, only=None):
    """Findet Steine aller aktivierten Farben (oder nur Farbe `only`)."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(frame, (5, 5), 0), cv2.COLOR_BGR2HSV)
    d = cfg["detect"]
    roi = np.float32(cfg["roi"]) if len(cfg.get("roi") or []) >= 3 else None
    out = []
    for ci, col in enumerate(cfg["colors"]):
        if (only is None and not col["enabled"]) or (only is not None and ci != only):
            continue
        m = color_mask(hsv, col["lo"], col["hi"], d["morph"])
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not d["min_area"] <= area <= d["max_area"]:
                continue
            M = cv2.moments(c)
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            if roi is not None and cv2.pointPolygonTest(roi, (float(u), float(v)), False) < 0:
                continue
            box = cv2.boxPoints(cv2.minAreaRect(c))
            e1, e2 = box[1] - box[0], box[2] - box[1]
            le = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
            a_img = math.atan2(float(le[1]), float(le[0]))       # Längsachse im Bild
            det = {"ci": ci, "name": col["name"], "px": (u, v), "area": area,
                   "box": box.tolist(), "ang_img": a_img, "xy": None, "theta": None}
            if calib is not None:
                x, y = calib.to_base(u, v)
                x2, y2 = calib.to_base(u + 10 * math.cos(a_img), v + 10 * math.sin(a_img))
                det["xy"] = (x, y)
                # Finger schließen quer zur Längsachse
                det["theta"] = math.atan2(y2 - y, x2 - x) + math.pi / 2
            out.append(det)
    return out


def draw_dets(img, dets, colors):
    for d in dets:
        bgr = color_bgr(colors[d["ci"]])
        box = np.int32(np.round(d["box"]))
        cv2.polylines(img, [box], True, (0, 0, 0), 4)
        cv2.polylines(img, [box], True, bgr, 2)
        u, v = d["px"]
        a = d["ang_img"] + math.pi / 2                         # Schließrichtung der Finger
        dx, dy = 18 * math.cos(a), 18 * math.sin(a)
        cv2.line(img, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)), (255, 255, 255), 1)
        cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)
        txt = d["name"]
        if d["xy"] is not None:
            txt += f" {d['xy'][0]*1000:.0f}/{d['xy'][1]*1000:.0f}"
        cv2.putText(img, txt, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, txt, (int(u) + 8, int(v) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def range_preview(lo, hi, w=256, h=56):
    """Oben: Farbtonleiste (gewählter H-Bereich hell). Unten: alle Farben im Bereich,
    links->rechts Farbton, oben S/V max, unten S/V min."""
    span = (hi[0] - lo[0]) % 180
    hues = (lo[0] + np.linspace(0, span, w)) % 180
    t = np.linspace(0, 1, h)[:, None]
    S = hi[1] + (lo[1] - hi[1]) * t
    V = hi[2] + (lo[2] - hi[2]) * t
    hsv = np.dstack([np.broadcast_to(hues, (h, w)), np.broadcast_to(S, (h, w)),
                     np.broadcast_to(V, (h, w))]).astype(np.uint8)
    patch = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    hb = np.linspace(0, 179, w)
    bar = cv2.cvtColor(np.dstack([np.broadcast_to(hb, (14, w)), np.full((14, w), 255),
                                  np.full((14, w), 255)]).astype(np.uint8), cv2.COLOR_HSV2BGR)
    bar[:, ~in_hue(hb, lo[0], hi[0])] //= 5
    return np.vstack([bar, np.full((4, w, 3), 32, np.uint8), patch])


def to_pixmap(bgr):
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QPixmap.fromImage(QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy())


# ====================================================================== Widgets
class ImageView(QWidget):
    """Kamerabild mit Zoom (Mausrad, zum Cursor hin), Verschieben (rechte/mittlere Maustaste)
    und subpixelgenauem Klick (Koordinaten in OpenCV-Konvention: Pixelmitte = ganzzahlig)."""
    clicked = pyqtSignal(float, float)
    hovered = pyqtSignal(float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.qimg = None
        self.zoom = 1.0
        self.cx = self.cy = 0.0
        self.markers = []      # (u, v, QColor, text)
        self.polygon = []      # [(u, v)]
        self.info = ""
        self._press = None
        self._drag = None

    def set_image(self, bgr):
        bgr = np.ascontiguousarray(bgr)
        h, w = bgr.shape[:2]
        if self.qimg is None or (self.qimg.width(), self.qimg.height()) != (w, h):
            self.cx, self.cy, self.zoom = w / 2, h / 2, 1.0
        self.qimg = QImage(bgr.data, w, h, bgr.strides[0], QImage.Format.Format_BGR888).copy()
        self.update()

    def reset_view(self):
        if self.qimg is not None:
            self.cx, self.cy, self.zoom = self.qimg.width() / 2, self.qimg.height() / 2, 1.0
            self.update()

    def _scale(self):
        return min(self.width() / self.qimg.width(), self.height() / self.qimg.height()) * self.zoom

    def _to_widget(self, u, v):
        s = self._scale()
        return QPointF((u - self.cx) * s + self.width() / 2, (v - self.cy) * s + self.height() / 2)

    def _to_image(self, p):
        s = self._scale()
        return (p.x() - self.width() / 2) / s + self.cx, (p.y() - self.height() / 2) / s + self.cy

    def _clamp(self):
        self.cx = min(max(self.cx, 0), self.qimg.width())
        self.cy = min(max(self.cy, 0), self.qimg.height())

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(12, 12, 12))
        if self.qimg is None:
            p.setPen(QColor(150, 150, 150))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Kein Kamerabild - Kamera starten")
            return
        s = self._scale()
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, s < 3)
        tl = self._to_widget(0, 0)
        p.drawImage(QRectF(tl.x(), tl.y(), self.qimg.width() * s, self.qimg.height() * s), self.qimg)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.polygon:
            pts = [self._to_widget(u + 0.5, v + 0.5) for u, v in self.polygon]
            pen = QPen(QColor(255, 220, 0), 1.5, Qt.PenStyle.DashLine)
            p.setPen(pen)
            for a, b in zip(pts, pts[1:] + pts[:1]):
                p.drawLine(a, b)
        for u, v, col, text in self.markers:
            q = self._to_widget(u + 0.5, v + 0.5)
            for pen in (QPen(QColor(0, 0, 0), 3.5), QPen(col, 1.5)):
                p.setPen(pen)
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    p.drawLine(QPointF(q.x() + dx * 4, q.y() + dy * 4), QPointF(q.x() + dx * 16, q.y() + dy * 16))
            p.setPen(col)
            p.drawText(QPointF(q.x() + 8, q.y() - 8), text)
        if self.zoom > 1.01 or self.info:
            p.setPen(QColor(230, 230, 230))
            p.drawText(QPointF(8, 18), (f"Zoom {self.zoom:.1f}x   " if self.zoom > 1.01 else "") + self.info)

    def mousePressEvent(self, e):
        if e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = e.position()
        elif e.button() == Qt.MouseButton.LeftButton:
            self._press = e.position()

    def mouseMoveEvent(self, e):
        if self.qimg is None:
            return
        if self._drag is not None:
            d = e.position() - self._drag
            s = self._scale()
            self.cx -= d.x() / s
            self.cy -= d.y() / s
            self._clamp()
            self._drag = e.position()
            self.update()
        u, v = self._to_image(e.position())
        self.hovered.emit(u - 0.5, v - 0.5)

    def mouseReleaseEvent(self, e):
        if e.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
            self._drag = None
        elif e.button() == Qt.MouseButton.LeftButton and self._press is not None:
            if self.qimg is not None and (e.position() - self._press).manhattanLength() < 5:
                u, v = self._to_image(e.position())
                if 0 <= u < self.qimg.width() and 0 <= v < self.qimg.height():
                    self.clicked.emit(u - 0.5, v - 0.5)
            self._press = None

    def wheelEvent(self, e):
        if self.qimg is None:
            return
        pos = e.position()
        u, v = self._to_image(pos)
        self.zoom = min(max(self.zoom * 1.25 ** (e.angleDelta().y() / 120), 1.0), 40.0)
        s = self._scale()
        self.cx = u - (pos.x() - self.width() / 2) / s
        self.cy = v - (pos.y() - self.height() / 2) / s
        self._clamp()
        self.update()


class IntSlider(QWidget):
    valueChanged = pyqtSignal(int)

    def __init__(self, lo, hi, val):
        super().__init__()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.spin = QSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setFixedWidth(60)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self.spin.setValue)
        self.spin.valueChanged.connect(self.slider.setValue)
        self.spin.valueChanged.connect(self.valueChanged.emit)
        self.setValue(val, quiet=True)

    def value(self):
        return self.spin.value()

    def setValue(self, v, quiet=False):
        self.blockSignals(quiet)
        self.spin.setValue(int(v))
        self.blockSignals(False)


# ====================================================================== Roboter-Abläufe
class Abort(Exception):
    pass


class MoveRejected(RuntimeError):
    """Ziel nicht erreichbar -> Stein wird übersprungen, Sortieren läuft weiter."""


class CollisionStop(RuntimeError):
    """Kraftgrenze überschritten -> zurückgefahren, Sortieren wird beendet."""


def finger_pt(cfg, calib):
    """Mitte der Fingerspitzen im TCP-KS: tip_len hinter der Kalibrierspitze auf Werkzeug-Z."""
    return calib.tip - np.array([0.0, 0.0, cfg["tip_len_mm"] / 1000])


def grasp_poses(cfg, calib, d, ref_yaw):
    """TCP-Posen (über dem Stein auf Transferhöhe, Annäherung, Greifen) für eine Erkennung d.
    Die Fingerspitzen (nicht der TCP) werden auf die Höhen gefahren."""
    x = d["xy"][0] + cfg["corr_mm"][0] / 1000
    y = d["xy"][1] + cfg["corr_mm"][1] / 1000
    z0 = calib.table_z(x, y)
    zg = z0 + max(cfg["grip_h_mm"], 1.0) / 1000
    za = zg + cfg["approach_mm"] / 1000
    zt = z0 + cfg["travel_mm"] / 1000
    t = d["theta"] + math.radians(cfg["gripper"]["angle_off_deg"])
    t += math.pi * round((ref_yaw - t) / math.pi)      # Greifer ist 180° symmetrisch
    rv = rotvec_down(t)
    f = finger_pt(cfg, calib)
    return (pose_for_point([x, y, zt], rv, f), pose_for_point([x, y, za], rv, f),
            pose_for_point([x, y, zg], rv, f))


def pose_reachable(c, pose, qnear):
    pose = [float(v) for v in pose]
    return bool(c.isPoseWithinSafetyLimits(pose)
                and c.getInverseKinematicsHasSolution(pose, list(qnear), 1e-5, 1e-4))


class RobotTask(QThread):
    """Bewegungsabläufe im Hintergrund. Nur dieser Thread benutzt rtde_c/Greifer, solange er läuft;
    Stopp über stop_ev (der Thread hält den Roboter dann selbst an)."""
    status = pyqtSignal(str)
    detections = pyqtSignal(object)
    result = pyqtSignal(str)

    def __init__(self, win, job, **kw):
        super().__init__()
        self.win, self.job, self.kw = win, job, kw
        self.c, self.r, self.gr = win.rtde_c, win.rtde_r, win.gripper
        self.cfg = json.loads(json.dumps(win.cfg))      # Schnappschuss
        self.calib = Calib.from_cfg(self.cfg)
        self.stop_ev = threading.Event()
        self.peak_force = 0.0          # max. gefilterte Kraft bei überwachten Bewegungen (zum Einstellen)
        self._contact_warned = False

    def run(self):
        try:
            msg = getattr(self, "job_" + self.job)(**self.kw) or "Fertig"
        except Abort:
            msg = "Abgebrochen"
        except Exception as e:
            msg = f"Fehler: {e}"
            try:
                self.c.stopL(2.0)
            except Exception:
                pass
        self.result.emit(msg)

    # ---------------------------------------------------------------- Hilfen
    def _check(self, stopfn=None):
        if self.stop_ev.is_set():
            if stopfn is not None:
                try:
                    stopfn(2.0)
                except Exception:
                    pass
            raise Abort()
        if self.r.isProtectiveStopped() or self.r.isEmergencyStopped():
            raise RuntimeError("Schutzstopp / Not-Halt aktiv - nach dem Entriegeln 'Wiederherstellen' drücken")

    def _reached(self, kind, target):
        if kind == "J":
            return max(abs(a - b) for a, b in zip(self.r.getActualQ(), target)) < 0.01
        return math.dist(self.r.getActualTCPPose()[:3], target[:3]) < 0.002

    def _require_reachable(self, poses, what):
        """Vor der Bewegung prüfen, ob die IK eine Lösung hat und die Posen in den Sicherheitsgrenzen liegen
        (sonst bricht das Skript auf dem UR mit IK-Fehler ab)."""
        q = list(self.r.getActualQ())
        for p in poses:
            if not pose_reachable(self.c, p, q):
                raise MoveRejected(f"{what} nicht erreichbar (IK / Sicherheitsgrenzen): {fmt_pose(p)}")

    def _move(self, kind, target, v=None, a=None, force_limit=None, retreat=None):
        """kind: 'L' = moveL, 'J' = moveJ (Gelenkwinkel), 'JIK' = moveJ zu kartesischer Pose.
        Mit `retreat` ist die Bewegung überwacht (Absenken): Bei Kontakt stoppen, zu `retreat`
        zurückfahren und mit CollisionStop abbrechen. Überwachung:
          1. UR-eigene Kontakterkennung (tool_contact) - rechnet Beschleunigung/Dynamik heraus
          2. optional Kraftgrenze [N] auf die genullte, gefilterte TCP-Kraft, nur bei konstanter
             Sollgeschwindigkeit (beim Beschleunigen/Bremsen ausgesetzt -> keine Fehlauslösung)"""
        sp, c, r = self.cfg["speed"], self.c, self.r
        target = [float(x) for x in target]
        if kind != "J":
            self._require_reachable([target], "Ziel")
        guard = retreat is not None
        use_contact = guard and self.cfg.get("contact_detect", True)
        base = None
        if guard and force_limit:
            try:
                c.zeroFtSensor()
            except Exception as e:
                raise RuntimeError(f"Kraftsensor nicht verfügbar ({e}) - Kraftgrenze auf 0 setzen zum Deaktivieren")
            time.sleep(0.1)
            samples = []
            for _ in range(10):
                samples.append(r.getActualTCPForce()[:3])
                time.sleep(0.004)
            base = np.mean(samples, axis=0)          # Rest-Offset nach dem Nullen
        if use_contact:
            d = np.array(target[:3]) - np.array(r.getActualTCPPose()[:3])
            n = float(np.linalg.norm(d))
            direction = [*(d / n), 0.0, 0.0, 0.0] if n > 1e-4 else [0.0] * 6
            try:
                use_contact = bool(c.startContactDetection(direction))
            except Exception:
                use_contact = False
            if not use_contact and not self._contact_warned:
                self._contact_warned = True
                self.status.emit("UR-Kontakterkennung nicht verfügbar - nur Kraftgrenze aktiv")
        try:
            if kind == "L":
                ok, stop = c.moveL(target, v or sp["v"], a or sp["a"], True), c.stopL
            elif kind == "J":
                ok, stop = c.moveJ(target, v or sp["vj"], a or sp["aj"], True), c.stopJ
            else:
                ok, stop = c.moveJ_IK(target, v or sp["vj"], a or sp["aj"], True), c.stopJ
            if not ok:
                raise MoveRejected("Ziel nicht erreichbar" + ("" if kind == "J" else f": {fmt_pose(target)}"))
            t0, idle = time.monotonic(), None
            fwin = []
            v_prev, t_prev, t_acc = None, None, time.monotonic()
            while True:
                self._check(stop)
                time.sleep(0.004 if guard else 0.01)
                if use_contact and c.readContactDetection():
                    self._collision(stop, retreat, use_contact, "Kontakt erkannt (UR-Kontakterkennung)")
                if base is not None:
                    now = time.monotonic()
                    vt = np.array(r.getTargetTCPSpeed()[:3])
                    if v_prev is None or (now > t_prev and
                                          float(np.linalg.norm(vt - v_prev)) / (now - t_prev) > 0.05):
                        t_acc = now                              # Soll beschleunigt/bremst gerade
                    v_prev, t_prev = vt, now
                    f = float(np.linalg.norm(np.array(r.getActualTCPForce()[:3]) - base))
                    if now - t_acc < 0.06:
                        fwin = []          # Werte aus Beschleunigung/Bremsen verwerfen (Trägheit)
                        continue_check = False
                    else:
                        fwin = (fwin + [f])[-4:]                 # gleitender Mittelwert
                        continue_check = len(fwin) >= 3
                    fm = sum(fwin) / len(fwin) if fwin else 0.0
                    self.peak_force = max(self.peak_force, fm)
                    if continue_check and fm > force_limit:
                        self._collision(stop, retreat, use_contact,
                                        f"Kraftgrenze überschritten ({fm:.1f} N > {force_limit:.0f} N)")
                if c.getAsyncOperationProgress() >= 0:
                    idle = None
                    continue
                if self._reached(kind, target):
                    return
                if time.monotonic() - t0 < 0.3:
                    continue
                idle = idle or time.monotonic()
                if time.monotonic() - idle > 1.5:
                    raise MoveRejected("Bewegung vorzeitig beendet (Ziel nicht erreicht)")
        finally:
            if use_contact:
                try:
                    c.stopContactDetection()
                except Exception:
                    pass

    def _collision(self, stop, retreat, use_contact, why):
        try:
            stop(5.0)
        except Exception:
            pass
        if use_contact:
            try:
                c = self.c
                c.stopContactDetection()
            except Exception:
                pass
        time.sleep(0.3)
        self._move("L", retreat, v=self.cfg["speed"]["v_slow"])
        raise CollisionStop(f"{why} - zurückgefahren und gestoppt. Hindernis prüfen.")

    def _grip(self, pos, speed=None, force=None):
        g = self.cfg["gripper"]
        _, st = self.gr.move_and_wait_for_pos(int(pos), int(speed or g["speed"]), int(force or g["force"]))
        return st

    def _flim(self):
        return self.cfg["force_limit_N"] or None

    def _ref_yaw(self):
        return yaw_of(self.cfg["home_pose"] or self.r.getActualTCPPose())

    def _retreat(self):
        """Senkrecht auf Transferhöhe, bevor eine Gelenkbewegung startet (kein Schleifen über den Tisch)."""
        if self.calib is None:
            return
        p = list(self.r.getActualTCPPose())
        f = tool_point(p, finger_pt(self.cfg, self.calib))
        dz = self.calib.table_z(f[0], f[1]) + self.cfg["travel_mm"] / 1000 - f[2]
        if dz > 0.002:
            self._move("L", [p[0], p[1], p[2] + dz, *p[3:]])

    def _home(self):
        self._retreat()
        self.status.emit("Fahre zur Beobachtungsposition")
        self._move("J", self.cfg["home_q"])

    def _detect(self):
        t0 = time.time()
        while True:
            self._check()
            with self.win.frame_lock:
                f, ts = self.win.frame, self.win.frame_ts
            if f is not None and ts > t0 + 0.3:      # Bild nach Stillstand des Roboters
                f = f.copy()
                break
            if time.time() - t0 > 5:
                raise RuntimeError("Keine Kamerabilder")
            time.sleep(0.02)
        dets = [d for d in detect_bricks(f, self.cfg, self.calib) if d["xy"] is not None]
        self.detections.emit(dets)
        return dets

    def _nearest(self, dets):
        hx, hy = (self.cfg["home_pose"] or self.r.getActualTCPPose())[:2]
        return min(dets, key=lambda d: math.dist(d["xy"], (hx, hy)))

    # ---------------------------------------------------------------- Jobs
    def job_home(self):
        self._home()
        return "Beobachtungsposition erreicht"

    def job_align(self):
        p = list(self.r.getActualTCPPose())
        self._move("L", [*p[:3], *rotvec_down(yaw_of(p))], v=0.2, a=0.5)
        return "Werkzeug senkrecht ausgerichtet"

    def job_goto_xy(self, x, y, h_mm):
        """Kalibrierspitze h_mm über den Tisch an (x, y) fahren (Kalibrierungstest)."""
        z = self.calib.table_z(x, y) + h_mm / 1000
        rv = rotvec_down(yaw_of(self.r.getActualTCPPose()))
        above = pose_for_point([x, y, z + 0.04], rv, self.calib.tip)
        target = pose_for_point([x, y, z], rv, self.calib.tip)
        self._require_reachable([above, target], "Testpunkt")
        self._retreat()
        self._move("JIK", above)
        self._move("L", target, v=self.cfg["speed"]["v_slow"], force_limit=self._flim(), retreat=above)
        return (f"Spitze {h_mm:.0f} mm über X {x*1000:.1f} Y {y*1000:.1f} mm "
                f"(max. Kraft beim Absenken {self.peak_force:.1f} N)")

    def job_drop_test(self, ci):
        col = self.cfg["colors"][ci]
        dp = col["drop"]
        above = [dp[0], dp[1], dp[2] + 0.03, *dp[3:]]
        self._require_reachable([above, dp], f"Ablage {col['name']}")
        self._retreat()
        self._move("JIK", above)
        self._move("L", dp, force_limit=self._flim(), retreat=above)
        return f"Ablage {col['name']} erreicht"

    def job_hover(self):
        self._grip(self.cfg["gripper"]["open_pos"])
        self._home()
        dets = self._detect()
        if not dets:
            return "Keine Steine erkannt"
        d = self._nearest(dets)
        above, _, grasp = grasp_poses(self.cfg, self.calib, d, self._ref_yaw())
        hover = [grasp[0], grasp[1], grasp[2] + 0.015, *grasp[3:]]
        self._require_reachable([above, hover], f"{d['name']}er Stein")
        self._move("JIK", above)
        self._move("L", hover, force_limit=self._flim(), retreat=above)
        return (f"Fingerspitzen 15 mm über Greifhöhe am {d['name']}en Stein - Lage prüfen, "
                f"ggf. Korrektur X/Y bzw. Winkel-Offset anpassen (max. Kraft beim Absenken {self.peak_force:.1f} N)")

    def job_sort(self):
        picked, fails = 0, []
        while True:
            self._home()
            dets = self._detect()
            todo = [d for d in dets if self.cfg["colors"][d["ci"]].get("drop")
                    and not any(math.dist(d["xy"], f) < 0.015 for f in fails)]
            if not todo:
                msg = f"Sortieren fertig: {picked} Steine abgelegt"
                return msg + (f", {len(fails)} Stelle(n) übersprungen" if fails else "")
            d = self._nearest(todo)
            self.status.emit(f"Greife {d['name']} bei X {d['xy'][0]*1000:.0f} Y {d['xy'][1]*1000:.0f} mm "
                             f"({len(todo)} erkannt)")
            try:
                ok = self._pick_place(d)
            except MoveRejected as e:          # unerreichbar -> Stein überspringen, weitermachen
                self.status.emit(f"Übersprungen: {e}")
                ok = False
            # CollisionStop wird NICHT abgefangen -> Sortieren endet
            if ok:
                picked += 1
            else:
                fails.append(d["xy"])

    def _pick_place(self, d):
        g, sp = self.cfg["gripper"], self.cfg["speed"]
        above, approach, grasp = grasp_poses(self.cfg, self.calib, d, self._ref_yaw())
        dp = self.cfg["colors"][d["ci"]]["drop"]
        drop_above = [dp[0], dp[1], max(above[2], dp[2]), *dp[3:]]
        # alles vorab prüfen, bevor sich der Roboter bewegt
        self._require_reachable([above, approach, grasp], f"{d['name']}er Stein")
        self._require_reachable([drop_above, dp], f"Ablage {d['name']}")
        self._grip(g["open_pos"])
        self._move("JIK", above)
        if approach[2] < above[2]:
            self._move("L", approach, force_limit=self._flim(), retreat=above)
        self._move("L", grasp, v=sp["v_slow"], force_limit=self._flim(), retreat=above)
        st = self._grip(255, force=g["force"])
        self._move("L", above)
        if st not in (RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
                      RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT):
            self.status.emit(f"Kein Stein gegriffen ({d['name']}) - nächster Versuch nach neuer Erkennung")
            self._grip(g["open_pos"])
            return False
        self._move("JIK", drop_above)
        if dp[2] < drop_above[2] - 1e-4:
            self._move("L", dp, force_limit=self._flim(), retreat=drop_above)
        self._grip(g["open_pos"])
        self.status.emit(f"{d['name']} abgelegt")
        return True

# ====================================================================== Hauptfenster
class LegoSorter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("P07 Lego-Sortierer (UR3e + Robotiq + Kamera)")
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, int(avail.width() * 0.95)), min(900, int(avail.height() * 0.92)))

        self.cfg = load_cfg()
        self.calib = Calib.from_cfg(self.cfg)
        self.rtde_c = self.rtde_r = self.gripper = None
        self.cap = None
        self.frame, self.frame_ts = None, 0.0
        self.frame_lock = threading.Lock()
        self.frozen = None                         # Standbild (Kalibrierung)
        self.pending_pt = None                     # Kalibrierpunkt, der auf Bildklick wartet
        self.calib_frames = [None] * len(self.cfg["calib"]["points"])
        self.last_click = None                     # (u, v, x, y) Test-Klick
        self.pip_samples = []
        self.roi_edit = False
        self.freedrive = None                      # None / "teach" / "constrained"
        self.clamp_state = 0
        self.task = None
        self._threads = set()
        self._size_warned = False

        self._build_ui()
        self._refresh_points()
        self._refresh_colors()
        self._refresh_home()
        self._update_calib_label()
        self._update_pivot_label()

        self.cam_timer = QTimer(self, interval=33)
        self.cam_timer.timeout.connect(self._cam_tick)
        self.cam_timer.start()
        self.pose_timer = QTimer(self, interval=150)
        self.pose_timer.timeout.connect(self._update_pose)
        self.refresh_cameras()
        self._update_enabled()

    # ================================================================== UI-Aufbau
    def _cfg_get(self, path):
        d = self.cfg
        for k in path:
            d = d[k]
        return d

    def _cfg_set(self, path, v):
        d = self.cfg
        for k in path[:-1]:
            d = d[k]
        d[path[-1]] = v

    def _num(self, path, lo, hi, step=1.0, dec=1, suffix=""):
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setDecimals(dec)
        sp.setSingleStep(step)
        sp.setSuffix(suffix)
        sp.setValue(self._cfg_get(path))
        sp.valueChanged.connect(lambda v, p=path, d=dec: self._cfg_set(p, int(v) if d == 0 else v))
        return sp

    @staticmethod
    def _hint(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color:#aaa;")
        return lbl

    @staticmethod
    def _scroll(w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setFrameShape(QScrollArea.Shape.NoFrame)
        s.setWidget(w)
        return s

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # ---- Kopfzeile: Roboter, Kamera, Freedrive
        top = QGroupBox()
        tl = QGridLayout(top)
        self.ip_edit = QLineEdit(self.cfg["robot_ip"])
        self.btn_connect = QPushButton("Verbinden")
        self.btn_disconnect = QPushButton("Trennen")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_disconnect.clicked.connect(self.disconnect_robot)
        self.cb_cam = QComboBox()
        self.btn_cam_refresh = QPushButton("⟳")
        self.btn_cam_refresh.setFixedWidth(30)
        self.btn_cam_refresh.clicked.connect(self.refresh_cameras)
        self.btn_cam = QPushButton("Kamera starten")
        self.btn_cam.clicked.connect(self.toggle_camera)
        self.btn_free = QPushButton("Freedrive EIN")
        self.btn_free.setCheckable(True)
        self.btn_free.toggled.connect(self.toggle_freedrive)
        self.chk_vertical = QCheckBox("senkrecht halten (nur XYZ + Rz)")
        self.chk_vertical.setChecked(False)
        self.btn_align = QPushButton("Senkrecht ausrichten")
        self.btn_align.clicked.connect(lambda: self._start_task("align"))
        tl.addWidget(QLabel("UR IP:"), 0, 0)
        tl.addWidget(self.ip_edit, 0, 1)
        tl.addWidget(self.btn_connect, 0, 2)
        tl.addWidget(self.btn_disconnect, 0, 3)
        tl.addWidget(QLabel("  Kamera:"), 0, 4)
        tl.addWidget(self.cb_cam, 0, 5)
        tl.addWidget(self.btn_cam_refresh, 0, 6)
        tl.addWidget(self.btn_cam, 0, 7)
        tl.addWidget(self.btn_free, 0, 8)
        tl.addWidget(self.chk_vertical, 0, 9)
        tl.addWidget(self.btn_align, 0, 10)
        self.lbl_pose = QLabel("TCP: --")
        self.lbl_grip = QLabel("Greifer: getrennt")
        self.lbl_tcp = QLabel("")
        tl.addWidget(self.lbl_pose, 1, 0, 1, 6)
        tl.addWidget(self.lbl_grip, 1, 6, 1, 3)
        tl.addWidget(self.lbl_tcp, 1, 9)
        self.btn_recover = QPushButton("Wiederherstellen")
        self.btn_recover.setToolTip("Nach Schutzstopp / Not-Halt / IK-Fehler: entsperren, einschalten, Bremsen "
                                    "lösen, Steuerskript neu laden, Greifer aktivieren - ohne neu zu verbinden")
        self.btn_recover.clicked.connect(self.recover_robot)
        tl.addWidget(self.btn_recover, 1, 10)
        outer.addWidget(top)

        # ---- Bild links, Reiter rechts
        split = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(split, 1)
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.view = ImageView()
        self.view.clicked.connect(self._on_view_click)
        self.view.hovered.connect(self._on_hover)
        lv.addWidget(self.view, 1)
        h = QHBoxLayout()
        self.cb_view = QComboBox()
        self.cb_view.addItems(["Kamera", "Maske (gewählte Farbe)", "Erkennung (alle Farben)"])
        self.btn_reset_view = QPushButton("Zoom zurücksetzen")
        self.btn_reset_view.clicked.connect(self.view.reset_view)
        self.btn_unfreeze = QPushButton("Live")
        self.btn_unfreeze.clicked.connect(self._unfreeze)
        self.lbl_hover = QLabel("")
        h.addWidget(QLabel("Anzeige:"))
        h.addWidget(self.cb_view)
        h.addWidget(self.btn_reset_view)
        h.addWidget(self.btn_unfreeze)
        h.addWidget(self.lbl_hover, 1)
        lv.addLayout(h)
        split.addWidget(left)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._scroll(self._tab_calib()), "1 Kalibrierung")
        self.tabs.addTab(self._scroll(self._tab_colors()), "2 Farben")
        self.tabs.addTab(self._scroll(self._tab_drops()), "3 Ablagen")
        self.tabs.addTab(self._scroll(self._tab_sort()), "4 Sortieren")
        self.tabs.currentChanged.connect(self._on_tab)
        split.addWidget(self.tabs)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)

        # ---- Statuszeile + Stopp
        h = QHBoxLayout()
        self.status = QLabel("Bereit")
        self.status.setWordWrap(True)
        btn_save = QPushButton("Einstellungen speichern")
        btn_save.clicked.connect(self.save)
        self.btn_stop = QPushButton("STOPP")
        self.btn_stop.setMinimumHeight(38)
        self.btn_stop.setMinimumWidth(140)
        self.btn_stop.setStyleSheet("background:#8b1a1a; font-weight:bold;")
        self.btn_stop.clicked.connect(self.stop_all)
        h.addWidget(self.status, 1)
        h.addWidget(btn_save)
        h.addWidget(self.btn_stop)
        outer.addLayout(h)

        # Labels mit ständig wechselndem Text dürfen das Layout nicht umverteilen
        # (sonst springt/skaliert das Kamerabild beim Bewegen der Maus)
        for lbl in (self.lbl_hover, self.lbl_pose, self.lbl_grip, self.lbl_tcp, self.status, self.lbl_count):
            lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            lbl.setMinimumWidth(0)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ------------------------------------------------------------ Reiter 1
    def _tab_calib(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._hint(
            "1) Kalibrierspitze einspannen (Knopf ①, Spitze einführen, Knopf ②).\n"
            "2) Spitzen-Offset (Pivot): Spitze auf einen markierten Punkt stellen, 4-6 Posen mit "
            "verschiedenen Neigungen (20-40°, in verschiedene Richtungen) aufnehmen, Spitze bleibt am Punkt.\n"
            "3) Kamera: Freedrive, Spitze auf den Tisch - Roboter so neigen, dass die Kamera die Spitze sieht.\n"
            "   'Pose übernehmen' -> Standbild -> Spitze anklicken (Mausrad zoomt, rechte Maustaste verschiebt).\n"
            "4) Für 4-6 weit verteilte Stellen wiederholen (min. 3), dann 'Kalibrierung berechnen'."))

        g = QGroupBox("Kalibrierspitze einspannen")
        f = QFormLayout(g)
        self.btn_clamp = QPushButton()
        self.btn_clamp.setMinimumHeight(32)
        self.btn_clamp.clicked.connect(self.clamp_cycle)
        self._clamp_label()
        self.sl_pre = IntSlider(0, 255, self.cfg["gripper"]["pre_pos"])
        self.sl_pre.valueChanged.connect(lambda val: self._cfg_set(("gripper", "pre_pos"), val))
        f.addRow(self.btn_clamp)
        f.addRow("Vorposition (0=offen, 255=zu):", self.sl_pre)
        f.addRow("Überstand Spitze über Finger:", self._num(("tip_len_mm",), 0, 100, 0.5, 1, " mm"))
        v.addWidget(g)

        g = QGroupBox("Spitzen-Offset (Pivot-Kalibrierung)")
        gv = QVBoxLayout(g)
        h = QHBoxLayout()
        self.btn_pivot_add = QPushButton("Pivot-Pose übernehmen")
        self.btn_pivot_add.clicked.connect(self.pivot_add)
        b1, b2 = QPushButton("Berechnen"), QPushButton("Posen löschen")
        b1.clicked.connect(self.pivot_compute)
        b2.clicked.connect(self.pivot_clear)
        h.addWidget(self.btn_pivot_add, 1)
        h.addWidget(b1)
        h.addWidget(b2)
        gv.addLayout(h)
        self.lbl_pivot = QLabel()
        self.lbl_pivot.setWordWrap(True)
        gv.addWidget(self.lbl_pivot)
        h = QHBoxLayout()
        self.sp_tip_z = QDoubleSpinBox()
        self.sp_tip_z.setRange(-500, 500)
        self.sp_tip_z.setDecimals(1)
        self.sp_tip_z.setSuffix(" mm")
        self.sp_tip_z.setValue(self.cfg["tip_offset"][2] * 1000)
        self.sp_tip_z.setToolTip("Abstand der Spitze vom TCP entlang Werkzeug-Z =\n"
                                 "Flansch->Fingerspitzen (Datenblatt) + 30 mm Überstand - TCP-Z am Pendant")
        b = QPushButton("Manuell übernehmen")
        b.clicked.connect(self.tip_manual)
        h.addWidget(QLabel("oder Spitze Z manuell:"))
        h.addWidget(self.sp_tip_z, 1)
        h.addWidget(b)
        gv.addLayout(h)
        v.addWidget(g)

        g = QGroupBox("Kamera-Kalibrierpunkte")
        gv = QVBoxLayout(g)
        self.btn_capture = QPushButton("Pose übernehmen + Standbild (Spitze steht auf dem Tisch)")
        self.btn_capture.setMinimumHeight(34)
        self.btn_capture.clicked.connect(self.calib_capture)
        gv.addWidget(self.btn_capture)
        self.list_points = QListWidget()
        self.list_points.setMinimumHeight(130)
        gv.addWidget(self.list_points)
        h = QHBoxLayout()
        b1, b2, b3 = QPushButton("Neu markieren"), QPushButton("Löschen"), QPushButton("Alle löschen")
        b1.clicked.connect(self.calib_remark)
        b2.clicked.connect(self.calib_delete)
        b3.clicked.connect(self.calib_clear)
        for b in (b1, b2, b3):
            h.addWidget(b)
        gv.addLayout(h)
        self.btn_compute = QPushButton("Kalibrierung berechnen")
        self.btn_compute.setMinimumHeight(32)
        self.btn_compute.clicked.connect(self.calib_compute)
        gv.addWidget(self.btn_compute)
        self.lbl_calib = QLabel()
        self.lbl_calib.setWordWrap(True)
        gv.addWidget(self.lbl_calib)
        v.addWidget(g)

        g = QGroupBox("Test")
        gv = QVBoxLayout(g)
        gv.addWidget(self._hint("Klick ins Livebild zeigt die Basis-Koordinaten. Mit eingespannter Spitze "
                                "kann der Roboter dorthin fahren - die Spitze sollte genau über dem Punkt stehen."))
        self.lbl_click = QLabel("Klickpunkt: —")
        gv.addWidget(self.lbl_click)
        h = QHBoxLayout()
        self.sp_test_h = QDoubleSpinBox()
        self.sp_test_h.setRange(2, 100)
        self.sp_test_h.setValue(10)
        self.sp_test_h.setSuffix(" mm über Tisch")
        self.btn_goto_click = QPushButton("Spitze zum Klickpunkt fahren")
        self.btn_goto_click.clicked.connect(self.goto_click)
        h.addWidget(self.btn_goto_click, 1)
        h.addWidget(self.sp_test_h)
        gv.addLayout(h)
        v.addWidget(g)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------ Reiter 2
    def _tab_colors(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = QGroupBox("Farben (Haken = wird sortiert)")
        gv = QVBoxLayout(g)
        self.list_colors = QListWidget()
        self.list_colors.setMaximumHeight(140)
        self.list_colors.currentRowChanged.connect(self._load_color)
        self.list_colors.itemChanged.connect(self._color_item_changed)
        gv.addWidget(self.list_colors)
        h = QHBoxLayout()
        b1, b2, b3 = QPushButton("Hinzufügen"), QPushButton("Umbenennen"), QPushButton("Entfernen")
        b1.clicked.connect(self.color_add)
        b2.clicked.connect(self.color_rename)
        b3.clicked.connect(self.color_remove)
        for b in (b1, b2, b3):
            h.addWidget(b)
        gv.addLayout(h)
        v.addWidget(g)

        g = QGroupBox("HSV-Bereich (OpenCV: H 0-179, S/V 0-255; H min > H max = über Rot hinweg)")
        gl = QGridLayout(g)
        self.hsv_sliders = []
        for i, (name, hi) in enumerate([("H min", 179), ("S min", 255), ("V min", 255),
                                        ("H max", 179), ("S max", 255), ("V max", 255)]):
            s = IntSlider(0, hi, 0)
            s.valueChanged.connect(self._hsv_changed)
            gl.addWidget(QLabel(name), i, 0)
            gl.addWidget(s, i, 1)
            self.hsv_sliders.append(s)
        self.lbl_range = QLabel()
        self.lbl_swatch = QLabel()
        self.lbl_swatch.setFixedSize(56, 74)
        h = QHBoxLayout()
        h.addWidget(self.lbl_range)
        h.addWidget(self.lbl_swatch)
        h.addStretch(1)
        gl.addLayout(h, 6, 0, 1, 2)
        self.lbl_range_txt = self._hint("")
        gl.addWidget(self.lbl_range_txt, 7, 0, 1, 2)
        v.addWidget(g)

        g = QGroupBox("Schnellauswahl")
        gl = QGridLayout(g)
        self.btn_pipette = QPushButton("Pipette: in Bild klicken (Shift+Klick = erweitern)")
        self.btn_pipette.setCheckable(True)
        self.btn_colordlg = QPushButton("Farbwähler …")
        self.btn_colordlg.clicked.connect(self.color_dialog)
        self.sp_tol = [QSpinBox() for _ in range(3)]
        for sp, val, hi in zip(self.sp_tol, (8, 50, 60), (90, 255, 255)):
            sp.setRange(0, hi)
            sp.setValue(val)
        gl.addWidget(self.btn_pipette, 0, 0, 1, 4)
        gl.addWidget(self.btn_colordlg, 1, 0)
        for i, (n, sp) in enumerate(zip(("±H", "±S", "±V"), self.sp_tol)):
            hh = QHBoxLayout()
            hh.addWidget(QLabel(n))
            hh.addWidget(sp)
            gl.addLayout(hh, 1, i + 1)
        v.addWidget(g)

        v.addWidget(self._form_box("Erkennung", [
            ("Min. Fläche [px²]:", self._num(("detect", "min_area"), 1, 1e6, 10, 0)),
            ("Max. Fläche [px²]:", self._num(("detect", "max_area"), 1, 1e7, 100, 0)),
            ("Morphologie (Rauschfilter):", self._num(("detect", "morph"), 0, 6, 1, 0)),
        ]))
        self.lbl_count = QLabel("")
        v.addWidget(self.lbl_count)
        v.addStretch(1)
        return w

    # Robotiq-Position 0 = ganz offen, 255 = zu. Einstellung in % Öffnung (100 % = ganz offen)
    def _open_spin(self):
        self.sp_open = QSpinBox()
        self.sp_open.setRange(0, 100)
        self.sp_open.setSingleStep(5)
        self.sp_open.setSuffix(" %")
        self.sp_open.setToolTip("100 % = ganz offen. Kleiner -> Finger öffnen weniger weit, weniger Kollisionen\n"
                                "mit Nachbarsteinen. Muss noch sicher über die Steinbreite passen.")
        self.sp_open.setValue(round((1 - self.cfg["gripper"]["open_pos"] / 255) * 100))
        self.sp_open.valueChanged.connect(
            lambda pct: self._cfg_set(("gripper", "open_pos"), int(round((1 - pct / 100) * 255))))
        self.btn_open_test = QPushButton("Testen")
        self.btn_open_test.setToolTip("Greifer auf diese Öffnung fahren - nachmessen bzw. Stein dazwischen halten")
        self.btn_open_test.clicked.connect(self.open_test)
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(self.sp_open, 1)
        h.addWidget(self.btn_open_test)
        return w

    def open_test(self):
        gr = self.gripper
        if gr is None or self.task is not None:
            self.status.setText("Greifer nicht verbunden" if gr is None else "Erst laufenden Ablauf beenden")
            return
        pct = self.sp_open.value()
        pos = int(self.cfg["gripper"]["open_pos"])
        self.btn_open_test.setEnabled(False)
        self.lbl_grip.setText("Greifer: fährt …")

        def done(res, err):
            self.btn_open_test.setEnabled(True)
            if err is not None:
                self.lbl_grip.setText(f"Greifer-Fehler: {err!r}")
                return
            p, obj = res
            self.lbl_grip.setText(f"Greifer: Position {p}")
            self._log(f"Öffnungstest: {pct} % = Position {pos}, erreicht {p}"
                      + (" (auf Objekt gestoppt)" if obj in (RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
                                                             RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT)
                         else "") + " - jetzt Fingerabstand messen")

        g = self.cfg["gripper"]
        self._run_async(lambda: gr.move_and_wait_for_pos(pos, int(g["speed"]), max(int(g["force"]), 1)), done)

    def _form_box(self, title, rows):
        g = QGroupBox(title)
        f = QFormLayout(g)
        for label, wdg in rows:
            f.addRow(label, wdg)
        return g

    # ------------------------------------------------------------ Reiter 3
    def _tab_drops(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._hint(
            "Freedrive einschalten (für Ablagen 'senkrecht halten' ausschalten, falls nötig), den Greifer "
            "über die Kiste der Farbe führen und speichern. Der Stein wird an dieser Pose losgelassen - "
            "hoch genug über dem Kistenrand wählen."))
        g = QGroupBox("Ablageposition je Farbe")
        gv = QVBoxLayout(g)
        self.list_drops = QListWidget()
        self.list_drops.setMinimumHeight(140)
        gv.addWidget(self.list_drops)
        h = QHBoxLayout()
        self.btn_drop_save = QPushButton("Aktuelle Pose speichern")
        self.btn_drop_goto = QPushButton("Anfahren (Test)")
        self.btn_drop_clear = QPushButton("Löschen")
        self.btn_drop_save.clicked.connect(self.drop_save)
        self.btn_drop_goto.clicked.connect(self.drop_goto)
        self.btn_drop_clear.clicked.connect(self.drop_clear)
        for b in (self.btn_drop_save, self.btn_drop_goto, self.btn_drop_clear):
            h.addWidget(b)
        gv.addLayout(h)
        v.addWidget(g)

        g = QGroupBox("Beobachtungsposition (Roboter außerhalb des Kamerabilds)")
        gv = QVBoxLayout(g)
        self.lbl_home = QLabel()
        self.lbl_home.setWordWrap(True)
        gv.addWidget(self.lbl_home)
        h = QHBoxLayout()
        self.btn_home_save = QPushButton("Aktuelle Pose speichern")
        self.btn_home_goto = QPushButton("Anfahren")
        self.btn_home_save.clicked.connect(self.home_save)
        self.btn_home_goto.clicked.connect(lambda: self._start_task("home"))
        h.addWidget(self.btn_home_save)
        h.addWidget(self.btn_home_goto)
        gv.addLayout(h)
        v.addWidget(g)
        v.addStretch(1)
        return w

    # ------------------------------------------------------------ Reiter 4
    def _tab_sort(self):
        w = QWidget()
        v = QVBoxLayout(w)
        g = self._form_box("Greifen", [
            ("Fingerspitzen über Tisch beim Greifen:", self._num(("grip_h_mm",), 1, 30, 0.5, 1, " mm")),
            ("Langsame Annäherung ab:", self._num(("approach_mm",), 0, 100, 5, 0, " mm")),
            ("Transferhöhe (Fingerspitzen):", self._num(("travel_mm",), 30, 300, 5, 0, " mm")),
            ("Korrektur X:", self._num(("corr_mm", 0), -30, 30, 0.5, 1, " mm")),
            ("Korrektur Y:", self._num(("corr_mm", 1), -30, 30, 0.5, 1, " mm")),
            ("Kraftgrenze beim Absenken (0 = aus):", self._num(("force_limit_N",), 0, 100, 1, 0, " N")),
            ("Greifer-Winkeloffset:",self._num(("gripper", "angle_off_deg"), -180, 180, 90, 0, " °")),
            ("Max. Öffnung beim Greifen/Ablegen:", self._open_spin()),
            ("Greifer Speed:", self._num(("gripper", "speed"), 0, 255, 5, 0)),
            ("Greifer Kraft:", self._num(("gripper", "force"), 0, 255, 5, 0)),
        ])
        v.addWidget(g)
        chk = QCheckBox("UR-Kontakterkennung beim Absenken (empfohlen, erkennt Kontakt unabhängig von Beschleunigung)")
        chk.setChecked(self.cfg.get("contact_detect", True))
        chk.toggled.connect(lambda on: self._cfg_set(("contact_detect",), on))
        v.addWidget(chk)
        v.addWidget(self._hint("Kraftgrenze: wirkt zusätzlich, nur bei konstanter Geschwindigkeit. Die aktuelle "
                               "Kraft steht oben in der Statuszeile ('F … N'). Testfahrten melden die maximale Kraft."))
        v.addWidget(self._form_box("Geschwindigkeiten", [
            ("moveL v [m/s]:", self._num(("speed", "v"), 0.01, 0.5, 0.01, 2)),
            ("moveL a [m/s²]:", self._num(("speed", "a"), 0.05, 2.0, 0.05, 2)),
            ("Annäherung v [m/s]:", self._num(("speed", "v_slow"), 0.005, 0.2, 0.005, 3)),
            ("moveJ v [rad/s]:", self._num(("speed", "vj"), 0.05, 2.0, 0.05, 2)),
            ("moveJ a [rad/s²]:", self._num(("speed", "aj"), 0.05, 3.0, 0.05, 2)),
        ]))

        g = QGroupBox("Arbeitsbereich (optional: nur Steine innerhalb des Polygons, z. B. Kisten ausblenden)")
        h = QHBoxLayout(g)
        self.btn_roi = QPushButton("Bereich im Bild klicken")
        self.btn_roi.setCheckable(True)
        self.btn_roi.toggled.connect(self.roi_toggle)
        b = QPushButton("Bereich löschen")
        b.clicked.connect(self.roi_clear)
        h.addWidget(self.btn_roi)
        h.addWidget(b)
        v.addWidget(g)

        g = QGroupBox("Ablauf")
        gv = QVBoxLayout(g)
        self.btn_detect = QPushButton("Steine erkennen (Vorschau, aktuelles Bild)")
        self.btn_detect.clicked.connect(self.detect_preview)
        self.list_dets = QListWidget()
        self.list_dets.setMinimumHeight(100)
        self.btn_hover = QPushButton("Test: Greifer über nächsten Stein fahren (ohne Greifen)")
        self.btn_hover.clicked.connect(lambda: self._start_task("hover"))
        self.btn_sort = QPushButton("▶ Sortieren starten")
        self.btn_sort.setMinimumHeight(42)
        self.btn_sort.setStyleSheet("font-weight:bold;")
        self.btn_sort.clicked.connect(self.start_sort)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        for wdg in (self.btn_detect, self.list_dets, self.btn_hover, self.btn_sort, self.log):
            gv.addWidget(wdg)
        v.addWidget(g)
        return w

    # ================================================================== Allgemein
    def _log(self, msg):
        self.status.setText(msg)
        self.log.appendPlainText(time.strftime("%H:%M:%S  ") + msg)

    def save(self):
        self.cfg["robot_ip"] = self.ip_edit.text().strip()
        try:
            save_cfg(self.cfg)
            self.status.setText(f"Gespeichert: {CONFIG_PATH}")
        except OSError as e:
            QMessageBox.critical(self, "Fehler", f"Speichern fehlgeschlagen:\n{e}")

    def _run_async(self, fn, on_done):
        w = Worker(fn)
        self._threads.add(w)
        w.done.connect(on_done)
        w.finished.connect(lambda: self._threads.discard(w))
        w.start()

    def _update_enabled(self):
        conn = self.rtde_c is not None
        busy = self.task is not None
        motion = conn and not busy and self.freedrive is None
        self.btn_connect.setEnabled(not conn)
        self.btn_disconnect.setEnabled(conn and not busy)
        self.ip_edit.setEnabled(not conn)
        self.btn_free.setEnabled(conn and not busy)
        self.btn_recover.setEnabled(conn and not busy)
        self.chk_vertical.setEnabled(self.freedrive is None)
        for b in (self.btn_align, self.btn_goto_click, self.btn_drop_goto, self.btn_home_goto, self.btn_hover):
            b.setEnabled(motion)
        for b in (self.btn_capture, self.btn_pivot_add, self.btn_drop_save, self.btn_home_save):
            b.setEnabled(conn and not busy)
        self.btn_clamp.setEnabled(self.gripper is not None and not busy)
        sorting = busy and self.task.job == "sort"
        self.btn_sort.setEnabled(sorting or motion)
        self.btn_sort.setText("■ Sortieren stoppen" if sorting else "▶ Sortieren starten")

    def _on_tab(self, i):
        self.cb_view.setCurrentIndex({0: 0, 1: 1, 2: 0, 3: 2}.get(i, 0))
        if i != 1:
            self.btn_pipette.setChecked(False)
        if i != 3 and self.btn_roi.isChecked():
            self.btn_roi.setChecked(False)

    # ================================================================== Kamera
    def refresh_cameras(self):
        self.stop_camera()
        self.cb_cam.clear()
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            for i in range(6):
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    self.cb_cam.addItem(f"Kamera {i}", i)
                cap.release()
        finally:
            QApplication.restoreOverrideCursor()
        if self.cb_cam.count() == 0:
            self.cb_cam.addItem("Keine Kamera", -1)
        k = self.cb_cam.findData(self.cfg["camera_index"])
        if k >= 0:
            self.cb_cam.setCurrentIndex(k)
        self.btn_cam.setEnabled(self.cb_cam.currentData() != -1)

    def toggle_camera(self):
        if self.cap is not None:
            self.stop_camera()
            return
        idx = self.cb_cam.currentData()
        if idx is None or idx < 0:
            return
        backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, backend)
        if not cap.isOpened():
            QMessageBox.warning(self, "Fehler", f"Kamera {idx} konnte nicht geöffnet werden.")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)      # Fokus fest, sonst verschiebt sich die Kalibrierung
        self.cap = cap
        self.cfg["camera_index"] = idx
        self.cb_cam.setEnabled(False)
        self.btn_cam.setText("Kamera stoppen")

    def stop_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.cb_cam.setEnabled(True)
        self.btn_cam.setText("Kamera starten")

    def _cam_tick(self):
        if self.cap is not None:
            ok, f = self.cap.read()
            if ok:
                with self.frame_lock:
                    self.frame, self.frame_ts = f, time.time()
                self._check_img_size(f)
        src = self.frozen if self.frozen is not None else self.frame
        if src is None:
            return
        self.view.set_image(self._render(src))
        self.view.markers = self._markers()
        self.view.polygon = [tuple(p) for p in self.cfg.get("roi") or []]
        self.view.info = "STANDBILD - Spitze anklicken" if self.pending_pt is not None else (
            "STANDBILD" if self.frozen is not None else "")

    def _check_img_size(self, f):
        size = self.cfg["calib"].get("img_size")
        if size and list(f.shape[1::-1]) != list(size) and not self._size_warned:
            self._size_warned = True
            self._log(f"WARNUNG: Kamerabild {f.shape[1]}x{f.shape[0]}, kalibriert wurde mit "
                      f"{size[0]}x{size[1]} - Kalibrierung neu durchführen!")

    def _render(self, src):
        mode = self.cb_view.currentIndex()
        if mode == 1:
            ci = self.list_colors.currentRow()
            if 0 <= ci < len(self.cfg["colors"]):
                col = self.cfg["colors"][ci]
                hsv = cv2.cvtColor(cv2.GaussianBlur(src, (5, 5), 0), cv2.COLOR_BGR2HSV)
                m = color_mask(hsv, col["lo"], col["hi"], self.cfg["detect"]["morph"]) > 0
                out = (src // 4).copy()
                out[m] = src[m]
                dets = detect_bricks(src, self.cfg, self.calib, only=ci)
                draw_dets(out, dets, self.cfg["colors"])
                self.lbl_count.setText(f"{col['name']}: {len(dets)} Stein(e) erkannt "
                                       f"({int(m.sum())} Pixel in der Maske)")
                return out
        elif mode == 2:
            out = src.copy()
            draw_dets(out, detect_bricks(src, self.cfg, self.calib), self.cfg["colors"])
            return out
        return src

    def _markers(self):
        if self.tabs.currentIndex() != 0:
            return []
        mk = []
        for i, p in enumerate(self.cfg["calib"]["points"]):
            if p["px"] is not None:
                mk.append((p["px"][0], p["px"][1], QColor(0, 255, 120), f"P{i+1}"))
        if self.last_click is not None:
            mk.append((self.last_click[0], self.last_click[1], QColor(255, 220, 0), "Test"))
        return mk

    def _unfreeze(self):
        self.frozen = None
        self.pending_pt = None

    def _on_hover(self, u, v):
        src = self.frozen if self.frozen is not None else self.frame
        if src is None:
            return
        x, y = int(round(u)), int(round(v))
        if not (0 <= x < src.shape[1] and 0 <= y < src.shape[0]):
            self.lbl_hover.setText("")
            return
        hh, ss, vv = cv2.cvtColor(np.uint8([[src[y, x]]]), cv2.COLOR_BGR2HSV)[0, 0]
        t = f"Pixel ({u:.1f}, {v:.1f})   HSV ({hh}, {ss}, {vv})"
        if self.calib is not None:
            bx, by = self.calib.to_base(u, v)
            t += f"   Basis X {bx*1000:.1f}  Y {by*1000:.1f} mm"
        self.lbl_hover.setText(t)

    def _on_view_click(self, u, v):
        tab = self.tabs.currentIndex()
        if tab == 0:
            if self.pending_pt is not None:
                i = self.pending_pt
                self.cfg["calib"]["points"][i]["px"] = [u, v]
                self.pending_pt = None
                self.frozen = None
                self._refresh_points()
                n = sum(p["px"] is not None for p in self.cfg["calib"]["points"])
                self._log(f"P{i+1} markiert ({n} Punkte). Nächste Stelle anfahren oder berechnen.")
                self.save()
            elif self.calib is not None:
                x, y = self.calib.to_base(u, v)
                self.last_click = (u, v, x, y)
                self.lbl_click.setText(f"Klickpunkt: Pixel ({u:.1f}, {v:.1f}) → X {x*1000:.1f}  Y {y*1000:.1f} mm, "
                                       f"Tischhöhe {self.calib.table_z(x, y)*1000:.1f} mm")
            else:
                self.status.setText("Erst 'Pose übernehmen' drücken (oder Kalibrierung berechnen)")
        elif tab == 1 and self.btn_pipette.isChecked():
            add = bool(QApplication.keyboardModifiers() & Qt.KeyboardModifier.ShiftModifier)
            self.pipette(u, v, add)
        elif tab == 3 and self.roi_edit:
            self.cfg["roi"].append([u, v])

    # ================================================================== Roboter
    def connect_robot(self):
        if RTDEControlInterface is None:
            QMessageBox.critical(self, "Fehler", "ur_rtde ist nicht installiert (pip install ur_rtde).")
            return
        ip = self.ip_edit.text().strip()
        self.cfg["robot_ip"] = ip
        self.status.setText(f"Verbinde mit {ip} …")
        self.btn_connect.setEnabled(False)

        def work():
            r = RTDEReceiveInterface(ip)
            c = RTDEControlInterface(ip)
            gr, gerr = None, None
            if RobotiqGripper is not None:
                try:
                    gr = RobotiqGripper()
                    gr.connect(ip, GRIPPER_PORT)
                    gr.activate(auto_calibrate=False)
                except Exception as e:
                    gr, gerr = None, e
            return c, r, gr, gerr

        def done(res, err):
            if err is not None:
                self.btn_connect.setEnabled(True)
                QMessageBox.critical(self, "Verbindung fehlgeschlagen", str(err))
                self.status.setText("Getrennt")
                return
            self.rtde_c, self.rtde_r, self.gripper, gerr = res
            self.lbl_grip.setText("Greifer: bereit" if self.gripper else f"Greifer: FEHLER ({gerr!r})")
            try:
                off = self.rtde_c.getTCPOffset()
                self.lbl_tcp.setText(f"TCP-Offset (Pendant): X {off[0]*1000:.0f} Y {off[1]*1000:.0f} "
                                     f"Z {off[2]*1000:.0f} mm")
                if math.hypot(off[0], off[1]) > 0.002:
                    self._log("WARNUNG: TCP-Offset hat X/Y-Anteil - TCP sollte auf der Greiferachse liegen")
            except Exception:
                pass
            self.pose_timer.start()
            self._log("Verbunden")
            self._update_enabled()

        self._run_async(work, done)

    def disconnect_robot(self):
        self.stop_all()
        self.pose_timer.stop()
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass
        for iface in (self.rtde_c, self.rtde_r):
            try:
                if iface is not None:
                    iface.disconnect()
            except Exception:
                pass
        self.rtde_c = self.rtde_r = self.gripper = None
        self.lbl_grip.setText("Greifer: getrennt")
        self.lbl_pose.setText("TCP: --")
        self._log("Getrennt")
        self._update_enabled()

    def _pose(self):
        return list(self.rtde_r.getActualTCPPose())

    def _update_pose(self):
        try:
            p = self._pose()
        except Exception as e:
            self.lbl_pose.setText(f"Lesefehler: {e}")
            return
        fd = {"teach": "  |  FREEDRIVE", "constrained": "  |  FREEDRIVE senkrecht"}.get(self.freedrive, "")
        issue = self._robot_issue()
        try:
            fx, fy, fz = self.rtde_r.getActualTCPForce()[:3]
            ftxt = f"  |  F {math.sqrt(fx*fx + fy*fy + fz*fz):.1f} N (Fz {fz:+.1f})"
        except Exception:
            ftxt = ""
        self.lbl_pose.setText(f"TCP: X {p[0]*1000:.1f}  Y {p[1]*1000:.1f}  Z {p[2]*1000:.1f} mm  |  "
                              f"Neigung {math.degrees(tilt_of(p)):.1f}°  Drehung {math.degrees(yaw_of(p)):.1f}°{ftxt}{fd}"
                              + (f"  |  ⚠ {issue}" if issue else ""))
        self.lbl_pose.setStyleSheet("color:#ff6060; font-weight:bold;" if issue else "")
        self.btn_recover.setStyleSheet("background:#8a6d00; font-weight:bold;" if issue else "")

    # ------------------------------------------------------------ Zustand / Wiederherstellen
    def _robot_issue(self):
        """None = bereit, sonst kurze Beschreibung des Problems."""
        r, c = self.rtde_r, self.rtde_c
        if r is None or c is None:
            return "nicht verbunden"
        try:
            if not r.isConnected():
                return "RTDE-Verbindung unterbrochen"
            if r.isEmergencyStopped():
                return "NOT-HALT"
            if r.isProtectiveStopped():
                return "SCHUTZSTOPP"
            if r.getRobotMode() != 7:
                return "Roboter aus / Bremsen fest"
            if not c.isConnected() or not c.isProgramRunning():
                return "Steuerskript gestoppt"
        except Exception as e:
            return f"Statusfehler: {e}"
        return None

    def _ensure_ready(self):
        """Vor jeder Roboteraktion: gestopptes Steuerskript (z. B. nach IK-Fehler) still neu laden.
        Schutzstopp / Not-Halt / ausgeschaltet -> Hinweis auf 'Wiederherstellen'."""
        issue = self._robot_issue()
        if issue == "Steuerskript gestoppt":
            try:
                if not self.rtde_c.isConnected():
                    self.rtde_c.reconnect()
                if not self.rtde_c.isProgramRunning():
                    self.rtde_c.reuploadScript()
                self._log("Steuerskript neu geladen")
            except Exception as e:
                self._log(f"Steuerskript neu laden fehlgeschlagen: {e}")
            issue = self._robot_issue()
        if issue:
            self._log(f"Roboter nicht bereit: {issue} - 'Wiederherstellen' drücken")
            return False
        return True

    def recover_robot(self):
        if self.rtde_c is None or self.task is not None:
            return
        issue = self._robot_issue()
        if issue in ("NOT-HALT", "SCHUTZSTOPP", "Roboter aus / Bremsen fest") and QMessageBox.question(
                self, "Wiederherstellen",
                f"Zustand: {issue}\n\nDas Programm entsperrt den Schutzstopp, schaltet den Roboter ein und löst "
                "die Bremsen (der Roboter ruckt dabei leicht), lädt das Steuerskript neu und aktiviert den Greifer."
                "\n\nUrsache beseitigt und Arbeitsbereich frei?") != QMessageBox.StandardButton.Yes:
            return
        ip = self.ip_edit.text().strip()
        r, c, gr = self.rtde_r, self.rtde_c, self.gripper
        self._end_freedrive()
        self.btn_recover.setEnabled(False)
        self.status.setText("Stelle Roboter wieder her …")

        def wait_for(cond, timeout, what):
            t0 = time.time()
            while not cond():
                if time.time() - t0 > timeout:
                    raise RuntimeError(f"Zeitüberschreitung: {what}")
                time.sleep(0.2)

        def work():
            log = []
            if not r.isConnected():
                r.reconnect()
                log.append("RTDE-Empfang neu verbunden")
            if r.isEmergencyStopped():
                raise RuntimeError("Not-Halt ist noch gedrückt - erst entriegeln, dann erneut 'Wiederherstellen'")
            dash = None
            if r.isProtectiveStopped() or r.getRobotMode() != 7:
                if DashboardClient is None:
                    raise RuntimeError("dashboard_client fehlt - am Teach Pendant entsperren/einschalten")
                dash = DashboardClient(ip)
                dash.connect()
            if r.isProtectiveStopped():
                dash.closeSafetyPopup()
                t0 = time.time()
                while r.isProtectiveStopped():       # UR erlaubt Entsperren erst 5 s nach dem Stopp
                    try:
                        dash.unlockProtectiveStop()
                    except Exception:
                        pass
                    if time.time() - t0 > 12:
                        raise RuntimeError("Schutzstopp ließ sich nicht entsperren - am Pendant prüfen")
                    time.sleep(1.0)
                log.append("Schutzstopp entsperrt")
            if r.getRobotMode() != 7:
                dash.closeSafetyPopup()
                if r.getRobotMode() < 5:             # POWER_OFF / POWER_ON
                    dash.powerOn()
                    wait_for(lambda: r.getRobotMode() >= 5, 30, "Einschalten")
                dash.brakeRelease()
                wait_for(lambda: r.getRobotMode() == 7, 30, "Bremsen lösen")
                log.append("Roboter eingeschaltet, Bremsen gelöst")
            if dash is not None:
                dash.disconnect()
            if not c.isConnected():
                c.reconnect()
                log.append("RTDE-Steuerung neu verbunden")
            if not c.isProgramRunning():
                c.reuploadScript()
                log.append("Steuerskript neu geladen")
            new_gr = gr
            if RobotiqGripper is not None:
                try:
                    if gr is None:
                        raise ConnectionError
                    if not gr.is_active():
                        gr.activate(auto_calibrate=False)
                        log.append("Greifer neu aktiviert")
                except Exception:
                    new_gr = RobotiqGripper()
                    new_gr.connect(ip, GRIPPER_PORT)
                    new_gr.activate(auto_calibrate=False)
                    log.append("Greifer neu verbunden + aktiviert")
            return log, new_gr

        def done(res, err):
            self.btn_recover.setEnabled(True)
            if err is not None:
                self._log(f"Wiederherstellen fehlgeschlagen: {err}")
                return
            log, self.gripper = res
            self.lbl_grip.setText("Greifer: bereit" if self.gripper else "Greifer: getrennt")
            self._log("Wiederhergestellt" + (": " + ", ".join(log) if log else " (war schon bereit)"))
            self._update_enabled()

        self._run_async(work, done)

    def _set_checked(self, btn, on):
        btn.blockSignals(True)
        btn.setChecked(on)
        btn.blockSignals(False)

    def toggle_freedrive(self, on):
        c = self.rtde_c
        if c is None or self.task is not None:
            self._set_checked(self.btn_free, False)
            return
        try:
            if on:
                if not self._ensure_ready():
                    raise RuntimeError("Roboter nicht bereit")
                c.speedStop()
                mode, ok = "teach", False
                if self.chk_vertical.isChecked():
                    # optional: nur X/Y/Z + Drehung um Z frei (Orientierung bleibt wie sie gerade ist)
                    try:
                        ok = bool(c.freedriveMode([1, 1, 1, 0, 0, 1], [0.0] * 6))
                        mode = "constrained"
                    except Exception as e:
                        self._log(f"Eingeschränkter Freedrive nicht möglich ({e}) - nutze normalen Freedrive")
                if not ok:
                    mode = "teach"
                    ok = bool(c.teachMode())      # wie in P06
                if not ok:
                    raise RuntimeError("teachMode fehlgeschlagen (Remote Control aktiv?)")
                self.freedrive = mode
                self.btn_free.setText("Freedrive AUS")
                self._log("Freedrive aktiv" + (" (nur X/Y/Z + Drehung um Z)" if mode == "constrained" else ""))
            else:
                self._end_freedrive()
                self._log("Freedrive beendet")
        except Exception as e:
            self.freedrive = None
            self._set_checked(self.btn_free, False)
            self.btn_free.setText("Freedrive EIN")
            self._log(f"Freedrive-Fehler: {e}")
        self._update_enabled()

    def _end_freedrive(self):
        if self.freedrive is not None and self.rtde_c is not None:
            try:
                if self.freedrive == "constrained":
                    self.rtde_c.endFreedriveMode()
                else:
                    self.rtde_c.endTeachMode()
            except Exception as e:
                self._log(f"Freedrive beenden: {e}")
        self.freedrive = None
        self._set_checked(self.btn_free, False)
        self.btn_free.setText("Freedrive EIN")

    def _start_task(self, job, **kw):
        if self.task is not None:
            self.status.setText("Es läuft bereits ein Ablauf")
            return
        if self.rtde_c is None:
            self.status.setText("Roboter nicht verbunden")
            return
        if self.freedrive is not None:
            self._end_freedrive()
        if not self._ensure_ready():
            self._update_enabled()
            return
        t = RobotTask(self, job, **kw)
        t.status.connect(self._log)
        t.detections.connect(self._show_dets)
        t.result.connect(self._task_done)
        t.finished.connect(lambda: self._threads.discard(t))
        self._threads.add(t)
        self.task = t
        self._update_enabled()
        t.start()

    def _task_done(self, msg):
        self.task = None
        self._log(msg)
        self._update_enabled()

    def stop_all(self):
        if self.task is not None:
            self.task.stop_ev.set()
            self.status.setText("Stoppe …")
        if self.freedrive is not None:
            self._end_freedrive()
            self._update_enabled()
        elif self.task is None and self.rtde_c is not None:
            try:
                self.rtde_c.stopL(2.0)
                self.rtde_c.stopJ(2.0)
            except Exception:
                pass

    # ================================================================== Greifer
    def _clamp_label(self):
        self.btn_clamp.setText(["① Vorpositionieren (Spitze passt gerade so hinein)",
                                "② Spitze festklemmen",
                                "Spitze lösen (Greifer öffnen)"][self.clamp_state])

    def clamp_cycle(self):
        gr = self.gripper
        if gr is None:
            return
        st = self.clamp_state
        pos, speed, force = [(int(self.sl_pre.value()), 80, 1), (255, 40, 150),
                             (0, 150, 50)][st]      # Spitze lösen: ganz auf (unabhängig von max. Öffnung)
        self.btn_clamp.setEnabled(False)
        self.lbl_grip.setText("Greifer: fährt …")

        def done(res, err):
            self.btn_clamp.setEnabled(self.task is None)
            if err is not None:
                self.lbl_grip.setText(f"Greifer-Fehler: {err!r}")
                return
            p, obj = res
            self.lbl_grip.setText(f"Greifer: Position {p}")
            if st == 1 and obj not in (RobotiqGripper.ObjectStatus.STOPPED_INNER_OBJECT,
                                       RobotiqGripper.ObjectStatus.STOPPED_OUTER_OBJECT):
                self._log("Achtung: Greifer hat ganz geschlossen - Spitze nicht gegriffen?")
            self.clamp_state = (st + 1) % 3
            self._clamp_label()
            self._log(["Spitze jetzt zwischen die Finger halten und ② drücken",
                       "Spitze eingespannt - Kalibrierung kann beginnen",
                       "Greifer geöffnet"][st])

        self._run_async(lambda: gr.move_and_wait_for_pos(pos, speed, force), done)

    # ================================================================== Kalibrierung
    def _refresh_points(self):
        self.list_points.clear()
        for i, p in enumerate(self.cfg["calib"]["points"]):
            x, y, z = (c * 1000 for c in p["pose"][:3])
            px = f"Pixel ({p['px'][0]:.1f}, {p['px'][1]:.1f})" if p["px"] is not None else "Pixel: — (anklicken!)"
            self.list_points.addItem(f"P{i+1}: {px}   |   X {x:.1f}  Y {y:.1f}  Z {z:.1f} mm")

    # ------------------------------------------------------------ Pivot
    def _update_pivot_label(self):
        n = len(self.cfg["pivot"]["poses"])
        t = np.array(self.cfg["tip_offset"]) * 1000
        rms = self.cfg["pivot"].get("rms_mm")
        if self.cfg["pivot"].get("manual"):
            self.lbl_pivot.setText(f"Manuell eingetragen: Spitze im TCP-KS X {t[0]:.1f}  Y {t[1]:.1f}  Z {t[2]:.1f} mm "
                                   "(nicht gemessen - Pivot ist genauer)")
            return
        if rms is None:
            txt = f"{n} Pose(n) aufgenommen. Kein Spitzen-Offset gemessen -> Kamerapunkte nur senkrecht aufnehmen!"
        else:
            txt = (f"{n} Pose(n). Spitze im TCP-KS: X {t[0]:.1f}  Y {t[1]:.1f}  Z {t[2]:.1f} mm "
                   f"(Länge {np.linalg.norm(t):.1f} mm), Restfehler RMS {rms:.2f} mm")
            if rms > 1.0:
                txt += "  ⚠ groß - Spitze ist verrutscht oder Posen zu ähnlich"
        self.lbl_pivot.setText(txt)

    def pivot_add(self):
        if self.rtde_r is None:
            return
        self.cfg["pivot"]["poses"].append(self._pose())
        self._update_pivot_label()
        n = len(self.cfg["pivot"]["poses"])
        self._log(f"Pivot-Pose {n} übernommen" + (" - weitere Neigungen aufnehmen (min. 4)" if n < 4 else ""))

    def pivot_clear(self):
        self.cfg["pivot"]["poses"] = []
        self._update_pivot_label()

    def pivot_compute(self):
        poses = self.cfg["pivot"]["poses"]
        if len(poses) < 3:
            QMessageBox.warning(self, "Pivot", "Mindestens 3 (besser 4-6) Posen mit verschiedenen Neigungen nötig.")
            return
        # Orientierungsvielfalt prüfen: größter Winkel zwischen zwei Werkzeug-Z-Achsen
        zs = [_rot(p)[:, 2] for p in poses]
        spread = max(math.degrees(math.acos(float(np.clip(a @ b, -1, 1)))) for a in zs for b in zs)
        if spread < 20 and QMessageBox.question(
                self, "Pivot", f"Die Neigungen unterscheiden sich nur um {spread:.0f}°. Das macht den Offset "
                "ungenau (besser > 30°). Trotzdem berechnen?") != QMessageBox.StandardButton.Yes:
            return
        t, q, res = pivot_calibrate(poses)
        self.cfg["tip_offset"] = [float(x) for x in t]
        self.cfg["pivot"]["rms_mm"] = float(np.sqrt((res ** 2).mean()) * 1000)
        self.cfg["pivot"]["manual"] = False
        self.sp_tip_z.setValue(t[2] * 1000)
        self._update_pivot_label()
        self.save()
        msg = f"Spitzen-Offset gemessen (Pivotpunkt X {q[0]*1000:.1f} Y {q[1]*1000:.1f} Z {q[2]*1000:.1f} mm)"
        if self.calib is not None:
            msg += " - Kamera-Kalibrierung jetzt neu berechnen!"
        self._log(msg)

    def tip_manual(self):
        self.cfg["tip_offset"] = [0.0, 0.0, self.sp_tip_z.value() / 1000]
        self.cfg["pivot"]["manual"] = True
        self._update_pivot_label()
        self.save()
        self._log("Spitzen-Offset manuell gesetzt" +
                  (" - Kamera-Kalibrierung jetzt neu berechnen!" if self.calib is not None else ""))

    def _tip_known(self):
        return self.cfg["pivot"].get("manual") or self.cfg["pivot"].get("rms_mm") is not None

    # ------------------------------------------------------------ Kamera-Punkte
    def calib_capture(self):
        if self.rtde_r is None:
            return
        if self.frame is None:
            QMessageBox.warning(self, "Kamera", "Kein Kamerabild - erst Kamera starten.")
            return
        pose = self._pose()
        tilt = math.degrees(tilt_of(pose))
        if tilt > 2 and not self._tip_known() and QMessageBox.question(
                self, "Werkzeug geneigt",
                f"Das Werkzeug ist {tilt:.1f}° geneigt, aber der Spitzen-Offset ist noch nicht gemessen.\n"
                "Erst Pivot-Kalibrierung machen - sonst wird die Spitzenposition falsch.\n"
                "Trotzdem übernehmen?") != QMessageBox.StandardButton.Yes:
            return
        with self.frame_lock:
            frame = self.frame.copy()
        self.cfg["calib"]["points"].append({"pose": pose, "px": None})
        self.calib_frames.append(frame)
        self.frozen = frame
        self.pending_pt = len(self.cfg["calib"]["points"]) - 1
        self.view.reset_view()
        self._refresh_points()
        self._log(f"P{self.pending_pt+1}: Pose übernommen - jetzt die Spitze im Standbild anklicken "
                  "(Mausrad = Zoom, rechte Maustaste = verschieben)")

    def calib_remark(self):
        i = self.list_points.currentRow()
        if i < 0:
            return
        if self.calib_frames[i] is None:
            QMessageBox.information(self, "Nicht möglich", "Für diesen Punkt ist kein Standbild mehr vorhanden "
                                    "(Programm neu gestartet). Punkt löschen und neu aufnehmen.")
            return
        self.frozen = self.calib_frames[i]
        self.pending_pt = i
        self._log(f"P{i+1} neu im Standbild anklicken")

    def calib_delete(self):
        i = self.list_points.currentRow()
        if i < 0:
            return
        del self.cfg["calib"]["points"][i]
        del self.calib_frames[i]
        self._unfreeze()
        self._refresh_points()

    def calib_clear(self):
        if QMessageBox.question(self, "Alle löschen", "Alle Kalibrierpunkte löschen?") != QMessageBox.StandardButton.Yes:
            return
        self.cfg["calib"]["points"] = []
        self.calib_frames = []
        self._unfreeze()
        self._refresh_points()

    def calib_compute(self):
        pts = [p for p in self.cfg["calib"]["points"] if p["px"] is not None]
        if len(pts) < 3:
            QMessageBox.warning(self, "Zu wenige Punkte", "Mindestens 3 markierte Punkte nötig (besser 5-6).")
            return
        px = np.array([p["px"] for p in pts], float)
        tip = self.cfg["tip_offset"]
        xyz = np.array([tool_point(p["pose"], tip) for p in pts], float)    # Spitzenpositionen
        hull = cv2.contourArea(cv2.convexHull(xyz[:, :2].astype(np.float32)))
        if hull < 0.004 and QMessageBox.question(
                self, "Punkte zu eng",
                f"Die Punkte überdecken nur {hull*1e4:.0f} cm². Für gute Genauigkeit über den ganzen "
                "Arbeitsbereich verteilen. Trotzdem berechnen?") != QMessageBox.StandardButton.Yes:
            return
        H, method = fit_mapping(px, xyz[:, :2])
        res = np.linalg.norm(map_pts(H, px) - xyz[:, :2], axis=1) * 1000
        loo = []
        if len(pts) >= 4:      # Kreuzvalidierung: jeden Punkt aus den übrigen vorhersagen
            for i in range(len(pts)):
                m = np.arange(len(pts)) != i
                Hi, _ = fit_mapping(px[m], xyz[m, :2])
                loo.append(float(np.linalg.norm(map_pts(Hi, px[i:i + 1])[0] - xyz[i, :2]) * 1000))
        A = np.hstack([xyz[:, :2], np.ones((len(pts), 1))])
        plane, *_ = np.linalg.lstsq(A, xyz[:, 2], rcond=None)
        zres = np.abs(A @ plane - xyz[:, 2]) * 1000
        c = self.cfg["calib"]
        c["H"] = H.tolist()
        c["plane"] = [float(x) for x in plane]
        c["method"] = method
        c["tip_offset"] = list(tip)
        img =self.calib_frames[-1] if self.calib_frames and self.calib_frames[-1] is not None else self.frame
        c["img_size"] = [int(img.shape[1]), int(img.shape[0])] if img is not None else None
        c["stats"] = {"n": len(pts), "rms": float(np.sqrt((res ** 2).mean())),
                      "loo_max": max(loo) if loo else None, "loo_mean": float(np.mean(loo)) if loo else None,
                      "tilt_deg": math.degrees(math.atan(math.hypot(plane[0], plane[1]))),
                      "z_res_max": float(zres.max())}
        self.calib = Calib.from_cfg(self.cfg)
        self._size_warned = False
        self._update_calib_label()
        self.save()
        self._log("Kalibrierung berechnet und gespeichert")

    def _update_calib_label(self):
        c = self.cfg["calib"]
        s = c.get("stats")
        if self.calib is None or not s:
            self.lbl_calib.setText("Noch nicht kalibriert.")
            return
        t = f"✔ {c['method']} aus {s['n']} Punkten, Restfehler RMS {s['rms']:.2f} mm"
        if s["loo_max"] is not None:
            t += f"\nKreuzvalidierung: Ø {s['loo_mean']:.2f} mm, max {s['loo_max']:.2f} mm"
            if s["loo_max"] > 3:
                t += "  ⚠ groß - Punkte prüfen / neu markieren"
        else:
            t += "\n(bei 3 Punkten keine Fehlerabschätzung möglich - 4+ Punkte empfohlen)"
        t += f"\nTischneigung {s['tilt_deg']:.2f}°, Höhenabweichung max {s['z_res_max']:.1f} mm"
        self.lbl_calib.setText(t)

    def goto_click(self):
        if self.last_click is None or self.calib is None:
            self.status.setText("Erst kalibrieren und ins Livebild klicken")
            return
        if QMessageBox.question(self, "Testfahrt", "Ist die Kalibrierspitze eingespannt und der Weg frei?") \
                != QMessageBox.StandardButton.Yes:
            return
        _, _, x, y = self.last_click
        self._start_task("goto_xy", x=x, y=y, h_mm=self.sp_test_h.value())

    # ================================================================== Farben
    def _refresh_colors(self):
        cur = self.list_colors.currentRow()
        self.list_colors.blockSignals(True)
        self.list_colors.clear()
        for col in self.cfg["colors"]:
            it = QListWidgetItem(self._swatch_icon(col), col["name"])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if col["enabled"] else Qt.CheckState.Unchecked)
            self.list_colors.addItem(it)
        self.list_colors.blockSignals(False)
        n = len(self.cfg["colors"])
        self.list_colors.setCurrentRow(min(max(cur, 0), n - 1) if n else -1)
        self._load_color(self.list_colors.currentRow())
        self._refresh_drops()

    @staticmethod
    def _swatch_icon(col):
        pm = QPixmap(16, 16)
        b, g, r = color_bgr(col)
        pm.fill(QColor(r, g, b))
        return QIcon(pm)

    def _load_color(self, i):
        if not 0 <= i < len(self.cfg["colors"]):
            return
        col = self.cfg["colors"][i]
        for s, val in zip(self.hsv_sliders, col["lo"] + col["hi"]):
            s.setValue(val, quiet=True)
        self.pip_samples = []
        self._update_preview()

    def _hsv_changed(self):
        i = self.list_colors.currentRow()
        if not 0 <= i < len(self.cfg["colors"]):
            return
        vals = [s.value() for s in self.hsv_sliders]
        col = self.cfg["colors"][i]
        col["lo"], col["hi"] = vals[:3], vals[3:]
        self.list_colors.blockSignals(True)
        self.list_colors.item(i).setIcon(self._swatch_icon(col))
        self.list_colors.blockSignals(False)
        self._update_preview()

    def _update_preview(self):
        i = self.list_colors.currentRow()
        if not 0 <= i < len(self.cfg["colors"]):
            return
        col = self.cfg["colors"][i]
        self.lbl_range.setPixmap(to_pixmap(range_preview(col["lo"], col["hi"])))
        b, g, r = color_bgr(col)
        self.lbl_swatch.setStyleSheet(f"background: rgb({r},{g},{b}); border:1px solid #888;")
        lo, hi = col["lo"], col["hi"]
        wrap = "  (über Rot/0 hinweg)" if lo[0] > hi[0] else ""
        self.lbl_range_txt.setText(f"Farbtonleiste: heller Teil = gewählter H-Bereich{wrap}. Feld darunter: "
                                   f"alle erkannten Farben (oben S {hi[1]}/V {hi[2]}, unten S {lo[1]}/V {lo[2]}). "
                                   f"Mitte ≈ #{r:02X}{g:02X}{b:02X}")

    def _color_item_changed(self, it):
        i = self.list_colors.row(it)
        self.cfg["colors"][i]["enabled"] = it.checkState() == Qt.CheckState.Checked

    def color_add(self):
        name, ok = QInputDialog.getText(self, "Neue Farbe", "Name:")
        if ok and name.strip():
            self.cfg["colors"].append({"name": name.strip(), "lo": [0, 100, 80], "hi": [10, 255, 255],
                                       "enabled": True, "drop": None})
            self._refresh_colors()
            self.list_colors.setCurrentRow(len(self.cfg["colors"]) - 1)

    def color_rename(self):
        i = self.list_colors.currentRow()
        if i < 0:
            return
        name, ok = QInputDialog.getText(self, "Umbenennen", "Name:", text=self.cfg["colors"][i]["name"])
        if ok and name.strip():
            self.cfg["colors"][i]["name"] = name.strip()
            self._refresh_colors()

    def color_remove(self):
        i = self.list_colors.currentRow()
        if i >= 0 and QMessageBox.question(self, "Entfernen", f"Farbe '{self.cfg['colors'][i]['name']}' "
                                           "entfernen?") == QMessageBox.StandardButton.Yes:
            del self.cfg["colors"][i]
            self._refresh_colors()

    def _sample(self, u, v):
        src = self.frozen if self.frozen is not None else self.frame
        x, y, r = int(round(u)), int(round(v)), 3
        patch = src[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1]
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(float)
        ang = hsv[:, 0] * np.pi / 90                      # Farbton als Winkel (Rot liegt bei 0/180)
        h = (math.atan2(np.sin(ang).mean(), np.cos(ang).mean()) * 90 / np.pi) % 180
        return h, float(np.median(hsv[:, 1])), float(np.median(hsv[:, 2]))

    def _apply_samples(self):
        th, ts, tv = (sp.value() for sp in self.sp_tol)
        hs, ss, vs = zip(*self.pip_samples)
        h_lo, h_hi = hue_range(hs, th)
        vals = [h_lo, max(0, min(ss) - ts), max(0, min(vs) - tv),
                h_hi, min(255, max(ss) + ts), min(255, max(vs) + tv)]
        for s, val in zip(self.hsv_sliders, vals):
            s.setValue(int(round(val)), quiet=True)
        self._hsv_changed()

    def pipette(self, u, v, add):
        if self.list_colors.currentRow() < 0:
            return
        if not add:
            self.pip_samples = []
        self.pip_samples.append(self._sample(u, v))
        self._apply_samples()
        h, s, val = self.pip_samples[-1]
        self.status.setText(f"Pipette: HSV ({h:.0f}, {s:.0f}, {val:.0f}) - {len(self.pip_samples)} Probe(n)")

    def color_dialog(self):
        i = self.list_colors.currentRow()
        if i < 0:
            return
        b, g, r = color_bgr(self.cfg["colors"][i])
        c = QColorDialog.getColor(QColor(r, g, b), self, "Steinfarbe wählen")
        if not c.isValid():
            return
        h, s, v = cv2.cvtColor(np.uint8([[[c.blue(), c.green(), c.red()]]]), cv2.COLOR_BGR2HSV)[0, 0]
        self.pip_samples = [(float(h), float(s), float(v))]
        self._apply_samples()

    # ================================================================== Ablagen
    def _refresh_drops(self):
        cur = self.list_drops.currentRow()
        self.list_drops.clear()
        for col in self.cfg["colors"]:
            d = col.get("drop")
            txt = (f"{col['name']}:  X {d[0]*1000:.0f}  Y {d[1]*1000:.0f}  Z {d[2]*1000:.0f} mm" if d
                   else f"{col['name']}:  — keine Ablage —")
            self.list_drops.addItem(QListWidgetItem(self._swatch_icon(col), txt))
        if self.list_drops.count():
            self.list_drops.setCurrentRow(min(max(cur, 0), self.list_drops.count() - 1))

    def drop_save(self):
        i = self.list_drops.currentRow()
        if i < 0 or self.rtde_r is None:
            return
        self.cfg["colors"][i]["drop"] = self._pose()
        self._refresh_drops()
        self.save()
        self._log(f"Ablage für {self.cfg['colors'][i]['name']} gespeichert")

    def drop_goto(self):
        i = self.list_drops.currentRow()
        if i >= 0 and self.cfg["colors"][i].get("drop"):
            self._start_task("drop_test", ci=i)

    def drop_clear(self):
        i = self.list_drops.currentRow()
        if i >= 0:
            self.cfg["colors"][i]["drop"] = None
            self._refresh_drops()

    def _refresh_home(self):
        p = self.cfg.get("home_pose")
        self.lbl_home.setText("Gespeichert: " + fmt_pose(p) if p else "Noch keine Beobachtungsposition gespeichert.")

    def home_save(self):
        if self.rtde_r is None:
            return
        self.cfg["home_q"] = list(self.rtde_r.getActualQ())
        self.cfg["home_pose"] = self._pose()
        self._refresh_home()
        self.save()
        self._log("Beobachtungsposition gespeichert - prüfen, dass der Roboter nicht im Kamerabild ist")

    # ================================================================== Sortieren
    def roi_toggle(self, on):
        self.roi_edit = on
        if on:
            self.cfg["roi"] = []
            self.tabs.setCurrentIndex(3)
            self.btn_roi.setText("Fertig (Ecken im Bild klicken)")
            self.status.setText("Ecken des Arbeitsbereichs nacheinander ins Bild klicken, dann 'Fertig'")
        else:
            self.btn_roi.setText("Bereich im Bild klicken")
            if 0 < len(self.cfg["roi"]) < 3:
                self.cfg["roi"] = []
                self.status.setText("Bereich braucht mindestens 3 Ecken - verworfen")

    def roi_clear(self):
        self._set_checked(self.btn_roi, False)
        self.roi_toggle(False)
        self.cfg["roi"] = []

    def _show_dets(self, dets):
        self.list_dets.clear()
        for d in dets:
            t = f"{d['name']}: Pixel ({d['px'][0]:.0f}, {d['px'][1]:.0f}), Fläche {d['area']:.0f}"
            if d["xy"] is not None:
                t += f"  →  X {d['xy'][0]*1000:.1f}  Y {d['xy'][1]*1000:.1f} mm, " \
                     f"Greifwinkel {math.degrees(d['theta']) % 180:.0f}°"
                if d.get("reach") is False:
                    t += "   ⚠ NICHT ERREICHBAR"
            self.list_dets.addItem(t)
        if not dets:
            self.list_dets.addItem("Keine Steine erkannt")

    def detect_preview(self):
        if self.frame is None:
            self.status.setText("Kein Kamerabild")
            return
        dets = detect_bricks(self.frame, self.cfg, self.calib)
        # Erreichbarkeit per IK vorab prüfen (nur wenn verbunden und kein Ablauf aktiv)
        if self.calib is not None and self.task is None and self._robot_issue() is None:
            try:
                q = list(self.rtde_r.getActualQ())
                ref = yaw_of(self.cfg["home_pose"] or self._pose())
                for d in dets:
                    d["reach"] = all(pose_reachable(self.rtde_c, p, q)
                                     for p in grasp_poses(self.cfg, self.calib, d, ref))
            except Exception as e:
                self._log(f"IK-Prüfung fehlgeschlagen: {e}")
        self._show_dets(dets)
        self.cb_view.setCurrentIndex(2)
        if self.calib is None:
            self.status.setText("Hinweis: ohne Kalibrierung keine Roboter-Koordinaten")

    def start_sort(self):
        if self.task is not None:
            self.stop_all()
            return
        problems = []
        if self.rtde_c is None:
            problems.append("Roboter nicht verbunden")
        if self.gripper is None:
            problems.append("Greifer nicht verbunden")
        if self.calib is None:
            problems.append("Kamera nicht kalibriert (Schritt 1)")
        if not self.cfg.get("home_q"):
            problems.append("Keine Beobachtungsposition gespeichert (Schritt 3)")
        if self.cap is None:
            problems.append("Kamera läuft nicht")
        enabled = [c for c in self.cfg["colors"] if c["enabled"]]
        if not any(c.get("drop") for c in enabled):
            problems.append("Keine aktivierte Farbe mit Ablageposition (Schritte 2/3)")
        if problems:
            QMessageBox.warning(self, "Sortieren nicht möglich", "\n".join(problems))
            return
        missing = [c["name"] for c in enabled if not c.get("drop")]
        txt = "Kalibrierspitze entfernt und Arbeitsbereich frei?"
        if missing:
            txt += f"\n\nOhne Ablage (werden ignoriert): {', '.join(missing)}"
        if QMessageBox.question(self, "Sortieren starten", txt) != QMessageBox.StandardButton.Yes:
            return
        self.save()
        self.cb_view.setCurrentIndex(2)
        self._start_task("sort")

    # ================================================================== Ende
    def closeEvent(self, ev):
        if self.task is not None:
            self.task.stop_ev.set()
            self.task.wait(5000)
        self._end_freedrive()
        self.disconnect_robot()
        self.stop_camera()
        self.save()
        super().closeEvent(ev)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark(app)
    win = LegoSorter()
    win.show()
    sys.exit(app.exec())
