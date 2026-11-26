# P08 Lego-/Duplo-Sortierer mit DNN-Objekterkennung (YOLO / Ultralytics)
#
# Wie P07, aber statt HSV-Farbschwellen erkennt ein trainiertes YOLO-Modell die Steine
# (Modell z. B. mit src/yolo_labeler.py labeln + trainieren).
#
# Greifwinkel: YOLO (detect) liefert nur achsparallele Boxen. Die Orientierung wird deshalb
# innerhalb jeder Box bestimmt: Stein vom Hintergrund trennen (Sättigung/Otsu, GrabCut oder
# Canny-Kanten) -> größte Kontur -> cv2.minAreaRect -> Längsachse.
# Ein YOLO-OBB-Modell (gedrehte Boxen) oder ein Segmentierungsmodell wird automatisch erkannt
# und liefert den Winkel direkt.
#
# pip install ultralytics opencv-python PyQt6 ur_rtde
import glob
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton, QScrollArea,
    QSizePolicy, QSlider, QSpinBox, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

try:
    from dashboard_client import DashboardClient    # Teil von ur_rtde
except ImportError:
    DashboardClient = None

from P06 import RTDEControlInterface, RTDEReceiveInterface, RobotiqGripper, Worker, apply_dark, fmt_pose

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "p08_config.json")
P07_CONFIG_PATH = os.path.join(SCRIPT_DIR, "p07_config.json")   # Kalibrierung etc. beim ersten Start übernehmen
GRIPPER_PORT = 63352

ORIENT_METHODS = [("grabcut", "GrabCut – robust, auch auf farbigem Untergrund (Standard)"),
                  ("otsu", "Sättigung (Otsu) – bunte Steine auf grauem/weißem Tisch"),
                  ("canny", "Kanten (Canny) + gefüllte Kontur"),
                  ("box", "Nur YOLO-Box (achsparallel, längere Seite)")]

DEFAULT_CFG = {
    "robot_ip": "192.168.0.100",
    "camera_index": 0,
    "calib": {"points": [], "H": None, "plane": None, "method": "", "img_size": None},
    "tip_len_mm": 30.0,      # Überstand der Kalibrierspitze über die geschlossenen Finger
    "tip_offset": [0.0, 0.0, 0.0],   # Spitze im TCP-KS [m] (Pivot-Kalibrierung)
    "pivot": {"poses": [], "rms_mm": None, "manual": False},
    "force_limit_N": 15.0,   # Kraftgrenze beim Absenken (0 = aus)
    "grip_h_mm": 4.0,        # Höhe der Fingerspitzen über dem Tisch beim Greifen
    "approach_mm": 30.0,     # ab hier langsam auf Greifhöhe fahren
    "travel_mm": 120.0,      # Transferhöhe der Fingerspitzen über dem Tisch
    "corr_mm": [0.0, 0.0],   # Feinkorrektur X/Y für Greifpunkte (z. B. Parallaxe)
    # pre_pos 205 = Kalibrierspitze passt gerade so zwischen die Finger (am Roboter getestet)
    # open_pos = max. Öffnung beim Greifen (Robotiq 0 = ganz offen .. 255 = zu), in der GUI als %
    "gripper": {"pre_pos": 205, "open_pos": 0, "speed": 150, "force": 60, "angle_off_deg": 0.0},
    "speed": {"v": 0.15, "a": 0.8, "v_slow": 0.03, "vj": 0.8, "aj": 1.0},
    # YOLO-Modell: Pfad absolut oder relativ zu diesem Skript ("" = neuestes best.pt unter ../DNN suchen)
    "model": {"path": "", "conf": 0.5, "iou": 0.5, "imgsz": 640},
    # Orientierung innerhalb der YOLO-Box (nur bei reinen Detect-Modellen)
    "orient": {"method": "grabcut", "pad_px": 8, "min_fill": 0.25, "canny_lo": 40, "canny_hi": 120},
    "roi": [],               # Arbeitsbereich als Polygon im Bild (Pixel)
    "home_q": None,
    "home_pose": None,
    # je Klassenname des Modells: {"enabled": bool, "drop": Pose oder None}
    "classes": {},
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
    """Lädt p08_config.json. Gibt es die noch nicht, werden Roboter-/Kamera-Kalibrierung,
    Posen und Greifer-Einstellungen aus p07_config.json übernommen (gleicher Aufbau)."""
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return _merge(DEFAULT_CFG, json.load(f))
    except FileNotFoundError:
        pass
    cfg = json.loads(json.dumps(DEFAULT_CFG))
    try:
        with open(P07_CONFIG_PATH, encoding="utf-8") as f:
            old = json.load(f)
        for k in ("robot_ip", "camera_index", "calib", "tip_len_mm", "tip_offset", "pivot", "force_limit_N",
                  "grip_h_mm", "approach_mm", "travel_mm", "corr_mm", "gripper", "speed", "roi",
                  "home_q", "home_pose"):
            if k in old:
                cfg[k] = _merge(DEFAULT_CFG[k], old[k])
        cfg["_imported_p07"] = True
    except (FileNotFoundError, ValueError):
        pass
    return cfg


def resolve_path(p):
    return p if not p or os.path.isabs(p) else os.path.normpath(os.path.join(SCRIPT_DIR, p))


def find_default_model():
    """Neuestes trainiertes best.pt unter ../DNN (vom yolo_labeler erzeugt)."""
    hits = glob.glob(os.path.join(SCRIPT_DIR, "..", "DNN", "**", "weights", "best.pt"), recursive=True)
    if not hits:
        return ""
    best = max(hits, key=os.path.getmtime)
    return os.path.relpath(best, SCRIPT_DIR)


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


# ====================================================================== Bildverarbeitung / DNN
DET_COLOR = (0, 220, 0)          # BGR, aktive Klassen
DET_COLOR_OFF = (150, 150, 150)  # deaktivierte Klassen (werden nicht sortiert)


def _long_axis(box):
    """Winkel der längeren Rechteckseite im Bild [rad] für 4 Eckpunkte."""
    box = np.asarray(box, float)
    e1, e2 = box[1] - box[0], box[2] - box[1]
    le = e1 if np.linalg.norm(e1) >= np.linalg.norm(e2) else e2
    return math.atan2(float(le[1]), float(le[0]))


def _axis_box(x1, y1, x2, y2):
    return np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def orient_in_box(frame, xyxy, ocfg):
    """Orientierung eines Steins innerhalb einer achsparallelen YOLO-Box.
    Gibt (Zentrum (u, v), 4 Eckpunkte, Winkel Längsachse, Kontur oder None, ok) zurück.
    ok = False -> Rückfall auf die YOLO-Box (achsparallel)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    fallback = (((x1 + x2) / 2, (y1 + y2) / 2), _axis_box(x1, y1, x2, y2),
                0.0 if (x2 - x1) >= (y2 - y1) else math.pi / 2, None, False)
    method = ocfg.get("method", "grabcut")
    if method == "box":
        return fallback
    pad = int(ocfg.get("pad_px", 8))
    rx1, ry1 = max(int(x1) - pad, 0), max(int(y1) - pad, 0)
    rx2, ry2 = min(int(math.ceil(x2)) + pad, w), min(int(math.ceil(y2)) + pad, h)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.shape[0] < 8 or roi.shape[1] < 8:
        return fallback
    k = np.ones((3, 3), np.uint8)

    if method == "grabcut":
        mask = np.zeros(roi.shape[:2], np.uint8)
        rect = (int(x1) - rx1, int(y1) - ry1, max(int(x2 - x1), 2), max(int(y2 - y1), 2))
        try:
            cv2.grabCut(roi, mask, rect, np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64),
                        3, cv2.GC_INIT_WITH_RECT)
        except cv2.error:
            return fallback
        m = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    elif method == "canny":
        gray = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        e = cv2.Canny(gray, int(ocfg.get("canny_lo", 40)), int(ocfg.get("canny_hi", 120)))
        e = cv2.dilate(e, k, iterations=2)              # Kantenlücken schließen
        cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        m = np.zeros_like(e)
        cv2.drawContours(m, cnts, -1, 255, -1)          # Umriss füllen (Noppen-Kanten verschwinden)
        m = cv2.erode(m, k, iterations=2)
    else:  # otsu: Sättigung trennt bunte Steine vom unbunten Tisch
        s = cv2.cvtColor(cv2.GaussianBlur(roi, (5, 5), 0), cv2.COLOR_BGR2HSV)[:, :, 1]
        _, m = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=2)

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return fallback
    # Kontur, die den größten Teil der YOLO-Box ausmacht (Nachbarsteine am Rand ignorieren)
    cx, cy = (x1 + x2) / 2 - rx1, (y1 + y2) / 2 - ry1
    def score(c):
        inside = cv2.pointPolygonTest(c, (float(cx), float(cy)), False) >= 0
        return (inside, cv2.contourArea(c))
    c = max(cnts, key=score)
    box_area = max((x2 - x1) * (y2 - y1), 1.0)
    if cv2.contourArea(c) < ocfg.get("min_fill", 0.25) * box_area:
        return fallback
    c = c + np.array([[rx1, ry1]], dtype=c.dtype)
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    return rect[0], box, _long_axis(box), c.reshape(-1, 2), True


class Detector:
    """YOLO-Modell (Ultralytics). Thread-sicher: Live-Vorschau und Roboter-Ablauf teilen sich das Modell."""

    def __init__(self):
        self.model = None
        self.path = ""
        self.task = ""
        self.names = {}
        self.lock = threading.Lock()
        self.last_ms = 0.0

    @property
    def ready(self):
        return self.model is not None

    def load(self, path):
        from ultralytics import YOLO       # erst hier importieren (dauert ein paar Sekunden)
        full = resolve_path(path)
        if not os.path.isfile(full):
            raise FileNotFoundError(f"Modell nicht gefunden: {full}")
        model = YOLO(full)
        # Aufwärmen: die erste Vorhersage lädt intern weitere Module und dauert mehrere Sekunden
        model.predict(np.zeros((480, 640, 3), np.uint8), verbose=False)
        with self.lock:
            self.model, self.path = model, path
            self.task = model.task
            self.names = dict(model.names)
        return self

    def detect(self, frame, cfg, calib=None):
        """Liste von Erkennungen (alle Klassen; 'enabled' sagt, ob die Klasse sortiert wird)."""
        if self.model is None:
            return []
        mc = cfg["model"]
        t0 = time.perf_counter()
        with self.lock:
            res = self.model.predict(frame, conf=mc["conf"], iou=mc["iou"], imgsz=int(mc["imgsz"]),
                                     verbose=False)[0]
        self.last_ms = (time.perf_counter() - t0) * 1000
        roi = np.float32(cfg["roi"]) if len(cfg.get("roi") or []) >= 3 else None
        out = []
        items = []
        if res.obb is not None and len(res.obb):             # Modell mit gedrehten Boxen
            for xywhr, poly, c, p in zip(res.obb.xywhr.cpu().numpy(), res.obb.xyxyxyxy.cpu().numpy(),
                                         res.obb.cls.cpu().numpy(), res.obb.conf.cpu().numpy()):
                xs, ys = poly[:, 0], poly[:, 1]
                items.append((int(c), float(p), (xs.min(), ys.min(), xs.max(), ys.max()),
                              ((float(xywhr[0]), float(xywhr[1])), poly, _long_axis(poly), None, True)))
        elif res.boxes is not None and len(res.boxes):
            masks = res.masks.xy if res.masks is not None else None
            for i, (xyxy, c, p) in enumerate(zip(res.boxes.xyxy.cpu().numpy(), res.boxes.cls.cpu().numpy(),
                                                 res.boxes.conf.cpu().numpy())):
                x1, y1, x2, y2 = (float(v) for v in xyxy)
                if masks is not None and len(masks[i]) >= 3:  # Segmentierungsmodell: Maske direkt nutzen
                    cnt = np.float32(masks[i])
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    ori = (rect[0], box, _long_axis(box), cnt, True)
                else:
                    ori = orient_in_box(frame, (x1, y1, x2, y2), cfg["orient"])
                items.append((int(c), float(p), (x1, y1, x2, y2), ori))

        for c, p, bb, (center, box, a_img, contour, ok) in items:
            u, v = float(center[0]), float(center[1])
            if roi is not None and cv2.pointPolygonTest(roi, (u, v), False) < 0:
                continue
            name = self.names.get(c, str(c))
            ccfg = cfg["classes"].get(name, {})
            det = {"cls": c, "name": name, "conf": p, "px": (u, v), "bbox": [float(x) for x in bb],
                   "box": np.asarray(box, float).tolist(), "ang_img": a_img, "orient_ok": ok,
                   "contour": None if contour is None else np.asarray(contour, float).tolist(),
                   "area": float((bb[2] - bb[0]) * (bb[3] - bb[1])),
                   "enabled": bool(ccfg.get("enabled", True)), "xy": None, "theta": None}
            if calib is not None:
                x, y = calib.to_base(u, v)
                x2, y2 = calib.to_base(u + 10 * math.cos(a_img), v + 10 * math.sin(a_img))
                det["xy"] = (x, y)
                # Finger schließen quer zur Längsachse
                det["theta"] = math.atan2(y2 - y, x2 - x) + math.pi / 2
            out.append(det)
        return out


def draw_dets(img, dets):
    for d in dets:
        col = DET_COLOR if d["enabled"] else DET_COLOR_OFF
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 1)
        if d.get("contour"):
            cv2.polylines(img, [np.int32(np.round(d["contour"]))], True, (255, 160, 0), 1)
        box = np.int32(np.round(d["box"]))
        cv2.polylines(img, [box], True, (0, 0, 0), 4)
        cv2.polylines(img, [box], True, col if d["orient_ok"] else (0, 140, 255), 2)
        u, v = d["px"]
        a = d["ang_img"] + math.pi / 2                         # Schließrichtung der Finger
        dx, dy = 18 * math.cos(a), 18 * math.sin(a)
        cv2.line(img, (int(u - dx), int(v - dy)), (int(u + dx), int(v + dy)), (255, 255, 255), 2)
        cv2.circle(img, (int(u), int(v)), 3, (255, 255, 255), -1)
        txt = f"{d['name']} {d['conf']:.2f}"
        if d["xy"] is not None:
            txt += f" {d['xy'][0]*1000:.0f}/{d['xy'][1]*1000:.0f}"
        if not d["orient_ok"]:
            txt += " (Box)"
        cv2.putText(img, txt, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(img, txt, (x1, max(y1 - 6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)


def frame_ok(f):
    """False für (fast) schwarze oder einfarbige Bilder - z. B. wenn die Kamera kurz aussetzt."""
    if f is None or f.size == 0:
        return False
    small = f[::8, ::8]
    return float(small.mean()) > 12 and float(small.std()) > 1.5


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
        zurückfahren und mit CollisionStop abbrechen. Überwachung: Kraftgrenze [N] auf die genullte,
        gefilterte TCP-Kraft, nur bei konstanter Sollgeschwindigkeit (beim Beschleunigen/Bremsen
        ausgesetzt -> keine Fehlauslösung durch Trägheit)"""
        sp, c, r = self.cfg["speed"], self.c, self.r
        target = [float(x) for x in target]
        if kind != "J":
            self._require_reachable([target], "Ziel")
        guard = retreat is not None
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
                    self._collision(stop, retreat,
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

    def _collision(self, stop, retreat, why):
        try:
            stop(5.0)
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

    def _fresh_frame(self, settle=0.3):
        """Nächstes gültiges Kamerabild, das mind. `settle` s nach dem Aufruf aufgenommen wurde
        (schwarze/kaputte Bilder filtert schon _cam_tick heraus)."""
        t0 = time.time()
        while True:
            self._check()
            with self.win.frame_lock:
                f, ts = self.win.frame, self.win.frame_ts
            if f is not None and ts > t0 + settle and frame_ok(f):
                return f.copy()
            if time.time() - t0 > 5:
                raise RuntimeError("Keine gültigen Kamerabilder (schwarz/abgerissen?)")
            time.sleep(0.02)

    def _detect(self):
        f = self._fresh_frame()
        dets = [d for d in self.win.detector.detect(f, self.cfg, self.calib) if d["xy"] is not None]
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

    def job_drop_test(self, name):
        dp = self.cfg["classes"][name]["drop"]
        above = [dp[0], dp[1], dp[2] + 0.03, *dp[3:]]
        self._require_reachable([above, dp], f"Ablage {name}")
        self._retreat()
        self._move("JIK", above)
        self._move("L", dp, force_limit=self._flim(), retreat=above)
        return f"Ablage {name} erreicht"

    def job_hover(self):
        self._grip(self.cfg["gripper"]["open_pos"])
        self._home()
        dets = [d for d in self._detect() if d["enabled"]]
        if not dets:
            return "Keine Steine erkannt"
        d = self._nearest(dets)
        above, _, grasp = grasp_poses(self.cfg, self.calib, d, self._ref_yaw())
        hover = [grasp[0], grasp[1], grasp[2] + 0.015, *grasp[3:]]
        self._require_reachable([above, hover], f"Stein {d['name']}")
        self._move("JIK", above)
        self._move("L", hover, force_limit=self._flim(), retreat=above)
        return (f"Fingerspitzen 15 mm über Greifhöhe am Stein {d['name']} - Lage prüfen, "
                f"ggf. Korrektur X/Y bzw. Winkel-Offset anpassen (max. Kraft beim Absenken {self.peak_force:.1f} N)")

    def job_sort(self):
        picked, fails = 0, []
        while True:
            self._home()

            def usable(dets):
                return [d for d in dets if d["enabled"]
                        and self.cfg["classes"].get(d["name"], {}).get("drop")
                        and not any(math.dist(d["xy"], f) < 0.015 for f in fails)]
            todo = usable(self._detect())
            # "leer" erst glauben, wenn auch 2 weitere frische Bilder nichts zeigen
            # (Kamera liefert gelegentlich gestörte/dunkle Einzelbilder)
            for _ in range(2):
                if todo:
                    break
                self.status.emit("Keine Steine gefunden - prüfe mit neuem Bild …")
                todo = usable(self._detect())
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
        dp = self.cfg["classes"][d["name"]]["drop"]
        drop_above = [dp[0], dp[1], max(above[2], dp[2]), *dp[3:]]
        # alles vorab prüfen, bevor sich der Roboter bewegt
        self._require_reachable([above, approach, grasp], f"Stein {d['name']}")
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
        self.setWindowTitle("P08 Lego-Sortierer mit DNN-Erkennung (UR3e + Robotiq + Kamera + YOLO)")
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
        self.roi_edit = False
        self.detector = Detector()
        self.live_dets = []                        # letzte Erkennung für die Anzeige
        self.dets_lock = threading.Lock()
        self.view_mode = 0                         # 0 = Kamera, 1 = Erkennung (vom Live-Thread gelesen)
        self._live_stop = threading.Event()
        self._live_err = None
        self.freedrive = None                      # None / "teach" / "constrained"
        self.clamp_state = 0
        self.task = None
        self._threads = set()
        self._size_warned = False
        self._bad_frames = 0                       # verworfene schwarze Kamerabilder

        self._build_ui()
        self._refresh_points()
        self._refresh_classes()
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

        # Erkennung läuft in eigenem Thread (YOLO auf der CPU braucht 50-300 ms pro Bild)
        threading.Thread(target=self._live_loop, daemon=True).start()
        if self.cfg.pop("_imported_p07", False):
            self._log("Kalibrierung, Posen und Greifer-Einstellungen aus p07_config.json übernommen")
        if not self.cfg["model"]["path"]:
            self.cfg["model"]["path"] = find_default_model()
            self.ed_model.setText(self.cfg["model"]["path"])
        if self.cfg["model"]["path"]:
            self.load_model()

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
        self.cb_view.addItems(["Kamera", "Erkennung (DNN)"])
        self.cb_view.currentIndexChanged.connect(self._view_changed)
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
        self.tabs.addTab(self._scroll(self._tab_model()), "2 Modell")
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
    def _tab_model(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(self._hint(
            "YOLO-Modell laden (z. B. mit src/yolo_labeler.py trainiert: <projekt>/runs/<name>/weights/best.pt). "
            "Die Klassen des Modells erscheinen unten; Haken = wird sortiert (z. B. 'ignore' abhaken). "
            "Anzeige 'Erkennung (DNN)' zeigt live: grüne Box = Greifrechteck, weiße Linie = Schließrichtung "
            "der Finger, orange Kontur = gefundener Steinumriss. Orange Rechteck = Orientierung nicht gefunden "
            "(achsparallele YOLO-Box als Rückfall)."))

        g = QGroupBox("Modell")
        gl = QGridLayout(g)
        self.ed_model = QLineEdit(self.cfg["model"]["path"])
        b_browse = QPushButton("…")
        b_browse.setFixedWidth(30)
        b_browse.clicked.connect(self.browse_model)
        self.btn_load_model = QPushButton("Laden")
        self.btn_load_model.clicked.connect(self.load_model)
        gl.addWidget(QLabel("Datei:"), 0, 0)
        gl.addWidget(self.ed_model, 0, 1)
        gl.addWidget(b_browse, 0, 2)
        gl.addWidget(self.btn_load_model, 0, 3)
        self.lbl_model = QLabel("Kein Modell geladen")
        self.lbl_model.setWordWrap(True)
        gl.addWidget(self.lbl_model, 1, 0, 1, 4)
        v.addWidget(g)

        v.addWidget(self._form_box("Erkennung", [
            ("Min. Konfidenz:", self._num(("model", "conf"), 0.05, 0.99, 0.05, 2)),
            ("NMS-IoU:", self._num(("model", "iou"), 0.1, 0.95, 0.05, 2)),
            ("Bildgröße (imgsz, wie beim Training):", self._num(("model", "imgsz"), 160, 1920, 32, 0)),
        ]))

        g = QGroupBox("Klassen (Haken = wird sortiert)")
        gv = QVBoxLayout(g)
        self.list_classes = QListWidget()
        self.list_classes.setMaximumHeight(150)
        self.list_classes.itemChanged.connect(self._class_item_changed)
        gv.addWidget(self.list_classes)
        v.addWidget(g)

        g = QGroupBox("Greifwinkel bestimmen (nur bei Detect-Modellen; OBB/Segmentierung liefern ihn direkt)")
        f = QFormLayout(g)
        self.cb_orient = QComboBox()
        for key, text in ORIENT_METHODS:
            self.cb_orient.addItem(text, key)
        self.cb_orient.setCurrentIndex(max(self.cb_orient.findData(self.cfg["orient"]["method"]), 0))
        self.cb_orient.currentIndexChanged.connect(
            lambda _: self._cfg_set(("orient", "method"), self.cb_orient.currentData()))
        f.addRow("Verfahren:", self.cb_orient)
        f.addRow("Rand um YOLO-Box:", self._num(("orient", "pad_px"), 0, 50, 1, 0, " px"))
        f.addRow("Min. Füllgrad der Box:", self._num(("orient", "min_fill"), 0.05, 1.0, 0.05, 2))
        f.addRow("Canny unten:", self._num(("orient", "canny_lo"), 0, 500, 5, 0))
        f.addRow("Canny oben:", self._num(("orient", "canny_hi"), 0, 500, 5, 0))
        f.addRow(self._hint("Sättigung (Otsu): schnell und robust bei bunten Steinen auf unbuntem Untergrund. "
                            "GrabCut: auch bei grauen/weißen Steinen oder buntem Tisch, ca. 10-30 ms pro Stein. "
                            "Canny: Kanten -> gefüllter Umriss; empfindlich auf Schatten/Texturen. "
                            "Min. Füllgrad: Umriss muss mind. diesen Anteil der Box bedecken, sonst Rückfall auf die Box."))
        v.addWidget(g)

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
            "über die Kiste der Klasse führen und speichern. Der Stein wird an dieser Pose losgelassen - "
            "hoch genug über dem Kistenrand wählen."))
        g = QGroupBox("Ablageposition je Klasse")
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
        v.addWidget(self._hint("Kraftgrenze beim Absenken: wirkt nur bei konstanter Geschwindigkeit (nicht beim Beschleunigen). Die aktuelle "
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
        self.cb_view.setCurrentIndex({0: 0, 1: 1, 2: 0, 3: 1}.get(i, 0))
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
            if ok and frame_ok(f):           # schwarze/kaputte Bilder gar nicht erst übernehmen
                with self.frame_lock:
                    self.frame, self.frame_ts = f, time.time()
                self._check_img_size(f)
            elif ok:
                self._bad_frames += 1
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

    def _view_changed(self, i):
        self.view_mode = i

    def _live_loop(self):
        """Hintergrund-Thread: erkennt auf dem jeweils neuesten Kamerabild, solange die Anzeige
        'Erkennung' aktiv ist. Während eines Roboter-Ablaufs erkennt der Ablauf selbst."""
        last_ts = 0.0
        while not self._live_stop.is_set():
            if self.view_mode == 1 and self.detector.ready and self.task is None and self.frozen is None:
                with self.frame_lock:
                    f, ts = self.frame, self.frame_ts
                if f is not None and ts != last_ts:
                    last_ts = ts
                    try:
                        dets = self.detector.detect(f, self.cfg, self.calib)
                        self._live_err = None
                    except Exception as e:
                        dets, self._live_err = [], str(e)
                    with self.dets_lock:
                        self.live_dets = dets
                    continue
            time.sleep(0.03)

    def _render(self, src):
        if self.view_mode != 1:
            return src
        out = src.copy()
        with self.dets_lock:
            dets = list(self.live_dets)
        draw_dets(out, dets)
        if not self.detector.ready:
            self.lbl_count.setText("Kein Modell geladen")
        elif self._live_err:
            self.lbl_count.setText(f"Erkennungsfehler: {self._live_err}")
        else:
            per = {}
            for d in dets:
                per[d["name"]] = per.get(d["name"], 0) + 1
            txt = ", ".join(f"{n}: {k}" for n, k in sorted(per.items())) or "nichts"
            no_ori = sum(not d["orient_ok"] for d in dets)
            self.lbl_count.setText(f"Erkannt: {txt}   |   Inferenz {self.detector.last_ms:.0f} ms"
                                   + (f"   |   {no_ori}x ohne Orientierung" if no_ori else ""))
        return out

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

    # ================================================================== Modell / Klassen
    def browse_model(self):
        start = os.path.dirname(resolve_path(self.ed_model.text().strip())) or os.path.join(SCRIPT_DIR, "..", "DNN")
        f, _ = QFileDialog.getOpenFileName(self, "YOLO-Modell wählen", start, "PyTorch-Modell (*.pt)")
        if f:
            try:      # relativ zum Skript speichern, wenn möglich (Repo bleibt verschiebbar)
                rel = os.path.relpath(f, SCRIPT_DIR)
                f = rel if not rel.startswith("..\\..") and not rel.startswith("../..") else f
            except ValueError:        # anderes Laufwerk
                pass
            self.ed_model.setText(f)
            self.load_model()

    def load_model(self):
        path = self.ed_model.text().strip()
        if not path:
            self.lbl_model.setText("Kein Modell angegeben")
            return
        self.btn_load_model.setEnabled(False)
        self.lbl_model.setText(f"Lade {path} … (Ultralytics-Import dauert beim ersten Mal einige Sekunden)")

        def done(res, err):
            self.btn_load_model.setEnabled(True)
            if err is not None:
                self.lbl_model.setText(f"⚠ Laden fehlgeschlagen: {err}")
                self._log(f"Modell laden fehlgeschlagen: {err}")
                return
            det = res
            self.cfg["model"]["path"] = path
            for name in det.names.values():          # neue Klassen übernehmen, bestehende behalten
                self.cfg["classes"].setdefault(name, {"enabled": name.lower() != "ignore", "drop": None})
            kind = {"detect": "Detect (achsparallele Boxen -> Winkel per Bildverarbeitung)",
                    "obb": "OBB (gedrehte Boxen -> Winkel direkt aus dem Modell)",
                    "segment": "Segmentierung (Maske -> Winkel aus der Maske)"}.get(det.task, det.task)
            self.lbl_model.setText(f"✔ {resolve_path(path)}\nTyp: {kind}\n{len(det.names)} Klassen: "
                                   + ", ".join(det.names.values()))
            self.cb_orient.setEnabled(det.task == "detect")
            self._refresh_classes()
            self._log(f"Modell geladen: {os.path.basename(path)} ({det.task}, {len(det.names)} Klassen)")

        self._run_async(lambda: self.detector.load(path), done)

    def class_names(self):
        """Klassen in Modell-Reihenfolge (ohne geladenes Modell: aus der Konfiguration)."""
        if self.detector.ready:
            return list(self.detector.names.values())
        return list(self.cfg["classes"].keys())

    def _refresh_classes(self):
        self.list_classes.blockSignals(True)
        self.list_classes.clear()
        for name in self.class_names():
            c = self.cfg["classes"].setdefault(name, {"enabled": name.lower() != "ignore", "drop": None})
            it = QListWidgetItem(name)
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if c["enabled"] else Qt.CheckState.Unchecked)
            self.list_classes.addItem(it)
        self.list_classes.blockSignals(False)
        self._refresh_drops()

    def _class_item_changed(self, it):
        name = it.data(Qt.ItemDataRole.UserRole)
        self.cfg["classes"][name]["enabled"] = it.checkState() == Qt.CheckState.Checked

    # ================================================================== Ablagen
    def _refresh_drops(self):
        cur = self.list_drops.currentRow()
        self.list_drops.clear()
        for name in self.class_names():
            c = self.cfg["classes"].get(name, {})
            d = c.get("drop")
            txt = (f"{name}:  X {d[0]*1000:.0f}  Y {d[1]*1000:.0f}  Z {d[2]*1000:.0f} mm" if d
                   else f"{name}:  — keine Ablage —")
            if not c.get("enabled", True):
                txt += "   (wird nicht sortiert)"
            it = QListWidgetItem(txt)
            it.setData(Qt.ItemDataRole.UserRole, name)
            self.list_drops.addItem(it)
        if self.list_drops.count():
            self.list_drops.setCurrentRow(min(max(cur, 0), self.list_drops.count() - 1))

    def _drop_name(self):
        it = self.list_drops.currentItem()
        return it.data(Qt.ItemDataRole.UserRole) if it is not None else None

    def drop_save(self):
        name = self._drop_name()
        if name is None or self.rtde_r is None:
            return
        self.cfg["classes"][name]["drop"] = self._pose()
        self._refresh_drops()
        self.save()
        self._log(f"Ablage für {name} gespeichert")

    def drop_goto(self):
        name = self._drop_name()
        if name is not None and self.cfg["classes"].get(name, {}).get("drop"):
            self._start_task("drop_test", name=name)

    def drop_clear(self):
        name = self._drop_name()
        if name is not None:
            self.cfg["classes"][name]["drop"] = None
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
        with self.dets_lock:            # auch im Bild anzeigen (z. B. Erkennung des Sortier-Ablaufs)
            self.live_dets = list(dets)
        self.list_dets.clear()
        for d in dets:
            t = f"{d['name']} ({d['conf']:.2f}): Pixel ({d['px'][0]:.0f}, {d['px'][1]:.0f})"
            if not d["enabled"]:
                t += " [nicht sortiert]"
            if not d["orient_ok"]:
                t += " [Winkel nur aus Box]"
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
        if not self.detector.ready:
            self.status.setText("Kein Modell geladen (Reiter 2)")
            return
        with self.frame_lock:
            frame = self.frame.copy()
        try:
            dets = self.detector.detect(frame, self.cfg, self.calib)
        except Exception as e:
            self._log(f"Erkennung fehlgeschlagen: {e}")
            return
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
        self.cb_view.setCurrentIndex(1)
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
        if not self.detector.ready:
            problems.append("Kein YOLO-Modell geladen (Schritt 2)")
        enabled = [n for n in self.class_names() if self.cfg["classes"].get(n, {}).get("enabled")]
        if not any(self.cfg["classes"][n].get("drop") for n in enabled):
            problems.append("Keine aktivierte Klasse mit Ablageposition (Schritte 2/3)")
        if problems:
            QMessageBox.warning(self, "Sortieren nicht möglich", "\n".join(problems))
            return
        missing = [n for n in enabled if not self.cfg["classes"][n].get("drop")]
        txt = "Kalibrierspitze entfernt und Arbeitsbereich frei?"
        if missing:
            txt += f"\n\nOhne Ablage (werden ignoriert): {', '.join(missing)}"
        if QMessageBox.question(self, "Sortieren starten", txt) != QMessageBox.StandardButton.Yes:
            return
        self.save()
        self.cb_view.setCurrentIndex(1)
        self._start_task("sort")

    # ================================================================== Ende
    def closeEvent(self, ev):
        if self.task is not None:
            self.task.stop_ev.set()
            self.task.wait(5000)
        self._live_stop.set()
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
