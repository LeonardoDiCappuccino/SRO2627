# YOLO-Labeling-Tool (PyQt6): Bilder mit der Kamera aufnehmen und direkt labeln.
#
# pip install opencv-python PyQt6
#
# Ablauf:
#   1. Beim Start neues Projekt anlegen (Speicherort + Klassen) oder bestehendes öffnen.
#   2. Leertaste: Foto aufnehmen
#   3. Mit der Maus Rechtecke ziehen -> Klasse im Dialog wählen (Zifferntasten 1-9, Enter, Doppelklick)
#   4. Enter: Bild + Labels speichern, zurück zum Livebild
#   5. Menü Training -> YOLO trainieren… (Strg+T): Basismodell, Epochen usw. wählen
#
# Kamera jederzeit über die Auswahl oben rechts wechselbar.
# Mausrad = Zoom, mittlere Maustaste = verschieben, Entf = Box löschen (Bild bleibt).
#
# Ordnerstruktur (YOLO / Ultralytics-kompatibel):
#   <projekt>/images/*.jpg
#   <projekt>/labels/*.txt     (class cx cy w h, normiert 0..1)
#   <projekt>/classes.txt
#   <projekt>/data.yaml        -> model.train(data="<projekt>/data.yaml")
#   <projekt>/project.json     (Einstellungen des Tools)

import glob
import json
import os
import random
import re
import sys
import tempfile
import time

import cv2
from PyQt6.QtCore import (QPointF, QProcess, QProcessEnvironment, QRectF, QSettings, Qt, QTimer,
                          QUrl, pyqtSignal)
from PyQt6.QtGui import (QAction, QColor, QCursor, QDesktopServices, QFont, QImage, QKeySequence,
                         QPainter, QPen, QShortcut, QTextCursor)
from PyQt6.QtWidgets import (QApplication, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
                             QFrame, QGroupBox, QHBoxLayout, QInputDialog, QComboBox, QLabel, QLineEdit,
                             QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
                             QPlainTextEdit, QPushButton, QSpinBox, QVBoxLayout, QWidget)

PROJECT_FILE = "project.json"
BOX_COLOR = QColor("#00e000")        # alle Boxen
SELECTED_COLOR = QColor("#ffcc00")   # in der Liste ausgewählte Box
MAX_ZOOM = 20.0


# --------------------------------------------------------------------------- #
# Projekt laden / speichern
# --------------------------------------------------------------------------- #
class Project:
    def __init__(self, root_dir, classes, camera_index=0, width=640, height=480):
        self.root = os.path.abspath(root_dir)
        self.classes = list(classes)
        self.camera_index = int(camera_index)
        self.width = int(width)
        self.height = int(height)

    @property
    def images_dir(self):
        return os.path.join(self.root, "images")

    @property
    def labels_dir(self):
        return os.path.join(self.root, "labels")

    def save(self):
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
        with open(os.path.join(self.root, PROJECT_FILE), "w", encoding="utf-8") as f:
            json.dump({"classes": self.classes, "camera_index": self.camera_index,
                       "width": self.width, "height": self.height}, f, indent=2, ensure_ascii=False)
        with open(os.path.join(self.root, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.classes) + "\n")
        # data.yaml für Ultralytics (train und val zeigen hier auf denselben Ordner).
        # Ohne "path:" löst Ultralytics die Ordner relativ zur data.yaml auf -> Projekt ist verschiebbar.
        with open(os.path.join(self.root, "data.yaml"), "w", encoding="utf-8") as f:
            f.write("train: images\nval: images\n\nnames:\n")
            for i, name in enumerate(self.classes):
                f.write(f"  {i}: {name}\n")

    @staticmethod
    def load(root_dir):
        cfg_path = os.path.join(root_dir, PROJECT_FILE)
        if os.path.isfile(cfg_path):
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            return Project(root_dir, cfg["classes"], cfg.get("camera_index", 0),
                           cfg.get("width", 640), cfg.get("height", 480))
        classes_path = os.path.join(root_dir, "classes.txt")
        if os.path.isfile(classes_path):  # Fremdprojekt mit classes.txt
            with open(classes_path, encoding="utf-8") as f:
                classes = [l.strip() for l in f if l.strip()]
            return Project(root_dir, classes)
        raise FileNotFoundError(f"Kein {PROJECT_FILE} oder classes.txt in\n{root_dir}")

    def image_list(self):
        if not os.path.isdir(self.images_dir):
            return []
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return sorted(f for f in os.listdir(self.images_dir) if f.lower().endswith(exts))

    def label_path(self, image_name):
        return os.path.join(self.labels_dir, os.path.splitext(image_name)[0] + ".txt")


def remember_project(path):
    s = QSettings("SRO", "YoloLabeler")
    recent = [p for p in (s.value("recent", [], type=list) or []) if p != path]
    s.setValue("recent", [path] + recent[:9])


def recent_projects():
    s = QSettings("SRO", "YoloLabeler")
    return [p for p in (s.value("recent", [], type=list) or [])
            if os.path.isfile(os.path.join(p, PROJECT_FILE)) or os.path.isfile(os.path.join(p, "classes.txt"))]


# --------------------------------------------------------------------------- #
# Startdialog
# --------------------------------------------------------------------------- #
class StartDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Labeler – Projekt")
        self.result_project = None
        layout = QVBoxLayout(self)

        # --- Neues Projekt ---
        new_box = QGroupBox("Neues Projekt")
        form = QFormLayout(new_box)
        self.dir_edit = QLineEdit(os.path.join(os.getcwd(), "dataset"))
        browse = QPushButton("…")
        browse.setFixedWidth(30)
        browse.clicked.connect(self.pick_dir)
        row = QHBoxLayout()
        row.addWidget(self.dir_edit)
        row.addWidget(browse)
        form.addRow("Speicherort:", row)

        self.classes_edit = QPlainTextEdit("duplo_4_green\n")
        self.classes_edit.setPlaceholderText("eine Klasse pro Zeile")
        self.classes_edit.setFixedHeight(130)
        form.addRow("Klassen\n(eine pro Zeile):", self.classes_edit)

        self.cam_spin = QSpinBox()
        self.cam_spin.setRange(0, 10)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(160, 4096)
        self.w_spin.setValue(640)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(120, 4096)
        self.h_spin.setValue(480)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Index"))
        cam_row.addWidget(self.cam_spin)
        cam_row.addSpacing(12)
        cam_row.addWidget(QLabel("Auflösung"))
        cam_row.addWidget(self.w_spin)
        cam_row.addWidget(QLabel("x"))
        cam_row.addWidget(self.h_spin)
        cam_row.addStretch()
        form.addRow("Kamera:", cam_row)

        create_btn = QPushButton("Projekt anlegen")
        create_btn.setDefault(True)
        create_btn.clicked.connect(self.create)
        form.addRow("", create_btn)
        layout.addWidget(new_box)

        # --- Bestehendes Projekt ---
        open_box = QGroupBox("Bestehendes Projekt")
        ol = QVBoxLayout(open_box)
        self.recent_list = QListWidget()
        self.recent_list.setMaximumHeight(110)
        for p in recent_projects():
            self.recent_list.addItem(p)
        self.recent_list.itemDoubleClicked.connect(lambda it: self._open_path(it.text()))
        if self.recent_list.count():
            ol.addWidget(QLabel("Zuletzt verwendet (Doppelklick):"))
            ol.addWidget(self.recent_list)
        else:
            self.recent_list.hide()
        open_btn = QPushButton("Projektordner öffnen…")
        open_btn.clicked.connect(self.open)
        ol.addWidget(open_btn)
        layout.addWidget(open_box)

        self.resize(560, 0)

    def pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Speicherort wählen", self.dir_edit.text())
        if d:
            self.dir_edit.setText(d)

    def create(self):
        root_dir = self.dir_edit.text().strip()
        classes = [c.strip() for c in self.classes_edit.toPlainText().splitlines() if c.strip()]
        if not root_dir:
            return QMessageBox.critical(self, "Fehler", "Bitte Speicherort angeben.")
        if not classes:
            return QMessageBox.critical(self, "Fehler", "Bitte mindestens eine Klasse angeben.")
        if len(set(classes)) != len(classes):
            return QMessageBox.critical(self, "Fehler", "Klassennamen müssen eindeutig sein.")
        if os.path.isfile(os.path.join(root_dir, PROJECT_FILE)):
            ans = QMessageBox.question(self, "Projekt existiert",
                                       "In diesem Ordner gibt es schon ein Projekt.\n"
                                       "Bestehendes Projekt öffnen (statt überschreiben)?")
            if ans == QMessageBox.StandardButton.Yes:
                self._open_path(root_dir)
            return
        try:
            p = Project(root_dir, classes, self.cam_spin.value(), self.w_spin.value(), self.h_spin.value())
            p.save()
        except OSError as e:
            return QMessageBox.critical(self, "Fehler", str(e))
        self._finish(p)

    def open(self):
        d = QFileDialog.getExistingDirectory(self, "Projektordner öffnen")
        if d:
            self._open_path(d)

    def _open_path(self, d):
        try:
            self._finish(Project.load(d))
        except (OSError, ValueError, KeyError) as e:
            QMessageBox.critical(self, "Fehler", str(e))

    def _finish(self, project):
        remember_project(project.root)
        self.result_project = project
        self.accept()


# --------------------------------------------------------------------------- #
# Klassenauswahl nach dem Markieren
# --------------------------------------------------------------------------- #
class ClassDialog(QDialog):
    def __init__(self, parent, classes, preselect=0):
        super().__init__(parent)
        self.setWindowTitle("Klasse wählen")
        self.selected = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Klasse wählen (1-9, Enter, Doppelklick):"))

        self.list = QListWidget()
        self.list.setFont(QFont("Segoe UI", 11))
        for i, name in enumerate(classes):
            self.list.addItem(f"{i + 1}: {name}" if i < 9 else f"     {name}")
        self.list.setCurrentRow(max(0, min(preselect, len(classes) - 1)))
        self.list.itemDoubleClicked.connect(lambda _: self.accept_current())
        rows = min(len(classes), 12)
        self.list.setFixedHeight(self.list.sizeHintForRow(0) * rows + 2 * self.list.frameWidth() + 4)
        layout.addWidget(self.list)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept_current)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for i in range(min(len(classes), 9)):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda idx=i: self._pick(idx))

        pos = QCursor.pos()
        self.move(pos.x() + 12, pos.y() + 12)
        self.list.setFocus()

    def _pick(self, idx):
        self.selected = idx
        self.accept()

    def accept_current(self):
        if self.list.currentRow() >= 0:
            self._pick(self.list.currentRow())

    @staticmethod
    def ask(parent, classes, preselect=0):
        dlg = ClassDialog(parent, classes, preselect)
        dlg.exec()
        return dlg.selected


# --------------------------------------------------------------------------- #
# Bildfläche mit Boxen
# --------------------------------------------------------------------------- #
class ImageCanvas(QWidget):
    box_drawn = pyqtSignal(float, float, float, float)   # x1, y1, x2, y2 in Bildpixeln
    right_clicked = pyqtSignal(float, float)             # x, y in Bildpixeln

    def __init__(self):
        super().__init__()
        self.setMinimumSize(320, 240)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.qimage = None
        self.boxes = []
        self.classes = []
        self.highlight = -1
        self.overlay_top = ""
        self.overlay_bottom = ""
        self.drawing_enabled = False
        self.zoom = 1.0              # 1.0 = Bild passt ins Fenster
        self.pan = QPointF(0, 0)     # Verschiebung in Widget-Pixeln
        self._drag_start = None
        self._drag_end = None
        self._pan_last = None

    def set_image(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        self.qimage = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        self._clamp_pan()
        self.update()

    def clear_image(self):
        self.qimage = None
        self.update()

    # --- Koordinaten / Zoom ---
    def _fit_scale(self):
        return min(self.width() / self.qimage.width(), self.height() / self.qimage.height())

    def _geometry(self):
        """Skalierung und Offset (Bild mittig, seitenrichtig, mit Zoom und Verschiebung)."""
        iw, ih = self.qimage.width(), self.qimage.height()
        s = self._fit_scale() * self.zoom
        return s, (self.width() - iw * s) / 2 + self.pan.x(), (self.height() - ih * s) / 2 + self.pan.y()

    def _clamp_pan(self):
        if self.qimage is None:
            return
        s = self._fit_scale() * self.zoom
        mx = max(0.0, (self.qimage.width() * s - self.width()) / 2)
        my = max(0.0, (self.qimage.height() * s - self.height()) / 2)
        self.pan = QPointF(min(max(self.pan.x(), -mx), mx), min(max(self.pan.y(), -my), my))

    def _to_img(self, p, clamp=True):
        s, ox, oy = self._geometry()
        x, y = (p.x() - ox) / s, (p.y() - oy) / s
        if clamp:
            x = min(max(x, 0), self.qimage.width() - 1)
            y = min(max(y, 0), self.qimage.height() - 1)
        return x, y

    def set_zoom(self, zoom, anchor=None):
        """Zoomt so, dass der Bildpunkt unter `anchor` (Widget-Koordinaten) stehen bleibt."""
        if self.qimage is None:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2, self.height() / 2)
        ix, iy = self._to_img(anchor, clamp=False)
        self.zoom = min(max(zoom, 1.0), MAX_ZOOM)
        s, ox, oy = self._geometry()
        self.pan += QPointF(anchor.x() - (ox + ix * s), anchor.y() - (oy + iy * s))
        self._clamp_pan()
        self.update()

    def reset_zoom(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def box_at(self, x, y):
        """Index der kleinsten Box, die den Bildpunkt (x, y) enthält, oder None."""
        hits = [i for i, (_, x1, y1, x2, y2) in enumerate(self.boxes) if x1 <= x <= x2 and y1 <= y <= y2]
        if not hits:
            return None
        b = self.boxes
        return min(hits, key=lambda k: (b[k][3] - b[k][1]) * (b[k][4] - b[k][2]))

    def box_under_cursor(self):
        if self.qimage is None:
            return None
        p = self.mapFromGlobal(QCursor.pos())
        if not self.rect().contains(p):
            return None
        return self.box_at(*self._to_img(QPointF(p), clamp=False))

    # --- Zeichnen ---
    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.GlobalColor.black)
        if self.qimage is None:
            painter.setPen(QColor("yellow"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.overlay_top or "Kein Bild")
            return
        # beim starken Reinzoomen Pixel scharf lassen, sonst glätten
        if self.zoom < 3:
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        s, ox, oy = self._geometry()
        painter.drawImage(QRectF(ox, oy, self.qimage.width() * s, self.qimage.height() * s), self.qimage)

        painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        fm = painter.fontMetrics()
        for i, (c, x1, y1, x2, y2) in enumerate(self.boxes):
            col = SELECTED_COLOR if i == self.highlight else BOX_COLOR
            painter.setPen(QPen(col, 3 if i == self.highlight else 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            r = QRectF(ox + x1 * s, oy + y1 * s, (x2 - x1) * s, (y2 - y1) * s)
            painter.drawRect(r)
            name = self.classes[c] if c < len(self.classes) else f"#{c}"
            tw, th = fm.horizontalAdvance(name) + 6, fm.height()
            ty = r.top() - th if r.top() - th >= 0 else r.top()
            painter.fillRect(QRectF(r.left(), ty, tw, th), col)
            painter.setPen(QColor("black"))
            painter.drawText(QRectF(r.left() + 3, ty, tw, th), Qt.AlignmentFlag.AlignVCenter, name)

        if self._drag_start is not None and self._drag_end is not None:
            painter.setPen(QPen(QColor("white"), 2, Qt.PenStyle.DashLine))
            painter.drawRect(QRectF(self._drag_start, self._drag_end).normalized())

        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.setPen(QColor("yellow"))
        top = self.overlay_top
        if self.zoom > 1.001:
            top = f"{top}   Zoom {self.zoom * 100:.0f}%".strip()
        if top:
            painter.drawText(QPointF(10, 24), top)
        if self.overlay_bottom:
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(QPointF(10, self.height() - 10), self.overlay_bottom)

    # --- Maus ---
    def wheelEvent(self, e):
        steps = e.angleDelta().y() / 120
        if steps:
            self.set_zoom(self.zoom * (1.25 ** steps), e.position())

    def resizeEvent(self, e):
        self._clamp_pan()
        super().resizeEvent(e)

    def mousePressEvent(self, e):
        if self.qimage is None:
            return
        # Verschieben: mittlere Maustaste, im Livebild auch linke Maustaste
        if e.button() == Qt.MouseButton.MiddleButton or (
                e.button() == Qt.MouseButton.LeftButton and not self.drawing_enabled):
            self._pan_last = e.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif not self.drawing_enabled:
            return
        elif e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = self._drag_end = e.position()
        elif e.button() == Qt.MouseButton.RightButton:
            self.right_clicked.emit(*self._to_img(e.position(), clamp=False))

    def mouseMoveEvent(self, e):
        if self._pan_last is not None:
            self.pan += e.position() - self._pan_last
            self._pan_last = e.position()
            self._clamp_pan()
            self.update()
        elif self._drag_start is not None:
            self._drag_end = e.position()
            self.update()

    def mouseReleaseEvent(self, e):
        if self._pan_last is not None:
            self._pan_last = None
            self.setCursor(Qt.CursorShape.CrossCursor)
            return
        if self._drag_start is None or e.button() != Qt.MouseButton.LeftButton:
            return
        start, end = self._drag_start, e.position()
        self._drag_start = self._drag_end = None
        self.update()
        if abs(end.x() - start.x()) < 5 or abs(end.y() - start.y()) < 5:
            return  # nur Klick / zu klein
        x1, y1 = self._to_img(QPointF(min(start.x(), end.x()), min(start.y(), end.y())))
        x2, y2 = self._to_img(QPointF(max(start.x(), end.x()), max(start.y(), end.y())))
        if x2 - x1 < 1 or y2 - y1 < 1:
            return  # komplett außerhalb des Bildes gezogen
        self.box_drawn.emit(x1, y1, x2, y2)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
BASE_MODELS = ["yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt",
               "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt",
               "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

# Wird als eigener Prozess gestartet, damit die GUI während des Trainings bedienbar bleibt.
TRAIN_SCRIPT = '''\
import json, sys
from ultralytics import YOLO

def main():
    cfg = json.loads(sys.argv[1])
    model = YOLO(cfg.pop("model"))
    model.train(**cfg)
    print("__SAVE_DIR__=" + str(model.trainer.save_dir), flush=True)

if __name__ == "__main__":  # nötig unter Windows (Dataloader-Worker)
    main()
'''

ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# offizielle Ultralytics-Gewichte, die automatisch heruntergeladen werden (yolov8n.pt, yolo11s.pt, yolo26m.pt …)
OFFICIAL_MODEL_RE = re.compile(r"^yolo(v\d+|\d+)[nsmlx]\.pt$", re.IGNORECASE)


def make_split(project, val_ratio, seed=0):
    """Schreibt train.txt / val.txt / data_train.yaml (Zufallssplit, Bilder bleiben wo sie sind)."""
    imgs = [f for f in project.image_list() if os.path.isfile(project.label_path(f))]
    rnd = random.Random(seed)
    rnd.shuffle(imgs)
    n_val = int(round(len(imgs) * val_ratio))
    if val_ratio > 0 and len(imgs) >= 2:
        n_val = min(max(n_val, 1), len(imgs) - 1)
    else:
        n_val = 0
    val, train = sorted(imgs[:n_val]), sorted(imgs[n_val:])
    if not val:
        val = train  # ohne Split: Validierung auf Trainingsdaten (Werte zu optimistisch)
    for name, items in (("train.txt", train), ("val.txt", val)):
        with open(os.path.join(project.root, name), "w", encoding="utf-8") as f:
            f.write("".join(f"./images/{i}\n" for i in items))
    yaml_path = os.path.join(project.root, "data_train.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("train: train.txt\nval: val.txt\n\nnames:\n")
        for i, name in enumerate(project.classes):
            f.write(f"  {i}: {name}\n")
    return yaml_path, len(train), (n_val if n_val else 0)


class TrainDialog(QDialog):
    def __init__(self, parent, project):
        super().__init__(parent)
        self.p = project
        self.proc = None
        self.save_dir = None
        self.setWindowTitle("YOLO trainieren")
        self.resize(820, 640)
        layout = QVBoxLayout(self)

        # --- Einstellungen ---
        box = QGroupBox("Einstellungen")
        form = QFormLayout(box)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(BASE_MODELS)
        for best in sorted(glob.glob(os.path.join(self.p.root, "runs", "*", "weights", "best.pt"))):
            self.model_combo.addItem(best)  # eigene Modelle zum Weitertrainieren
        self.model_combo.setCurrentText("yolo26n.pt")
        pick = QPushButton("Datei…")
        pick.clicked.connect(self._pick_model)
        row = QHBoxLayout()
        row.addWidget(self.model_combo, 1)
        row.addWidget(pick)
        form.addRow("Basismodell:", row)
        form.addRow("", self._hint("n = schnell/klein … l = genau/langsam. Standardmodelle werden beim "
                                   "ersten Mal automatisch heruntergeladen."))

        self.epochs = QSpinBox()
        self.epochs.setRange(1, 5000)
        self.epochs.setValue(100)
        form.addRow("Epochen:", self.epochs)

        self.patience = QSpinBox()
        self.patience.setRange(0, 1000)
        self.patience.setValue(30)
        self.patience.setSpecialValueText("aus")
        form.addRow("Early Stopping (Geduld):", self.patience)

        self.imgsz = QComboBox()
        self.imgsz.addItems(["320", "416", "512", "640", "800", "960", "1280"])
        self.imgsz.setCurrentText("640")
        form.addRow("Bildgröße (imgsz):", self.imgsz)

        self.batch = QSpinBox()
        self.batch.setRange(-1, 256)
        self.batch.setValue(8)
        self.batch.setSpecialValueText("auto")
        form.addRow("Batch-Größe:", self.batch)

        self.device = QComboBox()
        self.device.addItems(["auto", "cpu", "0", "1"])
        form.addRow("Gerät:", self.device)

        self.workers = QSpinBox()
        self.workers.setRange(0, 16)
        self.workers.setValue(2)
        form.addRow("Dataloader-Worker:", self.workers)

        self.val_split = QSpinBox()
        self.val_split.setRange(0, 50)
        self.val_split.setValue(20)
        self.val_split.setSuffix(" %")
        form.addRow("Validierungsanteil:", self.val_split)

        self.run_name = QLineEdit("train")
        form.addRow("Name des Laufs:", self.run_name)
        layout.addWidget(box)

        self.info = QLabel()
        layout.addWidget(self.info)
        self._update_info()

        # --- Log ---
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 9))
        self.log.setMaximumBlockCount(5000)
        layout.addWidget(self.log, 1)

        btns = QHBoxLayout()
        self.start_btn = QPushButton("▶ Training starten")
        self.start_btn.clicked.connect(self.start)
        self.stop_btn = QPushButton("■ Abbrechen")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        self.open_btn = QPushButton("📂 Ergebnisordner öffnen")
        self.open_btn.clicked.connect(self._open_results)
        close_btn = QPushButton("Schließen")
        close_btn.clicked.connect(self.close)
        btns.addWidget(self.start_btn)
        btns.addWidget(self.stop_btn)
        btns.addStretch()
        btns.addWidget(self.open_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    @staticmethod
    def _hint(text):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: gray;")
        return lbl

    def _update_info(self):
        imgs = self.p.image_list()
        labeled = sum(1 for i in imgs if os.path.isfile(self.p.label_path(i)))
        self.info.setText(f"Projekt: {self.p.root}   –   {labeled} gelabelte Bilder, "
                          f"{len(self.p.classes)} Klassen")

    def _model_search_dirs(self):
        here = os.path.dirname(os.path.abspath(__file__))
        dirs = [self.p.root, os.getcwd(), here, os.path.join(here, "..", "DNN")]
        return list(dict.fromkeys(os.path.normpath(d) for d in dirs))

    def _resolve_model(self, text):
        """Absoluter Pfad einer vorhandenen .pt-Datei, offizieller Modellname (Download) oder None."""
        if os.path.isabs(text):
            return text if os.path.isfile(text) else None
        for d in self._model_search_dirs():
            path = os.path.join(d, text)
            if os.path.isfile(path):
                return os.path.abspath(path)
        if OFFICIAL_MODEL_RE.match(text):
            return text  # Ultralytics lädt es herunter
        return None

    def _pick_model(self):
        f, _ = QFileDialog.getOpenFileName(self, "Modell wählen", self.p.root, "PyTorch-Modell (*.pt)")
        if f:
            self.model_combo.setCurrentText(f)

    def _open_results(self):
        d = self.save_dir or os.path.join(self.p.root, "runs")
        if not os.path.isdir(d):
            QMessageBox.information(self, "Ergebnisse", "Noch keine Ergebnisse vorhanden.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(d))

    # --- Prozess ---
    def start(self):
        self._update_info()
        text = self.model_combo.currentText().strip()
        if not text:
            return QMessageBox.critical(self, "Fehler", "Bitte ein Basismodell wählen.")
        model = self._resolve_model(text)
        if model is None:
            return QMessageBox.critical(
                self, "Modell nicht gefunden",
                f"„{text}“ wurde nicht gefunden.\n\n"
                "Entweder ein offizielles Modell wählen (z. B. yolo26n.pt – wird automatisch geladen)\n"
                "oder über „Datei…“ eine vorhandene .pt-Datei auswählen.\n\n"
                "Gesucht in:\n" + "\n".join(self._model_search_dirs()))
        yaml_path, n_train, n_val = make_split(self.p, self.val_split.value() / 100)
        if n_train == 0:
            return QMessageBox.critical(self, "Fehler", "Keine gelabelten Bilder im Projekt.")

        cfg = {
            "model": model,
            "data": yaml_path,
            "epochs": self.epochs.value(),
            "patience": self.patience.value() or self.epochs.value(),
            "imgsz": int(self.imgsz.currentText()),
            "batch": self.batch.value(),
            "workers": self.workers.value(),
            "project": os.path.join(self.p.root, "runs"),
            "name": self.run_name.text().strip() or "train",
        }
        if self.device.currentText() != "auto":
            cfg["device"] = self.device.currentText()

        script = os.path.join(tempfile.gettempdir(), "yolo_labeler_train.py")
        with open(script, "w", encoding="utf-8") as f:
            f.write(TRAIN_SCRIPT)

        self.log.clear()
        val_txt = f"{n_val} Validierung" if n_val else "keine Validierungsbilder (val = train)"
        self.log.appendPlainText(f"Split: {n_train} Training, {val_txt}\n"
                                 f"Konfiguration: {json.dumps(cfg, ensure_ascii=False)}\n")
        self.save_dir = None

        self.proc = QProcess(self)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(self.p.root)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        self.proc.readyReadStandardOutput.connect(self._read_output)
        self.proc.finished.connect(self._finished)
        self.proc.errorOccurred.connect(self._error)
        self.proc.start(sys.executable, ["-u", script, json.dumps(cfg)])
        self._set_running(True)

    def _set_running(self, running):
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)

    def _read_output(self):
        text = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        text = ANSI_RE.sub("", text).replace("\r\n", "\n")
        cursor = self.log.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for k, part in enumerate(text.split("\r")):
            if k > 0:  # Wagenrücklauf (Fortschrittsbalken): aktuelle Zeile überschreiben
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock, QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            cursor.insertText(part)
            for line in part.splitlines():
                if line.startswith("__SAVE_DIR__="):
                    self.save_dir = line.split("=", 1)[1].strip()
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _finished(self, code, _status):
        self._set_running(False)
        if code == 0 and self.save_dir:
            best = os.path.join(self.save_dir, "weights", "best.pt")
            self.log.appendPlainText(f"\n✔ Training fertig.\nBestes Modell: {best}")
            QMessageBox.information(self, "Training fertig", f"Bestes Modell:\n{best}")
        else:
            self.log.appendPlainText(f"\n✖ Training beendet (Exit-Code {code}).")

    def _error(self, err):
        if err == QProcess.ProcessError.FailedToStart:
            self._set_running(False)
            self.log.appendPlainText(f"Prozess konnte nicht gestartet werden: {sys.executable}")

    def stop(self):
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            self.log.appendPlainText("\nBreche ab …")
            self.proc.kill()

    def closeEvent(self, e):
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            ans = QMessageBox.question(self, "Training läuft", "Training läuft noch. Abbrechen und schließen?")
            if ans != QMessageBox.StandardButton.Yes:
                e.ignore()
                return
            self.proc.kill()
            self.proc.waitForFinished(3000)
        e.accept()


# --------------------------------------------------------------------------- #
# Hauptfenster
# --------------------------------------------------------------------------- #
class LabelerWindow(QMainWindow):
    def __init__(self, project):
        super().__init__()
        self.p = project
        self.setWindowTitle(f"YOLO Labeler – {self.p.root}")

        self.mode = "live"          # "live" oder "label"
        self.frame = None           # aktuelles Livebild (BGR)
        self.image = None           # Bild, das gelabelt wird (BGR)
        self.image_name = None      # Dateiname, falls schon gespeichert
        self.boxes = []             # [cls, x1, y1, x2, y2] in Bildpixeln
        self.dirty = False
        self.last_cls = 0

        self.train_dialog = None
        self._build_ui()
        self._build_menu()
        self._open_camera()
        self._refresh()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._grab_frame)
        self.timer.start(30)

    # ---------------- UI ---------------- #
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)

        self.canvas = ImageCanvas()
        self.canvas.classes = self.p.classes
        self.canvas.box_drawn.connect(self._on_box_drawn)
        self.canvas.right_clicked.connect(self._on_right_click)
        layout.addWidget(self.canvas, 1)

        side = QVBoxLayout()
        side_w = QWidget()
        side_w.setLayout(side)
        side_w.setFixedWidth(280)
        layout.addWidget(side_w)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Kamera:"))
        self.cam_combo = QComboBox()
        self.cam_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.cam_combo.addItems([f"Index {i}" for i in range(10)])
        self.cam_combo.setCurrentIndex(min(self.p.camera_index, 9))
        self.cam_combo.currentIndexChanged.connect(self._switch_camera)
        cam_row.addWidget(self.cam_combo, 1)
        side.addLayout(cam_row)

        def button(text, slot, key=None):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setStyleSheet("text-align: left; padding: 4px 8px;")
            b.clicked.connect(slot)
            side.addWidget(b)
            if key:
                QShortcut(QKeySequence(key), self, activated=slot)
            return b

        button("📷  Foto aufnehmen  [Leertaste]", self.capture, Qt.Key.Key_Space)
        button("💾  Speichern  [Enter]", self.save, Qt.Key.Key_Return)
        QShortcut(QKeySequence(Qt.Key.Key_Enter), self, activated=self.save)
        button("✖  Verwerfen / Livebild  [Esc]", self.back_to_live, Qt.Key.Key_Escape)
        button("↶  Letzte Box löschen  [Strg+Z]", self.undo, "Ctrl+Z")

        nav = QHBoxLayout()
        for text, step, key in (("◀ Vorheriges [←]", -1, Qt.Key.Key_Left), ("Nächstes [→] ▶", 1, Qt.Key.Key_Right)):
            b = QPushButton(text)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(lambda _=False, s=step: self.navigate(s))
            QShortcut(QKeySequence(key), self, activated=lambda s=step: self.navigate(s))
            nav.addWidget(b)
        side.addLayout(nav)
        button("❌  Box löschen  [Entf]", self.delete_box, Qt.Key.Key_Delete)
        button("🗑  Ganzes Bild löschen  [Umschalt+Entf]", self.delete_image, "Shift+Del")
        button("🔍  Zoom zurücksetzen  [0]", self.canvas.reset_zoom, Qt.Key.Key_0)
        QShortcut(QKeySequence(Qt.Key.Key_Plus), self, activated=lambda: self.canvas.set_zoom(self.canvas.zoom * 1.25))
        QShortcut(QKeySequence(Qt.Key.Key_Minus), self, activated=lambda: self.canvas.set_zoom(self.canvas.zoom / 1.25))
        button("➕  Klasse hinzufügen", self.add_class)

        side.addWidget(QLabel("Boxen im Bild (Doppelklick = Klasse ändern):"))
        self.box_list = QListWidget()
        self.box_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.box_list.itemDoubleClicked.connect(self._change_box_class)
        self.box_list.currentRowChanged.connect(self._on_box_selected)
        side.addWidget(self.box_list, 1)

        side.addWidget(QLabel("Klassen (Anzahl Boxen gesamt):"))
        self.class_list = QListWidget()
        self.class_list.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        side.addWidget(self.class_list, 1)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        side.addWidget(line)
        self.status = QLabel()
        self.status.setWordWrap(True)
        side.addWidget(self.status)
        hint = QLabel("Linke Maustaste ziehen: Box zeichnen\n"
                      "Rechtsklick auf Box / Entf: Box löschen\n"
                      "Mausrad: Zoom (+ / − / 0)\n"
                      "Mittlere Maustaste ziehen: Bild verschieben")
        hint.setStyleSheet("color: gray;")
        side.addWidget(hint)

        self.resize(self.p.width + 330, max(self.p.height + 40, 600))

    def _build_menu(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("&Datei")
        act = QAction("Projektordner im Explorer öffnen", self)
        act.triggered.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(self.p.root)))
        file_menu.addAction(act)
        file_menu.addSeparator()
        act = QAction("Beenden", self)
        act.triggered.connect(self.close)
        file_menu.addAction(act)

        train_menu = mb.addMenu("&Training")
        act = QAction("YOLO trainieren…", self)
        act.setShortcut(QKeySequence("Ctrl+T"))
        act.triggered.connect(self.open_training)
        train_menu.addAction(act)
        act = QAction("Ergebnisordner (runs) öffnen", self)
        act.triggered.connect(self._open_runs)
        train_menu.addAction(act)

    def open_training(self):
        if self.mode == "label" and self.dirty and self.boxes:
            if self._ask("Ungespeichert", "Aktuelles Bild vorher speichern?") == QMessageBox.StandardButton.Yes:
                self._write()
                self._refresh()
        if self.train_dialog is None:
            self.train_dialog = TrainDialog(self, self.p)
        self.train_dialog._update_info()
        self.train_dialog.show()
        self.train_dialog.raise_()
        self.train_dialog.activateWindow()

    def _open_runs(self):
        d = os.path.join(self.p.root, "runs")
        if os.path.isdir(d):
            QDesktopServices.openUrl(QUrl.fromLocalFile(d))
        else:
            QMessageBox.information(self, "Ergebnisse", "Noch keine Trainingsläufe vorhanden.")

    # ---------------- Kamera ---------------- #
    def _open_camera(self):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self.cap = cv2.VideoCapture(self.p.camera_index, backend)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.p.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.p.height)
        finally:
            QApplication.restoreOverrideCursor()

    def _switch_camera(self, index):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.frame = None
        self.p.camera_index = index
        self._open_camera()
        try:
            self.p.save()  # gewählten Index im Projekt merken
        except OSError:
            pass
        if self.mode == "live":
            self.canvas.clear_image()
        self._refresh()

    def _grab_frame(self):
        if self.mode != "live" or self.cap is None:
            return
        ok, frame = self.cap.read()
        if ok and frame is not None:
            self.frame = frame
            self.canvas.set_image(frame)

    # ---------------- Anzeige ---------------- #
    def _refresh(self):
        """Canvas-Overlay und Seitenleiste aktualisieren."""
        c = self.canvas
        c.boxes = self.boxes
        c.drawing_enabled = self.mode == "label"
        if self.mode == "live":
            c.highlight = -1
            c.overlay_top = ("LIVE – Leertaste zum Aufnehmen" if self.cap else
                             f"Kamera {self.p.camera_index} nicht verfügbar – andere wählen oder mit ← / → Bilder öffnen")
            c.overlay_bottom = ""
            if self.cap is None:
                c.clear_image()
        else:
            c.overlay_top = ""
            c.overlay_bottom = (self.image_name or "(neu, noch nicht gespeichert)") + ("  *" if self.dirty else "")
            c.set_image(self.image)

        self.box_list.blockSignals(True)
        self.box_list.clear()
        for cls, x1, y1, x2, y2 in self.boxes:
            name = self.p.classes[cls] if cls < len(self.p.classes) else f"#{cls}"
            self.box_list.addItem(f"{name}  ({int(x2 - x1)}x{int(y2 - y1)})")
        if 0 <= c.highlight < len(self.boxes):
            self.box_list.setCurrentRow(c.highlight)
        self.box_list.blockSignals(False)

        imgs = self.p.image_list()
        counts = [0] * len(self.p.classes)
        for img in imgs:
            lp = self.p.label_path(img)
            if os.path.isfile(lp):
                with open(lp) as f:
                    for line in f:
                        parts = line.split()
                        if parts and parts[0].isdigit() and int(parts[0]) < len(counts):
                            counts[int(parts[0])] += 1
        self.class_list.clear()
        for i, name in enumerate(self.p.classes):
            self.class_list.addItem(f"{i + 1}: {name}  ({counts[i]})")

        pos = f"   Bild {imgs.index(self.image_name) + 1}/{len(imgs)}" if self.image_name in imgs else ""
        mode = "Livebild" if self.mode == "live" else "Labeln"
        cam = "ok" if self.cap is not None else "<span style='color:red'>nicht verfügbar</span>"
        self.status.setText(f"<b>Modus:</b> {mode}<br><b>Kamera {self.p.camera_index}:</b> {cam}"
                            f"<br><b>Bilder im Projekt:</b> {len(imgs)}{pos}"
                            f"<br><b>Ordner:</b> {self.p.root}")

    def _on_box_selected(self, row):
        self.canvas.highlight = row
        self.canvas.update()

    # ---------------- Boxen ---------------- #
    def _on_box_drawn(self, x1, y1, x2, y2):
        self.canvas.boxes = self.boxes + [[self.last_cls, x1, y1, x2, y2]]  # Vorschau während der Auswahl
        self.canvas.update()
        cls = ClassDialog.ask(self, self.p.classes, self.last_cls)
        if cls is not None:
            self.last_cls = cls
            self.boxes.append([cls, x1, y1, x2, y2])
            self.dirty = True
        self._refresh()

    def _on_right_click(self, x, y):
        self._remove_box(self.canvas.box_at(x, y))

    def delete_box(self):
        """Entf: ausgewählte Box löschen, sonst die Box unter dem Mauszeiger. Das Bild bleibt erhalten."""
        if self.mode != "label" or not self.boxes:
            return
        i = self.box_list.currentRow()
        if not 0 <= i < len(self.boxes):
            i = self.canvas.box_under_cursor()
        self._remove_box(i)

    def _remove_box(self, i):
        if i is None or not 0 <= i < len(self.boxes):
            return
        del self.boxes[i]
        self.canvas.highlight = -1
        self.dirty = True
        self._refresh()

    def _change_box_class(self, _item):
        i = self.box_list.currentRow()
        if i < 0:
            return
        cls = ClassDialog.ask(self, self.p.classes, self.boxes[i][0])
        if cls is not None:
            self.boxes[i][0] = cls
            self.canvas.highlight = i
            self.dirty = True
            self._refresh()

    def undo(self):
        if self.mode == "label" and self.boxes:
            self.boxes.pop()
            self.canvas.highlight = -1
            self.dirty = True
            self._refresh()

    # ---------------- Aktionen ---------------- #
    def _ask(self, title, text, cancel=False):
        buttons = QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        if cancel:
            buttons |= QMessageBox.StandardButton.Cancel
        return QMessageBox.question(self, title, text, buttons)

    def _resolve_unsaved(self):
        """Fragt bei ungespeicherten Änderungen nach. False = Aktion abbrechen."""
        if self.mode != "label" or not self.dirty:
            return True
        if self.image_name is None:  # frisch aufgenommen
            if not self.boxes:
                return True
            return self._ask("Verwerfen", "Aufgenommenes Bild samt Boxen verwerfen?") == QMessageBox.StandardButton.Yes
        ans = self._ask("Ungespeichert", "Änderungen speichern?", cancel=True)
        if ans == QMessageBox.StandardButton.Cancel:
            return False
        if ans == QMessageBox.StandardButton.Yes:
            self._write()
        return True

    def capture(self):
        if self.mode == "label":
            return  # erst speichern oder verwerfen
        if self.frame is None:
            QMessageBox.information(self, "Kamera", "Kein Kamerabild verfügbar.")
            return
        self.image = self.frame.copy()
        self.image_name = None
        self.boxes = []
        self.canvas.highlight = -1
        self.dirty = True
        self.mode = "label"
        self._refresh()

    def _write(self):
        os.makedirs(self.p.images_dir, exist_ok=True)
        os.makedirs(self.p.labels_dir, exist_ok=True)
        if self.image_name is None:
            self.image_name = time.strftime("img_%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}.jpg"
            cv2.imwrite(os.path.join(self.p.images_dir, self.image_name), self.image,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = self.image.shape[:2]
        with open(self.p.label_path(self.image_name), "w") as f:
            for c, x1, y1, x2, y2 in self.boxes:
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                f.write(f"{c} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        self.dirty = False

    def save(self):
        if self.mode != "label":
            return
        if not self.boxes and self._ask("Keine Boxen", "Keine Boxen markiert.\n"
                                        "Als Hintergrundbild (ohne Objekte) speichern?") != QMessageBox.StandardButton.Yes:
            return
        self._write()
        self._go_live()

    def back_to_live(self):
        if self.mode == "label" and self._resolve_unsaved():
            self._go_live()

    def _go_live(self):
        self.mode = "live"
        self.image = None
        self.image_name = None
        self.boxes = []
        self.dirty = False
        self._refresh()

    def _load_image(self, name):
        img = cv2.imread(os.path.join(self.p.images_dir, name))
        if img is None:
            QMessageBox.critical(self, "Fehler", f"Bild konnte nicht gelesen werden:\n{name}")
            return
        h, w = img.shape[:2]
        boxes = []
        lp = self.p.label_path(name)
        if os.path.isfile(lp):
            with open(lp) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    c = int(parts[0])
                    cx, cy, bw, bh = (float(v) for v in parts[1:])
                    boxes.append([c, (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
        self.image, self.image_name, self.boxes = img, name, boxes
        self.canvas.highlight = -1
        self.dirty = False
        self.mode = "label"
        self._refresh()

    def navigate(self, step):
        imgs = self.p.image_list()
        if not imgs or not self._resolve_unsaved():
            return
        if self.image_name in imgs:
            idx = (imgs.index(self.image_name) + step) % len(imgs)
        else:
            idx = len(imgs) - 1 if step < 0 else 0
        self._load_image(imgs[idx])

    def delete_image(self):
        if self.mode != "label":
            return
        if self.image_name is None:
            if self._ask("Verwerfen", "Aufgenommenes Bild verwerfen?") == QMessageBox.StandardButton.Yes:
                self._go_live()
            return
        if self._ask("Löschen", f"{self.image_name} und Labels wirklich löschen?") != QMessageBox.StandardButton.Yes:
            return
        imgs = self.p.image_list()
        idx = imgs.index(self.image_name) if self.image_name in imgs else 0
        for path in (os.path.join(self.p.images_dir, self.image_name), self.p.label_path(self.image_name)):
            if os.path.isfile(path):
                os.remove(path)
        imgs = self.p.image_list()
        if imgs:
            self._load_image(imgs[min(idx, len(imgs) - 1)])
        else:
            self._go_live()

    def add_class(self):
        name, ok = QInputDialog.getText(self, "Klasse hinzufügen", "Name der neuen Klasse:")
        name = name.strip()
        if not ok or not name:
            return
        if name in self.p.classes:
            QMessageBox.information(self, "Klasse", "Diese Klasse gibt es schon.")
            return
        self.p.classes.append(name)  # nur anhängen, damit bestehende IDs gültig bleiben
        self.p.save()
        self._refresh()

    def closeEvent(self, e):
        if not self._resolve_unsaved():
            e.ignore()
            return
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
        e.accept()


def main():
    app = QApplication(sys.argv)
    start = StartDialog()
    if start.exec() != QDialog.DialogCode.Accepted or start.result_project is None:
        return
    win = LabelerWindow(start.result_project)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
