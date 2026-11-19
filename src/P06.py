"""UR3e Jog GUI (ur_rtde) - TCP Jog + Pose + Teach/Hold

Start:  python src/P06.py
Benötigt: PyQt6, ur_rtde, opencv-python (siehe requirements.txt)

Der Roboter muss im Remote Control Modus sein (Teach Pendant), RTDE aktiv.
ACHTUNG: Die Software bewegt einen echten Roboter. Not-Halt in Reichweite halten!
"""
import csv
import math
import sys

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QVBoxLayout, QWidget, QComboBox,
)

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:  # GUI soll auch ohne ur_rtde starten
    RTDEControlInterface = RTDEReceiveInterface = None

JOG_PERIOD_MS = 20      # 50 Hz Jog-Kommandos
HOLD_PERIOD_MS = 8      # 125 Hz servoL (UR e-Series)
POSE_PERIOD_MS = 100    # Anzeige-Update


def fmt_pose(p):
    return (f"X {p[0]*1000:.1f}  Y {p[1]*1000:.1f}  Z {p[2]*1000:.1f} mm | "
            f"Rx {math.degrees(p[3]):.1f}  Ry {math.degrees(p[4]):.1f}  "
            f"Rz {math.degrees(p[5]):.1f} °")


class JogGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR3e Jog GUI (ur_rtde) - TCP Jog + Pose + Teach/Hold")
        self.resize(820, 780)

        self.rtde_c = None
        self.rtde_r = None
        self.jog_dir = np.zeros(6)      # aktuelle Jog-Richtung (x,y,z,rx,ry,rz)
        self.hold_target = None
        self.hold_active = False
        self.points = []                # [(name, [6 floats])]

        self._build_ui()

        self.jog_timer = QTimer(self, interval=JOG_PERIOD_MS)
        self.jog_timer.timeout.connect(self._jog_tick)
        self.hold_timer = QTimer(self, interval=HOLD_PERIOD_MS)
        self.hold_timer.timeout.connect(self._hold_tick)
        self.pose_timer = QTimer(self, interval=POSE_PERIOD_MS)
        self.pose_timer.timeout.connect(self._update_pose)
        self._set_connected(False)

    # ------------------------------------------------------------------ UI
    def _spin(self, val, lo, hi, step, dec, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(dec)
        s.setValue(val)
        s.setSuffix(suffix)
        return s

    def _build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QVBoxLayout(root)

        # Verbindung
        g = QGroupBox("Verbindung")
        h = QHBoxLayout(g)
        h.addWidget(QLabel("UR3e IP:"))
        self.ip_edit = QLineEdit("192.168.0.100")
        h.addWidget(self.ip_edit, 1)
        self.btn_connect = QPushButton("Verbinden")
        self.btn_disconnect = QPushButton("Trennen")
        self.btn_connect.clicked.connect(self.connect_robot)
        self.btn_disconnect.clicked.connect(self.disconnect_robot)
        h.addWidget(self.btn_connect)
        h.addWidget(self.btn_disconnect)
        main.addWidget(g)

        # Jog Parameter
        g = QGroupBox("Jog Parameter (TCP)")
        gl = QGridLayout(g)
        self.sp_lin_v = self._spin(0.05, 0.001, 0.5, 0.01, 3)
        self.sp_ang_w = self._spin(0.2, 0.01, 1.5, 0.05, 3)
        self.sp_jog_a = self._spin(0.5, 0.05, 2.0, 0.05, 2)
        self.sp_movel_v = self._spin(0.05, 0.001, 0.5, 0.01, 3)
        self.sp_movel_a = self._spin(0.5, 0.05, 2.0, 0.05, 2)
        self.sp_servo_a = self._spin(1.2, 0.01, 5.0, 0.05, 2)
        self.sp_servo_v = self._spin(0.25, 0.01, 2.0, 0.05, 2)
        self.sp_look = self._spin(0.1, 0.03, 0.2, 0.01, 3)
        self.sp_gain = self._spin(300, 100, 2000, 10, 1)
        self.cb_frame = QComboBox()
        self.cb_frame.addItems(["Tool-KS (TCP)", "Basis-KS"])
        rows = [
            ("Linear v [m/s]:", self.sp_lin_v, "Angular ω [rad/s]:", self.sp_ang_w),
            ("Jog Beschl. [m/s²]:", self.sp_jog_a, "Jog-Koordinatensystem:", self.cb_frame),
            ("moveL v [m/s]:", self.sp_movel_v, "moveL a [m/s²]:", self.sp_movel_a),
            ("servoL a:", self.sp_servo_a, "servoL v:", self.sp_servo_v),
            ("servoL lookahead [s]:", self.sp_look, "servoL gain:", self.sp_gain),
        ]
        for r, (l1, w1, l2, w2) in enumerate(rows):
            gl.addWidget(QLabel(l1), r, 0)
            gl.addWidget(w1, r, 1)
            gl.addWidget(QLabel(l2), r, 2)
            gl.addWidget(w2, r, 3)
        main.addWidget(g)

        mid = QHBoxLayout()
        left = QVBoxLayout()
        right = QVBoxLayout()
        mid.addLayout(left, 1)
        mid.addLayout(right, 1)
        main.addLayout(mid, 1)

        # Jog Translation / Rotation
        self.jog_buttons = []
        left.addWidget(self._jog_group("Jog (Translation TCP)",
                                       [("+X", 0, 1), ("+Y", 1, 1), ("+Z", 2, 1),
                                        ("-X", 0, -1), ("-Y", 1, -1), ("-Z", 2, -1)]))
        left.addWidget(self._jog_group("Jog (Rotation TCP)",
                                       [("+Rx", 3, 1), ("+Ry", 4, 1), ("+Rz", 5, 1),
                                        ("-Rx", 3, -1), ("-Ry", 4, -1), ("-Rz", 5, -1)]))

        # Aktuelle Pose
        g = QGroupBox("Aktuelle TCP Pose")
        gl = QGridLayout(g)
        self.pose_labels = []
        names = ["X [mm]:", "Y [mm]:", "Z [mm]:", "Rx [°]:", "Ry [°]:", "Rz [°]:"]
        for i, n in enumerate(names):
            col, row = (i % 3) * 2, i // 3
            gl.addWidget(QLabel(n), row, col)
            lbl = QLabel("--")
            self.pose_labels.append(lbl)
            gl.addWidget(lbl, row, col + 1)
        left.addWidget(g)

        # Teach-Punkte
        g = QGroupBox("Teach-Punkte")
        v = QVBoxLayout(g)
        h = QHBoxLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name des Teach-Punkts (z. B. P1)")
        self.btn_teach = QPushButton("Teach: aktuelle Pose speichern")
        self.btn_teach.clicked.connect(self.teach_point)
        h.addWidget(self.name_edit, 1)
        h.addWidget(self.btn_teach)
        v.addLayout(h)
        self.list = QListWidget()
        v.addWidget(self.list, 1)
        h = QHBoxLayout()
        self.btn_goto = QPushButton("Anfahren (moveL)")
        self.btn_hold_sel = QPushButton("Hold-Ziel = Auswahl")
        self.btn_del = QPushButton("Löschen")
        self.btn_goto.clicked.connect(self.goto_point)
        self.btn_hold_sel.clicked.connect(self.hold_to_selection)
        self.btn_del.clicked.connect(self.delete_point)
        for b in (self.btn_goto, self.btn_hold_sel, self.btn_del):
            h.addWidget(b)
        v.addLayout(h)
        h = QHBoxLayout()
        b1, b2 = QPushButton("CSV speichern"), QPushButton("CSV laden")
        b1.clicked.connect(self.save_csv)
        b2.clicked.connect(self.load_csv)
        h.addWidget(b1)
        h.addWidget(b2)
        v.addLayout(h)
        right.addWidget(g, 1)

        # Pose Hold
        g = QGroupBox("Pose Hold")
        gl = QGridLayout(g)
        self.btn_hold = QPushButton("Hold EIN")
        self.btn_hold.setCheckable(True)
        self.btn_hold.toggled.connect(self.toggle_hold)
        self.btn_hold_now = QPushButton("Hold-Ziel = aktuelle Pose")
        self.btn_hold_now.clicked.connect(self.hold_to_current)
        self.lbl_hold = QLabel("Ziel: —")
        gl.addWidget(self.btn_hold, 0, 0)
        gl.addWidget(self.btn_hold_now, 0, 1)
        gl.addWidget(QLabel("Aktuelles Hold-Ziel:"), 1, 0)
        gl.addWidget(self.lbl_hold, 1, 1)
        right.addWidget(g)

        # Statuszeile + Stopp
        h = QHBoxLayout()
        self.status = QLabel("Getrennt")
        self.btn_stop = QPushButton("Stopp (speedStop)")
        self.btn_stop.setStyleSheet("background:#8b1a1a; font-weight:bold;")
        self.btn_stop.clicked.connect(self.stop_all)
        h.addWidget(self.status, 1)
        h.addWidget(self.btn_stop)
        main.addLayout(h)

    def _jog_group(self, title, items):
        g = QGroupBox(title)
        gl = QGridLayout(g)
        for i, (text, axis, sign) in enumerate(items):
            b = QPushButton(text)
            b.setMinimumHeight(38)
            b.pressed.connect(lambda a=axis, s=sign: self._jog_start(a, s))
            b.released.connect(self._jog_stop)
            self.jog_buttons.append(b)
            gl.addWidget(b, i // 3, i % 3)
        return g

    # -------------------------------------------------------- Verbindung
    def _set_connected(self, ok):
        self.btn_connect.setEnabled(not ok)
        self.btn_disconnect.setEnabled(ok)
        self.ip_edit.setEnabled(not ok)
        for w in (self.btn_stop, self.btn_teach, self.btn_goto, self.btn_hold,
                  self.btn_hold_now, self.btn_hold_sel, *self.jog_buttons):
            w.setEnabled(ok)
        self.status.setText("Verbunden" if ok else "Getrennt")

    def connect_robot(self):
        if RTDEControlInterface is None:
            QMessageBox.critical(self, "Fehler", "ur_rtde ist nicht installiert (pip install ur_rtde).")
            return
        ip = self.ip_edit.text().strip()
        self.status.setText(f"Verbinde mit {ip} ...")
        QApplication.processEvents()
        try:
            self.rtde_r = RTDEReceiveInterface(ip)
            self.rtde_c = RTDEControlInterface(ip)
        except Exception as e:
            self.rtde_c = self.rtde_r = None
            self._set_connected(False)
            QMessageBox.critical(self, "Verbindung fehlgeschlagen", str(e))
            return
        self._set_connected(True)
        self.pose_timer.start()

    def disconnect_robot(self):
        self.stop_all()
        self.pose_timer.stop()
        for iface in (self.rtde_c, self.rtde_r):
            try:
                if iface is not None:
                    iface.disconnect()
            except Exception:
                pass
        self.rtde_c = self.rtde_r = None
        self._set_connected(False)

    # --------------------------------------------------------------- Pose
    def _update_pose(self):
        try:
            p = self.rtde_r.getActualTCPPose()
        except Exception as e:
            self.status.setText(f"Lesefehler: {e}")
            return
        vals = [p[0] * 1000, p[1] * 1000, p[2] * 1000,
                math.degrees(p[3]), math.degrees(p[4]), math.degrees(p[5])]
        for lbl, v in zip(self.pose_labels, vals):
            lbl.setText(f"{v:.2f}")

    def _current_pose(self):
        return list(self.rtde_r.getActualTCPPose())

    # ---------------------------------------------------------------- Jog
    def _jog_start(self, axis, sign):
        if self.hold_active:
            self.status.setText("Jog gesperrt: Hold ist aktiv")
            return
        d = np.zeros(6)
        d[axis] = sign
        self.jog_dir = d
        self.jog_timer.start()
        self._jog_tick()

    def _jog_stop(self):
        self.jog_timer.stop()
        self.jog_dir = np.zeros(6)
        if self.rtde_c is not None:
            try:
                self.rtde_c.speedStop()
            except Exception as e:
                self.status.setText(f"Fehler: {e}")

    def _jog_tick(self):
        if self.rtde_c is None:
            return
        v = self.jog_dir[:3] * self.sp_lin_v.value()
        w = self.jog_dir[3:] * self.sp_ang_w.value()
        if self.cb_frame.currentIndex() == 0:  # Tool-KS -> in Basis-KS drehen
            pose = self._current_pose()
            R, _ = cv2.Rodrigues(np.array(pose[3:6], dtype=float))
            v, w = R @ v, R @ w
        try:
            self.rtde_c.speedL([*v, *w], self.sp_jog_a.value(), JOG_PERIOD_MS * 3 / 1000)
        except Exception as e:
            self.status.setText(f"Jog-Fehler: {e}")
            self.jog_timer.stop()

    # --------------------------------------------------------------- Hold
    def _set_hold_target(self, pose):
        self.hold_target = list(pose)
        self.lbl_hold.setText("Ziel: " + fmt_pose(pose))

    def hold_to_current(self):
        self._set_hold_target(self._current_pose())

    def hold_to_selection(self):
        i = self.list.currentRow()
        if i < 0:
            self.status.setText("Kein Punkt ausgewählt")
            return
        self._set_hold_target(self.points[i][1])

    def toggle_hold(self, on):
        if self.rtde_c is None:
            self.btn_hold.blockSignals(True)
            self.btn_hold.setChecked(False)
            self.btn_hold.blockSignals(False)
            return
        if on:
            if self.hold_target is None:
                self.hold_to_current()
            self.hold_active = True
            self.btn_hold.setText("Hold AUS")
            self.hold_timer.start()
        else:
            self.hold_active = False
            self.hold_timer.stop()
            self.btn_hold.setText("Hold EIN")
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass

    def _hold_tick(self):
        try:
            self.rtde_c.servoL(self.hold_target, self.sp_servo_v.value(), self.sp_servo_a.value(),
                               HOLD_PERIOD_MS / 1000, self.sp_look.value(), self.sp_gain.value())
        except Exception as e:
            self.status.setText(f"Hold-Fehler: {e}")
            self.btn_hold.setChecked(False)

    # -------------------------------------------------------- Teach-Punkte
    def _refresh_list(self):
        self.list.clear()
        for name, p in self.points:
            self.list.addItem(f"{name}:  {fmt_pose(p)}")

    def teach_point(self):
        name = self.name_edit.text().strip() or f"P{len(self.points) + 1}"
        self.points.append((name, self._current_pose()))
        self.name_edit.clear()
        self._refresh_list()

    def delete_point(self):
        i = self.list.currentRow()
        if i >= 0:
            del self.points[i]
            self._refresh_list()

    def goto_point(self):
        i = self.list.currentRow()
        if i < 0:
            self.status.setText("Kein Punkt ausgewählt")
            return
        if self.hold_active:
            self.btn_hold.setChecked(False)  # Hold aus, sonst Konflikt mit moveL
        name, p = self.points[i]
        try:
            self.rtde_c.moveL(p, self.sp_movel_v.value(), self.sp_movel_a.value(), True)
            self.status.setText(f"Fahre zu {name} ...")
        except Exception as e:
            self.status.setText(f"moveL-Fehler: {e}")

    def save_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "Teach-Punkte speichern", "teach_points.csv", "CSV (*.csv)")
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "x", "y", "z", "rx", "ry", "rz"])
            for name, p in self.points:
                w.writerow([name, *p])

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(self, "Teach-Punkte laden", "", "CSV (*.csv)")
        if not path:
            return
        try:
            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))[1:]
            self.points = [(r[0], [float(x) for x in r[1:7]]) for r in rows if len(r) >= 7]
        except Exception as e:
            QMessageBox.critical(self, "Fehler", f"CSV konnte nicht gelesen werden:\n{e}")
            return
        self._refresh_list()

    # --------------------------------------------------------------- Stopp
    def stop_all(self):
        self.jog_timer.stop()
        self.jog_dir = np.zeros(6)
        if self.hold_active:
            self.btn_hold.setChecked(False)
        if self.rtde_c is not None:
            try:
                self.rtde_c.speedStop()
                self.rtde_c.stopL(2.0)
            except Exception as e:
                self.status.setText(f"Stopp-Fehler: {e}")
                return
            self.status.setText("Gestoppt")

    def closeEvent(self, ev):
        self.disconnect_robot()
        super().closeEvent(ev)


def apply_dark(app):
    app.setStyle("Fusion")
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window, QColor(32, 32, 32))
    p.setColor(QPalette.ColorRole.WindowText, QColor(225, 225, 225))
    p.setColor(QPalette.ColorRole.Base, QColor(20, 20, 20))
    p.setColor(QPalette.ColorRole.Text, QColor(225, 225, 225))
    p.setColor(QPalette.ColorRole.Button, QColor(52, 52, 52))
    p.setColor(QPalette.ColorRole.ButtonText, QColor(225, 225, 225))
    p.setColor(QPalette.ColorRole.Highlight, QColor(60, 110, 170))
    p.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(p)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_dark(app)
    win = JogGUI()
    win.show()
    sys.exit(app.exec())
