"""UR3e Jog GUI (ur_rtde) - TCP Jog + Pose + Teach/Hold

Start:  python src/P06.py
Benötigt: PyQt6, ur_rtde, opencv-python (siehe requirements.txt)

Der Roboter muss im Remote Control Modus sein (Teach Pendant), RTDE aktiv.
ACHTUNG: Die Software bewegt einen echten Roboter. Not-Halt in Reichweite halten!
"""
import csv
import math
import sys
import time

import cv2
import numpy as np
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication, QDoubleSpinBox, QFileDialog, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox,
    QPushButton, QVBoxLayout, QWidget, QComboBox, QSlider, QScrollArea,
    QFormLayout, QSizePolicy, QCheckBox,
)

try:
    from rtde_control import RTDEControlInterface
    from rtde_receive import RTDEReceiveInterface
except ImportError:  # GUI soll auch ohne ur_rtde starten
    RTDEControlInterface = RTDEReceiveInterface = None
try:
    from robotiq_gripper import RobotiqGripper  # liegt neben dieser Datei
except ImportError:
    RobotiqGripper = None

JOG_PERIOD_MS = 20      # 50 Hz Jog-Kommandos
HOLD_PERIOD_MS = 8      # 125 Hz servoL (UR e-Series)
POSE_PERIOD_MS = 100    # Anzeige-Update


class SliderSpin(QWidget):
    """Slider + Spinbox, gekoppelt. Schnittstelle wie QDoubleSpinBox (value/setValue)."""
    STEPS = 1000

    def __init__(self, val, lo, hi, step, dec):
        super().__init__()
        self.lo, self.hi = lo, hi
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self.STEPS)
        self.slider.setMinimumWidth(110)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setSingleStep(step)
        self.spin.setDecimals(dec)
        self.spin.setFixedWidth(80)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(val)

    def _from_slider(self, i):
        self.spin.blockSignals(True)
        self.spin.setValue(self.lo + (self.hi - self.lo) * i / self.STEPS)
        self.spin.blockSignals(False)

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(round((v - self.lo) / (self.hi - self.lo) * self.STEPS))
        self.slider.blockSignals(False)

    def value(self):
        return self.spin.value()

    def setValue(self, v):
        self.spin.setValue(v)
        self._from_spin(self.spin.value())


class Worker(QThread):
    """Führt blockierende Aufrufe (z. B. Verbindungsaufbau) außerhalb des GUI-Threads aus."""
    done = pyqtSignal(object, object)  # (Ergebnis, Fehler)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.done.emit(self.fn(), None)
        except Exception as e:
            self.done.emit(None, e)


def fmt_pose(p):
    return (f"X {p[0]*1000:.1f}  Y {p[1]*1000:.1f}  Z {p[2]*1000:.1f} mm | "
            f"Rx {math.degrees(p[3]):.1f}  Ry {math.degrees(p[4]):.1f}  "
            f"Rz {math.degrees(p[5]):.1f} °")


class JogGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR3e Jog GUI (ur_rtde) - TCP Jog + Pose + Teach/Hold")
        avail = QApplication.primaryScreen().availableGeometry()
        self.resize(min(1280, int(avail.width() * 0.95)), min(780, int(avail.height() * 0.92)))

        self.rtde_c = None
        self.rtde_r = None
        self.jog_dir = np.zeros(6)      # aktuelle Jog-Richtung (x,y,z,rx,ry,rz)
        self.jog_target = None          # integrierte Zielpose im moveJ-Modus
        self.hold_paused = False
        self.hold_cmd = None
        self.hold_speed = 0.0
        self.playing = False
        self.play_idx = 0
        self.play_state = ""
        self.play_t0 = 0.0
        self.freedrive = False
        self.gripper = None
        self._workers = set()
        self.hold_target = None
        self.hold_active = False
        self.points = []                # [(name, [6 floats])]

        self._build_ui()

        self.jog_timer = QTimer(self, interval=JOG_PERIOD_MS)
        self.jog_timer.timeout.connect(self._jog_tick)
        self.hold_timer = QTimer(self, interval=HOLD_PERIOD_MS)
        self.hold_timer.timeout.connect(self._hold_tick)
        self.play_timer = QTimer(self, interval=50)
        self.play_timer.timeout.connect(self._play_tick)
        self.pose_timer = QTimer(self, interval=POSE_PERIOD_MS)
        self.pose_timer.timeout.connect(self._update_pose)
        self._set_connected(False)

    # ------------------------------------------------------------------ UI
    def _spin(self, val, lo, hi, step, dec, suffix=""):
        return SliderSpin(val, lo, hi, step, dec)

    def _form(self, title, rows):
        g = QGroupBox(title)
        f = QFormLayout(g)
        f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        for label, w in rows:
            f.addRow(label, w)
        return g

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)

        # scrollbarer Bereich: nichts wird abgeschnitten, falls das Fenster klein ist
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        root = QWidget()
        scroll.setWidget(root)
        outer.addWidget(scroll, 1)
        main = QVBoxLayout(root)

        # ---- Verbindung (oben, volle Breite)
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

        cols = QHBoxLayout()
        main.addLayout(cols, 1)
        colA, colB, colC = QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        cols.addLayout(colA, 4)
        cols.addLayout(colB, 4)
        cols.addLayout(colC, 5)

        # ================= Spalte A: Jog =================
        self.sp_lin_v = self._spin(0.05, 0.001, 0.5, 0.01, 3)
        self.sp_ang_w = self._spin(0.2, 0.01, 1.5, 0.05, 3)
        self.sp_jog_a = self._spin(0.5, 0.05, 2.0, 0.05, 2)
        self.cb_frame = QComboBox()
        self.cb_frame.addItems(["Tool-KS (TCP)", "Basis-KS"])
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["moveL (linear, speedL)", "moveJ (Gelenkraum, servoJ+IK)"])
        colA.addWidget(self._form("Jog-Parameter", [
            ("Linear v [m/s]:", self.sp_lin_v),
            ("Angular ω [rad/s]:", self.sp_ang_w),
            ("Jog Beschl. [m/s²]:", self.sp_jog_a),
            ("Koordinatensystem:", self.cb_frame),
            ("Jog-Modus:", self.cb_mode),
        ]))

        self.jog_buttons = []
        colA.addWidget(self._jog_group("Jog (Translation)",
                                       [("+X", 0, 1), ("+Y", 1, 1), ("+Z", 2, 1),
                                        ("-X", 0, -1), ("-Y", 1, -1), ("-Z", 2, -1)]))
        colA.addWidget(self._jog_group("Jog (Rotation)",
                                       [("+Rx", 3, 1), ("+Ry", 4, 1), ("+Rz", 5, 1),
                                        ("-Rx", 3, -1), ("-Ry", 4, -1), ("-Rz", 5, -1)]))

        g = QGroupBox("TCP Pose: Ist / Zielpose (mm, °)")
        gl = QGridLayout(g)
        gl.setHorizontalSpacing(4)
        self.pose_labels = []
        self.target_spins = []
        for k, n in enumerate(["X", "Y", "Z", "Rx", "Ry", "Rz"]):
            hd = QLabel(n)
            hd.setAlignment(Qt.AlignmentFlag.AlignCenter)
            gl.addWidget(hd, 0, k + 1)
            lbl = QLabel("--")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.pose_labels.append(lbl)
            gl.addWidget(lbl, 1, k + 1)
            sp = QDoubleSpinBox()
            sp.setRange(-2000, 2000) if k < 3 else sp.setRange(-360, 360)
            sp.setDecimals(1)
            sp.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            sp.setMinimumWidth(56)
            sp.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.target_spins.append(sp)
            gl.addWidget(sp, 2, k + 1)
        gl.addWidget(QLabel("Ist"), 1, 0)
        gl.addWidget(QLabel("Ziel"), 2, 0)
        self.btn_t_cur = QPushButton("Ist übernehmen")
        self.btn_t_move = QPushButton("Zielpose anfahren (moveL)")
        self.btn_t_teach = QPushButton("Als Teach-Punkt")
        self.btn_t_hold = QPushButton("Als Hold-Ziel")
        self.btn_t_cur.clicked.connect(self.target_from_current)
        self.btn_t_move.clicked.connect(self.target_move)
        self.btn_t_teach.clicked.connect(self.target_teach)
        self.btn_t_hold.clicked.connect(self.target_hold)
        gl.addWidget(self.btn_t_cur, 3, 0, 1, 3)
        gl.addWidget(self.btn_t_teach, 3, 3, 1, 4)
        gl.addWidget(self.btn_t_move, 4, 0, 1, 4)
        gl.addWidget(self.btn_t_hold, 4, 4, 1, 3)
        colA.addWidget(g)
        colA.addStretch(1)

        # ================= Spalte B: Bewegung / Hold / Freedrive =================
        self.sp_movel_v = self._spin(0.05, 0.001, 0.5, 0.01, 3)
        self.sp_movel_a = self._spin(0.5, 0.05, 2.0, 0.05, 2)
        self.sp_servo_a = self._spin(1.2, 0.01, 5.0, 0.05, 2)
        self.sp_servo_v = self._spin(0.25, 0.01, 2.0, 0.05, 2)
        self.sp_look = self._spin(0.1, 0.03, 0.2, 0.01, 3)
        self.sp_gain = self._spin(300, 100, 2000, 10, 1)
        colB.addWidget(self._form("Bewegungs-Parameter", [
            ("moveL v [m/s]:", self.sp_movel_v),
            ("moveL a [m/s²]:", self.sp_movel_a),
            ("servoL v:", self.sp_servo_v),
            ("servoL a:", self.sp_servo_a),
            ("servoL lookahead [s]:", self.sp_look),
            ("servoL gain:", self.sp_gain),
        ]))

        g = QGroupBox("Pose Hold / Handführung")
        v = QVBoxLayout(g)
        self.btn_hold = QPushButton("Hold EIN")
        self.btn_hold.setCheckable(True)
        self.btn_hold.toggled.connect(self.toggle_hold)
        self.btn_free = QPushButton("Freedrive EIN")
        self.btn_free.setCheckable(True)
        self.btn_free.toggled.connect(self.toggle_freedrive)
        h = QHBoxLayout()
        h.addWidget(self.btn_hold)
        h.addWidget(self.btn_free)
        self.btn_hold_now = QPushButton("Hold-Ziel = aktuelle Pose")
        self.btn_hold_now.clicked.connect(self.hold_to_current)
        self.lbl_hold = QLabel("Ziel: —")
        self.lbl_hold.setWordWrap(True)
        v.addLayout(h)
        v.addWidget(self.btn_hold_now)
        v.addWidget(QLabel("Aktuelles Hold-Ziel:"))
        v.addWidget(self.lbl_hold)
        colB.addWidget(g)


        # ================= Spalte C: Teach + Greifer =================
        g = QGroupBox("Teach-Punkte")
        v = QVBoxLayout(g)
        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("Name des Teach-Punkts (z. B. P1)")
        self.btn_teach = QPushButton("Teach: aktuelle Pose speichern")
        self.btn_teach.clicked.connect(self.teach_point)
        v.addWidget(self.name_edit)
        v.addWidget(self.btn_teach)
        self.list = QListWidget()
        self.list.setMinimumHeight(150)
        v.addWidget(self.list, 1)
        self.btn_goto = QPushButton("Anfahren (moveL)")
        self.btn_hold_sel = QPushButton("Hold-Ziel = Auswahl")
        self.btn_del = QPushButton("Löschen")
        self.btn_goto.clicked.connect(self.goto_point)
        self.btn_hold_sel.clicked.connect(self.hold_to_selection)
        self.btn_del.clicked.connect(self.delete_point)
        h = QHBoxLayout()
        for b in (self.btn_goto, self.btn_hold_sel, self.btn_del):
            h.addWidget(b)
        v.addLayout(h)
        b1, b2 = QPushButton("CSV speichern"), QPushButton("CSV laden")
        b1.clicked.connect(self.save_csv)
        b2.clicked.connect(self.load_csv)
        h = QHBoxLayout()
        h.addWidget(b1)
        h.addWidget(b2)
        v.addLayout(h)

        self.btn_play = QPushButton("▶ Abspielen")
        self.btn_play.setCheckable(True)
        self.btn_play.setMinimumHeight(32)
        self.btn_play.toggled.connect(self.toggle_play)
        self.chk_loop = QCheckBox("Wiederholen")
        self.sp_pause = SliderSpin(0.5, 0.0, 5.0, 0.1, 1)
        h = QHBoxLayout()
        h.addWidget(self.btn_play, 1)
        h.addWidget(self.chk_loop)
        v.addLayout(h)
        h = QHBoxLayout()
        h.addWidget(QLabel("Pause pro Punkt [s]:"))
        h.addWidget(self.sp_pause, 1)
        v.addLayout(h)
        colC.addWidget(g, 1)

        g = QGroupBox("Greifer")
        gl = QGridLayout(g)
        self.gr_port = QLineEdit("63352")
        self.gr_port.setFixedWidth(65)
        self.btn_gr_conn = QPushButton("Verbinden")
        self.btn_gr_conn.clicked.connect(self.gripper_connect)
        self.btn_gr_act = QPushButton("Aktivieren")
        self.btn_gr_act.clicked.connect(self.gripper_activate)
        self.btn_gr_open = QPushButton("Öffnen")
        self.btn_gr_close = QPushButton("Schließen")
        self.btn_gr_open.clicked.connect(lambda: self._gripper_move(lambda gr: gr.get_open_position()))
        self.btn_gr_close.clicked.connect(lambda: self._gripper_move(lambda gr: gr.get_closed_position()))
        self.sp_gr_pos = SliderSpin(0, 0, 255, 1, 0)
        self.sp_gr_speed = SliderSpin(150, 0, 255, 1, 0)
        self.sp_gr_force = SliderSpin(100, 0, 255, 1, 0)
        self.btn_gr_move = QPushButton("Fahre zu Position")
        self.btn_gr_move.clicked.connect(lambda: self._gripper_move(lambda gr: int(self.sp_gr_pos.value())))
        self.lbl_gr = QLabel("Greifer: getrennt")
        self.lbl_gr.setWordWrap(True)
        gl.addWidget(QLabel("Port:"), 0, 0)
        gl.addWidget(self.gr_port, 0, 1)
        gl.addWidget(self.btn_gr_conn, 0, 2)
        gl.addWidget(self.btn_gr_act, 0, 3)
        gl.addWidget(self.btn_gr_open, 1, 0, 1, 2)
        gl.addWidget(self.btn_gr_close, 1, 2, 1, 2)
        gl.addWidget(QLabel("Position:"), 2, 0)
        gl.addWidget(self.sp_gr_pos, 2, 1, 1, 3)
        gl.addWidget(QLabel("Speed:"), 3, 0)
        gl.addWidget(self.sp_gr_speed, 3, 1, 1, 3)
        gl.addWidget(QLabel("Kraft:"), 4, 0)
        gl.addWidget(self.sp_gr_force, 4, 1, 1, 3)
        gl.addWidget(self.btn_gr_move, 5, 0, 1, 4)
        gl.addWidget(self.lbl_gr, 6, 0, 1, 4)
        colB.addWidget(g)
        colB.addStretch(1)

        # ---- Statuszeile + Stopp (immer sichtbar, außerhalb des Scrollbereichs)
        h = QHBoxLayout()
        self.status = QLabel("Getrennt")
        self.btn_stop = QPushButton("Stopp (speedStop)")
        self.btn_stop.setMinimumHeight(34)
        self.btn_stop.setStyleSheet("background:#8b1a1a; font-weight:bold;")
        self.btn_stop.clicked.connect(self.stop_all)
        h.addWidget(self.status, 1)
        h.addWidget(self.btn_stop)
        outer.addLayout(h)

    def _jog_group(self, title, items):
        g = QGroupBox(title)
        gl = QGridLayout(g)
        for i, (text, axis, sign) in enumerate(items):
            b = QPushButton(text)
            b.setMinimumHeight(34)
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
                  self.btn_hold_now, self.btn_hold_sel, self.btn_free, *self.jog_buttons):
            w.setEnabled(ok)
        self._gripper_enable(self.gripper is not None)
        self.btn_gr_conn.setEnabled(True)
        self.btn_play.setEnabled(ok)
        self.btn_t_cur.setEnabled(ok)
        self.btn_t_move.setEnabled(ok)
        self.status.setText("Verbunden" if ok else "Getrennt")

    def _run_async(self, fn, on_done):
        w = Worker(fn)
        self._workers.add(w)

        def finished(res, err):
            self._workers.discard(w)
            on_done(res, err)
        w.done.connect(finished)
        w.start()

    def connect_robot(self):
        if RTDEControlInterface is None:
            QMessageBox.critical(self, "Fehler", "ur_rtde ist nicht installiert (pip install ur_rtde).")
            return
        ip = self.ip_edit.text().strip()
        self.status.setText(f"Verbinde mit {ip} ...")
        self.btn_connect.setEnabled(False)
        self.ip_edit.setEnabled(False)

        def work():
            r = RTDEReceiveInterface(ip)
            c = RTDEControlInterface(ip)
            return c, r

        def done(res, err):
            if err is not None:
                self.rtde_c = self.rtde_r = None
                self._set_connected(False)
                QMessageBox.critical(self, "Verbindung fehlgeschlagen", str(err))
                return
            self.rtde_c, self.rtde_r = res
            self._set_connected(True)
            self.target_from_current()
            self.pose_timer.start()

        self._run_async(work, done)

    def disconnect_robot(self):
        self.stop_all()
        self.gripper_disconnect()
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
    def _pause_hold(self):
        """Hold kurz aussetzen (servoL und speedL/servoJ dürfen nicht gleichzeitig laufen)."""
        if self.hold_active and not self.hold_paused:
            self.hold_timer.stop()
            self.hold_paused = True
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass

    def _resume_hold(self):
        if self.hold_paused:
            self.hold_paused = False
            if self.hold_active:
                self.hold_to_current()  # neue Pose halten
                self._hold_reset()
                self.hold_timer.start()

    def _jog_start(self, axis, sign):
        if self.freedrive:
            return
        if self.playing:
            self.status.setText("Jog gesperrt: Wiedergabe läuft")
            return
        self._pause_hold()
        d = np.zeros(6)
        d[axis] = sign
        self.jog_dir = d
        self.jog_target = self._current_pose()
        self.jog_timer.start()
        self._jog_tick()

    def _jog_stop(self):
        was_jogging = self.jog_timer.isActive()
        self.jog_timer.stop()
        self.jog_dir = np.zeros(6)
        if self.rtde_c is not None and was_jogging:
            self._jog_stop_safe()

    def _jog_tick(self):
        if self.rtde_c is None:
            return
        tool = self.cb_frame.currentIndex() == 0
        v = self.jog_dir[:3] * self.sp_lin_v.value()
        w = self.jog_dir[3:] * self.sp_ang_w.value()
        try:
            if self.cb_mode.currentIndex() == 0:
                # moveL: kartesisch linear per speedL
                if tool:
                    pose = self._current_pose()
                    R, _ = cv2.Rodrigues(np.array(pose[3:6], dtype=float))
                    v, w = R @ v, R @ w
                self.rtde_c.speedL([*v, *w], self.sp_jog_a.value(), JOG_PERIOD_MS * 3 / 1000)
            else:
                # moveJ: Zielpose integrieren, per IK in Gelenkwinkel, servoJ
                dt = JOG_PERIOD_MS / 1000
                t = self.jog_target
                R, _ = cv2.Rodrigues(np.array(t[3:6], dtype=float))
                Rd, _ = cv2.Rodrigues(np.array(w * dt, dtype=float))
                if tool:
                    t[:3] = list(np.array(t[:3]) + R @ v * dt)
                    Rn = R @ Rd
                else:
                    t[:3] = list(np.array(t[:3]) + v * dt)
                    Rn = Rd @ R
                t[3:6] = [float(x) for x in cv2.Rodrigues(Rn)[0].ravel()]
                q = self.rtde_c.getInverseKinematics(t, list(self.rtde_r.getActualQ()))
                if not q:
                    raise RuntimeError("keine IK-Lösung (Zielpose nicht erreichbar)")
                self.rtde_c.servoJ(list(q), 0.0, 0.0, dt, self.sp_look.value(), self.sp_gain.value())
        except Exception as e:
            self.status.setText(f"Jog-Fehler: {e}")
            self.jog_timer.stop()
            self._jog_stop_safe()

    def _jog_stop_safe(self):
        try:
            self.rtde_c.speedStop()
            self.rtde_c.servoStop()
        except Exception as e:
            self.status.setText(f"Fehler: {e}")
        self._resume_hold()

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
        if on and self.playing:
            self.status.setText("Hold gesperrt: Wiedergabe läuft")
            self.btn_hold.blockSignals(True)
            self.btn_hold.setChecked(False)
            self.btn_hold.blockSignals(False)
            return
        if self.rtde_c is None:
            self.btn_hold.blockSignals(True)
            self.btn_hold.setChecked(False)
            self.btn_hold.blockSignals(False)
            return
        if on:
            if self.freedrive:
                self.btn_hold.setChecked(False)
                return
            if self.hold_target is None:
                self.hold_to_current()
            self.hold_active = True
            self._hold_reset()
            self.btn_hold.setText("Hold AUS")
            self.hold_timer.start()
        else:
            self.hold_active = False
            self.hold_paused = False
            self.hold_timer.stop()
            self.btn_hold.setText("Hold EIN")
            try:
                self.rtde_c.servoStop()
            except Exception:
                pass

    def _hold_reset(self):
        """Hold-Sollwert startet an der aktuellen Pose, Geschwindigkeit 0."""
        self.hold_cmd = self._current_pose()
        self.hold_speed = 0.0

    def _hold_tick(self):
        """Fährt einen Sollwert mit servoL v / servoL a (linear) bzw. Angular ω (Rotation)
        zum Hold-Ziel. ur_rtde-servoL ignoriert v/a selbst, daher begrenzen wir hier."""
        try:
            dt = HOLD_PERIOD_MS / 1000
            v_lim = max(self.sp_servo_v.value(), 1e-4)
            a_lim = max(self.sp_servo_a.value(), 1e-4)
            w_lim = max(self.sp_ang_w.value(), 1e-4)
            c = np.array(self.hold_cmd, dtype=float)
            t = np.array(self.hold_target, dtype=float)

            err_lin = t[:3] - c[:3]
            Rc, _ = cv2.Rodrigues(c[3:6])
            Rt, _ = cv2.Rodrigues(t[3:6])
            err_rot = cv2.Rodrigues(Rt @ Rc.T)[0].ravel()   # Basis-KS Drehvektor
            d_lin = float(np.linalg.norm(err_lin))
            d_ang = float(np.linalg.norm(err_rot))
            # Winkel in "äquivalente Strecke" umrechnen (v_lim entspricht w_lim)
            D = max(d_lin, d_ang * v_lim / w_lim)
            if D < 1e-6:
                self.hold_cmd = list(t)
                self.hold_speed = 0.0
            else:
                v_brake = math.sqrt(2 * a_lim * D)          # rechtzeitig abbremsen
                self.hold_speed = min(self.hold_speed + a_lim * dt, v_lim, v_brake)
                f = min(1.0, self.hold_speed * dt / D)
                c[:3] += err_lin * f
                Rn = cv2.Rodrigues(err_rot * f)[0] @ Rc
                c[3:6] = cv2.Rodrigues(Rn)[0].ravel()
                self.hold_cmd = [float(x) for x in c]
            self.rtde_c.servoL(self.hold_cmd, v_lim, a_lim, dt, self.sp_look.value(), self.sp_gain.value())
        except Exception as e:
            self.status.setText(f"Hold-Fehler: {e}")
            self.btn_hold.setChecked(False)

    # ------------------------------------------------- Zielpose eingeben
    def _target_pose(self):
        v = [sp.value() for sp in self.target_spins]
        return [v[0] / 1000, v[1] / 1000, v[2] / 1000,
                math.radians(v[3]), math.radians(v[4]), math.radians(v[5])]

    def target_from_current(self):
        """Ist-Pose in die Eingabefelder übernehmen (keine Bewegung)."""
        if self.rtde_r is None:
            return
        p = self._current_pose()
        vals = [p[0] * 1000, p[1] * 1000, p[2] * 1000,
                math.degrees(p[3]), math.degrees(p[4]), math.degrees(p[5])]
        for sp, v in zip(self.target_spins, vals):
            sp.setValue(v)

    def target_move(self):
        """Erst auf Knopfdruck: eingegebene Zielpose per moveL anfahren."""
        if self.rtde_c is None:
            return
        if self.playing:
            self.status.setText("Wiedergabe läuft - erst stoppen")
            return
        if self.freedrive:
            self.status.setText("Erst Freedrive ausschalten")
            return
        if self.hold_active:
            self.btn_hold.setChecked(False)
        try:
            self.rtde_c.moveL(self._target_pose(), self.sp_movel_v.value(), self.sp_movel_a.value(), True)
            self.status.setText("Fahre zur eingegebenen Zielpose ...")
        except Exception as e:
            self.status.setText(f"moveL-Fehler: {e}")

    def target_teach(self):
        name = self.name_edit.text().strip() or f"P{len(self.points) + 1}"
        self.points.append((name, self._target_pose()))
        self.name_edit.clear()
        self._refresh_list()

    def target_hold(self):
        self._set_hold_target(self._target_pose())

    # -------------------------------------------------------- Teach-Punkte
    def _refresh_list(self):
        self.list.clear()
        for name, p in self.points:
            self.list.addItem(f"{name}:  {p[0]*1000:.0f} / {p[1]*1000:.0f} / {p[2]*1000:.0f} mm  |  "
                              f"{math.degrees(p[3]):.0f} / {math.degrees(p[4]):.0f} / {math.degrees(p[5]):.0f} °")

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
        if self.playing:
            self.status.setText("Wiedergabe läuft - erst stoppen")
            return
        i = self.list.currentRow()
        if i < 0:
            self.status.setText("Kein Punkt ausgewählt")
            return
        if self.freedrive:
            self.status.setText("Erst Freedrive ausschalten")
            return
        if self.hold_active:
            self.btn_hold.setChecked(False)  # Hold aus, sonst Konflikt mit moveL
        name, p = self.points[i]
        try:
            self.rtde_c.moveL(p, self.sp_movel_v.value(), self.sp_movel_a.value(), True)
            self.status.setText(f"Fahre zu {name} ...")
        except Exception as e:
            self.status.setText(f"moveL-Fehler: {e}")

    def toggle_play(self, on):
        if on:
            if self.rtde_c is None or not self.points:
                self.status.setText("Keine Teach-Punkte zum Abspielen")
                self._set_play_button(False)
                return
            if self.freedrive:
                self.status.setText("Erst Freedrive ausschalten")
                self._set_play_button(False)
                return
            if self.hold_active:
                self.btn_hold.setChecked(False)
            self.playing = True
            self.btn_play.setText("■ Wiedergabe stoppen")
            self.play_idx = max(0, self.list.currentRow())
            self._play_start_move()
            self.play_timer.start()
        else:
            self._play_end("Wiedergabe gestoppt", stop_robot=True)

    def _set_play_button(self, on):
        self.btn_play.blockSignals(True)
        self.btn_play.setChecked(on)
        self.btn_play.blockSignals(False)
        self.btn_play.setText("■ Wiedergabe stoppen" if on else "▶ Abspielen")

    def _play_start_move(self):
        name, pose = self.points[self.play_idx]
        self.list.setCurrentRow(self.play_idx)
        try:
            self.rtde_c.moveL(pose, self.sp_movel_v.value(), self.sp_movel_a.value(), True)
        except Exception as e:
            self._play_end(f"Wiedergabe-Fehler: {e}", stop_robot=False)
            return
        self.play_state = "moving"
        self.play_t0 = time.monotonic()
        self.status.setText(f"Wiedergabe: {name} ({self.play_idx + 1}/{len(self.points)})")

    def _play_tick(self):
        if not self.playing:
            return
        try:
            if self.play_state == "moving":
                if time.monotonic() - self.play_t0 < 0.15:   # Async-Start abwarten
                    return
                if self.rtde_c.getAsyncOperationProgress() >= 0:
                    return                                   # läuft noch
                self.play_state = "waiting"
                self.play_t0 = time.monotonic()
            elif self.play_state == "waiting":
                if time.monotonic() - self.play_t0 < self.sp_pause.value():
                    return
                self.play_idx += 1
                if self.play_idx >= len(self.points):
                    if not self.chk_loop.isChecked():
                        self._play_end("Wiedergabe beendet", stop_robot=False)
                        return
                    self.play_idx = 0
                self._play_start_move()
        except Exception as e:
            self._play_end(f"Wiedergabe-Fehler: {e}", stop_robot=True)

    def _play_end(self, msg, stop_robot):
        was = self.playing
        self.playing = False
        self.play_state = ""
        self.play_timer.stop()
        self._set_play_button(False)
        if was and stop_robot and self.rtde_c is not None:
            try:
                self.rtde_c.stopL(2.0)
            except Exception:
                pass
        if was:
            self.status.setText(msg)

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

    # ----------------------------------------------------------- Freedrive
    def toggle_freedrive(self, on):
        if self.rtde_c is None:
            return
        if on and self.playing:
            self._play_end("Wiedergabe gestoppt", stop_robot=True)
        try:
            if on:
                if self.hold_active:
                    self.btn_hold.setChecked(False)
                self.jog_timer.stop()
                self.rtde_c.speedStop()
                self.freedrive = bool(self.rtde_c.teachMode())
                if not self.freedrive:
                    raise RuntimeError("teachMode fehlgeschlagen")
                self.btn_free.setText("Freedrive AUS")
                self.status.setText("Freedrive aktiv - Roboter von Hand führbar")
            else:
                self.rtde_c.endTeachMode()
                self.freedrive = False
                self.btn_free.setText("Freedrive EIN")
                self.status.setText("Freedrive beendet")
        except Exception as e:
            self.freedrive = False
            self.btn_free.blockSignals(True)
            self.btn_free.setChecked(False)
            self.btn_free.blockSignals(False)
            self.btn_free.setText("Freedrive EIN")
            self.status.setText(f"Freedrive-Fehler: {e}")
        for b in self.jog_buttons:
            b.setEnabled(not self.freedrive)

    # -------------------------------------------------------------- Greifer
    def _gr_sf(self):
        return int(self.sp_gr_speed.value()), int(self.sp_gr_force.value())

    def _gripper_enable(self, ok):
        for w in (self.btn_gr_act, self.btn_gr_open, self.btn_gr_close, self.btn_gr_move):
            w.setEnabled(ok)

    def gripper_connect(self):
        if RobotiqGripper is None:
            QMessageBox.critical(self, "Fehler", "robotiq_gripper.py nicht gefunden (muss neben P06.py liegen).")
            return
        try:
            port = int(self.gr_port.text())
        except ValueError:
            self.status.setText("Ungültiger Greifer-Port")
            return
        ip = self.ip_edit.text().strip()
        self.lbl_gr.setText("Greifer: verbinde ...")
        self.btn_gr_conn.setEnabled(False)

        def work():
            gr = RobotiqGripper()
            gr.connect(ip, port)
            return gr

        def done(gr, err):
            self.btn_gr_conn.setEnabled(True)
            if err is not None:
                self.gripper = None
                self.lbl_gr.setText(f"Greifer: Verbindung fehlgeschlagen ({err!r}) - Port 63352 "
                                    "(Robotiq-URCap) prüfen, Greifer eingeschaltet?")
                return
            self.gripper = gr
            self._gripper_enable(True)
            self.lbl_gr.setText(f"Greifer: verbunden (Port {port})")

        self._run_async(work, done)

    def gripper_disconnect(self):
        if self.gripper is not None:
            try:
                self.gripper.disconnect()
            except Exception:
                pass
        self.gripper = None
        self._gripper_enable(False)
        self.lbl_gr.setText("Greifer: getrennt")

    def gripper_activate(self):
        # blockiert bis aktiv (ca. 1-3 s) -> im Hintergrund-Thread
        self._gripper_cmd(lambda gr: gr.activate(auto_calibrate=False), "aktiviere ...")

    def _gripper_move(self, target_fn):
        speed, force = self._gr_sf()
        self._gripper_cmd(lambda gr: gr.move_and_wait_for_pos(target_fn(gr), speed, force), "fahre ...")

    def _gripper_cmd(self, fn, msg=""):
        """Greifer-Befehl im Hintergrund, damit die GUI nicht blockiert."""
        gr = self.gripper
        if gr is None:
            return
        self._gripper_enable(False)
        self.lbl_gr.setText(f"Greifer: {msg}")

        def work():
            fn(gr)
            return gr.get_current_position()

        def done(pos, err):
            self._gripper_enable(self.gripper is not None)
            if err is not None:
                self.lbl_gr.setText(f"Greifer-Fehler: {err!r}")
            else:
                self.lbl_gr.setText(f"Greifer: Position {pos}")

        self._run_async(work, done)

    # --------------------------------------------------------------- Stopp
    def stop_all(self):
        self.jog_timer.stop()
        self.jog_dir = np.zeros(6)
        self.hold_paused = False
        self._play_end("Wiedergabe gestoppt", stop_robot=False)
        if self.hold_active:
            self.btn_hold.setChecked(False)
        if self.freedrive:
            self.btn_free.setChecked(False)
        if self.rtde_c is not None:
            try:
                self.rtde_c.speedStop()
                self.rtde_c.servoStop()
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
