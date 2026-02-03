
# filepath: tello_altitude_hmi.py

from __future__ import annotations

import csv
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import messagebox, ttk

from djitellopy import Tello


# -----------------------------
# Utilidades
# -----------------------------
class MedianFilter:
    def __init__(self, n: int = 5) -> None:
        self.buf: Deque[float] = deque(maxlen=n)

    def update(self, x: float) -> float:
        self.buf.append(float(x))
        return float(np.median(self.buf))


def clip(v: float, vmin: float, vmax: float) -> float:
    return float(np.clip(v, vmin, vmax))


def to_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        s = str(val).strip().lower()
        if s in ("", "none"):
            return None
        s = s.replace("cm", "").replace("mm", "").strip()
        return float(s)
    except Exception:
        return None


# -----------------------------
# PID digital
# -----------------------------
@dataclass
class PIDGains:
    Kp: float = 1.2
    Ki: float = 0.03
    Kd: float = 0.10


class PID:
    def __init__(
        self,
        gains: PIDGains,
        umin: float = -100.0,
        umax: float = 100.0,
        deriv_filter_alpha: float = 0.25,
        aw_gain: float = 0.2,
    ) -> None:
        self.g = gains
        self.umin = float(umin)
        self.umax = float(umax)
        self.alpha = float(deriv_filter_alpha)
        self.aw_gain = float(aw_gain)

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t: Optional[float] = None
        self.d_filt = 0.0

    def reset(self) -> None:
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_t = None
        self.d_filt = 0.0

    def update(self, error: float, tnow: float) -> Tuple[float, float]:
        if self.prev_t is None:
            self.prev_t = tnow
            self.prev_error = error
            return 0.0, 0.0

        dt = tnow - self.prev_t
        if dt <= 1e-6:
            return 0.0, dt

        d_raw = (error - self.prev_error) / dt
        self.d_filt = (1.0 - self.alpha) * self.d_filt + self.alpha * d_raw

        u_unsat = self.g.Kp * error + self.g.Ki * self.integral + self.g.Kd * self.d_filt
        u_sat = clip(u_unsat, self.umin, self.umax)

        self.integral += error * dt + self.aw_gain * (u_sat - u_unsat) * dt

        self.prev_error = error
        self.prev_t = tnow
        return u_sat, dt


# -----------------------------
# Métricas (Mp y Ts)
# -----------------------------
def compute_overshoot_percent(h: np.ndarray, ref: float) -> float:
    if len(h) == 0 or abs(ref) < 1e-6:
        return 0.0
    hmax = float(np.max(h))
    return max(0.0, (hmax - ref) / abs(ref) * 100.0)


def compute_settling_time(
    t: np.ndarray,
    h: np.ndarray,
    ref: float,
    band_ratio: float = 0.02,
    hold_s: float = 2.0,
) -> Optional[float]:
    if len(t) < 5 or abs(ref) < 1e-6:
        return None
    band = band_ratio * abs(ref)
    dt = float(np.mean(np.diff(t))) if len(t) > 1 else 0.0
    if dt <= 0:
        return None
    hold_n = max(1, int(hold_s / dt))
    err = np.abs(h - ref)
    inside = err <= band
    for i in range(len(inside)):
        j = i + hold_n
        if j <= len(inside) and bool(np.all(inside[i:j])):
            return float(t[i])
    return None


# -----------------------------
# Estimar Tu por autocorrelación (para relay)
# -----------------------------
def estimate_tu_autocorr(t: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(t) < 150:
        return None
    y = y.astype(float)
    y = y - np.mean(y)

    # detrend leve
    x = np.linspace(-1.0, 1.0, len(y))
    p = np.polyfit(x, y, deg=1)
    y = y - (p[0] * x + p[1])

    if float(np.std(y)) < 0.8:
        return None

    ac = np.correlate(y, y, mode="full")
    ac = ac[ac.size // 2 :]
    if ac[0] <= 1e-9:
        return None
    ac = ac / ac[0]

    dt = float(np.mean(np.diff(t)))
    if dt <= 0:
        return None

    min_lag = int(0.5 / dt)
    max_lag = min(len(ac) - 2, int(8.0 / dt))
    if min_lag >= max_lag:
        return None

    best_lag = None
    best_val = -1.0
    for k in range(max(min_lag, 2), max_lag):
        if ac[k] > ac[k - 1] and ac[k] > ac[k + 1] and float(ac[k]) > best_val:
            best_val = float(ac[k])
            best_lag = k

    if best_lag is None or best_val < 0.35:
        return None
    return float(best_lag * dt)


# -----------------------------
# APP
# -----------------------------
class TelloAltitudeApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("HMI Control de Altura - DJI Tello (PID + ZN)")

        # Setpoint: ahora es DELTA sobre takeoff
        self.SP_DELTA_MIN = 0.0
        self.SP_DELTA_MAX = 150.0

        # sensor validación
        self.H_VALID_MIN = 5.0
        self.H_VALID_MAX = 300.0

        # control authority
        self.U_MIN = -100.0
        self.U_MAX = 100.0

        self.DT_CTRL = 0.10
        self.LOG_MAXLEN = 4000

        # relay tuning params (robustos)
        self.relay_amp = 40.0      # d (cmd)
        self.relay_hyst = 1.0      # histéresis en cm
        self.relay_timeout_s = 45.0

        # tello
        self.tello: Optional[Tello] = None
        self.connected = False
        self.flying = False

        # threads
        self.mode = "IDLE"  # IDLE | KU_SEARCH | CONTROL
        self.stop_event = threading.Event()
        self.ku_thread: Optional[threading.Thread] = None
        self.ctrl_thread: Optional[threading.Thread] = None

        # UI queue (thread-safe)
        self.uiq: "queue.Queue[Callable[[], None]]" = queue.Queue()

        # shared state
        self.lock = threading.Lock()
        self.height_cm = 0.0
        self.height_valid = False

        self.h0_takeoff: Optional[float] = None  # baseline
        self.sp_delta_cm = 100.0                 # Δ sobre h0
        self.ref_abs_cm = 0.0

        self.error_cm = 0.0
        self.u_cmd = 0.0
        self.battery: Optional[int] = None

        # sensor health
        self.last_good_height: Optional[float] = None
        self.last_good_time: Optional[float] = None
        self.takeoff_time: Optional[float] = None

        self.medf = MedianFilter(n=5)

        # ZN outputs
        self.Ku: Optional[float] = None
        self.Tu: Optional[float] = None

        # PID
        self.pid = PID(PIDGains(), umin=self.U_MIN, umax=self.U_MAX)

        # logs
        self.t_log: Deque[float] = deque(maxlen=self.LOG_MAXLEN)
        self.h_log: Deque[float] = deque(maxlen=self.LOG_MAXLEN)
        self.r_log: Deque[float] = deque(maxlen=self.LOG_MAXLEN)
        self.u_log: Deque[float] = deque(maxlen=self.LOG_MAXLEN)

        # métricas en vivo (desde último step)
        self.step_start_time: Optional[float] = None
        self.step_ref: Optional[float] = None
        self.mp_live: float = 0.0
        self.ts_live: Optional[float] = None

        self._build_gui()
        self._build_plot()
        self._ui_pump()
        self._ui_update()
        self._battery_poll()

    # ---------- UI thread-safe ----------
    def _post_ui(self, fn: Callable[[], None]) -> None:
        self.uiq.put(fn)

    def _ui_pump(self) -> None:
        while True:
            try:
                fn = self.uiq.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
            except Exception:
                pass
        self.root.after(50, self._ui_pump)

    # ---------- tello helpers ----------
    def _ensure_sdk_command(self) -> None:
        if self.tello is None:
            return
        try:
            self.tello.send_command_with_return("command")
        except Exception:
            pass

    def _safe_rc_zero(self, times: int = 5, dt: float = 0.08) -> None:
        if self.tello is None:
            return
        for _ in range(times):
            try:
                self.tello.send_rc_control(0, 0, 0, 0)
            except Exception:
                pass
            time.sleep(dt)

    def _send_vz(self, vz_cmd: float) -> None:
        if self.tello is None:
            return
        vz = int(clip(vz_cmd, self.U_MIN, self.U_MAX))
        self.tello.send_rc_control(0, 0, vz, 0)

    def _read_height_cm(self) -> Tuple[Optional[float], bool]:
        if self.tello is None:
            return None, False

        h_cm: Optional[float] = None
        st = None
        try:
            st = self.tello.get_current_state()
        except Exception:
            st = None

        if isinstance(st, dict):
            tof = to_float(st.get("tof"))
            if tof is not None and tof > 0:
                h_cm = tof / 10.0 if tof > 300 else tof
            if h_cm is None:
                h_bar = to_float(st.get("h"))
                if h_bar is not None:
                    h_cm = h_bar

        if h_cm is None:
            try:
                h_cm = float(self.tello.get_height())
            except Exception:
                h_cm = None

        if h_cm is None or not (self.H_VALID_MIN <= h_cm <= self.H_VALID_MAX):
            return None, False

        return self.medf.update(h_cm), True

    # ---------- takeoff baseline ----------
    def _capture_takeoff_baseline(self) -> bool:
        """
        Captura h0 (altura estabilizada post-takeoff).
        Retorna True si se logró.
        """
        samples: list[float] = []
        t0 = time.perf_counter()
        while (time.perf_counter() - t0) < 3.0 and len(samples) < 12:
            h, ok = self._read_height_cm()
            if ok and h is not None:
                samples.append(h)
                self.last_good_height = h
                self.last_good_time = time.perf_counter()
            time.sleep(0.1)

        if len(samples) < 6:
            return False

        h0 = float(np.median(samples[-6:]))
        with self.lock:
            self.h0_takeoff = h0
            self.ref_abs_cm = h0 + self.sp_delta_cm
        return True

    # ---------- step metrics ----------
    def _mark_step(self, ref_abs: float) -> None:
        self.step_start_time = time.perf_counter()
        self.step_ref = float(ref_abs)
        self.mp_live = 0.0
        self.ts_live = None

    def _compute_live_metrics(self) -> None:
        if self.step_start_time is None or self.step_ref is None:
            return
        with self.lock:
            t = np.array(self.t_log, dtype=float)
            h = np.array(self.h_log, dtype=float)
            r = np.array(self.r_log, dtype=float)

        if len(t) < 10:
            return

        # métrica respecto al ref objetivo actual (último)
        ref = float(self.step_ref)
        self.mp_live = compute_overshoot_percent(h, ref)
        self.ts_live = compute_settling_time(t, h, ref, band_ratio=0.02, hold_s=2.0)

    # ---------- threads stop ----------
    def _stop_all_threads(self) -> None:
        self.stop_event.set()
        self.mode = "IDLE"

        if self.ctrl_thread is not None and self.ctrl_thread.is_alive():
            self.ctrl_thread.join(timeout=2.0)
        self.ctrl_thread = None

        if self.ku_thread is not None and self.ku_thread.is_alive():
            self.ku_thread.join(timeout=2.0)
        self.ku_thread = None

        self.stop_event.clear()

    # ---------------- GUI ----------------
    def _build_gui(self) -> None:
        style = ttk.Style()
        style.configure("Danger.TButton", foreground="red")

        frm = ttk.Frame(self.root, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(frm)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        frm.columnconfigure(1, weight=1)

        ttk.Label(left, text="Conexión").grid(row=0, column=0, sticky="w")

        self.btn_connect = ttk.Button(left, text="Conectar", command=self.connect)
        self.btn_connect.grid(row=1, column=0, sticky="ew", pady=2)

        self.btn_reconnect = ttk.Button(left, text="Reconectar", command=self.reconnect, state="disabled")
        self.btn_reconnect.grid(row=2, column=0, sticky="ew", pady=2)

        self.btn_takeoff = ttk.Button(left, text="Despegar", command=self.takeoff, state="disabled")
        self.btn_takeoff.grid(row=3, column=0, sticky="ew", pady=2)

        self.btn_land = ttk.Button(left, text="Aterrizar", command=self.land, state="disabled")
        self.btn_land.grid(row=4, column=0, sticky="ew", pady=2)

        self.btn_emg = ttk.Button(left, text="EMERGENCIA (Land)", command=self.emergency_land, style="Danger.TButton")
        self.btn_emg.grid(row=5, column=0, sticky="ew", pady=(6, 8))

        ttk.Separator(left).grid(row=6, column=0, sticky="ew", pady=6)

        ttk.Label(left, text="Setpoint Δaltura sobre takeoff [cm]").grid(row=7, column=0, sticky="w")
        self.sp_var = tk.DoubleVar(value=100.0)
        self.sp_scale = ttk.Scale(
            left,
            from_=self.SP_DELTA_MIN,
            to=self.SP_DELTA_MAX,
            orient="horizontal",
            variable=self.sp_var,
            command=self._on_sp_change,
        )
        self.sp_scale.grid(row=8, column=0, sticky="ew")

        self.sp_entry = ttk.Entry(left, width=8)
        self.sp_entry.insert(0, "100.0")
        self.sp_entry.grid(row=9, column=0, sticky="w", pady=2)
        ttk.Button(left, text="Aplicar Setpoint", command=self.apply_setpoint_entry).grid(row=9, column=0, sticky="e")

        self.lbl_refabs = ttk.Label(left, text="h0: — cm | ref_abs: — cm")
        self.lbl_refabs.grid(row=10, column=0, sticky="w")

        ttk.Separator(left).grid(row=11, column=0, sticky="ew", pady=6)

        ttk.Label(left, text="Auto-tuning (Ku, Tu) - Relay").grid(row=12, column=0, sticky="w")
        self.btn_ku = ttk.Button(left, text="Iniciar búsqueda Ku", command=self.start_ku_search, state="disabled")
        self.btn_ku.grid(row=13, column=0, sticky="ew", pady=2)

        self.btn_stop_ku = ttk.Button(left, text="Detener Ku", command=self.stop_ku_search, state="disabled")
        self.btn_stop_ku.grid(row=14, column=0, sticky="ew", pady=2)

        self.lbl_ku = ttk.Label(left, text="Ku: —   Tu: —")
        self.lbl_ku.grid(row=15, column=0, sticky="w")

        self.btn_apply_pid = ttk.Button(left, text="Calcular PID (ZN)", command=self.apply_zn_pid, state="disabled")
        self.btn_apply_pid.grid(row=16, column=0, sticky="ew", pady=2)

        ttk.Separator(left).grid(row=17, column=0, sticky="ew", pady=6)
        ttk.Label(left, text="PID (manual)").grid(row=18, column=0, sticky="w")

        pid_grid = ttk.Frame(left)
        pid_grid.grid(row=19, column=0, sticky="ew")

        ttk.Label(pid_grid, text="Kp").grid(row=0, column=0)
        ttk.Label(pid_grid, text="Ki").grid(row=0, column=1)
        ttk.Label(pid_grid, text="Kd").grid(row=0, column=2)

        self.kp_var = tk.DoubleVar(value=self.pid.g.Kp)
        self.ki_var = tk.DoubleVar(value=self.pid.g.Ki)
        self.kd_var = tk.DoubleVar(value=self.pid.g.Kd)

        ttk.Entry(pid_grid, textvariable=self.kp_var, width=7).grid(row=1, column=0, padx=2)
        ttk.Entry(pid_grid, textvariable=self.ki_var, width=7).grid(row=1, column=1, padx=2)
        ttk.Entry(pid_grid, textvariable=self.kd_var, width=7).grid(row=1, column=2, padx=2)

        ttk.Button(left, text="Aplicar PID manual", command=self.apply_manual_pid).grid(row=20, column=0, sticky="ew", pady=2)

        self.btn_start_ctrl = ttk.Button(left, text="Iniciar Control", command=self.start_control, state="disabled")
        self.btn_start_ctrl.grid(row=21, column=0, sticky="ew", pady=2)

        self.btn_stop_ctrl = ttk.Button(left, text="Detener Control", command=self.stop_control, state="disabled")
        self.btn_stop_ctrl.grid(row=22, column=0, sticky="ew", pady=2)

        ttk.Separator(left).grid(row=23, column=0, sticky="ew", pady=6)
        ttk.Label(left, text="Telemetría + Métricas").grid(row=24, column=0, sticky="w")

        self.lbl_tel = ttk.Label(left, text="h: — | e: — | u: — | bat: — | mode: IDLE")
        self.lbl_tel.grid(row=25, column=0, sticky="w")

        self.lbl_metrics = ttk.Label(left, text="Mp: — % | Ts: — s")
        self.lbl_metrics.grid(row=26, column=0, sticky="w")

        ttk.Button(left, text="Guardar CSV + Métricas", command=self.export_report_csv).grid(row=27, column=0, sticky="ew", pady=(8, 0))

        right = ttk.Frame(frm)
        right.grid(row=0, column=1, sticky="nsew")
        self.right = right

    def _build_plot(self) -> None:
        self.fig, self.ax = plt.subplots(figsize=(7.5, 4.5))
        self.ax.set_title("Altura vs Referencia + Control")
        self.ax.set_xlabel("Tiempo [s]")
        self.ax.set_ylabel("Altura [cm]")
        self.ax.grid(True)

        (self.line_h,) = self.ax.plot([], [], label="Altura")
        (self.line_r,) = self.ax.plot([], [], "--", label="Referencia abs")

        self.ax2 = self.ax.twinx()
        (self.line_u,) = self.ax2.plot([], [], label="u (vz cmd)")
        self.ax2.set_ylabel("u [cmd]")

        lines = [self.line_h, self.line_r, self.line_u]
        labels = [l.get_label() for l in lines]
        self.ax.legend(lines, labels, loc="upper right")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    # ---------------- Connection ----------------
    def connect(self) -> None:
        if self.connected:
            return
        try:
            self.tello = Tello()
            self.tello.connect()
            self._ensure_sdk_command()
            self.battery = int(self.tello.get_battery())
            self.connected = True
            messagebox.showinfo("OK", f"Conectado. Batería: {self.battery}%")

            self.btn_takeoff.config(state="normal")
            self.btn_reconnect.config(state="normal")
            self.btn_ku.config(state="normal")
            self.btn_start_ctrl.config(state="normal")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo conectar: {e}")

    def reconnect(self) -> None:
        try:
            self._stop_all_threads()
            if self.tello is not None:
                try:
                    self._safe_rc_zero()
                except Exception:
                    pass
                try:
                    self.tello.end()
                except Exception:
                    pass
            self.tello = None
            self.connected = False
            self.flying = False
            self.battery = None

            time.sleep(0.3)
            self.connect()
        except Exception as e:
            messagebox.showerror("Error", f"Reconectar falló: {e}")

    # ---------------- Flight ----------------
    def takeoff(self) -> None:
        if not self.connected:
            messagebox.showwarning("Aviso", "Primero conecta.")
            return
        if self.flying:
            return
        try:
            self._stop_all_threads()
            self._ensure_sdk_command()
            self.tello.takeoff()
            self.flying = True
            self.takeoff_time = time.perf_counter()

            ok = self._capture_takeoff_baseline()
            if not ok:
                messagebox.showwarning("Sensor", "No pude capturar h0 (takeoff). Reintenta o cambia superficie/luz.")
            else:
                with self.lock:
                    ref = self.ref_abs_cm
                self._mark_step(ref)

            self.btn_land.config(state="normal")
            self.btn_takeoff.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo despegar: {e}")

    def land(self) -> None:
        if not self.connected:
            return
        try:
            self._stop_all_threads()
            self._safe_rc_zero(times=6, dt=0.08)
            if self.flying and self.tello is not None:
                self.tello.land()
            self.flying = False
            time.sleep(0.4)
            self._ensure_sdk_command()

            self.btn_takeoff.config(state="normal")
            self.btn_land.config(state="disabled")
            self.btn_stop_ctrl.config(state="disabled")
            self.btn_stop_ku.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo aterrizar: {e}")

    def emergency_land(self) -> None:
        try:
            self._stop_all_threads()
            self._safe_rc_zero(times=6, dt=0.06)
            if self.tello is not None:
                try:
                    self.tello.land()
                except Exception:
                    pass
            self.flying = False
            self.mode = "IDLE"
            messagebox.showwarning("EMERGENCIA", "LAND enviado.")
            self.btn_takeoff.config(state="normal")
            self.btn_land.config(state="disabled")
        except Exception as e:
            messagebox.showerror("Error", f"Emergencia falló: {e}")

    # ---------------- Setpoint (Δ) ----------------
    def _on_sp_change(self, _: str) -> None:
        val = clip(float(self.sp_var.get()), self.SP_DELTA_MIN, self.SP_DELTA_MAX)
        with self.lock:
            self.sp_delta_cm = val
            if self.h0_takeoff is not None:
                self.ref_abs_cm = self.h0_takeoff + self.sp_delta_cm
        self.sp_entry.delete(0, tk.END)
        self.sp_entry.insert(0, f"{val:.1f}")

    def apply_setpoint_entry(self) -> None:
        try:
            val = clip(float(self.sp_entry.get()), self.SP_DELTA_MIN, self.SP_DELTA_MAX)
            self.sp_var.set(val)
            with self.lock:
                self.sp_delta_cm = val
                if self.h0_takeoff is not None:
                    self.ref_abs_cm = self.h0_takeoff + self.sp_delta_cm
                    ref = self.ref_abs_cm
                else:
                    ref = None
            if ref is not None:
                self._mark_step(ref)
        except Exception:
            messagebox.showerror("Error", "Setpoint inválido.")

    # ---------------- Auto-tuning Ku/Tu (Relay) ----------------
    def start_ku_search(self) -> None:
        if not (self.connected and self.flying):
            messagebox.showwarning("Aviso", "Conecta y despega primero.")
            return
        if self.h0_takeoff is None:
            messagebox.showwarning("Aviso", "No hay h0. Despega y espera a que capture takeoff.")
            return
        if self.mode != "IDLE":
            messagebox.showwarning("Aviso", "Detén el modo actual antes.")
            return

        self._stop_all_threads()
        self.stop_event.clear()
        self.mode = "KU_SEARCH"
        self.Ku, self.Tu = None, None

        with self.lock:
            self.t_log.clear()
            self.h_log.clear()
            self.r_log.clear()
            self.u_log.clear()
            ref = self.ref_abs_cm

        self._mark_step(ref)

        self.btn_stop_ku.config(state="normal")
        self.btn_apply_pid.config(state="disabled")

        self.ku_thread = threading.Thread(target=self._relay_tune_thread, daemon=True)
        self.ku_thread.start()

    def stop_ku_search(self) -> None:
        if self.mode != "KU_SEARCH":
            return
        self.mode = "IDLE"
        self.stop_event.set()
        if self.ku_thread is not None and self.ku_thread.is_alive():
            self.ku_thread.join(timeout=2.0)
        self._safe_rc_zero(times=3, dt=0.06)
        self.btn_stop_ku.config(state="disabled")
        self.btn_apply_pid.config(state="normal" if (self.Ku is not None and self.Tu is not None) else "disabled")
        self.stop_event.clear()

    def _relay_tune_thread(self) -> None:
        """
        Relay autotuning:
        vz = +d si error > +hyst
        vz = -d si error < -hyst

        Estima:
        - Tu por autocorrelación
        - amplitud a ~ std*sqrt(2) (aprox) usando señal reciente
        - Ku = 4d/(pi a)
        """
        t0 = time.perf_counter()
        sign = 1.0  # estado del relay
        d = float(self.relay_amp)
        hyst = float(self.relay_hyst)

        # necesitamos suficiente oscilación
        while (not self.stop_event.is_set()) and self.mode == "KU_SEARCH":
            tnow = time.perf_counter()
            if (tnow - t0) > self.relay_timeout_s:
                self._post_ui(lambda: messagebox.showwarning("ZN", "Timeout en auto-tuning (relay)."))
                break

            h, ok = self._read_height_cm()
            if ok and h is not None:
                self.last_good_height = h
                self.last_good_time = tnow
            else:
                if self.last_good_time is None or (tnow - self.last_good_time) > 3.0:
                    self._post_ui(lambda: messagebox.showwarning("Sensor", "Altura inválida > 3s. Deteniendo Ku."))
                    break
                h = self.last_good_height

            if h is None:
                time.sleep(self.DT_CTRL)
                continue

            with self.lock:
                ref = self.ref_abs_cm

            error = ref - h

            # relay con histéresis
            if error > hyst:
                sign = 1.0
            elif error < -hyst:
                sign = -1.0

            u = sign * d
            try:
                self._send_vz(u)
            except Exception:
                pass

            ts = tnow - t0
            with self.lock:
                self.height_cm = h
                self.height_valid = ok
                self.error_cm = error
                self.u_cmd = u

                self.t_log.append(ts)
                self.h_log.append(h)
                self.r_log.append(ref)
                self.u_log.append(u)

            # cada ~6s intentamos estimar Tu y Ku
            if len(self.t_log) > 250:
                with self.lock:
                    t = np.array(self.t_log, dtype=float)
                    y = np.array(self.h_log, dtype=float)

                Tu = estimate_tu_autocorr(t, y)
                if Tu is not None and Tu >= 0.5:
                    # amplitud "a" de oscilación: estimación robusta con percentiles
                    y_recent = y[-200:]
                    a = float((np.percentile(y_recent, 95) - np.percentile(y_recent, 5)) / 2.0)
                    if a > 0.8:
                        Ku = float((4.0 * d) / (math.pi * a))
                        self.Ku, self.Tu = Ku, float(Tu)

                        def _finish() -> None:
                            self.btn_stop_ku.config(state="disabled")
                            self.btn_apply_pid.config(state="normal")
                            messagebox.showinfo("ZN", f"Relay OK.\nKu={Ku:.2f}\nTu={Tu:.2f}s\n(d={d:.0f}, a≈{a:.1f}cm)")

                        self._post_ui(_finish)
                        break

            time.sleep(self.DT_CTRL)

        # salida segura
        try:
            self._send_vz(0.0)
        except Exception:
            pass
        self._safe_rc_zero(times=4, dt=0.06)
        self.mode = "IDLE"
        self.stop_event.set()

    def apply_zn_pid(self) -> None:
        if self.Ku is None or self.Tu is None:
            messagebox.showwarning("Aviso", "Primero obtén Ku y Tu.")
            return

        Ku = float(self.Ku)
        Tu = float(self.Tu)

        # ZN PID clásico
        Kp = 0.6 * Ku
        Ti = 0.5 * Tu
        Td = 0.125 * Tu
        Ki = Kp / max(Ti, 1e-6)
        Kd = Kp * Td

        self.pid.g = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)
        self.pid.reset()

        self.kp_var.set(Kp)
        self.ki_var.set(Ki)
        self.kd_var.set(Kd)

        messagebox.showinfo("PID (ZN)", f"PID calculado:\nKp={Kp:.3f}\nKi={Ki:.3f}\nKd={Kd:.3f}")

    # ---------------- CONTROL ----------------
    def apply_manual_pid(self) -> None:
        try:
            Kp = float(self.kp_var.get())
            Ki = float(self.ki_var.get())
            Kd = float(self.kd_var.get())
        except Exception:
            messagebox.showerror("Error", "Ganancias inválidas.")
            return
        self.pid.g = PIDGains(Kp=Kp, Ki=Ki, Kd=Kd)
        self.pid.reset()
        messagebox.showinfo("OK", f"PID aplicado:\nKp={Kp:.3f}\nKi={Ki:.3f}\nKd={Kd:.3f}")

    def start_control(self) -> None:
        if not (self.connected and self.flying):
            messagebox.showwarning("Aviso", "Conecta y despega primero.")
            return
        if self.h0_takeoff is None:
            messagebox.showwarning("Aviso", "No hay h0. Despega y espera a que capture takeoff.")
            return
        if self.mode != "IDLE":
            messagebox.showwarning("Aviso", "Detén el modo actual antes.")
            return

        self._stop_all_threads()
        self.stop_event.clear()
        self.mode = "CONTROL"
        self.pid.reset()

        with self.lock:
            self.t_log.clear()
            self.h_log.clear()
            self.r_log.clear()
            self.u_log.clear()
            ref = self.ref_abs_cm

        self._mark_step(ref)

        self.btn_stop_ctrl.config(state="normal")
        self.ctrl_thread = threading.Thread(target=self._control_thread, daemon=True)
        self.ctrl_thread.start()

    def stop_control(self) -> None:
        if self.mode != "CONTROL":
            return
        self.mode = "IDLE"
        self.stop_event.set()
        if self.ctrl_thread is not None and self.ctrl_thread.is_alive():
            self.ctrl_thread.join(timeout=2.0)
        self.ctrl_thread = None
        self._safe_rc_zero(times=4, dt=0.06)
        self.btn_stop_ctrl.config(state="disabled")
        self.stop_event.clear()

    def _control_thread(self) -> None:
        t0 = time.perf_counter()
        while (not self.stop_event.is_set()) and self.mode == "CONTROL":
            tnow = time.perf_counter()

            h, ok = self._read_height_cm()
            if ok and h is not None:
                self.last_good_height = h
                self.last_good_time = tnow
            else:
                if self.last_good_time is None or (tnow - self.last_good_time) > 3.0:
                    self._post_ui(lambda: messagebox.showwarning("Sensor", "Altura inválida > 3s. Deteniendo control."))
                    break
                h = self.last_good_height

            if h is None:
                time.sleep(self.DT_CTRL)
                continue

            with self.lock:
                ref = self.ref_abs_cm

            error = ref - h
            u, _ = self.pid.update(error, tnow)

            # deadband mínima
            if abs(error) < 0.8:
                u = 0.0

            try:
                self._send_vz(u)
            except Exception:
                pass

            ts = tnow - t0
            with self.lock:
                self.height_cm = h
                self.height_valid = ok
                self.error_cm = error
                self.u_cmd = u
                self.t_log.append(ts)
                self.h_log.append(h)
                self.r_log.append(ref)
                self.u_log.append(u)

            time.sleep(self.DT_CTRL)

        try:
            self._send_vz(0.0)
        except Exception:
            pass
        self._safe_rc_zero(times=4, dt=0.06)
        self.mode = "IDLE"
        self.stop_event.set()
        self._post_ui(lambda: self.btn_stop_ctrl.config(state="disabled"))

    # ---------------- Export ----------------
    def export_report_csv(self) -> None:
        with self.lock:
            t = np.array(self.t_log, dtype=float)
            h = np.array(self.h_log, dtype=float)
            r = np.array(self.r_log, dtype=float)
            u = np.array(self.u_log, dtype=float)

        if len(t) < 20:
            messagebox.showwarning("Aviso", "No hay suficientes datos.")
            return

        ref = float(np.mean(r[-10:]))
        Mp = compute_overshoot_percent(h, ref)
        Ts = compute_settling_time(t, h, ref, band_ratio=0.02, hold_s=2.0)

        fname = f"tello_altitude_log_{int(time.time())}.csv"
        try:
            with open(fname, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["t_s", "height_cm", "ref_abs_cm", "u_cmd"])
                for i in range(len(t)):
                    w.writerow([t[i], h[i], r[i], u[i]])
                w.writerow([])
                w.writerow(["Mp_%", Mp])
                w.writerow(["Ts_s", Ts if Ts is not None else "NA"])
                w.writerow(["Ku", self.Ku if self.Ku is not None else "NA"])
                w.writerow(["Tu", self.Tu if self.Tu is not None else "NA"])
                w.writerow(["Kp", self.pid.g.Kp])
                w.writerow(["Ki", self.pid.g.Ki])
                w.writerow(["Kd", self.pid.g.Kd])

            messagebox.showinfo("Exportado", f"CSV: {fname}\nMp={Mp:.2f}% | Ts={Ts if Ts else 'NA'} s")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo guardar CSV: {e}")

    # ---------------- UI update ----------------
    def _ui_update(self) -> None:
        self._compute_live_metrics()

        with self.lock:
            h = self.height_cm
            e = self.error_cm
            u = self.u_cmd
            bat = self.battery if self.battery is not None else "—"
            mode = self.mode
            h0 = self.h0_takeoff
            sp = self.sp_delta_cm
            ref = self.ref_abs_cm if h0 is not None else None

        self.lbl_tel.config(text=f"h: {h:.1f} cm | e: {e:.1f} cm | u: {u:.0f} | bat: {bat} | mode: {mode}")
        if h0 is None or ref is None:
            self.lbl_refabs.config(text=f"h0: — cm | ref_abs: — cm (Δ={sp:.0f})")
        else:
            self.lbl_refabs.config(text=f"h0: {h0:.1f} cm | ref_abs: {ref:.1f} cm (Δ={sp:.0f})")

        if self.Ku is not None and self.Tu is not None:
            self.lbl_ku.config(text=f"Ku: {self.Ku:.2f}   Tu: {self.Tu:.2f} s")
        else:
            self.lbl_ku.config(text="Ku: —   Tu: —")

        ts_txt = f"{self.ts_live:.2f}" if self.ts_live is not None else "—"
        self.lbl_metrics.config(text=f"Mp: {self.mp_live:.2f} % | Ts: {ts_txt} s")

        self._update_plot()
        self.root.after(200, self._ui_update)

    def _update_plot(self) -> None:
        with self.lock:
            t = np.array(self.t_log, dtype=float)
            h = np.array(self.h_log, dtype=float)
            r = np.array(self.r_log, dtype=float)
            u = np.array(self.u_log, dtype=float)

        if len(t) < 2:
            return

        self.line_h.set_data(t, h)
        self.line_r.set_data(t, r)
        self.line_u.set_data(t, u)

        self.ax.set_xlim(max(0.0, t[-1] - 20.0), t[-1] + 0.1)

        ymin = float(min(np.min(h), np.min(r)) - 10.0)
        ymax = float(max(np.max(h), np.max(r)) + 10.0)
        self.ax.set_ylim(ymin, ymax)
        self.ax2.set_ylim(self.U_MIN - 5.0, self.U_MAX + 5.0)
        self.canvas.draw_idle()

    def _battery_poll(self) -> None:
        if self.connected and self.tello is not None:
            try:
                self.battery = int(self.tello.get_battery())
            except Exception:
                pass
        self.root.after(2000, self._battery_poll)


def main() -> None:
    root = tk.Tk()
    app = TelloAltitudeApp(root)

    def on_close() -> None:
        try:
            app._stop_all_threads()
            app._safe_rc_zero(times=6, dt=0.06)
            if app.tello is not None:
                try:
                    if app.flying:
                        app.tello.land()
                except Exception:
                    pass
                try:
                    app.tello.end()
                except Exception:
                    pass
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
