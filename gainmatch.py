#!/usr/bin/env python3
"""
Qt-based interactive gain matcher for Maestro .Spe spectra.

Usage:
  gainmatch.py REF.Spe MOB1.Spe [MOB2.Spe ...] [--norm peak]

Dependencies:
  pip install pyside6 pyqtgraph numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import find_peaks
from scipy.optimize import least_squares

try:
    from PySide6 import QtCore, QtGui, QtWidgets
    import pyqtgraph as pg
except Exception as exc:  # pragma: no cover
    QtCore = None  # type: ignore[assignment]
    QtGui = None  # type: ignore[assignment]
    QtWidgets = None  # type: ignore[assignment]
    pg = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

BaseMainWindow = QtWidgets.QMainWindow if QtWidgets is not None else object


def parse_spe(path: str) -> tuple[np.ndarray, dict[str, object]]:
    """Parse Maestro .Spe ASCII file into counts and metadata."""
    meta: dict[str, object] = {}
    counts: list[int] = []
    in_data = False
    current_key: Optional[str] = None

    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()

            if line.startswith("$") and line.endswith(":"):
                current_key = line[1:-1]
                in_data = current_key == "DATA"
                continue

            if in_data:
                if line.startswith("$"):
                    in_data = False
                    continue

                parts = line.split()
                if len(parts) == 2 and not counts:
                    try:
                        meta["ch_start"] = int(parts[0])
                        meta["ch_end"] = int(parts[1])
                    except ValueError:
                        meta["ch_start"] = 0
                        meta["ch_end"] = 0
                    continue

                for p in parts:
                    try:
                        counts.append(int(p))
                    except ValueError:
                        pass
            else:
                if current_key and line:
                    if current_key not in meta or not isinstance(meta[current_key], list):
                        meta[current_key] = []
                    section = meta[current_key]
                    if isinstance(section, list):
                        section.append(line)

    return np.array(counts, dtype=np.float64), meta


def _parse_time_seconds(value: str) -> float:
    text = str(value).strip()
    if not text:
        return 0.0
    try:
        if ":" in text:
            parts = [float(x) for x in text.split(":")]
            return sum(p * (60 ** (len(parts) - i - 1)) for i, p in enumerate(parts))
        return float(text)
    except (ValueError, IndexError):
        return 0.0


def _read_meas_tim(meta: dict[str, object]) -> tuple[float, float]:
    section = meta.get("MEAS_TIM")
    if isinstance(section, list) and section:
        parts: list[str] = []
        for line in section:
            parts.extend(str(line).split())
        if len(parts) >= 2:
            return _parse_time_seconds(parts[0]), _parse_time_seconds(parts[1])
    return 0.0, 0.0


def write_spe(
    path: str,
    counts: np.ndarray,
    meta: dict[str, object],
    is_difference: bool = False,
    other_meta: Optional[dict[str, object]] = None,
) -> None:
    """Write Maestro .Spe file preserving metadata and enforcing clean newlines.

    For difference spectra, MEAS_TIM is written as two integers:
      live_time real_time_diff
    where
      real_time_diff = real_time_1 + real_time_2 - 2 * live_time
    """
    known_order = [
        "SPEC_ID",
        "SPEC_REM",
        "DATE_MEA",
        "MEAS_TIM",
        "DATA",
        "ROI",
        "PRESETS",
        "ENER_FIT",
        "MCA_CAL",
        "SHAPE_CAL",
    ]

    meta_keys = [k for k in meta.keys() if k not in ("ch_start", "ch_end")]
    ordered = [k for k in known_order if (k in meta_keys or k == "DATA")]
    ordered += [k for k in meta_keys if k not in ordered]

    ch_start_raw = meta.get("ch_start", 0)
    ch_end_raw = meta.get("ch_end", len(counts) - 1)
    ch_start = int(ch_start_raw) if isinstance(ch_start_raw, (int, float, np.integer)) else 0
    ch_end = int(ch_end_raw) if isinstance(ch_end_raw, (int, float, np.integer)) else len(counts) - 1

    live_a, real_a = _read_meas_tim(meta)
    live_b, real_b = _read_meas_tim(other_meta) if isinstance(other_meta, dict) else (0.0, 0.0)

    live_time = int(round(live_a))
    real_time = int(round(real_a))
    if is_difference:
        live_time = int(round(live_a))
        real_time = int(round(max(0.0, real_a + real_b - 2.0 * float(live_time))))

    with open(path, "w", newline="\n") as fh:
        for key in ordered:
            fh.write(f"${key}:\n")

            if key == "DATA":
                fh.write(f"{ch_start} {ch_end}\n")
                for c in counts:
                    fh.write(f"       {int(round(float(c)))}\n")
                continue

            if key == "DATE_MEA":
                fh.write("0:00:00\n")
                continue

            if key == "MEAS_TIM":
                fh.write(f"{int(live_time)} {int(real_time)}\n")
                continue

            section = meta.get(key)
            if not isinstance(section, list):
                continue

            if key == "SPEC_REM" and is_difference:
                replaced = False
                for line in section:
                    clean = str(line).strip()
                    if not clean:
                        continue
                    if clean.startswith("DETDESC#"):
                        fh.write("DETDESC# Difference of spectra scaled by live time\n")
                        replaced = True
                    else:
                        fh.write(f"{clean}\n")
                if not replaced:
                    fh.write("DETDESC# Difference of spectra scaled by live time\n")
                continue

            for line in section:
                clean = str(line).strip()
                if clean:
                    fh.write(f"{clean}\n")


def remap_spectrum(counts: np.ndarray, m: float, q: float) -> np.ndarray:
    """Remap counts by ch_out = m*ch_in + q (inverse sampled with cubic spline)."""
    n = len(counts)
    if n == 0:
        return counts.copy()

    m_safe = m if abs(m) > 1e-12 else 1.0
    i = np.arange(n, dtype=np.float64)
    src = (i - q) / m_safe

    if n < 4:
        return np.interp(src, i, counts, left=0.0, right=0.0)

    spline = CubicSpline(i, counts, bc_type="natural", extrapolate=False)
    out = np.zeros(n, dtype=np.float64)
    valid = (src >= 0.0) & (src <= float(n - 1))
    vals = spline(src[valid])
    out[valid] = np.where(np.isfinite(vals), vals, 0.0)
    return out


def peak_centroid_fwhm(counts: np.ndarray, center: int, half_win: int) -> tuple[float, float, float, float]:
    """Gaussian-fit centroid/FWHM (fallback to weighted), peak max, and area in window."""
    n = len(counts)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    c = int(np.clip(center, 0, n - 1))
    lo = max(0, c - half_win)
    hi = min(n - 1, c + half_win)

    region = counts[lo : hi + 1]
    ch = np.arange(lo, hi + 1, dtype=np.float64)

    total = float(region.sum())
    centroid = float((ch * region).sum() / total) if total > 0 else float(c)

    peak_val = float(region.max()) if region.size else 0.0

    # Simple local Gaussian fit: y = b + a*exp(-0.5*((x-mu)/sigma)^2)
    # Fallback to threshold-based width if fit fails.
    if region.size >= 5 and peak_val > 0.0:
        base0 = float(np.median(region))
        amp0 = float(max(1e-6, peak_val - base0))
        sigma0 = float(max(1.0, (hi - lo + 1) / 6.0))
        x0 = np.array([amp0, centroid, sigma0, base0], dtype=np.float64)

        max_amp = float(max(amp0 * 10.0, peak_val * 10.0, 1.0))
        max_sigma = float(max(2.0, hi - lo + 1))
        y_min = float(region.min())
        y_max = float(region.max())
        span = float(max(1.0, y_max - y_min))
        lb = np.array([0.0, float(lo), 0.5, y_min - span], dtype=np.float64)
        ub = np.array([max_amp, float(hi), max_sigma, y_max + span], dtype=np.float64)

        def gresid(p: np.ndarray) -> np.ndarray:
            a, mu, sigma, b = float(p[0]), float(p[1]), float(p[2]), float(p[3])
            expo = -0.5 * np.square((ch - mu) / max(sigma, 1e-6))
            yhat = b + a * np.exp(expo)
            return yhat - region

        try:
            fit = least_squares(gresid, x0=x0, bounds=(lb, ub))
            if fit.success and fit.x.size == 4 and np.all(np.isfinite(fit.x)):
                mu_fit = float(np.clip(fit.x[1], lo, hi))
                sigma_fit = float(max(1e-6, fit.x[2]))
                centroid = mu_fit
                return centroid, float(2.35482 * sigma_fit), peak_val, float(total)
        except Exception:
            pass

    half = peak_val / 2.0

    ic = int(np.clip(round(centroid), lo, hi))
    left = lo
    right = hi

    for i in range(ic, lo - 1, -1):
        if counts[i] <= half:
            left = i
            break

    for i in range(ic, hi + 1):
        if counts[i] <= half:
            right = i
            break

    return centroid, float(max(0, right - left)), peak_val, float(total)


def detect_peaks_scipy(spec: np.ndarray, max_peaks: int = 8, min_distance: int = 20) -> np.ndarray:
    """Detect significant peaks with scipy and return channels sorted by x position."""
    if len(spec) < 3:
        return np.array([], dtype=np.int64)

    arr = np.asarray(spec, dtype=np.float64)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vmax = float(np.max(arr)) if arr.size else 0.0
    if vmax <= 0:
        return np.array([], dtype=np.int64)

    prominence = max(1.0, 0.02 * vmax)
    peaks, props = find_peaks(arr, prominence=prominence, distance=max(1, int(min_distance)))
    if peaks.size == 0:
        return np.array([], dtype=np.int64)

    prom = props.get("prominences", np.ones_like(peaks, dtype=np.float64))
    order = np.argsort(prom)[::-1]
    keep = order[: max(1, int(max_peaks))]
    selected = np.sort(peaks[keep])
    return selected.astype(np.int64)


def _fit_mq_from_peak_lists(ref_peaks: np.ndarray, mob_peaks: np.ndarray) -> Optional[tuple[float, float]]:
    """Fit linear map ref ~= m*mob + q from ordered peak lists."""
    k = min(len(ref_peaks), len(mob_peaks))
    if k < 2:
        return None

    x = np.asarray(mob_peaks[:k], dtype=np.float64)
    y = np.asarray(ref_peaks[:k], dtype=np.float64)
    if np.allclose(x, x[0]):
        return None

    m, q = np.polyfit(x, y, 1)
    if not np.isfinite(m) or not np.isfinite(q):
        return None
    return float(m), float(q)


def pair_peaks_nearest_mapped(
    ref_peaks: np.ndarray,
    mob_peaks: np.ndarray,
    m: float,
    q: float,
    max_delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pair peaks by nearest mapped distance while preserving order."""
    if len(ref_peaks) == 0 or len(mob_peaks) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    rp = np.sort(np.asarray(ref_peaks, dtype=np.int64))
    mp = np.sort(np.asarray(mob_peaks, dtype=np.int64))
    mapped = m * mp.astype(np.float64) + q

    out_r: list[int] = []
    out_m: list[int] = []
    start = 0

    for r in rp:
        if start >= len(mp):
            break
        seg = mapped[start:]
        j_rel = int(np.argmin(np.abs(seg - float(r))))
        j = start + j_rel
        dist = abs(mapped[j] - float(r))
        if dist <= float(max_delta):
            out_r.append(int(r))
            out_m.append(int(mp[j]))
            start = j + 1

    return np.array(out_r, dtype=np.int64), np.array(out_m, dtype=np.int64)


def auto_calibrate(
    ref: np.ndarray,
    mobile_raw: np.ndarray,
    half_win: int,
    ref_center: Optional[int] = None,
    mob_center_raw: Optional[int] = None,
) -> tuple[float, float, tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Estimate m,q mapping selected mobile peak center to selected reference peak center."""
    if ref_center is None:
        ref_center = int(ref.argmax()) if len(ref) else 0
    if mob_center_raw is None:
        mob_center_raw = int(mobile_raw.argmax()) if len(mobile_raw) else 0

    sr = peak_centroid_fwhm(ref, ref_center, half_win)
    sm = peak_centroid_fwhm(mobile_raw, mob_center_raw, half_win)

    m = sr[1] / sm[1] if sm[1] > 0 else 1.0
    q = sr[0] - m * sm[0]
    return m, q, sr, sm


def scale_factor(
    ref: np.ndarray,
    remapped: np.ndarray,
    mode: str,
    half_win: int,
    ref_center: int,
    mapped_mobile_center: int,
) -> float:
    """Compute normalization factor according to mode."""
    if mode == "none":
        return 1.0
    if len(ref) == 0 or len(remapped) == 0:
        return 1.0

    rc = int(np.clip(ref_center, 0, len(ref) - 1))
    mc = int(np.clip(mapped_mobile_center, 0, len(remapped) - 1))

    r_lo = max(0, rc - half_win)
    r_hi = min(len(ref) - 1, rc + half_win)
    m_lo = max(0, mc - half_win)
    m_hi = min(len(remapped) - 1, mc + half_win)

    if mode == "peak":
        ref_v = float(ref[r_lo : r_hi + 1].max())
        mob_v = float(remapped[m_lo : m_hi + 1].max())
        return ref_v / mob_v if mob_v > 0 else 1.0
    if mode == "area":
        ref_v = float(ref[r_lo : r_hi + 1].sum())
        mob_v = float(remapped[m_lo : m_hi + 1].sum())
        return ref_v / mob_v if mob_v > 0 else 1.0
    if mode == "integral":
        ref_v = float(ref.sum())
        mob_v = float(remapped.sum())
        return ref_v / mob_v if mob_v > 0 else 1.0
    return 1.0


def detect_primary_peak(spec: np.ndarray) -> int:
    return int(spec.argmax()) if len(spec) else 0


def make_output_path(src_path: str, suffix: str) -> str:
    src_dir = os.path.dirname(src_path)
    base = os.path.basename(src_path)
    stem, ext = os.path.splitext(base)
    out_dir = os.path.join(src_dir, "matched") if src_dir else "matched"
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{stem}_{suffix}{ext if ext else '.Spe'}")


class GainMatchWindow(BaseMainWindow):
    def __init__(
        self,
        ref_counts: np.ndarray,
        mobile_items: list[dict[str, object]],
        ref_path: str,
        ref_meta: Optional[dict[str, object]],
        norm_mode: str,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Gain Matcher (Qt)")
        self.resize(1500, 900)

        self.ref_path = ref_path
        self.ref_counts = np.array(ref_counts, dtype=np.float64)
        self.ref_meta = dict(ref_meta) if isinstance(ref_meta, dict) else {}
        self.mobile_items = mobile_items
        self.index1 = 0
        self.index2 = 1 if len(self.mobile_items) > 1 else 0
        self.norm_mode = norm_mode
        self.show_sub = False

        self.ref_center = detect_primary_peak(self.ref_counts)
        self.mob_center_raw = 0
        self.mob_center_raw2 = 0
        self.remapped_unscaled = np.zeros_like(self.ref_counts)
        self.remapped_unscaled2 = np.zeros_like(self.ref_counts)
        self.scaled = np.zeros_like(self.ref_counts)
        self.scaled2 = np.zeros_like(self.ref_counts)
        self.subtracted = np.zeros_like(self.ref_counts)
        self.subtracted2 = np.zeros_like(self.ref_counts)
        self.subtracted_dy = np.zeros_like(self.ref_counts)
        self.subtracted_dy2 = np.zeros_like(self.ref_counts)

        self.pick_mode: Optional[str] = None
        self.pending_pair_ref: Optional[int] = None
        self.selection_mode: Optional[str] = None
        self.selection_start: Optional[int] = None
        self.sum_ref: Optional[tuple[int, int]] = None
        self.sum_mob: Optional[tuple[int, int]] = None
        self.sum_mob2: Optional[tuple[int, int]] = None
        self.sum_sub: Optional[tuple[int, int]] = None
        self.sum_sub2: Optional[tuple[int, int]] = None
        self.ref_peaks = np.array([], dtype=np.int64)
        self.mob_peaks_raw = np.array([], dtype=np.int64)
        self.mob_peaks_raw2 = np.array([], dtype=np.int64)
        self.fit_mode = "peaks_lsq"
        self.fit_mode2 = "peaks_lsq"
        self._drag_guard = False

        self._build_ui(os.path.basename(ref_path))
        self._refresh_mobile_list()
        self._set_m(1.0)
        self._set_q(0.0)
        self._set_m2(1.0)
        self._set_q2(0.0)
        self._load_mobile_slot(1, self.index1, apply_auto=False)
        self._load_mobile_slot(2, self.index2, apply_auto=False)

    def _build_ui(self, ref_label: str) -> None:
        root = QtWidgets.QWidget(self)
        self.setCentralWidget(root)

        main = QtWidgets.QVBoxLayout(root)

        self.top_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.bottom_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main.addWidget(self.top_splitter, 3)
        main.addWidget(self.bottom_splitter, 2)

        top_plot_panel = QtWidgets.QWidget()
        top_plot_lay = QtWidgets.QVBoxLayout(top_plot_panel)
        top_plot_lay.setContentsMargins(0, 0, 0, 0)

        top_stats_panel = QtWidgets.QWidget()
        top_stats_lay = QtWidgets.QVBoxLayout(top_stats_panel)
        top_stats_lay.setContentsMargins(0, 0, 0, 0)

        self.top_splitter.addWidget(top_plot_panel)
        self.top_splitter.addWidget(top_stats_panel)
        self.top_splitter.setStretchFactor(0, 3)
        self.top_splitter.setStretchFactor(1, 1)

        general_panel = QtWidgets.QWidget()
        general_lay = QtWidgets.QVBoxLayout(general_panel)
        general_lay.setContentsMargins(0, 0, 0, 0)

        spec1_panel = QtWidgets.QWidget()
        spec1_lay = QtWidgets.QVBoxLayout(spec1_panel)
        spec1_lay.setContentsMargins(0, 0, 0, 0)

        spec2_panel = QtWidgets.QWidget()
        spec2_lay = QtWidgets.QVBoxLayout(spec2_panel)
        spec2_lay.setContentsMargins(0, 0, 0, 0)

        self.bottom_splitter.addWidget(general_panel)
        self.bottom_splitter.addWidget(spec1_panel)
        self.bottom_splitter.addWidget(spec2_panel)
        self.bottom_splitter.setStretchFactor(0, 2)
        self.bottom_splitter.setStretchFactor(1, 3)
        self.bottom_splitter.setStretchFactor(2, 3)

        pg.setConfigOptions(antialias=True)
        self.plot = pg.PlotWidget(background="#12151d")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend()
        self.plot.setLabel("bottom", "Channel")
        self.plot.setLabel("left", "Counts")
        self.plot.setTitle("Spectrum Gain Matching")
        self.plot.setMouseEnabled(x=True, y=True)
        self.plot.setLimits(xMin=0)

        self.curve_ref = self.plot.plot(pen=pg.mkPen("#4fc3f7", width=1.2), name=f"Reference: {ref_label}")
        self.curve_mob = self.plot.plot(pen=pg.mkPen("#f4a261", width=1.2), name="Matched 1")
        self.curve_mob2 = self.plot.plot(pen=pg.mkPen("#e9c46a", width=1.2), name="Matched 2")
        self.curve_sub = self.plot.plot(pen=pg.mkPen("#d66cff", width=1.0), name="Sub 1-2", visible=False)
        self.curve_sub2 = self.plot.plot(pen=pg.mkPen("#9b5de5", width=1.0), name="Sub 2-1", visible=False)

        self.vline_ref = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#4fc3f7", style=QtCore.Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#8ad8ff", width=2),
        )
        self.vline_mob = pg.InfiniteLine(
            angle=90,
            movable=True,
            pen=pg.mkPen("#f4a261", style=QtCore.Qt.PenStyle.DashLine),
            hoverPen=pg.mkPen("#ffc48c", width=2),
        )
        self.plot.addItem(self.vline_ref)
        self.plot.addItem(self.vline_mob)

        self.ref_peak_dot = pg.ScatterPlotItem(size=10, pen=pg.mkPen("#4fc3f7"), brush=pg.mkBrush("#4fc3f7"))
        self.mob_peak_dot = pg.ScatterPlotItem(size=10, pen=pg.mkPen("#f4a261"), brush=pg.mkBrush("#f4a261"))
        self.plot.addItem(self.ref_peak_dot)
        self.plot.addItem(self.mob_peak_dot)
        self.ref_peaks_scatter = pg.ScatterPlotItem(size=7, symbol="t", pen=pg.mkPen("#4fc3f7"), brush=pg.mkBrush("#4fc3f7"))
        self.mob_peaks_scatter = pg.ScatterPlotItem(size=7, symbol="t", pen=pg.mkPen("#f4a261"), brush=pg.mkBrush("#f4a261"))
        self.plot.addItem(self.ref_peaks_scatter)
        self.plot.addItem(self.mob_peaks_scatter)

        self.ref_target = None
        self.mob_target = None
        if hasattr(pg, "TargetItem"):
            self.ref_target = pg.TargetItem(pos=(0, 0), size=11, movable=True, symbol="o", pen=pg.mkPen("#4fc3f7"), brush=pg.mkBrush("#4fc3f7"))
            self.mob_target = pg.TargetItem(pos=(0, 0), size=11, movable=True, symbol="o", pen=pg.mkPen("#f4a261"), brush=pg.mkBrush("#f4a261"))
            self.plot.addItem(self.ref_target)
            self.plot.addItem(self.mob_target)

        self.region_ref = pg.LinearRegionItem(values=(0, 1), orientation=pg.LinearRegionItem.Vertical, movable=False, brush=(79, 195, 247, 40))
        self.region_mob = pg.LinearRegionItem(values=(0, 1), orientation=pg.LinearRegionItem.Vertical, movable=False, brush=(244, 162, 97, 40))
        self.region_sub = pg.LinearRegionItem(values=(0, 1), orientation=pg.LinearRegionItem.Vertical, movable=False, brush=(214, 108, 255, 35))
        self.plot.addItem(self.region_ref)
        self.plot.addItem(self.region_mob)
        self.plot.addItem(self.region_sub)
        self.region_ref.setVisible(False)
        self.region_mob.setVisible(False)
        self.region_sub.setVisible(False)

        self.plot.scene().sigMouseClicked.connect(self._on_plot_click)
        self.vline_ref.sigPositionChangeFinished.connect(self._on_ref_line_drag)
        self.vline_mob.sigPositionChangeFinished.connect(self._on_mob_line_drag)
        if self.ref_target is not None:
            sig_ref = getattr(self.ref_target, "sigPositionChangeFinished", None) or getattr(self.ref_target, "sigPositionChanged", None)
            if sig_ref is not None:
                sig_ref.connect(self._on_ref_target_drag)
        if self.mob_target is not None:
            sig_mob = getattr(self.mob_target, "sigPositionChangeFinished", None) or getattr(self.mob_target, "sigPositionChanged", None)
            if sig_mob is not None:
                sig_mob.connect(self._on_mob_target_drag)

        top_plot_lay.addWidget(self.plot, 1)

        control_grid = QtWidgets.QGridLayout()
        general_lay.addLayout(control_grid)

        self.m_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.m_slider.setRange(8000, 12000)
        self.m_spin = QtWidgets.QDoubleSpinBox()
        self.m_spin.setRange(0.8, 1.2)
        self.m_spin.setDecimals(5)
        self.m_spin.setSingleStep(0.0001)

        self.q_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.q_slider.setRange(-1000, 1000)
        self.q_spin = QtWidgets.QDoubleSpinBox()
        self.q_spin.setRange(-100.0, 100.0)
        self.q_spin.setDecimals(2)
        self.q_spin.setSingleStep(0.1)

        self.hw_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.hw_slider.setRange(10, 300)
        self.hw_spin = QtWidgets.QSpinBox()
        self.hw_spin.setRange(10, 300)

        self.ref_spin = QtWidgets.QSpinBox()
        self.ref_spin.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin = QtWidgets.QSpinBox()
        self.mob_spin.setRange(0, len(self.ref_counts) - 1)
        self.ref_spin.setFixedWidth(95)
        self.mob_spin.setFixedWidth(95)

        self.m_slider2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.m_slider2.setRange(8000, 12000)
        self.m_spin2 = QtWidgets.QDoubleSpinBox()
        self.m_spin2.setRange(0.8, 1.2)
        self.m_spin2.setDecimals(5)
        self.m_spin2.setSingleStep(0.0001)

        self.q_slider2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.q_slider2.setRange(-1000, 1000)
        self.q_spin2 = QtWidgets.QDoubleSpinBox()
        self.q_spin2.setRange(-100.0, 100.0)
        self.q_spin2.setDecimals(2)
        self.q_spin2.setSingleStep(0.1)

        self.hw_slider2 = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.hw_slider2.setRange(10, 300)
        self.hw_spin2 = QtWidgets.QSpinBox()
        self.hw_spin2.setRange(10, 300)
        self.mob_spin2 = QtWidgets.QSpinBox()
        self.mob_spin2.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin2.setFixedWidth(95)

        self.norm_modes = ["none", "peak", "area", "integral"]
        self.norm_mode_state = self.norm_mode if self.norm_mode in set(self.norm_modes) else "peak"
        self.y_scale_mode = "linear"

        self.norm_group = QtWidgets.QButtonGroup(self)
        self.rb_norm_none = QtWidgets.QRadioButton("None")
        self.rb_norm_peak = QtWidgets.QRadioButton("Peak")
        self.rb_norm_area = QtWidgets.QRadioButton("Area")
        self.rb_norm_integral = QtWidgets.QRadioButton("Integral")
        for rb in (self.rb_norm_none, self.rb_norm_peak, self.rb_norm_area, self.rb_norm_integral):
            self.norm_group.addButton(rb)
        if self.norm_mode_state == "none":
            self.rb_norm_none.setChecked(True)
        elif self.norm_mode_state == "area":
            self.rb_norm_area.setChecked(True)
        elif self.norm_mode_state == "integral":
            self.rb_norm_integral.setChecked(True)
        else:
            self.rb_norm_peak.setChecked(True)

        self.ys_group = QtWidgets.QButtonGroup(self)
        self.rb_y_linear = QtWidgets.QRadioButton("Linear")
        self.rb_y_log = QtWidgets.QRadioButton("Log")
        self.ys_group.addButton(self.rb_y_linear)
        self.ys_group.addButton(self.rb_y_log)
        self.rb_y_linear.setChecked(True)

        self.status = QtWidgets.QLabel("")
        self.nav_label = QtWidgets.QLabel("")
        self.range_edit = QtWidgets.QLineEdit()
        self.range_edit.setPlaceholderText("Range a:b")
        self.range_edit.setFixedWidth(140)

        norm_widget = QtWidgets.QWidget()
        norm_lay = QtWidgets.QHBoxLayout(norm_widget)
        norm_lay.setContentsMargins(0, 0, 0, 0)
        norm_lay.setSpacing(8)
        for rb in (self.rb_norm_none, self.rb_norm_peak, self.rb_norm_area, self.rb_norm_integral):
            norm_lay.addWidget(rb)

        ys_widget = QtWidgets.QWidget()
        ys_lay = QtWidgets.QHBoxLayout(ys_widget)
        ys_lay.setContentsMargins(0, 0, 0, 0)
        ys_lay.setSpacing(8)
        ys_lay.addWidget(self.rb_y_linear)
        ys_lay.addWidget(self.rb_y_log)

        control_grid.addWidget(QtWidgets.QLabel("Normalize"), 0, 0)
        control_grid.addWidget(norm_widget, 0, 1, 1, 3)
        control_grid.addWidget(QtWidgets.QLabel("Y scale"), 1, 0)
        control_grid.addWidget(ys_widget, 1, 1, 1, 3)
        control_grid.addWidget(self.status, 2, 0, 1, 2)
        control_grid.addWidget(self.nav_label, 2, 2, 1, 2)
        control_grid.addWidget(QtWidgets.QLabel("Range"), 3, 0)
        control_grid.addWidget(self.range_edit, 3, 1, 1, 1)

        control_grid.addWidget(QtWidgets.QLabel("Match1"), 4, 0)
        self.mob_select1 = QtWidgets.QComboBox()
        control_grid.addWidget(self.mob_select1, 4, 1, 1, 3)
        control_grid.addWidget(QtWidgets.QLabel("Match2"), 5, 0)
        self.mob_select2 = QtWidgets.QComboBox()
        control_grid.addWidget(self.mob_select2, 5, 1, 1, 3)

        self.btn_auto = QtWidgets.QPushButton("Auto")
        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_detect = QtWidgets.QPushButton("Detect")
        self.btn_save = QtWidgets.QPushButton("Save1")
        self.btn_save2 = QtWidgets.QPushButton("Save2")
        self.btn_subtract = QtWidgets.QPushButton("Subtract")
        self.btn_save_sub = QtWidgets.QPushButton("Save Sub")
        self.btn_save_sub3 = QtWidgets.QPushButton("Save Sub 3col")
        self.btn_clear = QtWidgets.QPushButton("Clear Sub")
        self.sub_mode_combo = QtWidgets.QComboBox()
        self.sub_mode_combo.addItem("1 - 2", "1-2")
        self.sub_mode_combo.addItem("2 - 1", "2-1")
        self.sub_mode_combo.addItem("Ref - 1", "ref-1")
        self.sub_mode_combo.addItem("Ref - 2", "ref-2")
        self.btn_prev = QtWidgets.QPushButton("Prev")
        self.btn_next = QtWidgets.QPushButton("Next")
        self.btn_pick_ref = QtWidgets.QPushButton("RefPk")
        self.btn_pick_mob = QtWidgets.QPushButton("MobPk")
        self.btn_pick_range = QtWidgets.QPushButton("Pick Range")
        self.btn_sum_all = QtWidgets.QPushButton("Sum")
        self.btn_print_params = QtWidgets.QPushButton("Print Params")

        cal1_box = QtWidgets.QGroupBox("Matching 1")
        cal1_lay = QtWidgets.QGridLayout(cal1_box)
        cal1_lay.addWidget(QtWidgets.QLabel("m"), 0, 0)
        cal1_lay.addWidget(self.m_slider, 0, 1)
        cal1_lay.addWidget(self.m_spin, 0, 3)
        cal1_lay.addWidget(QtWidgets.QLabel("q"), 1, 0)
        cal1_lay.addWidget(self.q_slider, 1, 1)
        cal1_lay.addWidget(self.q_spin, 1, 3)
        cal1_lay.addWidget(QtWidgets.QLabel("Half-window"), 2, 0)
        cal1_lay.addWidget(self.hw_slider, 2, 1)
        cal1_lay.addWidget(self.hw_spin, 2, 3)
        cal1_lay.addWidget(QtWidgets.QLabel("Ref peak"), 3, 0)
        cal1_lay.addWidget(self.ref_spin, 3, 1)
        cal1_lay.addWidget(QtWidgets.QLabel("Mob peak"), 3, 2)
        cal1_lay.addWidget(self.mob_spin, 3, 3)
        row1 = QtWidgets.QHBoxLayout()
        for b in (self.btn_auto, self.btn_reset, self.btn_detect, self.btn_pick_ref, self.btn_pick_mob, self.btn_save):
            row1.addWidget(b)
        cal1_lay.addLayout(row1, 5, 0, 1, 3)

        box_sub = QtWidgets.QGroupBox("Subtract")
        lay_sub = QtWidgets.QHBoxLayout(box_sub)
        lay_sub.addWidget(QtWidgets.QLabel("Mode"))
        lay_sub.addWidget(self.sub_mode_combo)
        for b in (self.btn_subtract, self.btn_clear, self.btn_save_sub, self.btn_save_sub3):
            lay_sub.addWidget(b)

        box_range = QtWidgets.QGroupBox("Range Tools")
        lay_range = QtWidgets.QHBoxLayout(box_range)
        for b in (self.btn_pick_range, self.btn_sum_all, self.btn_print_params):
            lay_range.addWidget(b)

        spec1_lay.addWidget(cal1_box, 1)
        general_lay.addWidget(box_sub)
        general_lay.addWidget(box_range)

        self.stats = QtWidgets.QPlainTextEdit()
        self.stats.setReadOnly(True)
        font = QtGui.QFont("monospace")
        font.setStyleHint(QtGui.QFont.StyleHint.TypeWriter)
        self.stats.setFont(font)
        stats_box = QtWidgets.QGroupBox("Stats")
        stats_lay = QtWidgets.QVBoxLayout(stats_box)
        stats_lay.addWidget(self.stats)
        top_stats_lay.addWidget(stats_box, 1)

        fit_box = QtWidgets.QGroupBox("Fit Controls")
        fit_lay = QtWidgets.QVBoxLayout(fit_box)
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.fit_mode_combo = QtWidgets.QComboBox()
        self.fit_mode_combo.addItems(["Peak LSQ", "Spectrum LSQ"])
        mode_row.addWidget(self.fit_mode_combo)
        fit_lay.addLayout(mode_row)
        self.peak_table = QtWidgets.QTableWidget(0, 4)
        self.peak_table.setHorizontalHeaderLabels(["Use", "Ref ch", "Mob ch", "dCh"])
        self.peak_table.horizontalHeader().setStretchLastSection(True)
        self.peak_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.peak_table.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.peak_table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        fit_lay.addWidget(self.peak_table)
        pair_btns = QtWidgets.QHBoxLayout()
        self.btn_add_pair = QtWidgets.QPushButton("Add Pair")
        self.btn_remove_pair = QtWidgets.QPushButton("Remove Pair")
        self.btn_clear_pairs = QtWidgets.QPushButton("Clear Pairs")
        pair_btns.addWidget(self.btn_add_pair)
        pair_btns.addWidget(self.btn_remove_pair)
        pair_btns.addWidget(self.btn_clear_pairs)
        fit_lay.addLayout(pair_btns)
        spec1_lay.addWidget(fit_box, 1)

        cal2_box = QtWidgets.QGroupBox("Matching 2")
        cal2_lay = QtWidgets.QGridLayout(cal2_box)
        cal2_lay.addWidget(QtWidgets.QLabel("m2"), 0, 0)
        cal2_lay.addWidget(self.m_slider2, 0, 1)
        cal2_lay.addWidget(self.m_spin2, 0, 2)
        cal2_lay.addWidget(QtWidgets.QLabel("q2"), 1, 0)
        cal2_lay.addWidget(self.q_slider2, 1, 1)
        cal2_lay.addWidget(self.q_spin2, 1, 2)
        cal2_lay.addWidget(QtWidgets.QLabel("Half-window2"), 2, 0)
        cal2_lay.addWidget(self.hw_slider2, 2, 1)
        cal2_lay.addWidget(self.hw_spin2, 2, 2)
        cal2_lay.addWidget(QtWidgets.QLabel("Mob2 peak"), 3, 0)
        cal2_lay.addWidget(self.mob_spin2, 3, 1)
        self.btn_auto2 = QtWidgets.QPushButton("Auto2")
        self.btn_reset2 = QtWidgets.QPushButton("Reset2")
        self.btn_detect2 = QtWidgets.QPushButton("Detect2")
        self.btn_pick_mob2 = QtWidgets.QPushButton("Mob2Pk")
        row2 = QtWidgets.QHBoxLayout()
        for b in (self.btn_auto2, self.btn_reset2, self.btn_detect2, self.btn_pick_mob2, self.btn_save2):
            row2.addWidget(b)
        cal2_lay.addLayout(row2, 4, 0, 1, 3)
        spec2_lay.addWidget(cal2_box, 1)

        fit2_box = QtWidgets.QGroupBox("Fit Controls 2")
        fit2_lay = QtWidgets.QVBoxLayout(fit2_box)
        mode2_row = QtWidgets.QHBoxLayout()
        mode2_row.addWidget(QtWidgets.QLabel("Mode2:"))
        self.fit_mode_combo2 = QtWidgets.QComboBox()
        self.fit_mode_combo2.addItems(["Peak LSQ", "Spectrum LSQ"])
        mode2_row.addWidget(self.fit_mode_combo2)
        fit2_lay.addLayout(mode2_row)
        self.peak_table2 = QtWidgets.QTableWidget(0, 4)
        self.peak_table2.setHorizontalHeaderLabels(["Use", "Ref ch", "Mob ch", "dCh"])
        self.peak_table2.horizontalHeader().setStretchLastSection(True)
        self.peak_table2.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.peak_table2.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.peak_table2.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        fit2_lay.addWidget(self.peak_table2)
        pair2_btns = QtWidgets.QHBoxLayout()
        self.btn_add_pair2 = QtWidgets.QPushButton("Add Pair2")
        self.btn_remove_pair2 = QtWidgets.QPushButton("Remove Pair2")
        self.btn_clear_pairs2 = QtWidgets.QPushButton("Clear Pairs2")
        pair2_btns.addWidget(self.btn_add_pair2)
        pair2_btns.addWidget(self.btn_remove_pair2)
        pair2_btns.addWidget(self.btn_clear_pairs2)
        fit2_lay.addLayout(pair2_btns)
        spec2_lay.addWidget(fit2_box, 1)

        list_box = QtWidgets.QGroupBox("Loaded Mobile Spectra")
        list_lay = QtWidgets.QVBoxLayout(list_box)
        self.mob_list = QtWidgets.QListWidget()
        self.btn_load_mob = QtWidgets.QPushButton("Load...")
        self.btn_remove_mob = QtWidgets.QPushButton("Remove Selected")
        self.btn_clear_mob = QtWidgets.QPushButton("Clear List")
        self.btn_to_ref = QtWidgets.QPushButton("To Ref")
        self.btn_to_s1 = QtWidgets.QPushButton("To S1")
        self.btn_to_s2 = QtWidgets.QPushButton("To S2")
        list_btns = QtWidgets.QHBoxLayout()
        list_btns.addWidget(self.btn_load_mob)
        list_btns.addWidget(self.btn_remove_mob)
        list_btns.addWidget(self.btn_clear_mob)
        list_btns.addWidget(self.btn_to_ref)
        list_btns.addWidget(self.btn_to_s1)
        list_btns.addWidget(self.btn_to_s2)
        list_lay.addWidget(self.mob_list)
        list_lay.addLayout(list_btns)
        general_lay.addWidget(list_box, 1)

        self._sync_guard = False

        self.m_slider.valueChanged.connect(self._on_m_slider)
        self.q_slider.valueChanged.connect(self._on_q_slider)
        self.hw_slider.valueChanged.connect(self._on_hw_slider)
        self.m_spin.valueChanged.connect(self._on_m_spin)
        self.q_spin.valueChanged.connect(self._on_q_spin)
        self.hw_spin.valueChanged.connect(self._on_hw_spin)
        self.m_slider2.valueChanged.connect(self._on_m_slider2)
        self.q_slider2.valueChanged.connect(self._on_q_slider2)
        self.hw_slider2.valueChanged.connect(self._on_hw_slider2)
        self.m_spin2.valueChanged.connect(self._on_m_spin2)
        self.q_spin2.valueChanged.connect(self._on_q_spin2)
        self.hw_spin2.valueChanged.connect(self._on_hw_spin2)

        for rb in (self.rb_norm_none, self.rb_norm_peak, self.rb_norm_area, self.rb_norm_integral):
            rb.toggled.connect(lambda _checked: self.update_plot())
        for rb in (self.rb_y_linear, self.rb_y_log):
            rb.toggled.connect(lambda _checked: self.update_plot())

        self.ref_spin.valueChanged.connect(lambda _v: self.update_plot())
        self.mob_spin.valueChanged.connect(lambda _v: self.update_plot())
        self.mob_spin2.valueChanged.connect(lambda _v: self.update_plot())

        self.btn_auto.clicked.connect(self.do_auto)
        self.btn_reset.clicked.connect(self.do_reset)
        self.btn_detect.clicked.connect(self.do_detect)
        self.btn_save.clicked.connect(self.do_save)
        self.btn_save2.clicked.connect(self.do_save2)
        self.btn_subtract.clicked.connect(self.do_subtract)
        self.btn_save_sub.clicked.connect(self.do_save_subtract)
        self.btn_save_sub3.clicked.connect(self.do_save_subtract_3col)
        self.btn_clear.clicked.connect(self.do_clear)
        self.btn_pick_ref.clicked.connect(self.pick_ref_peak)
        self.btn_pick_mob.clicked.connect(self.pick_mob_peak)
        self.btn_pick_mob2.clicked.connect(self.pick_mob2_peak)
        self.btn_pick_range.clicked.connect(self.pick_sum_range)
        self.btn_sum_all.clicked.connect(self.do_sum_all)
        self.btn_print_params.clicked.connect(self.do_print_sum_params)
        self.sub_mode_combo.currentIndexChanged.connect(lambda _idx: self.update_plot())
        self.btn_auto2.clicked.connect(self.do_auto2)
        self.btn_reset2.clicked.connect(self.do_reset2)
        self.btn_detect2.clicked.connect(self.do_detect2)
        self.btn_load_mob.clicked.connect(self.load_mobile_files)
        self.btn_remove_mob.clicked.connect(self.remove_selected_mobile)
        self.btn_clear_mob.clicked.connect(self.clear_mobile_list)
        self.btn_to_ref.clicked.connect(self.assign_selected_to_ref)
        self.btn_to_s1.clicked.connect(self.assign_selected_to_s1)
        self.btn_to_s2.clicked.connect(self.assign_selected_to_s2)
        self.mob_list.currentRowChanged.connect(self.on_mobile_selected)
        self.mob_select1.currentIndexChanged.connect(lambda row: self.on_mobile_selected_slot(1, row))
        self.mob_select2.currentIndexChanged.connect(lambda row: self.on_mobile_selected_slot(2, row))
        self.fit_mode_combo.currentTextChanged.connect(self._on_fit_mode_changed)
        self.fit_mode_combo2.currentTextChanged.connect(self._on_fit_mode2_changed)
        self.peak_table.itemChanged.connect(self._on_peak_table_item_changed)
        self.peak_table2.itemChanged.connect(self._on_peak_table2_item_changed)
        self.btn_add_pair.clicked.connect(self.begin_add_peak_pair)
        self.btn_remove_pair.clicked.connect(self.remove_selected_pair)
        self.btn_clear_pairs.clicked.connect(self.clear_peak_pairs)
        self.btn_add_pair2.clicked.connect(self.begin_add_peak_pair2)
        self.btn_remove_pair2.clicked.connect(self.remove_selected_pair2)
        self.btn_clear_pairs2.clicked.connect(self.clear_peak_pairs2)

    @property
    def mob_counts(self) -> np.ndarray:
        if not self.mobile_items:
            return np.zeros_like(self.ref_counts)
        return np.array(self.mobile_items[self.index1]["counts"], dtype=np.float64)

    @property
    def mob_counts2(self) -> np.ndarray:
        if not self.mobile_items:
            return np.zeros_like(self.ref_counts)
        return np.array(self.mobile_items[self.index2]["counts"], dtype=np.float64)

    @property
    def mob_meta(self) -> dict[str, object]:
        if not self.mobile_items:
            return {}
        raw = self.mobile_items[self.index1]["meta"]
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def mob_meta2(self) -> dict[str, object]:
        if not self.mobile_items:
            return {}
        raw = self.mobile_items[self.index2]["meta"]
        return dict(raw) if isinstance(raw, dict) else {}

    @property
    def mob_path(self) -> str:
        if not self.mobile_items:
            return ""
        return str(self.mobile_items[self.index1]["path"])

    @property
    def mob_path2(self) -> str:
        if not self.mobile_items:
            return ""
        return str(self.mobile_items[self.index2]["path"])

    def _set_status(self, text: str, ok: bool = True) -> None:
        color = "#9ee0b5" if ok else "#ff9a9a"
        self.status.setText(f"<span style='color:{color}'>{text}</span>")

    def _update_nav(self) -> None:
        if not self.mobile_items:
            self.nav_label.setText("Match1 0/0 | Match2 0/0")
        else:
            self.nav_label.setText(
                f"Match1 {self.index1 + 1}/{len(self.mobile_items)} | Match2 {self.index2 + 1}/{len(self.mobile_items)}"
            )

    def _on_fit_mode_changed(self, text: str) -> None:
        self.fit_mode = "spectrum_lsq" if text.lower().startswith("spectrum") else "peaks_lsq"
        self.update_plot()

    def _on_fit_mode2_changed(self, text: str) -> None:
        self.fit_mode2 = "spectrum_lsq" if text.lower().startswith("spectrum") else "peaks_lsq"
        self.update_plot()

    def _selected_peak_pairs(self) -> tuple[np.ndarray, np.ndarray]:
        return self._selected_peak_pairs_from_table(self.peak_table)

    def _selected_peak_pairs2(self) -> tuple[np.ndarray, np.ndarray]:
        return self._selected_peak_pairs_from_table(self.peak_table2)

    def _selected_peak_pairs_from_table(self, table: QtWidgets.QTableWidget) -> tuple[np.ndarray, np.ndarray]:
        ref_vals: list[int] = []
        mob_vals: list[int] = []
        for r in range(table.rowCount()):
            use_item = table.item(r, 0)
            ref_item = table.item(r, 1)
            mob_item = table.item(r, 2)
            if use_item is None or ref_item is None or mob_item is None:
                continue
            if use_item.checkState() != QtCore.Qt.CheckState.Checked:
                continue
            try:
                rv = int(ref_item.text())
                mv = int(mob_item.text())
            except ValueError:
                continue
            ref_vals.append(rv)
            mob_vals.append(mv)
        return np.array(ref_vals, dtype=np.int64), np.array(mob_vals, dtype=np.int64)

    def _on_peak_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() in (0, 1, 2):
            self._update_peak_pair_deltas()
            self.update_plot()

    def _on_peak_table2_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item.column() in (0, 1, 2):
            self._update_peak_pair_deltas2()
            self.update_plot()

    def _append_peak_pair(self, ref_ch: int, mob_ch: int, checked: bool = True) -> None:
        self._append_peak_pair_to_table(self.peak_table, ref_ch, mob_ch, checked=checked)

    def _append_peak_pair2(self, ref_ch: int, mob_ch: int, checked: bool = True) -> None:
        self._append_peak_pair_to_table(self.peak_table2, ref_ch, mob_ch, checked=checked)

    def _append_peak_pair_to_table(self, table: QtWidgets.QTableWidget, ref_ch: int, mob_ch: int, checked: bool = True) -> None:
        row = table.rowCount()
        table.blockSignals(True)
        table.insertRow(row)

        use = QtWidgets.QTableWidgetItem("")
        use.setFlags(use.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
        use.setCheckState(QtCore.Qt.CheckState.Checked if checked else QtCore.Qt.CheckState.Unchecked)
        table.setItem(row, 0, use)
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(str(int(ref_ch))))
        table.setItem(row, 2, QtWidgets.QTableWidgetItem(str(int(mob_ch))))
        table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))
        table.blockSignals(False)
        if table is self.peak_table:
            self._update_peak_pair_deltas()
        else:
            self._update_peak_pair_deltas2()

    def _update_peak_pair_deltas(self) -> None:
        m = float(self.m_spin.value()) if hasattr(self, "m_spin") else 1.0
        q = float(self.q_spin.value()) if hasattr(self, "q_spin") else 0.0
        self.peak_table.blockSignals(True)
        for r in range(self.peak_table.rowCount()):
            ref_item = self.peak_table.item(r, 1)
            mob_item = self.peak_table.item(r, 2)
            if ref_item is None or mob_item is None:
                continue
            try:
                rch = int(ref_item.text())
                mch = int(mob_item.text())
            except ValueError:
                continue
            dch = float(rch) - (m * float(mch) + q)
            self.peak_table.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{dch:+.2f}"))
        self.peak_table.blockSignals(False)

    def _update_peak_pair_deltas2(self) -> None:
        m = float(self.m_spin2.value()) if hasattr(self, "m_spin2") else 1.0
        q = float(self.q_spin2.value()) if hasattr(self, "q_spin2") else 0.0
        self.peak_table2.blockSignals(True)
        for r in range(self.peak_table2.rowCount()):
            ref_item = self.peak_table2.item(r, 1)
            mob_item = self.peak_table2.item(r, 2)
            if ref_item is None or mob_item is None:
                continue
            try:
                rch = int(ref_item.text())
                mch = int(mob_item.text())
            except ValueError:
                continue
            dch = float(rch) - (m * float(mch) + q)
            self.peak_table2.setItem(r, 3, QtWidgets.QTableWidgetItem(f"{dch:+.2f}"))
        self.peak_table2.blockSignals(False)

    def begin_add_peak_pair(self) -> None:
        self.pick_mode = "pair_ref"
        self.pending_pair_ref = None
        self._set_status("Click reference peak, then matched peak", ok=True)

    def begin_add_peak_pair2(self) -> None:
        self.pick_mode = "pair_ref2"
        self.pending_pair_ref = None
        self._set_status("Click reference peak, then matched2 peak", ok=True)

    def remove_selected_pair(self) -> None:
        row = self.peak_table.currentRow()
        if row < 0:
            self._set_status("No peak pair selected", ok=False)
            return
        self.peak_table.removeRow(row)
        self._update_peak_pair_deltas()
        self._set_status("Removed selected peak pair", ok=True)
        self.update_plot()

    def remove_selected_pair2(self) -> None:
        row = self.peak_table2.currentRow()
        if row < 0:
            self._set_status("No peak pair2 selected", ok=False)
            return
        self.peak_table2.removeRow(row)
        self._update_peak_pair_deltas2()
        self._set_status("Removed selected peak pair2", ok=True)
        self.update_plot()

    def clear_peak_pairs(self) -> None:
        self.peak_table.setRowCount(0)
        self._set_status("Cleared peak pairs", ok=True)
        self.update_plot()

    def clear_peak_pairs2(self) -> None:
        self.peak_table2.setRowCount(0)
        self._set_status("Cleared peak pairs2", ok=True)
        self.update_plot()

    def _refresh_peak_pairs_table(self) -> None:
        self.peak_table.blockSignals(True)
        self.peak_table.setRowCount(0)
        hw = int(self.hw_spin.value()) if hasattr(self, "hw_spin") else 60
        m = float(self.m_spin.value()) if hasattr(self, "m_spin") else 1.0
        q = float(self.q_spin.value()) if hasattr(self, "q_spin") else 0.0
        paired_ref, paired_mob = pair_peaks_nearest_mapped(
            self.ref_peaks,
            self.mob_peaks_raw,
            m,
            q,
            max_delta=max(20.0, float(2 * hw)),
        )
        k = min(len(paired_ref), len(paired_mob))
        for i in range(k):
            self.peak_table.insertRow(i)
            use = QtWidgets.QTableWidgetItem("")
            use.setFlags(use.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked)
            self.peak_table.setItem(i, 0, use)

            rch = int(paired_ref[i])
            mch = int(paired_mob[i])
            self.peak_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(rch)))
            self.peak_table.setItem(i, 2, QtWidgets.QTableWidgetItem(str(mch)))
            self.peak_table.setItem(i, 3, QtWidgets.QTableWidgetItem(""))
        self.peak_table.blockSignals(False)
        self._update_peak_pair_deltas()

    def _refresh_peak_pairs_table2(self) -> None:
        self.peak_table2.blockSignals(True)
        self.peak_table2.setRowCount(0)
        hw = int(self.hw_spin2.value()) if hasattr(self, "hw_spin2") else 60
        m = float(self.m_spin2.value()) if hasattr(self, "m_spin2") else 1.0
        q = float(self.q_spin2.value()) if hasattr(self, "q_spin2") else 0.0
        paired_ref, paired_mob = pair_peaks_nearest_mapped(
            self.ref_peaks,
            self.mob_peaks_raw2,
            m,
            q,
            max_delta=max(20.0, float(2 * hw)),
        )
        k = min(len(paired_ref), len(paired_mob))
        for i in range(k):
            self.peak_table2.insertRow(i)
            use = QtWidgets.QTableWidgetItem("")
            use.setFlags(use.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable)
            use.setCheckState(QtCore.Qt.CheckState.Checked)
            self.peak_table2.setItem(i, 0, use)

            rch = int(paired_ref[i])
            mch = int(paired_mob[i])
            self.peak_table2.setItem(i, 1, QtWidgets.QTableWidgetItem(str(rch)))
            self.peak_table2.setItem(i, 2, QtWidgets.QTableWidgetItem(str(mch)))
            self.peak_table2.setItem(i, 3, QtWidgets.QTableWidgetItem(""))
        self.peak_table2.blockSignals(False)
        self._update_peak_pair_deltas2()

    def _fit_by_spectrum_lsq(
        self,
        m0: float,
        q0: float,
        ref_peaks: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        ref = np.array(self.ref_counts, dtype=np.float64)
        mob = np.array(self.mob_counts, dtype=np.float64)
        if len(ref) == 0 or len(mob) == 0:
            return None

        hw = int(self.hw_spin.value())
        mask = np.zeros(len(ref), dtype=bool)
        if len(ref_peaks) >= 1:
            for p in ref_peaks:
                lo = max(0, int(p) - hw)
                hi = min(len(ref) - 1, int(p) + hw)
                mask[lo : hi + 1] = True
        else:
            mask[:] = True

        idx = np.where(mask)[0]
        if idx.size < 5:
            return None

        def resid(params: np.ndarray) -> np.ndarray:
            m = float(params[0])
            q = float(params[1])
            a = float(params[2])
            rem = remap_spectrum(mob, m, q)
            return ref[idx] - a * rem[idx]

        try:
            res = least_squares(
                resid,
                x0=np.array([m0, q0, 1.0], dtype=np.float64),
                bounds=(np.array([0.5, -600.0, 0.01]), np.array([1.5, 600.0, 100.0])),
            )
        except Exception:
            return None

        if not res.success or res.x.size < 2:
            return None
        m = float(res.x[0])
        q = float(res.x[1])
        if not np.isfinite(m) or not np.isfinite(q):
            return None
        return m, q

    def _fit_by_spectrum_lsq2(
        self,
        m0: float,
        q0: float,
        ref_peaks: np.ndarray,
    ) -> Optional[tuple[float, float]]:
        ref = np.array(self.ref_counts, dtype=np.float64)
        mob = np.array(self.mob_counts2, dtype=np.float64)
        if len(ref) == 0 or len(mob) == 0:
            return None

        hw = int(self.hw_spin2.value())
        mask = np.zeros(len(ref), dtype=bool)
        if len(ref_peaks) >= 1:
            for p in ref_peaks:
                lo = max(0, int(p) - hw)
                hi = min(len(ref) - 1, int(p) + hw)
                mask[lo : hi + 1] = True
        else:
            mask[:] = True

        idx = np.where(mask)[0]
        if idx.size < 5:
            return None

        def resid(params: np.ndarray) -> np.ndarray:
            m = float(params[0])
            q = float(params[1])
            a = float(params[2])
            rem = remap_spectrum(mob, m, q)
            return ref[idx] - a * rem[idx]

        try:
            res = least_squares(
                resid,
                x0=np.array([m0, q0, 1.0], dtype=np.float64),
                bounds=(np.array([0.5, -600.0, 0.01]), np.array([1.5, 600.0, 100.0])),
            )
        except Exception:
            return None

        if not res.success or res.x.size < 2:
            return None
        m = float(res.x[0])
        q = float(res.x[1])
        if not np.isfinite(m) or not np.isfinite(q):
            return None
        return m, q

    def _refresh_mobile_list(self) -> None:
        self.mob_list.blockSignals(True)
        self.mob_select1.blockSignals(True)
        self.mob_select2.blockSignals(True)
        self.mob_list.clear()
        self.mob_select1.clear()
        self.mob_select2.clear()
        for item in self.mobile_items:
            p = str(item.get("path", ""))
            name = os.path.basename(p) if p else "<unnamed>"
            self.mob_list.addItem(name)
            self.mob_select1.addItem(name)
            self.mob_select2.addItem(name)
        if self.mobile_items:
            self.index1 = int(np.clip(self.index1, 0, len(self.mobile_items) - 1))
            self.index2 = int(np.clip(self.index2, 0, len(self.mobile_items) - 1))
            self.mob_list.setCurrentRow(self.index1)
            self.mob_select1.setCurrentIndex(self.index1)
            self.mob_select2.setCurrentIndex(self.index2)
        self.mob_select1.blockSignals(False)
        self.mob_select2.blockSignals(False)
        self.mob_list.blockSignals(False)

    def _pad_all_to_max_len(self) -> None:
        max_len = len(self.ref_counts)
        for item in self.mobile_items:
            c = np.array(item["counts"], dtype=np.float64)
            max_len = max(max_len, len(c))

        if len(self.ref_counts) < max_len:
            self.ref_counts = np.pad(self.ref_counts, (0, max_len - len(self.ref_counts)))
            self.ref_spin.setRange(0, len(self.ref_counts) - 1)
            self.mob_spin.setRange(0, len(self.ref_counts) - 1)
            self.mob_spin2.setRange(0, len(self.ref_counts) - 1)

        for item in self.mobile_items:
            c = np.array(item["counts"], dtype=np.float64)
            if len(c) < max_len:
                item["counts"] = np.pad(c, (0, max_len - len(c)))

    def _refresh_reference_curve_label(self) -> None:
        ref_label = os.path.basename(str(self.ref_path)) if hasattr(self, "ref_path") else "Reference"
        x = np.arange(len(self.ref_counts), dtype=np.float64)
        try:
            self.plot.removeItem(self.curve_ref)
        except Exception:
            pass
        self.curve_ref = self.plot.plot(x=x, y=self.ref_counts, pen=pg.mkPen("#4fc3f7", width=1.2), name=f"Reference: {ref_label}")

    def load_reference_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Reference Spectrum",
            "",
            "Spectrum files (*.Spe *.spe);;All files (*)",
        )
        if not file_path:
            return

        try:
            ref_counts, ref_meta = parse_spe(file_path)
        except Exception as exc:
            self._set_status(f"Failed to load reference: {exc}", ok=False)
            return

        self.ref_path = file_path
        self.ref_counts = np.array(ref_counts, dtype=np.float64)
        self.ref_meta = dict(ref_meta) if isinstance(ref_meta, dict) else {}
        self.ref_center = detect_primary_peak(self.ref_counts)
        self.ref_peaks = detect_peaks_scipy(self.ref_counts, max_peaks=12, min_distance=20)

        self._pad_all_to_max_len()
        self.ref_spin.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin2.setRange(0, len(self.ref_counts) - 1)
        self.ref_spin.setValue(self.ref_center)

        self.remapped_unscaled = np.zeros_like(self.ref_counts)
        self.remapped_unscaled2 = np.zeros_like(self.ref_counts)
        self.scaled = np.zeros_like(self.ref_counts)
        self.scaled2 = np.zeros_like(self.ref_counts)
        self.subtracted = np.zeros_like(self.ref_counts)
        self.subtracted2 = np.zeros_like(self.ref_counts)
        self.subtracted_dy = np.zeros_like(self.ref_counts)
        self.subtracted_dy2 = np.zeros_like(self.ref_counts)

        self._refresh_reference_curve_label()
        if self.mobile_items:
            self._load_mobile_slot(1, self.index1 if self.index1 < len(self.mobile_items) else 0)
            self._load_mobile_slot(2, self.index2 if self.index2 < len(self.mobile_items) else 0)
        self._set_status(f"Loaded reference {os.path.basename(file_path)}", ok=True)

    def load_mobile_files(self) -> None:
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Load Mobile Spectra",
            "",
            "Spectrum files (*.Spe *.spe);;All files (*)",
        )
        if not files:
            return

        added = 0
        for p in files:
            try:
                counts, meta = parse_spe(p)
            except Exception as exc:
                self._set_status(f"Failed to load {os.path.basename(p)}: {exc}", ok=False)
                continue
            self.mobile_items.append({"path": p, "counts": counts, "meta": meta})
            added += 1

        if added == 0:
            return

        self._pad_all_to_max_len()
        if len(self.mobile_items) == added:
            self.index1 = 0
            self.index2 = 1 if len(self.mobile_items) > 1 else 0
        self._refresh_mobile_list()
        self._load_mobile_slot(1, self.index1)
        self._load_mobile_slot(2, self.index2)
        self._set_status(f"Loaded {added} spectrum(s)", ok=True)

    def remove_selected_mobile(self) -> None:
        row = self.mob_list.currentRow()
        if row < 0 or row >= len(self.mobile_items):
            self._set_status("No spectrum selected", ok=False)
            return

        removed_name = os.path.basename(str(self.mobile_items[row].get("path", "")))
        del self.mobile_items[row]

        if not self.mobile_items:
            self.index1 = 0
            self.index2 = 0
            self._refresh_mobile_list()
            self._update_nav()
            self._set_status("No mobile spectra loaded", ok=False)
            return

        self.index1 = min(self.index1, len(self.mobile_items) - 1)
        self.index2 = min(self.index2, len(self.mobile_items) - 1)
        self._refresh_mobile_list()
        self._load_mobile_slot(1, self.index1)
        self._load_mobile_slot(2, self.index2)
        self._set_status(f"Removed {removed_name}", ok=True)

    def clear_mobile_list(self) -> None:
        self.mobile_items = []
        self.index1 = 0
        self.index2 = 0
        self.ref_peaks = np.array([], dtype=np.int64)
        self.mob_peaks_raw = np.array([], dtype=np.int64)
        self.mob_peaks_raw2 = np.array([], dtype=np.int64)
        self._refresh_peak_pairs_table()
        self._refresh_peak_pairs_table2()
        self._refresh_mobile_list()
        self._update_nav()
        self._set_status("Cleared mobile spectra list", ok=True)

    def on_mobile_selected(self, row: int) -> None:
        if row < 0 or row >= len(self.mobile_items):
            return
        self.mob_select1.blockSignals(True)
        self.mob_select1.setCurrentIndex(row)
        self.mob_select1.blockSignals(False)
        self.on_mobile_selected_slot(1, row)

    def on_mobile_selected_slot(self, slot: int, row: int) -> None:
        if row < 0 or row >= len(self.mobile_items):
            return
        if slot == 1 and row == self.index1:
            return
        if slot == 2 and row == self.index2:
            return
        self._load_mobile_slot(slot, row)
        self._set_status(f"Switched match{slot} spectrum", ok=True)

    def assign_selected_to_ref(self) -> None:
        row = self.mob_list.currentRow()
        if row < 0 or row >= len(self.mobile_items):
            self._set_status("No spectrum selected", ok=False)
            return

        item = self.mobile_items[row]
        self.ref_path = str(item.get("path", self.ref_path))
        self.ref_counts = np.array(item.get("counts", np.zeros_like(self.ref_counts)), dtype=np.float64)
        raw_meta = item.get("meta", {})
        self.ref_meta = dict(raw_meta) if isinstance(raw_meta, dict) else {}
        self.ref_center = detect_primary_peak(self.ref_counts)
        self.ref_peaks = detect_peaks_scipy(self.ref_counts, max_peaks=12, min_distance=20)

        self._pad_all_to_max_len()
        self.ref_spin.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin.setRange(0, len(self.ref_counts) - 1)
        self.mob_spin2.setRange(0, len(self.ref_counts) - 1)
        self.ref_spin.setValue(self.ref_center)

        self.remapped_unscaled = np.zeros_like(self.ref_counts)
        self.remapped_unscaled2 = np.zeros_like(self.ref_counts)
        self.scaled = np.zeros_like(self.ref_counts)
        self.scaled2 = np.zeros_like(self.ref_counts)
        self.subtracted = np.zeros_like(self.ref_counts)
        self.subtracted2 = np.zeros_like(self.ref_counts)
        self.subtracted_dy = np.zeros_like(self.ref_counts)
        self.subtracted_dy2 = np.zeros_like(self.ref_counts)

        self._refresh_reference_curve_label()
        if self.mobile_items:
            self._load_mobile_slot(1, self.index1 if self.index1 < len(self.mobile_items) else 0)
            self._load_mobile_slot(2, self.index2 if self.index2 < len(self.mobile_items) else 0)
        self._set_status(f"Set reference from {os.path.basename(self.ref_path)}", ok=True)

    def assign_selected_to_s1(self) -> None:
        row = self.mob_list.currentRow()
        if row < 0 or row >= len(self.mobile_items):
            self._set_status("No spectrum selected", ok=False)
            return
        self._load_mobile_slot(1, row)
        self._set_status(f"Set Spectrum1 from {os.path.basename(self.mob_path)}", ok=True)

    def assign_selected_to_s2(self) -> None:
        row = self.mob_list.currentRow()
        if row < 0 or row >= len(self.mobile_items):
            self._set_status("No spectrum selected", ok=False)
            return
        self._load_mobile_slot(2, row)
        self._set_status(f"Set Spectrum2 from {os.path.basename(self.mob_path2)}", ok=True)

    def _set_m(self, val: float) -> None:
        self._sync_guard = True
        self.m_spin.setValue(val)
        self.m_slider.setValue(int(round(val * 10000.0)))
        self._sync_guard = False

    def _set_m2(self, val: float) -> None:
        self._sync_guard = True
        self.m_spin2.setValue(val)
        self.m_slider2.setValue(int(round(val * 10000.0)))
        self._sync_guard = False

    def _set_q(self, val: float) -> None:
        self._sync_guard = True
        self.q_spin.setValue(val)
        self.q_slider.setValue(int(round(val * 10.0)))
        self._sync_guard = False

    def _set_q2(self, val: float) -> None:
        self._sync_guard = True
        self.q_spin2.setValue(val)
        self.q_slider2.setValue(int(round(val * 10.0)))
        self._sync_guard = False

    def _set_hw(self, val: int) -> None:
        self._sync_guard = True
        self.hw_spin.setValue(int(val))
        self.hw_slider.setValue(int(val))
        self._sync_guard = False

    def _set_hw2(self, val: int) -> None:
        self._sync_guard = True
        self.hw_spin2.setValue(int(val))
        self.hw_slider2.setValue(int(val))
        self._sync_guard = False

    def _on_m_slider(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_m(float(value) / 10000.0)
        self._update_peak_pair_deltas()
        self.update_plot()

    def _on_q_slider(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_q(float(value) / 10.0)
        self._update_peak_pair_deltas()
        self.update_plot()

    def _on_hw_slider(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_hw(int(value))
        if self.peak_table.rowCount() == 0:
            self._refresh_peak_pairs_table()
        else:
            self._update_peak_pair_deltas()
        self.update_plot()

    def _on_m_spin(self, value: float) -> None:
        if self._sync_guard:
            return
        self._set_m(float(value))
        self._update_peak_pair_deltas()
        self.update_plot()

    def _on_m_slider2(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_m2(float(value) / 10000.0)
        self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_q_slider2(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_q2(float(value) / 10.0)
        self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_hw_slider2(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_hw2(int(value))
        if self.peak_table2.rowCount() == 0:
            self._refresh_peak_pairs_table2()
        else:
            self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_m_spin2(self, value: float) -> None:
        if self._sync_guard:
            return
        self._set_m2(float(value))
        self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_q_spin2(self, value: float) -> None:
        if self._sync_guard:
            return
        self._set_q2(float(value))
        self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_hw_spin2(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_hw2(int(value))
        if self.peak_table2.rowCount() == 0:
            self._refresh_peak_pairs_table2()
        else:
            self._update_peak_pair_deltas2()
        self.update_plot()

    def _on_q_spin(self, value: float) -> None:
        if self._sync_guard:
            return
        self._set_q(float(value))
        self._update_peak_pair_deltas()
        self.update_plot()

    def _on_hw_spin(self, value: int) -> None:
        if self._sync_guard:
            return
        self._set_hw(int(value))
        if self.peak_table.rowCount() == 0:
            self._refresh_peak_pairs_table()
        else:
            self._update_peak_pair_deltas()
        self.update_plot()

    def _selected_norm_mode(self) -> str:
        if self.rb_norm_none.isChecked():
            return "none"
        if self.rb_norm_area.isChecked():
            return "area"
        if self.rb_norm_integral.isChecked():
            return "integral"
        return "peak"

    def _selected_y_mode(self) -> str:
        return "log" if self.rb_y_log.isChecked() else "linear"

    def _selected_sub_mode(self) -> str:
        data = self.sub_mode_combo.currentData() if hasattr(self, "sub_mode_combo") else None
        return str(data) if data else "1-2"

    def _selected_sub_mode_label(self) -> str:
        text = self.sub_mode_combo.currentText() if hasattr(self, "sub_mode_combo") else "1 - 2"
        return str(text).replace(" ", "")

    def _sum_report_lines(self, bounds: tuple[int, int]) -> list[str]:
        (sum_r, sig_r), (sum_m1, sig_m1), (sum_m2, sig_m2), (sum_s, sig_s), _other = self._sum_quintet_for_range(bounds)
        ref_name = os.path.basename(str(self.ref_path))
        m1_name = os.path.basename(self.mob_path)
        m2_name = os.path.basename(self.mob_path2)
        sub_label = f"Sub({self._selected_sub_mode_label()})"
        return [
            f"{ref_name} SumR = {sum_r:.6g} +/- {sig_r:.6g}",
            f"{m1_name} SumM1 = {sum_m1:.6g} +/- {sig_m1:.6g}",
            f"{m2_name} SumM2 = {sum_m2:.6g} +/- {sig_m2:.6g}",
            f"{sub_label} = {sum_s:.6g} +/- {sig_s:.6g}",
        ]

    def _load_mobile(self, idx: int) -> None:
        self._load_mobile_slot(1, idx)

    def _load_mobile_slot(self, slot: int, idx: int, apply_auto: bool = True) -> None:
        if not self.mobile_items:
            return

        idx = int(np.clip(idx, 0, len(self.mobile_items) - 1))
        if slot == 1:
            self.index1 = idx
            mob = self.mob_counts
            self.mob_center_raw = detect_primary_peak(mob)
            self.mob_spin.setValue(self.mob_center_raw)
            if apply_auto:
                m0, q0, _, _ = auto_calibrate(self.ref_counts, mob, 60, self.ref_center, self.mob_center_raw)
                self._set_m(float(np.clip(m0, 0.8, 1.2)))
                self._set_q(float(np.clip(q0, -300.0, 300.0)))
            else:
                self._set_m(1.0)
                self._set_q(0.0)
            self._set_hw(60)
            self.mob_peaks_raw = detect_peaks_scipy(mob, max_peaks=12, min_distance=20)
            self._refresh_peak_pairs_table()
        else:
            self.index2 = idx
            mob = self.mob_counts2
            self.mob_center_raw2 = detect_primary_peak(mob)
            self.mob_spin2.setValue(self.mob_center_raw2)
            if apply_auto:
                m0, q0, _, _ = auto_calibrate(self.ref_counts, mob, 60, self.ref_center, self.mob_center_raw2)
                self._set_m2(float(np.clip(m0, 0.8, 1.2)))
                self._set_q2(float(np.clip(q0, -300.0, 300.0)))
            else:
                self._set_m2(1.0)
                self._set_q2(0.0)
            self._set_hw2(60)
            self.mob_peaks_raw2 = detect_peaks_scipy(mob, max_peaks=12, min_distance=20)
            self._refresh_peak_pairs_table2()

        self.ref_spin.setValue(self.ref_center)
        self.show_sub = False
        self.curve_sub.setVisible(False)
        self.curve_sub2.setVisible(False)
        self._refresh_matched_curve_label()
        self.ref_peaks = detect_peaks_scipy(self.ref_counts, max_peaks=12, min_distance=20)
        self.sum_ref = None
        self.sum_mob = None
        self.sum_mob2 = None
        self.sum_sub = None
        self.sum_sub2 = None
        self.selection_mode = None
        self.selection_start = None
        self._update_nav()

        self.mob_list.blockSignals(True)
        self.mob_list.setCurrentRow(self.index1)
        self.mob_list.blockSignals(False)
        self.mob_select1.blockSignals(True)
        self.mob_select2.blockSignals(True)
        self.mob_select1.setCurrentIndex(self.index1)
        self.mob_select2.setCurrentIndex(self.index2)
        self.mob_select1.blockSignals(False)
        self.mob_select2.blockSignals(False)
        self.update_plot()

    def _refresh_matched_curve_label(self) -> None:
        label = f"Matched 1: {os.path.basename(self.mob_path)}"
        label2 = f"Matched 2: {os.path.basename(self.mob_path2)}"
        try:
            self.plot.removeItem(self.curve_mob)
        except Exception:
            pass
        try:
            self.plot.removeItem(self.curve_mob2)
        except Exception:
            pass
        self.curve_mob = self.plot.plot(pen=pg.mkPen("#f4a261", width=1.2), name=label)
        self.curve_mob2 = self.plot.plot(pen=pg.mkPen("#e9c46a", width=1.2), name=label2)

    def _set_peak_channel(self, which: str, ch: int) -> None:
        chv = int(np.clip(ch, 0, len(self.ref_counts) - 1))
        if which == "ref":
            self.ref_spin.setValue(chv)
        elif which == "mob2":
            self.mob_spin2.setValue(chv)
        else:
            self.mob_spin.setValue(chv)

    def _fit_click_peak_channel(self, which: str, ch: int) -> int:
        hw = int(self.hw_spin.value()) if hasattr(self, "hw_spin") else 60
        chv = int(np.clip(ch, 0, len(self.ref_counts) - 1))
        if which == "ref":
            mu, _fwhm, _p, _a = peak_centroid_fwhm(self.ref_counts, chv, hw)
        elif which == "mob2":
            hw = int(self.hw_spin2.value()) if hasattr(self, "hw_spin2") else hw
            mu, _fwhm, _p, _a = peak_centroid_fwhm(self.mob_counts2, chv, hw)
        else:
            mu, _fwhm, _p, _a = peak_centroid_fwhm(self.mob_counts, chv, hw)
        return int(np.clip(round(mu), 0, len(self.ref_counts) - 1))

    def _on_ref_line_drag(self) -> None:
        if self._drag_guard:
            return
        self._set_peak_channel("ref", int(round(self.vline_ref.value())))
        self._set_status("Ref peak moved", ok=True)

    def _on_mob_line_drag(self) -> None:
        if self._drag_guard:
            return
        self._set_peak_channel("mob", int(round(self.vline_mob.value())))
        self._set_status("Mob peak moved", ok=True)

    def _on_ref_target_drag(self) -> None:
        if self._drag_guard or self.ref_target is None:
            return
        self._set_peak_channel("ref", int(round(self.ref_target.pos().x())))
        self._set_status("Ref peak moved", ok=True)

    def _on_mob_target_drag(self) -> None:
        if self._drag_guard or self.mob_target is None:
            return
        self._set_peak_channel("mob", int(round(self.mob_target.pos().x())))
        self._set_status("Mob peak moved", ok=True)

    def _parse_range(self, text: str) -> Optional[tuple[int, int]]:
        t = text.strip().replace(",", ":").replace("-", ":")
        parts = [p.strip() for p in t.split(":") if p.strip()]
        if len(parts) != 2:
            return None
        try:
            a = int(float(parts[0]))
            b = int(float(parts[1]))
        except ValueError:
            return None
        lo = int(np.clip(min(a, b), 0, len(self.ref_counts) - 1))
        hi = int(np.clip(max(a, b), 0, len(self.ref_counts) - 1))
        return lo, hi

    def _set_sum(self, which: str, bounds: tuple[int, int]) -> None:
        if which == "ref":
            self.sum_ref = bounds
        elif which == "mob":
            self.sum_mob = bounds
        else:
            self.sum_sub = bounds

    def _set_sum_range_all(self, bounds: tuple[int, int]) -> None:
        self.sum_ref = bounds
        self.sum_mob = bounds
        self.sum_mob2 = bounds
        self.sum_sub = bounds
        self.sum_sub2 = bounds
        self.range_edit.setText(f"{bounds[0]}:{bounds[1]}")

    def pick_sum_range(self) -> None:
        mode = "sum_all"
        if self.selection_mode == mode:
            self.selection_mode = None
            self.selection_start = None
            self._set_status("Range selection cancelled", ok=True)
            return
        self.selection_mode = mode
        self.selection_start = None
        self._set_status("Click two points for sum range", ok=True)

    def do_sum_all(self) -> None:
        if not self.mobile_items:
            self._set_status("No mobile spectra loaded", ok=False)
            return

        rng = self._active_sum_range()

        if rng is None:
            self._set_status("Set range in field (a:b) or click Pick Range", ok=False)
            return

        self._set_sum_range_all(rng)
        self.selection_mode = None
        self.selection_start = None
        self.update_plot()
        print("----- GainMatch Sum -----")
        print(f"Range: {rng[0]}:{rng[1]}")
        for line in self._sum_report_lines(rng):
            print(line)
        self._set_status(f"Summed range {rng[0]}:{rng[1]} and printed to terminal", ok=True)

    def _active_sum_range(self) -> Optional[tuple[int, int]]:
        rng = self._parse_range(self.range_edit.text())
        if rng is not None:
            return rng
        if isinstance(self.sum_ref, tuple):
            return self.sum_ref
        if isinstance(self.sum_mob, tuple):
            return self.sum_mob
        if isinstance(self.sum_mob2, tuple):
            return self.sum_mob2
        if isinstance(self.sum_sub, tuple):
            return self.sum_sub
        if isinstance(self.sum_sub2, tuple):
            return self.sum_sub2
        return None

    def _sum_triplet_for_range(self, bounds: tuple[int, int]) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        a, b = bounds
        yr = np.asarray(self.ref_counts[a : b + 1], dtype=np.float64)
        ym = np.asarray(self.remapped_unscaled[a : b + 1], dtype=np.float64)
        ys = np.asarray(self.subtracted[a : b + 1], dtype=np.float64)
        dys = np.asarray(self.subtracted_dy[a : b + 1], dtype=np.float64)

        sum_r = float(yr.sum())
        sig_r = float(np.sqrt(np.clip(yr, 0.0, None).sum()))
        sum_m = float(ym.sum())
        sig_m = float(np.sqrt(np.clip(ym, 0.0, None).sum()))
        sum_s = float(ys.sum())
        sig_s = float(np.sqrt(np.square(dys).sum()))
        return (sum_r, sig_r), (sum_m, sig_m), (sum_s, sig_s)

    def _sum_quintet_for_range(
        self, bounds: tuple[int, int]
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
        a, b = bounds
        yr = np.asarray(self.ref_counts[a : b + 1], dtype=np.float64)
        ym1 = np.asarray(self.remapped_unscaled[a : b + 1], dtype=np.float64)
        ym2 = np.asarray(self.remapped_unscaled2[a : b + 1], dtype=np.float64)
        ys12 = np.asarray(self.subtracted[a : b + 1], dtype=np.float64)
        ys21 = np.asarray(self.subtracted2[a : b + 1], dtype=np.float64)
        dys12 = np.asarray(self.subtracted_dy[a : b + 1], dtype=np.float64)
        dys21 = np.asarray(self.subtracted_dy2[a : b + 1], dtype=np.float64)

        sum_r = float(yr.sum())
        sig_r = float(np.sqrt(np.clip(yr, 0.0, None).sum()))
        sum_m1 = float(ym1.sum())
        sig_m1 = float(np.sqrt(np.clip(ym1, 0.0, None).sum()))
        sum_m2 = float(ym2.sum())
        sig_m2 = float(np.sqrt(np.clip(ym2, 0.0, None).sum()))
        sum_s12 = float(ys12.sum())
        sig_s12 = float(np.sqrt(np.square(dys12).sum()))
        sum_s21 = float(ys21.sum())
        sig_s21 = float(np.sqrt(np.square(dys21).sum()))
        return (sum_r, sig_r), (sum_m1, sig_m1), (sum_m2, sig_m2), (sum_s12, sig_s12), (sum_s21, sig_s21)

    def do_print_sum_params(self) -> None:
        if not self.mobile_items:
            self._set_status("No mobile spectra loaded", ok=False)
            return

        rng = self._active_sum_range()
        if rng is None:
            self._set_status("Set range in field (a:b) or click Pick Range", ok=False)
            return

        self._set_sum_range_all(rng)
        self.selection_mode = None
        self.selection_start = None
        self.update_plot()

        (sum_r, sig_r), (sum_m1, sig_m1), (sum_m2, sig_m2), (sum_s12, sig_s12), (sum_s21, sig_s21) = self._sum_quintet_for_range(rng)
        a, b = rng
        print("----- GainMatch Sum Parameters -----")
        print(f"Reference: {os.path.basename(str(self.ref_path))}")
        print(f"Target1:   {os.path.basename(self.mob_path)}")
        print(f"Target2:   {os.path.basename(self.mob_path2)}")
        print(f"Range:     {a}:{b}")
        print(f"SumR:      {sum_r:.6g} +/- {sig_r:.6g}")
        print(f"SumM1:     {sum_m1:.6g} +/- {sig_m1:.6g}")
        print(f"SumM2:     {sum_m2:.6g} +/- {sig_m2:.6g}")
        print(f"SumS12:    {sum_s12:.6g} +/- {sig_s12:.6g}")
        print(f"SumS21:    {sum_s21:.6g} +/- {sig_s21:.6g}")
        self._set_status("Printed sum parameters to terminal", ok=True)

    def update_plot(self) -> None:
        if not self.mobile_items:
            self.curve_mob.setData([], [])
            self.curve_mob2.setData([], [])
            self.curve_sub.setData([], [])
            self.curve_sub2.setData([], [])
            self.ref_peaks_scatter.setData([], [])
            self.mob_peaks_scatter.setData([], [])
            self.stats.setPlainText("No mobile spectra loaded.")
            return

        m = float(self.m_spin.value())
        q = float(self.q_spin.value())
        hw = int(self.hw_spin.value())
        m2 = float(self.m_spin2.value())
        q2 = float(self.q_spin2.value())
        hw2 = int(self.hw_spin2.value())
        ref_center = int(self.ref_spin.value())
        mob_center_raw = int(self.mob_spin.value())
        mob_center_raw2 = int(self.mob_spin2.value())
        norm_mode = self._selected_norm_mode()
        y_mode = self._selected_y_mode()
        self.plot.getPlotItem().setLogMode(x=False, y=(y_mode == "log"))
        
        # Set y-axis range when in log mode
        if y_mode == "log":
            y_max = float(np.max(self.ref_counts)) + 500
            self.plot.getPlotItem().setYRange(0.9, y_max, padding=0)

        mob_raw = self.mob_counts
        mob_raw2 = self.mob_counts2
        remapped = remap_spectrum(mob_raw, m, q)
        remapped2 = remap_spectrum(mob_raw2, m2, q2)
        mapped_center = int(np.clip(round(m * mob_center_raw + q), 0, len(self.ref_counts) - 1))
        mapped_center2 = int(np.clip(round(m2 * mob_center_raw2 + q2), 0, len(self.ref_counts) - 1))

        sf = scale_factor(self.ref_counts, remapped, norm_mode, hw, ref_center, mapped_center)
        sf2 = scale_factor(self.ref_counts, remapped2, norm_mode, hw2, ref_center, mapped_center2)
        self.remapped_unscaled = remapped
        self.remapped_unscaled2 = remapped2
        self.scaled = remapped * sf
        self.scaled2 = remapped2 * sf2

        live_ref, _real_ref = _read_meas_tim(self.ref_meta)
        live1, _real1 = _read_meas_tim(self.mob_meta)
        live2, _real2 = _read_meas_tim(self.mob_meta2)
        ratio12 = (live1 / live2) if live2 > 0 else 1.0
        ratio21 = (live2 / live1) if live1 > 0 else 1.0
        ratio_r1 = (live_ref / live1) if live1 > 0 else 1.0
        ratio_1r = (live1 / live_ref) if live_ref > 0 else 1.0
        ratio_r2 = (live_ref / live2) if live2 > 0 else 1.0
        ratio_2r = (live2 / live_ref) if live_ref > 0 else 1.0
        m1 = self.remapped_unscaled
        m2arr = self.remapped_unscaled2
        s12 = m1 - (m2arr * float(ratio12))
        s21 = m2arr - (m1 * float(ratio21))
        ds12 = np.sqrt(np.clip(m1, 0.0, None) + np.clip(m2arr * float(ratio12), 0.0, None))
        ds21 = np.sqrt(np.clip(m2arr, 0.0, None) + np.clip(m1 * float(ratio21), 0.0, None))

        sr1 = self.ref_counts - (m1 * float(ratio_r1))
        s1r = m1 - (self.ref_counts * float(ratio_1r))
        dsr1 = np.sqrt(np.clip(self.ref_counts, 0.0, None) + np.clip(m1 * float(ratio_r1), 0.0, None))
        ds1r = np.sqrt(np.clip(m1, 0.0, None) + np.clip(self.ref_counts * float(ratio_1r), 0.0, None))

        sr2 = self.ref_counts - (m2arr * float(ratio_r2))
        s2r = m2arr - (self.ref_counts * float(ratio_2r))
        dsr2 = np.sqrt(np.clip(self.ref_counts, 0.0, None) + np.clip(m2arr * float(ratio_r2), 0.0, None))
        ds2r = np.sqrt(np.clip(m2arr, 0.0, None) + np.clip(self.ref_counts * float(ratio_2r), 0.0, None))

        mode = self._selected_sub_mode()
        if mode == "2-1":
            self.subtracted, self.subtracted_dy = s21, ds21
            self.subtracted2, self.subtracted_dy2 = s12, ds12
        elif mode == "ref-1":
            self.subtracted, self.subtracted_dy = sr1, dsr1
            self.subtracted2, self.subtracted_dy2 = s1r, ds1r
        elif mode == "ref-2":
            self.subtracted, self.subtracted_dy = sr2, dsr2
            self.subtracted2, self.subtracted_dy2 = s2r, ds2r
        else:
            self.subtracted, self.subtracted_dy = s12, ds12
            self.subtracted2, self.subtracted_dy2 = s21, ds21

        x = np.arange(len(self.ref_counts), dtype=np.float64)
        self.curve_ref.setData(x=x, y=self.ref_counts)
        self.curve_mob.setData(x=x, y=self.scaled)
        self.curve_mob2.setData(x=x, y=self.scaled2)
        self.curve_sub.setData(x=x, y=self.subtracted)
        self.curve_sub2.setData(x=x, y=self.subtracted2)
        self.curve_sub.setVisible(self.show_sub)
        self.curve_sub2.setVisible(False)

        sr = peak_centroid_fwhm(self.ref_counts, ref_center, hw)
        sm = peak_centroid_fwhm(self.scaled, mapped_center, hw)
        sm2 = peak_centroid_fwhm(self.scaled2, mapped_center2, hw2)
        self._drag_guard = True
        self.vline_ref.setValue(sr[0])
        self.vline_mob.setValue(sm[0])
        self._drag_guard = False

        self.ref_peak_dot.setData([sr[0]], [sr[2]])
        self.mob_peak_dot.setData([sm[0]], [sm[2]])

        if self.ref_peaks.size:
            rp = self.ref_peaks.astype(np.int64)
            self.ref_peaks_scatter.setData(rp.astype(np.float64), self.ref_counts[rp].astype(np.float64))
        else:
            self.ref_peaks_scatter.setData([], [])

        if self.mob_peaks_raw.size:
            mapped = m * self.mob_peaks_raw.astype(np.float64) + q
            mapped = np.clip(np.rint(mapped).astype(np.int64), 0, len(self.scaled) - 1)
            self.mob_peaks_scatter.setData(mapped.astype(np.float64), self.scaled[mapped].astype(np.float64))
        else:
            self.mob_peaks_scatter.setData([], [])

        if self.ref_target is not None:
            self.ref_target.setPos((float(sr[0]), float(sr[2])))
        if self.mob_target is not None:
            self.mob_target.setPos((float(sm[0]), float(sm[2])))

        if isinstance(self.sum_ref, tuple):
            self.region_ref.setRegion(self.sum_ref)
            self.region_ref.setVisible(True)
        else:
            self.region_ref.setVisible(False)
        if isinstance(self.sum_mob, tuple):
            self.region_mob.setRegion(self.sum_mob)
            self.region_mob.setVisible(True)
        else:
            self.region_mob.setVisible(False)
        if isinstance(self.sum_sub, tuple):
            self.region_sub.setRegion(self.sum_sub)
            self.region_sub.setVisible(True)
        else:
            self.region_sub.setVisible(False)

        stats = [
            "Reference",
            f"  Centroid : {sr[0]:.2f} ch",
            f"  FWHM     : {sr[1]:.2f} ch",
            f"  Peak     : {sr[2]:,.0f}",
            f"  Area     : {sr[3]:,.0f}",
            "",
            "Matched",
            f"  Centroid : {sm[0]:.2f} ch",
            f"  FWHM     : {sm[1]:.2f} ch",
            f"  Peak     : {sm[2]:,.0f}",
            f"  Area     : {sm[3]:,.0f}",
            "",
            "Matched2",
            f"  Centroid : {sm2[0]:.2f} ch",
            f"  FWHM     : {sm2[1]:.2f} ch",
            f"  Peak     : {sm2[2]:,.0f}",
            f"  Area     : {sm2[3]:,.0f}",
            "",
            "Residual",
            f"  dCentroid: {sm[0] - sr[0]:+.3f} ch",
            f"  dFWHM    : {sm[1] - sr[1]:+.3f} ch",
            f"  dCentroid2: {sm2[0] - sr[0]:+.3f} ch",
            f"  dFWHM2    : {sm2[1] - sr[1]:+.3f} ch",
            "",
            "Cal",
            f"  m = {m:.5f}",
            f"  q = {q:.2f}",
            f"  scale = {sf:.5f}",
            f"  m2 = {m2:.5f}",
            f"  q2 = {q2:.2f}",
            f"  scale2 = {sf2:.5f}",
            f"  norm = {norm_mode}",
            f"  sub ratio 1->2 = {ratio12:.5f}",
            f"  sub ratio 2->1 = {ratio21:.5f}",
            f"  peaks(ref/m1/m2) = {len(self.ref_peaks)}/{len(self.mob_peaks_raw)}/{len(self.mob_peaks_raw2)}",
        ]

        rng = self._active_sum_range()
        if isinstance(rng, tuple):
            stats.append("")
            stats.append(f"Sums ({rng[0]}:{rng[1]})")
            for line in self._sum_report_lines(rng):
                stats.append(f"  {line}")
        self.stats.setPlainText("\n".join(stats))
        self._update_peak_pair_deltas()
        self._update_peak_pair_deltas2()

    def do_auto(self) -> None:
        hw = int(self.hw_spin.value())
        ref_center = int(self.ref_spin.value())
        mob_center = int(self.mob_spin.value())

        sel_ref, sel_mob = self._selected_peak_pairs()
        if len(sel_ref) >= 2 and len(sel_mob) >= 2:
            ref_peaks = sel_ref
            mob_peaks = sel_mob
        else:
            ref_auto = self.ref_peaks if self.ref_peaks.size else detect_peaks_scipy(self.ref_counts, max_peaks=10, min_distance=max(6, hw // 2))
            mob_auto = self.mob_peaks_raw if self.mob_peaks_raw.size else detect_peaks_scipy(self.mob_counts, max_peaks=10, min_distance=max(6, hw // 2))
            ref_peaks, mob_peaks = pair_peaks_nearest_mapped(
                ref_auto,
                mob_auto,
                float(self.m_spin.value()),
                float(self.q_spin.value()),
                max_delta=max(20.0, float(2 * hw)),
            )
        fit = _fit_mq_from_peak_lists(ref_peaks, mob_peaks)

        if self.fit_mode == "spectrum_lsq":
            m_init = float(fit[0]) if fit is not None else float(self.m_spin.value())
            q_init = float(fit[1]) if fit is not None else float(self.q_spin.value())
            spec_fit = self._fit_by_spectrum_lsq(m_init, q_init, ref_peaks)
            if spec_fit is not None:
                fit = spec_fit

        if fit is not None:
            m, q = fit
        else:
            m, q, _, _ = auto_calibrate(self.ref_counts, self.mob_counts, hw, ref_center, mob_center)

        self._set_m(float(np.clip(m, 0.8, 1.2)))
        self._set_q(float(np.clip(q, -300.0, 300.0)))
        if fit is not None:
            self._set_status(f"Auto calibration done (multi-peak N={min(len(ref_peaks), len(mob_peaks))})", ok=True)
        else:
            self._set_status("Auto calibration done (single-peak fallback)", ok=True)
        self.update_plot()

    def do_auto2(self) -> None:
        hw = int(self.hw_spin2.value())
        ref_center = int(self.ref_spin.value())
        mob_center = int(self.mob_spin2.value())

        sel_ref, sel_mob = self._selected_peak_pairs2()
        if len(sel_ref) >= 2 and len(sel_mob) >= 2:
            ref_peaks = sel_ref
            mob_peaks = sel_mob
        else:
            ref_auto = self.ref_peaks if self.ref_peaks.size else detect_peaks_scipy(self.ref_counts, max_peaks=10, min_distance=max(6, hw // 2))
            mob_auto = self.mob_peaks_raw2 if self.mob_peaks_raw2.size else detect_peaks_scipy(self.mob_counts2, max_peaks=10, min_distance=max(6, hw // 2))
            ref_peaks, mob_peaks = pair_peaks_nearest_mapped(
                ref_auto,
                mob_auto,
                float(self.m_spin2.value()),
                float(self.q_spin2.value()),
                max_delta=max(20.0, float(2 * hw)),
            )
        fit = _fit_mq_from_peak_lists(ref_peaks, mob_peaks)

        if self.fit_mode2 == "spectrum_lsq":
            m_init = float(fit[0]) if fit is not None else float(self.m_spin2.value())
            q_init = float(fit[1]) if fit is not None else float(self.q_spin2.value())
            spec_fit = self._fit_by_spectrum_lsq2(m_init, q_init, ref_peaks)
            if spec_fit is not None:
                fit = spec_fit

        if fit is not None:
            m, q = fit
        else:
            m, q, _, _ = auto_calibrate(self.ref_counts, self.mob_counts2, hw, ref_center, mob_center)

        self._set_m2(float(np.clip(m, 0.8, 1.2)))
        self._set_q2(float(np.clip(q, -300.0, 300.0)))
        self._set_status("Auto2 calibration done", ok=True)
        self.update_plot()

    def do_reset(self) -> None:
        self._set_m(1.0)
        self._set_q(0.0)
        self._set_status("Reset m=1, q=0", ok=True)
        self.update_plot()

    def do_reset2(self) -> None:
        self._set_m2(1.0)
        self._set_q2(0.0)
        self._set_status("Reset2 m2=1, q2=0", ok=True)
        self.update_plot()

    def do_detect(self) -> None:
        hw = int(self.hw_spin.value())
        self.ref_peaks = detect_peaks_scipy(self.ref_counts, max_peaks=12, min_distance=max(6, hw // 2))
        self.mob_peaks_raw = detect_peaks_scipy(self.mob_counts, max_peaks=12, min_distance=max(6, hw // 2))
        self._refresh_peak_pairs_table()

        if self.ref_peaks.size:
            self.ref_center = int(self.ref_peaks[np.argmax(self.ref_counts[self.ref_peaks])])
        else:
            self.ref_center = detect_primary_peak(self.ref_counts)

        if self.mob_peaks_raw.size:
            self.mob_center_raw = int(self.mob_peaks_raw[np.argmax(self.mob_counts[self.mob_peaks_raw])])
        else:
            self.mob_center_raw = detect_primary_peak(self.mob_counts)

        self.ref_spin.setValue(self.ref_center)
        self.mob_spin.setValue(self.mob_center_raw)
        self._set_status(f"Peaks detected: ref={len(self.ref_peaks)} mob={len(self.mob_peaks_raw)}", ok=True)
        self.update_plot()

    def do_detect2(self) -> None:
        hw = int(self.hw_spin2.value())
        self.ref_peaks = detect_peaks_scipy(self.ref_counts, max_peaks=12, min_distance=max(6, hw // 2))
        self.mob_peaks_raw2 = detect_peaks_scipy(self.mob_counts2, max_peaks=12, min_distance=max(6, hw // 2))
        self._refresh_peak_pairs_table2()

        if self.mob_peaks_raw2.size:
            self.mob_center_raw2 = int(self.mob_peaks_raw2[np.argmax(self.mob_counts2[self.mob_peaks_raw2])])
        else:
            self.mob_center_raw2 = detect_primary_peak(self.mob_counts2)

        self.mob_spin2.setValue(self.mob_center_raw2)
        self._set_status(f"Peaks2 detected: ref={len(self.ref_peaks)} mob2={len(self.mob_peaks_raw2)}", ok=True)
        self.update_plot()

    def do_save(self) -> None:
        out = make_output_path(self.mob_path, "matched")
        meta = self.mob_meta
        meta["ch_start"] = 0
        meta["ch_end"] = len(self.ref_counts) - 1
        write_spe(out, np.array(self.remapped_unscaled, dtype=np.float64), meta)
        self._set_status(f"Saved {os.path.basename(out)}", ok=True)
        print(f"Saved: {out}")

    def do_save2(self) -> None:
        out = make_output_path(self.mob_path2, "matched")
        meta = self.mob_meta2
        meta["ch_start"] = 0
        meta["ch_end"] = len(self.ref_counts) - 1
        write_spe(out, np.array(self.remapped_unscaled2, dtype=np.float64), meta)
        self._set_status(f"Saved {os.path.basename(out)}", ok=True)
        print(f"Saved: {out}")

    def do_subtract(self) -> None:
        self.show_sub = True
        self.curve_sub.setVisible(True)
        self.curve_sub2.setVisible(False)
        mode = self._selected_sub_mode().replace("-", " - ")
        self._set_status(f"Subtracted view {mode} updated (not saved)", ok=True)
        self.update_plot()

    def do_save_subtract(self) -> None:
        mode = self._selected_sub_mode()
        if mode in ("ref-2",):
            base_path = self.mob_path2
        elif mode in ("1-2", "2-1", "ref-1"):
            base_path = self.mob_path
        else:
            base_path = self.ref_path
        out = make_output_path(base_path, f"subtracted_{mode.replace('-', '')}")
        meta = dict(self.ref_meta)
        meta["ch_start"] = 0
        meta["ch_end"] = len(self.ref_counts) - 1
        write_spe(
            out,
            np.array(self.subtracted, dtype=np.float64),
            meta,
            is_difference=True,
            other_meta=self.mob_meta,
        )
        self._set_status(f"Saved {os.path.basename(out)} ({mode})", ok=True)
        print(f"Saved: {out}")

    def do_save_subtract_3col(self) -> None:
        mode = self._selected_sub_mode()
        if mode in ("ref-2",):
            base_path = self.mob_path2
        elif mode in ("1-2", "2-1", "ref-1"):
            base_path = self.mob_path
        else:
            base_path = self.ref_path
        out = make_output_path(base_path, f"subtracted3col_{mode.replace('-', '')}")
        base, _ext = os.path.splitext(out)
        out_txt = f"{base}.txt"
        x = np.arange(len(self.subtracted), dtype=np.int64)
        arr = np.column_stack((x, self.subtracted, self.subtracted_dy))
        np.savetxt(out_txt, arr, fmt=["%d", "%.8g", "%.8g"], header="x y dy", comments="")
        self._set_status(f"Saved {os.path.basename(out_txt)} ({mode})", ok=True)
        print(f"Saved: {out_txt}")

    def do_clear(self) -> None:
        self.show_sub = False
        self.curve_sub.setVisible(False)
        self.curve_sub2.setVisible(False)
        self.sum_ref = None
        self.sum_mob = None
        self.sum_mob2 = None
        self.sum_sub = None
        self.sum_sub2 = None
        self.selection_mode = None
        self.selection_start = None
        self._set_status("Subtracted view hidden", ok=True)
        self.update_plot()

    def switch_spectrum(self, delta: int) -> None:
        if not self.mobile_items:
            return
        self._load_mobile_slot(1, (self.index1 + delta) % len(self.mobile_items))
        self._set_status("Switched spectrum", ok=True)

    def pick_ref_peak(self) -> None:
        self.pick_mode = "ref"
        self._set_status("Click plot to set Ref peak", ok=True)

    def pick_mob_peak(self) -> None:
        self.pick_mode = "mob"
        self._set_status("Click plot to set Mob peak", ok=True)

    def pick_mob2_peak(self) -> None:
        self.pick_mode = "mob2"
        self._set_status("Click plot to set Mob2 peak", ok=True)

    def _on_plot_click(self, evt) -> None:
        mouse_point = self.plot.getPlotItem().vb.mapSceneToView(evt.scenePos())
        ch = int(np.clip(round(mouse_point.x()), 0, len(self.ref_counts) - 1))
        if self.pick_mode == "pair_ref":
            self.pending_pair_ref = self._fit_click_peak_channel("ref", ch)
            self.pick_mode = "pair_mob"
            self._set_status(f"Reference peak fit at {self.pending_pair_ref}; click matched peak", ok=True)
            return

        if self.pick_mode == "pair_ref2":
            self.pending_pair_ref = self._fit_click_peak_channel("ref", ch)
            self.pick_mode = "pair_mob2"
            self._set_status(f"Reference peak fit at {self.pending_pair_ref}; click matched2 peak", ok=True)
            return

        if self.pick_mode == "pair_mob":
            if self.pending_pair_ref is None:
                self.pick_mode = "pair_ref"
                self._set_status("Reference peak missing; click reference peak", ok=False)
                return
            mob_fit = self._fit_click_peak_channel("mob", ch)
            self._append_peak_pair(self.pending_pair_ref, mob_fit, checked=True)
            self.pick_mode = None
            self.pending_pair_ref = None
            self._set_status("Added manual peak pair", ok=True)
            self.update_plot()
            return

        if self.pick_mode == "pair_mob2":
            if self.pending_pair_ref is None:
                self.pick_mode = "pair_ref2"
                self._set_status("Reference peak missing; click reference peak", ok=False)
                return
            mob_fit = self._fit_click_peak_channel("mob2", ch)
            self._append_peak_pair2(self.pending_pair_ref, mob_fit, checked=True)
            self.pick_mode = None
            self.pending_pair_ref = None
            self._set_status("Added manual peak pair2", ok=True)
            self.update_plot()
            return

        if self.pick_mode in {"ref", "mob", "mob2"}:
            if self.pick_mode == "ref":
                self.ref_spin.setValue(ch)
            elif self.pick_mode == "mob2":
                self.mob_spin2.setValue(ch)
            else:
                self.mob_spin.setValue(ch)
            self.pick_mode = None
            self._set_status(f"Peak set to ch {ch}", ok=True)
            self.update_plot()
            return

        if self.selection_mode == "sum_all":
            if self.selection_start is None:
                self.selection_start = ch
                self._set_status(f"Start at {ch}; click end", ok=True)
                return
            lo = int(min(self.selection_start, ch))
            hi = int(max(self.selection_start, ch))
            self._set_sum_range_all((lo, hi))
            self.selection_start = None
            self.selection_mode = None
            self._set_status(f"Set sum range {lo}:{hi}", ok=True)
            self.update_plot()


def run_gui(
    ref_counts: np.ndarray,
    mobile_items: list[dict[str, object]],
    ref_path: str,
    ref_meta: Optional[dict[str, object]],
    norm_mode: str,
) -> None:
    if QtWidgets is None or pg is None:
        print(
            "Qt dependencies are missing. Install with: pip install pyside6 pyqtgraph",
            file=sys.stderr,
        )
        if _IMPORT_ERROR is not None:
            print(f"Import error: {_IMPORT_ERROR}", file=sys.stderr)
        raise SystemExit(2)

    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)

    win = GainMatchWindow(ref_counts, mobile_items, ref_path, ref_meta, norm_mode)
    win.show()
    app.exec()


def main() -> None:
    parser = argparse.ArgumentParser(description="Qt interactive gain matcher for Maestro .Spe files")
    parser.add_argument(
        "spectra",
        nargs="+",
        metavar="FILE.Spe",
        help="Reference .Spe followed by one or more mobile .Spe files",
    )
    parser.add_argument(
        "--norm",
        default="peak",
        choices=["none", "peak", "area", "integral"],
        help="Normalization mode (default: peak)",
    )
    args = parser.parse_args()

    files = args.spectra
    if len(files) == 1:
        files = files * 2

    ref_path = files[0]
    mob_paths = files[1:]

    print(f"Loading {ref_path} ...")
    ref_counts, ref_meta = parse_spe(ref_path)
    print(f"  {len(ref_counts)} channels, peak at ch {int(ref_counts.argmax())}, total counts {ref_counts.sum():,.0f}")

    mobile_items: list[dict[str, object]] = []
    max_len = len(ref_counts)

    for i, p in enumerate(mob_paths, start=1):
        print(f"Loading mobile [{i}] {p} ...")
        c, m = parse_spe(p)
        print(f"  {len(c)} channels, peak at ch {int(c.argmax())}, total counts {c.sum():,.0f}")
        mobile_items.append({"path": p, "counts": c, "meta": m})
        max_len = max(max_len, len(c))

    if len(ref_counts) < max_len:
        ref_counts = np.pad(ref_counts, (0, max_len - len(ref_counts)))

    for item in mobile_items:
        c = np.array(item["counts"], dtype=np.float64)
        if len(c) < max_len:
            item["counts"] = np.pad(c, (0, max_len - len(c)))

    run_gui(ref_counts=ref_counts, mobile_items=mobile_items, ref_path=ref_path, ref_meta=ref_meta, norm_mode=args.norm)


if __name__ == "__main__":
    main()
