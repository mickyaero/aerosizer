"""
AeroSizer Web Application
FastAPI backend for the aircraft conceptual design tool.

Exposes REST endpoints that wrap the core sizing, aerodynamics,
weights, performance, cost, and maneuvering analysis modules (Raymer method).

Run:
    uvicorn app:app --host 0.0.0.0 --port 8080 --reload
"""

from __future__ import annotations

import math
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
from core.atmosphere import (
    atmosphere,
    true_airspeed,
    kg_to_lb,
    lb_to_kg,
    ft_to_m,
    m_to_ft,
    m2_to_ft2,
    ft2_to_m2,
    G0,
    RHO0,
)
from core.mission import (
    MissionProfile,
    MissionSegment,
    SegmentType,
    AircraftType,
)
from core.sizing.initial_sizing import size_aircraft, SizingResult
from core.sizing.tw_ws_selection import ConstraintAnalysis
from core.aerodynamics.drag_polar import DragPolar
from core.weights.statistical_weights import TransportWeights

# Performance imports
try:
    from core.performance.climb import (
        max_rate_of_climb,
        service_ceiling as _service_ceiling_func,
        absolute_ceiling as _absolute_ceiling_func,
        generate_roc_curve,
    )
    from core.performance.range_endurance import range_payload_diagram
    HAS_PERFORMANCE = True
except ImportError:
    HAS_PERFORMANCE = False

# Cost imports
try:
    from core.cost.dapca_iv import DAPCAInputs, compute_dapca
    HAS_DAPCA = True
except ImportError:
    HAS_DAPCA = False

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AeroSizer",
    description="Aircraft Conceptual Design - Raymer Method",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).resolve().parent / "static"

# Mount the static directory for CSS/JS/images
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------

class SizingRequest(BaseModel):
    aircraft_type: str = "jet_transport"
    range_km: float = 5000.0
    passengers: int = 180
    cruise_mach: float = 0.78
    cruise_alt_m: float = 10668.0
    loiter_hr: float = 0.5
    ld_cruise: Optional[float] = None
    ld_loiter: Optional[float] = None
    sfc_cruise: Optional[float] = None
    sfc_loiter: Optional[float] = None
    crew_weight_kg: Optional[float] = None
    payload_extra_kg: float = 0.0


class ConstraintRequest(BaseModel):
    CD0: float = 0.020
    K: float = 0.04
    CLmax_clean: float = 1.6
    CLmax_takeoff: float = 2.0
    CLmax_landing: float = 2.4
    cruise_mach: float = 0.85
    cruise_alt_m: float = 10668.0
    takeoff_distance_m: float = 2500.0
    landing_distance_m: float = 1800.0
    climb_gradient_oei: float = 0.024
    service_ceiling_m: float = 12192.0
    n_engines: int = 2
    thrust_lapse_cruise: float = 0.25


class DragPolarRequest(BaseModel):
    CD0: float = 0.020
    K: float = 0.04
    cl_min: float = 0.0
    cl_max: float = 1.5
    cl_step: float = 0.01
    CD0_takeoff_increment: float = 0.02
    CD0_landing_increment: float = 0.07


class WeightsRequest(BaseModel):
    W0_kg: float
    Wf_kg: float
    AR: float = 9.0
    Sref_m2: float = 125.0
    sweep_quarter_deg: float = 25.0
    taper_ratio: float = 0.3
    t_c_root: float = 0.12
    Nz: float = 3.75
    fuselage_length_m: float = 38.0
    fuselage_width_m: float = 3.76
    fuselage_depth_m: float = 4.01
    Sht_m2: float = 32.0
    Svt_m2: float = 22.0
    sweep_ht_deg: float = 30.0
    sweep_vt_deg: float = 35.0
    AR_ht: float = 4.0
    AR_vt: float = 1.5
    taper_ht: float = 0.4
    taper_vt: float = 0.4
    t_c_ht: float = 0.10
    t_c_vt: float = 0.10
    n_engines: int = 2
    engine_weight_each_kg: float = 2500.0
    thrust_each_N: float = 120000.0
    n_crew: int = 2
    n_pax: int = 180
    weight_growth_factor: float = 1.0


class TakeoffLandingRequest(BaseModel):
    W_kg: float
    S_m2: float
    T_N: float
    n_engines: int = 2
    CD0: float = 0.020
    K: float = 0.04
    CLmax_takeoff: float = 2.0
    CLmax_landing: float = 2.4
    altitude_m: float = 0.0
    mu_ground: float = 0.04
    mu_braking: float = 0.40


class TurnPerformanceRequest(BaseModel):
    W_kg: float
    S_m2: float
    T_N: float
    CD0: float = 0.020
    K: float = 0.04
    CLmax: float = 1.5
    n_limit: float = 3.0
    altitude_m: float = 0.0
    velocity_range_start: float = 50.0
    velocity_range_end: float = 350.0


class PsDiagramRequest(BaseModel):
    W_kg: float
    S_m2: float
    T_N: float
    CD0: float = 0.020
    K: float = 0.04
    altitude_m: float = 0.0
    load_factors: List[float] = [1, 3, 5]


class DAPCACostRequest(BaseModel):
    We_kg: float
    V_max_ms: float
    Q: int = 500
    FTA: int = 4
    N_engines: int = 2
    T_max_N: float = 120000.0
    M_max_engine: float = 0.85
    T_turbine_inlet_K: float = 1400.0
    C_avionics: float = 5000000.0
    n_pax: int = 180
    cargo_aircraft: bool = False
    CPI_factor: float = 1.35


class OperatingCostRequest(BaseModel):
    W0_kg: float
    We_kg: float
    Wf_mission_kg: float
    V_cruise_ms: float
    range_km: float
    n_pax: int = 180
    n_engines: int = 2
    aircraft_cost: float = 100000000.0
    engine_cost_each: float = 15000000.0
    fuel_price_per_kg: float = 0.80


class ClimbProfileRequest(BaseModel):
    W_kg: float
    S_m2: float
    T_N: float
    CD0: float = 0.020
    K: float = 0.04
    alt_range_start: float = 0.0
    alt_range_end: float = 25000.0


class RangePayloadRequest(BaseModel):
    W0_kg: float
    We_kg: float
    Wf_max_kg: float
    sfc_per_hr: float
    V_cruise_ms: float
    CD0: float = 0.020
    K: float = 0.04
    W_crew_kg: float = 450.0


class EnvelopeRequest(BaseModel):
    W_kg: float
    S_m2: float
    T_N: float
    CD0: float = 0.020
    K: float = 0.04
    CLmax: float = 1.5


class AircraftSVGRequest(BaseModel):
    span_m: float = 34.0
    root_chord_m: float = 7.0
    tip_chord_m: float = 1.8
    sweep_quarter_deg: float = 25.0
    fuselage_length_m: float = 38.0
    fuselage_width_m: float = 3.76
    tail_span_m: float = 12.0
    tail_root_chord_m: float = 3.5
    tail_tip_chord_m: float = 1.2
    tail_sweep_deg: float = 30.0
    vtail_height_m: float = 6.0
    vtail_root_chord_m: float = 5.0
    vtail_tip_chord_m: float = 2.0
    vtail_sweep_deg: float = 40.0
    engine_diameter_m: float = 1.8
    engine_length_m: float = 3.5
    n_engines: int = 2
    wing_position_fraction: float = 0.40


class GeometryFromSizingRequest(BaseModel):
    W0_kg: float = 80000.0
    S_m2: float = 125.0
    AR: float = 9.0
    taper_ratio: float = 0.3
    sweep_quarter_deg: float = 25.0
    fuselage_length_m: float = 38.0
    fuselage_width_m: float = 3.76
    n_engines: int = 2


# ---------------------------------------------------------------------------
# Preset aircraft data
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "b787": {
        "name": "B787-8 Dreamliner",
        "aircraft_type": "jet_transport",
        "range_km": 13621,
        "passengers": 242,
        "cruise_mach": 0.85,
        "cruise_alt_m": 10668,
        "cruise_alt_ft": 35000,
        "loiter_hr": 0.75,
        "ld_cruise": 19.0,
        "ld_loiter": 20.0,
        "sfc_cruise": 0.525,
        "sfc_loiter": 0.43,
        "crew_weight_kg": 630,
        "payload_extra_kg": 0,
        # Constraint parameters
        "CD0": 0.0175,
        "K": 0.0377,
        "CLmax_clean": 1.6,
        "CLmax_takeoff": 2.0,
        "CLmax_landing": 2.6,
        "takeoff_distance_m": 2600,
        "landing_distance_m": 1700,
        "climb_gradient_oei": 0.024,
        "service_ceiling_m": 13106,
        "n_engines": 2,
        "thrust_lapse_cruise": 0.24,
        # Weights parameters
        "AR": 10.58,
        "Sref_m2": 377.0,
        "sweep_quarter_deg": 32.2,
        "taper_ratio": 0.18,
        "t_c_root": 0.148,
        "fuselage_length_m": 56.7,
        "fuselage_width_m": 5.77,
        "fuselage_depth_m": 5.97,
        "Sht_m2": 65.0,
        "Svt_m2": 39.0,
        "sweep_ht_deg": 36.0,
        "sweep_vt_deg": 40.0,
        "n_crew": 2,
        "n_pax": 242,
        "engine_weight_each_kg": 5860,
        "thrust_each_N": 296000,
        # Cost parameters
        "V_max_ms": 255.0,
        "T_turbine_inlet_K": 1677,
        "M_max_engine": 0.90,
        "C_avionics": 8000000,
        "Q_production": 800,
        "engine_cost_each": 18000000,
        "aircraft_cost_est": 248300000,
    },
    "b737": {
        "name": "B737-800",
        "aircraft_type": "jet_transport",
        "range_km": 5765,
        "passengers": 162,
        "cruise_mach": 0.785,
        "cruise_alt_m": 10668,
        "cruise_alt_ft": 35000,
        "loiter_hr": 0.5,
        "ld_cruise": 17.5,
        "ld_loiter": 18.5,
        "sfc_cruise": 0.575,
        "sfc_loiter": 0.47,
        "crew_weight_kg": 450,
        "payload_extra_kg": 0,
        # Constraint parameters
        "CD0": 0.022,
        "K": 0.042,
        "CLmax_clean": 1.5,
        "CLmax_takeoff": 1.8,
        "CLmax_landing": 2.3,
        "takeoff_distance_m": 2300,
        "landing_distance_m": 1600,
        "climb_gradient_oei": 0.024,
        "service_ceiling_m": 12496,
        "n_engines": 2,
        "thrust_lapse_cruise": 0.26,
        # Weights parameters
        "AR": 9.45,
        "Sref_m2": 124.6,
        "sweep_quarter_deg": 25.0,
        "taper_ratio": 0.278,
        "t_c_root": 0.128,
        "fuselage_length_m": 38.02,
        "fuselage_width_m": 3.76,
        "fuselage_depth_m": 4.01,
        "Sht_m2": 32.4,
        "Svt_m2": 23.1,
        "sweep_ht_deg": 30.0,
        "sweep_vt_deg": 35.0,
        "n_crew": 2,
        "n_pax": 162,
        "engine_weight_each_kg": 2370,
        "thrust_each_N": 121400,
        # Cost parameters
        "V_max_ms": 236.0,
        "T_turbine_inlet_K": 1560,
        "M_max_engine": 0.82,
        "C_avionics": 5000000,
        "Q_production": 5000,
        "engine_cost_each": 12000000,
        "aircraft_cost_est": 106100000,
    },
    "a320": {
        "name": "A320neo",
        "aircraft_type": "jet_transport",
        "range_km": 6300,
        "passengers": 150,
        "cruise_mach": 0.78,
        "cruise_alt_m": 11278,
        "cruise_alt_ft": 37000,
        "loiter_hr": 0.5,
        "ld_cruise": 17.0,
        "ld_loiter": 18.0,
        "sfc_cruise": 0.545,
        "sfc_loiter": 0.45,
        "crew_weight_kg": 450,
        "payload_extra_kg": 0,
        # Constraint parameters
        "CD0": 0.021,
        "K": 0.041,
        "CLmax_clean": 1.5,
        "CLmax_takeoff": 1.9,
        "CLmax_landing": 2.4,
        "takeoff_distance_m": 2200,
        "landing_distance_m": 1500,
        "climb_gradient_oei": 0.024,
        "service_ceiling_m": 11887,
        "n_engines": 2,
        "thrust_lapse_cruise": 0.25,
        # Weights parameters
        "AR": 9.39,
        "Sref_m2": 122.6,
        "sweep_quarter_deg": 25.0,
        "taper_ratio": 0.24,
        "t_c_root": 0.153,
        "fuselage_length_m": 37.57,
        "fuselage_width_m": 3.95,
        "fuselage_depth_m": 4.14,
        "Sht_m2": 31.0,
        "Svt_m2": 21.5,
        "sweep_ht_deg": 29.0,
        "sweep_vt_deg": 34.0,
        "n_crew": 2,
        "n_pax": 150,
        "engine_weight_each_kg": 2450,
        "thrust_each_N": 120000,
        # Cost parameters
        "V_max_ms": 234.0,
        "T_turbine_inlet_K": 1580,
        "M_max_engine": 0.82,
        "C_avionics": 5500000,
        "Q_production": 4000,
        "engine_cost_each": 13000000,
        "aircraft_cost_est": 110600000,
    },
}


# ---------------------------------------------------------------------------
# Helper: map aircraft_type string to enum
# ---------------------------------------------------------------------------

def _parse_aircraft_type(s: str) -> AircraftType:
    """Convert a string to AircraftType enum, case-insensitive."""
    s = s.lower().strip()
    for member in AircraftType:
        if member.value == s:
            return member
    return AircraftType.JET_TRANSPORT


# ---------------------------------------------------------------------------
# Helper: thrust lapse model (used by several endpoints)
# ---------------------------------------------------------------------------

def _thrust_lapse(T_SL: float, altitude_m: float) -> float:
    """Simple thrust lapse model: T = T_SL * sigma^0.6."""
    atm = atmosphere(altitude_m)
    sigma = atm["sigma"]
    return T_SL * (sigma ** 0.6)


# ---------------------------------------------------------------------------
# SVG Generation Helpers
# ---------------------------------------------------------------------------

def _generate_top_view_svg(p: AircraftSVGRequest) -> str:
    """Generate top-view SVG of the aircraft."""
    # All coordinates in metres, then scaled to fit 400x300 viewport
    fuse_len = p.fuselage_length_m
    fuse_w = p.fuselage_width_m
    span = p.span_m
    half_span = span / 2.0

    # Determine the bounding box
    total_width = span + 4  # some margin
    total_height = fuse_len + 4
    scale = min(380.0 / total_width, 280.0 / total_height)

    cx = 200.0  # center x of viewport
    cy = 150.0  # center y of viewport

    def sx(x_m):
        return cx + x_m * scale

    def sy(y_m):
        # y_m: 0 = nose, positive toward tail
        return 20 + y_m * scale

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" '
                 f'width="400" height="300" style="background:#1e1e2e">')
    parts.append('<defs><style>'
                 'text{font-family:Share Tech Mono,Courier New,monospace;fill:#bac2de;font-size:9px;}'
                 '</style></defs>')

    green = "#89b4fa"
    fill_g = "rgba(137,180,250,0.08)"
    amber = "#fab387"

    # --- Fuselage (top view): elongated rounded rect ---
    fw_half = fuse_w / 2.0
    nose_y = 0.0
    tail_y = fuse_len
    # Fuselage outline as a path with rounded nose and tapered tail
    nose_r = fuse_w * 0.8  # nose rounding
    tail_taper = fuse_w * 0.3
    pts_fuse = [
        (sx(-fw_half), sy(nose_r)),
        (sx(-fw_half), sy(tail_y - tail_taper)),
        (sx(0), sy(tail_y)),
        (sx(fw_half), sy(tail_y - tail_taper)),
        (sx(fw_half), sy(nose_r)),
    ]
    # Nose arc
    fuse_path = (f'M {sx(0):.1f},{sy(0):.1f} '
                 f'C {sx(-fw_half):.1f},{sy(0):.1f} {sx(-fw_half):.1f},{sy(nose_r * 0.5):.1f} '
                 f'{sx(-fw_half):.1f},{sy(nose_r):.1f} '
                 f'L {sx(-fw_half):.1f},{sy(tail_y - tail_taper):.1f} '
                 f'L {sx(0):.1f},{sy(tail_y):.1f} '
                 f'L {sx(fw_half):.1f},{sy(tail_y - tail_taper):.1f} '
                 f'L {sx(fw_half):.1f},{sy(nose_r):.1f} '
                 f'C {sx(fw_half):.1f},{sy(nose_r * 0.5):.1f} {sx(fw_half):.1f},{sy(0):.1f} '
                 f'{sx(0):.1f},{sy(0):.1f} Z')
    parts.append(f'<path d="{fuse_path}" stroke="{green}" stroke-width="1.5" fill="{fill_g}"/>')

    # --- Wing (top view): trapezoid with sweep ---
    wing_y = fuse_len * p.wing_position_fraction  # wing root leading edge
    sweep_rad = math.radians(p.sweep_quarter_deg)
    root_c = p.root_chord_m
    tip_c = p.tip_chord_m
    # Quarter chord sweep: the quarter-chord line sweeps back
    # LE sweep offset at tip = half_span * tan(sweep_at_LE)
    # sweep_at_LE from quarter chord: tan(LE_sweep) = tan(qc_sweep) + (root_c - tip_c)/(2*half_span)
    le_sweep_offset = half_span * math.tan(sweep_rad) - 0.25 * (root_c - tip_c)

    # Left wing
    w_pts_l = [
        (sx(-fw_half), sy(wing_y)),
        (sx(-half_span), sy(wing_y + le_sweep_offset)),
        (sx(-half_span), sy(wing_y + le_sweep_offset + tip_c)),
        (sx(-fw_half), sy(wing_y + root_c)),
    ]
    poly_l = " ".join(f"{x:.1f},{y:.1f}" for x, y in w_pts_l)
    parts.append(f'<polygon points="{poly_l}" stroke="{green}" stroke-width="1.2" fill="{fill_g}"/>')

    # Right wing
    w_pts_r = [
        (sx(fw_half), sy(wing_y)),
        (sx(half_span), sy(wing_y + le_sweep_offset)),
        (sx(half_span), sy(wing_y + le_sweep_offset + tip_c)),
        (sx(fw_half), sy(wing_y + root_c)),
    ]
    poly_r = " ".join(f"{x:.1f},{y:.1f}" for x, y in w_pts_r)
    parts.append(f'<polygon points="{poly_r}" stroke="{green}" stroke-width="1.2" fill="{fill_g}"/>')

    # --- Horizontal Tail ---
    ht_y = fuse_len * 0.88
    ht_half = p.tail_span_m / 2.0
    ht_root = p.tail_root_chord_m
    ht_tip = p.tail_tip_chord_m
    ht_sweep_rad = math.radians(p.tail_sweep_deg)
    ht_le_offset = ht_half * math.tan(ht_sweep_rad) - 0.25 * (ht_root - ht_tip)

    for sign in [-1, 1]:
        ht_pts = [
            (sx(sign * fw_half * 0.3), sy(ht_y)),
            (sx(sign * ht_half), sy(ht_y + ht_le_offset)),
            (sx(sign * ht_half), sy(ht_y + ht_le_offset + ht_tip)),
            (sx(sign * fw_half * 0.3), sy(ht_y + ht_root)),
        ]
        poly_ht = " ".join(f"{x:.1f},{y:.1f}" for x, y in ht_pts)
        parts.append(f'<polygon points="{poly_ht}" stroke="{green}" stroke-width="1" fill="{fill_g}"/>')

    # --- Engines (top view): ellipses under wings ---
    if p.n_engines >= 2:
        eng_y_pos = wing_y + root_c * 0.3
        eng_span_frac = 0.35  # engines at 35% of half-span
        for sign in [-1, 1]:
            eng_x = sign * half_span * eng_span_frac
            ew = p.engine_diameter_m / 2.0
            el = p.engine_length_m / 2.0
            parts.append(f'<ellipse cx="{sx(eng_x):.1f}" cy="{sy(eng_y_pos):.1f}" '
                         f'rx="{ew * scale:.1f}" ry="{el * scale:.1f}" '
                         f'stroke="{amber}" stroke-width="1" fill="rgba(255,170,0,0.08)"/>')
        if p.n_engines == 4:
            for sign in [-1, 1]:
                eng_x = sign * half_span * 0.65
                parts.append(f'<ellipse cx="{sx(eng_x):.1f}" cy="{sy(eng_y_pos):.1f}" '
                             f'rx="{ew * scale:.1f}" ry="{el * scale:.1f}" '
                             f'stroke="{amber}" stroke-width="1" fill="rgba(255,170,0,0.08)"/>')

    # --- Dimension annotations ---
    # Span line
    ann_y = wing_y + root_c / 2.0
    parts.append(f'<line x1="{sx(-half_span):.1f}" y1="{sy(ann_y) + 15:.1f}" '
                 f'x2="{sx(half_span):.1f}" y2="{sy(ann_y) + 15:.1f}" '
                 f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="3,2"/>')
    parts.append(f'<text x="{cx:.1f}" y="{sy(ann_y) + 25:.1f}" text-anchor="middle">'
                 f'SPAN {p.span_m:.1f}m</text>')

    # Length line
    parts.append(f'<line x1="{sx(half_span) + 8:.1f}" y1="{sy(0):.1f}" '
                 f'x2="{sx(half_span) + 8:.1f}" y2="{sy(fuse_len):.1f}" '
                 f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="3,2"/>')
    parts.append(f'<text x="{sx(half_span) + 12:.1f}" y="{sy(fuse_len / 2):.1f}" '
                 f'text-anchor="start" transform="rotate(90,{sx(half_span) + 12:.1f},{sy(fuse_len / 2):.1f})">'
                 f'LEN {fuse_len:.1f}m</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def _generate_side_view_svg(p: AircraftSVGRequest) -> str:
    """Generate side-view SVG of the aircraft."""
    fuse_len = p.fuselage_length_m
    fuse_w = p.fuselage_width_m  # used as height in side view
    fuse_h = fuse_w * 1.07  # fuselage depth slightly > width

    total_height_m = fuse_h + p.vtail_height_m + 4
    total_width_m = fuse_len + 4
    scale = min(380.0 / total_width_m, 250.0 / total_height_m)

    # Coordinate origin: top-left of viewport. Fuselage centerline at a fixed y.
    x_off = 10.0
    fuse_center_y = 180.0  # pixel y of fuselage centerline

    def sx(x_m):
        return x_off + x_m * scale

    def sy(y_m):
        # y_m positive = up from fuselage center
        return fuse_center_y - y_m * scale

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" '
                 f'width="400" height="300" style="background:#1e1e2e">')
    parts.append('<defs><style>'
                 'text{font-family:Share Tech Mono,Courier New,monospace;fill:#bac2de;font-size:9px;}'
                 '</style></defs>')

    green = "#89b4fa"
    fill_g = "rgba(137,180,250,0.08)"
    amber = "#fab387"

    fh = fuse_h / 2.0

    # --- Fuselage (side view): rounded rectangle ---
    nose_r = fuse_h * 0.8
    tail_taper_len = fuse_len * 0.15
    fuse_path = (
        f'M {sx(nose_r):.1f},{sy(fh):.1f} '
        f'C {sx(0):.1f},{sy(fh):.1f} {sx(0):.1f},{sy(-fh):.1f} '
        f'{sx(nose_r):.1f},{sy(-fh):.1f} '
        f'L {sx(fuse_len - tail_taper_len):.1f},{sy(-fh):.1f} '
        f'L {sx(fuse_len):.1f},{sy(0):.1f} '
        f'L {sx(fuse_len - tail_taper_len):.1f},{sy(fh):.1f} '
        f'Z'
    )
    parts.append(f'<path d="{fuse_path}" stroke="{green}" stroke-width="1.5" fill="{fill_g}"/>')

    # --- Window dots ---
    win_start = fuse_len * 0.12
    win_end = fuse_len * 0.75
    win_y = fh * 0.4
    n_windows = int((win_end - win_start) / (fuse_len * 0.015))
    for i in range(n_windows):
        wx = win_start + i * (win_end - win_start) / n_windows
        parts.append(f'<circle cx="{sx(wx):.1f}" cy="{sy(win_y):.1f}" r="1" '
                     f'fill="{green}" opacity="0.4"/>')

    # --- Cockpit windows ---
    ckpt_x = fuse_len * 0.04
    parts.append(f'<line x1="{sx(ckpt_x):.1f}" y1="{sy(fh * 0.6):.1f}" '
                 f'x2="{sx(ckpt_x + fuse_len * 0.03):.1f}" y2="{sy(fh * 0.3):.1f}" '
                 f'stroke="{green}" stroke-width="1.5"/>')

    # --- Wing (side view): airfoil-like bump ---
    wing_x = fuse_len * p.wing_position_fraction
    root_c = p.root_chord_m
    wing_thickness = root_c * 0.12  # t/c ~12%
    wing_path = (
        f'M {sx(wing_x):.1f},{sy(-fh):.1f} '
        f'Q {sx(wing_x + root_c * 0.3):.1f},{sy(-fh - wing_thickness):.1f} '
        f'{sx(wing_x + root_c):.1f},{sy(-fh):.1f}'
    )
    parts.append(f'<path d="{wing_path}" stroke="{green}" stroke-width="1.2" fill="none"/>')

    # --- Vertical Tail ---
    vt_x = fuse_len * 0.85
    vt_root = p.vtail_root_chord_m
    vt_tip = p.vtail_tip_chord_m
    vt_h = p.vtail_height_m
    vt_sweep_rad = math.radians(p.vtail_sweep_deg)
    vt_le_offset = vt_h * math.tan(vt_sweep_rad)

    vt_pts = [
        (sx(vt_x), sy(fh)),
        (sx(vt_x + vt_le_offset), sy(fh + vt_h)),
        (sx(vt_x + vt_le_offset + vt_tip), sy(fh + vt_h)),
        (sx(vt_x + vt_root), sy(fh)),
    ]
    poly_vt = " ".join(f"{x:.1f},{y:.1f}" for x, y in vt_pts)
    parts.append(f'<polygon points="{poly_vt}" stroke="{green}" stroke-width="1.2" fill="{fill_g}"/>')

    # --- Horizontal Tail (side view): thin line ---
    ht_x = fuse_len * 0.88
    ht_root = p.tail_root_chord_m
    parts.append(f'<line x1="{sx(ht_x):.1f}" y1="{sy(fh * 0.9):.1f}" '
                 f'x2="{sx(ht_x + ht_root):.1f}" y2="{sy(fh * 0.9):.1f}" '
                 f'stroke="{green}" stroke-width="1.5"/>')

    # --- Engine (side view): circle below wing ---
    if p.n_engines >= 2:
        eng_x = wing_x + root_c * 0.2
        eng_r = p.engine_diameter_m / 2.0
        eng_y = -fh - eng_r * 1.5  # below fuselage
        # Engine nacelle
        parts.append(f'<ellipse cx="{sx(eng_x + p.engine_length_m / 2):.1f}" cy="{sy(eng_y):.1f}" '
                     f'rx="{p.engine_length_m / 2 * scale:.1f}" ry="{eng_r * scale:.1f}" '
                     f'stroke="{amber}" stroke-width="1" fill="rgba(255,170,0,0.08)"/>')
        # Pylon
        parts.append(f'<line x1="{sx(eng_x + p.engine_length_m / 2):.1f}" y1="{sy(-fh):.1f}" '
                     f'x2="{sx(eng_x + p.engine_length_m / 2):.1f}" y2="{sy(eng_y + eng_r):.1f}" '
                     f'stroke="{green}" stroke-width="0.8"/>')

    # --- Landing Gear indication ---
    # Nose gear
    ng_x = fuse_len * 0.10
    parts.append(f'<line x1="{sx(ng_x):.1f}" y1="{sy(-fh):.1f}" '
                 f'x2="{sx(ng_x):.1f}" y2="{sy(-fh - fuse_h * 0.3):.1f}" '
                 f'stroke="#45475a" stroke-width="1"/>')
    parts.append(f'<circle cx="{sx(ng_x):.1f}" cy="{sy(-fh - fuse_h * 0.3):.1f}" r="3" '
                 f'stroke="#45475a" fill="none"/>')
    # Main gear
    mg_x = wing_x + root_c * 0.5
    parts.append(f'<line x1="{sx(mg_x):.1f}" y1="{sy(-fh):.1f}" '
                 f'x2="{sx(mg_x):.1f}" y2="{sy(-fh - fuse_h * 0.35):.1f}" '
                 f'stroke="#45475a" stroke-width="1.2"/>')
    parts.append(f'<circle cx="{sx(mg_x):.1f}" cy="{sy(-fh - fuse_h * 0.35):.1f}" r="4" '
                 f'stroke="#45475a" fill="none"/>')
    parts.append(f'<circle cx="{sx(mg_x + fuse_h * 0.08):.1f}" cy="{sy(-fh - fuse_h * 0.35):.1f}" r="4" '
                 f'stroke="#45475a" fill="none"/>')

    # --- Length annotation ---
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(-fh - fuse_h * 0.6):.1f}" '
                 f'x2="{sx(fuse_len):.1f}" y2="{sy(-fh - fuse_h * 0.6):.1f}" '
                 f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="3,2"/>')
    parts.append(f'<text x="{sx(fuse_len / 2):.1f}" y="{sy(-fh - fuse_h * 0.6) + 12:.1f}" '
                 f'text-anchor="middle">LENGTH {fuse_len:.1f}m</text>')

    parts.append('</svg>')
    return "\n".join(parts)


def _generate_front_view_svg(p: AircraftSVGRequest) -> str:
    """Generate front-view SVG of the aircraft."""
    span = p.span_m
    half_span = span / 2.0
    fuse_w = p.fuselage_width_m
    fuse_h = fuse_w * 1.07
    vt_h = p.vtail_height_m

    total_width_m = span + 4
    total_height_m = fuse_h + vt_h + fuse_h * 0.6 + 4  # include gear
    scale = min(380.0 / total_width_m, 260.0 / total_height_m)

    cx = 200.0
    fuse_center_y = 170.0

    def sx(x_m):
        return cx + x_m * scale

    def sy(y_m):
        return fuse_center_y - y_m * scale

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 400 300" '
                 f'width="400" height="300" style="background:#1e1e2e">')
    parts.append('<defs><style>'
                 'text{font-family:Share Tech Mono,Courier New,monospace;fill:#bac2de;font-size:9px;}'
                 '</style></defs>')

    green = "#89b4fa"
    fill_g = "rgba(137,180,250,0.08)"
    amber = "#fab387"

    fw = fuse_w / 2.0
    fh = fuse_h / 2.0

    # --- Fuselage cross-section (ellipse) ---
    parts.append(f'<ellipse cx="{sx(0):.1f}" cy="{sy(0):.1f}" '
                 f'rx="{fw * scale:.1f}" ry="{fh * scale:.1f}" '
                 f'stroke="{green}" stroke-width="1.5" fill="{fill_g}"/>')

    # --- Wings (front view): thin lines with dihedral ---
    dihedral_deg = 5.0
    dihedral_rad = math.radians(dihedral_deg)
    wing_thickness = p.root_chord_m * 0.12 * scale  # visual thickness

    for sign in [-1, 1]:
        tip_y_offset = half_span * math.sin(dihedral_rad)
        wing_y_base = 0.0  # at mid fuselage
        # Wing as a thin trapezoid
        x1 = sx(sign * fw)
        x2 = sx(sign * half_span)
        y1_top = sy(wing_y_base) - wing_thickness / 2
        y1_bot = sy(wing_y_base) + wing_thickness / 2
        y2_top = sy(tip_y_offset) - wing_thickness / 4
        y2_bot = sy(tip_y_offset) + wing_thickness / 4

        wing_poly = (f'{x1:.1f},{y1_top:.1f} {x2:.1f},{y2_top:.1f} '
                     f'{x2:.1f},{y2_bot:.1f} {x1:.1f},{y1_bot:.1f}')
        parts.append(f'<polygon points="{wing_poly}" stroke="{green}" '
                     f'stroke-width="1" fill="{fill_g}"/>')

    # --- Vertical Tail (front view): on top ---
    vt_w = p.vtail_tip_chord_m * 0.1 * scale  # thin in front view
    parts.append(f'<rect x="{sx(0) - vt_w / 2:.1f}" y="{sy(fh):.1f}" '
                 f'width="{vt_w:.1f}" height="{vt_h * scale:.1f}" '
                 f'transform="translate(0,{-vt_h * scale:.1f})" '
                 f'stroke="{green}" stroke-width="1" fill="{fill_g}"/>')
    # Simpler: line for VT
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(fh):.1f}" '
                 f'x2="{sx(0):.1f}" y2="{sy(fh + vt_h):.1f}" '
                 f'stroke="{green}" stroke-width="2"/>')

    # --- Horizontal Tail (front view): short line ---
    ht_half = p.tail_span_m / 2.0
    ht_y = fh + vt_h * 0.05  # near base of VT
    for sign in [-1, 1]:
        parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(ht_y):.1f}" '
                     f'x2="{sx(sign * ht_half):.1f}" y2="{sy(ht_y + ht_half * math.sin(dihedral_rad)):.1f}" '
                     f'stroke="{green}" stroke-width="1.5"/>')

    # --- Engines (front view): circles under wings ---
    if p.n_engines >= 2:
        eng_r = p.engine_diameter_m / 2.0
        eng_span_frac = 0.35
        for sign in [-1, 1]:
            eng_x = sign * half_span * eng_span_frac
            eng_y_off = -eng_r * 1.5 + half_span * eng_span_frac * math.sin(dihedral_rad)
            parts.append(f'<circle cx="{sx(eng_x):.1f}" cy="{sy(eng_y_off):.1f}" '
                         f'r="{eng_r * scale:.1f}" '
                         f'stroke="{amber}" stroke-width="1" fill="rgba(255,170,0,0.08)"/>')
        if p.n_engines == 4:
            for sign in [-1, 1]:
                eng_x = sign * half_span * 0.65
                eng_y_off = -eng_r * 1.5 + half_span * 0.65 * math.sin(dihedral_rad)
                parts.append(f'<circle cx="{sx(eng_x):.1f}" cy="{sy(eng_y_off):.1f}" '
                             f'r="{eng_r * scale:.1f}" '
                             f'stroke="{amber}" stroke-width="1" fill="rgba(255,170,0,0.08)"/>')

    # --- Landing Gear (front view) ---
    gear_y = -fh - fuse_h * 0.3
    # Nose gear
    parts.append(f'<line x1="{sx(0):.1f}" y1="{sy(-fh):.1f}" '
                 f'x2="{sx(0):.1f}" y2="{sy(gear_y):.1f}" '
                 f'stroke="#45475a" stroke-width="0.8"/>')
    parts.append(f'<circle cx="{sx(0):.1f}" cy="{sy(gear_y):.1f}" r="2.5" stroke="#45475a" fill="none"/>')
    # Main gear
    for sign in [-1, 1]:
        mg_x = sign * fuse_w * 0.8
        parts.append(f'<line x1="{sx(sign * fw * 0.5):.1f}" y1="{sy(-fh):.1f}" '
                     f'x2="{sx(mg_x):.1f}" y2="{sy(gear_y):.1f}" '
                     f'stroke="#45475a" stroke-width="1"/>')
        parts.append(f'<circle cx="{sx(mg_x):.1f}" cy="{sy(gear_y):.1f}" r="3" stroke="#45475a" fill="none"/>')

    # --- Span annotation ---
    ann_y = 0.0
    parts.append(f'<line x1="{sx(-half_span):.1f}" y1="{sy(ann_y) + 20:.1f}" '
                 f'x2="{sx(half_span):.1f}" y2="{sy(ann_y) + 20:.1f}" '
                 f'stroke="#585b70" stroke-width="0.5" stroke-dasharray="3,2"/>')
    parts.append(f'<text x="{cx:.1f}" y="{sy(ann_y) + 30:.1f}" text-anchor="middle">'
                 f'SPAN {p.span_m:.1f}m</text>')

    parts.append('</svg>')
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main SPA."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path), media_type="text/html")
    return HTMLResponse("<h1>AeroSizer</h1><p>static/index.html not found.</p>")


@app.post("/api/size")
async def api_size(req: SizingRequest):
    """Run initial weight sizing (Raymer Ch 3/6)."""
    try:
        ac_type = _parse_aircraft_type(req.aircraft_type)

        mission = MissionProfile.transport_default(
            range_km=req.range_km,
            passengers=req.passengers,
            cruise_mach=req.cruise_mach,
            cruise_alt_m=req.cruise_alt_m,
            loiter_hr=req.loiter_hr,
        )
        mission.aircraft_type = ac_type

        if req.crew_weight_kg is not None:
            mission.crew_weight_kg = req.crew_weight_kg

        mission.payload_weight_kg = req.payload_extra_kg

        result: SizingResult = size_aircraft(
            mission,
            ld_cruise=req.ld_cruise,
            ld_loiter=req.ld_loiter,
            sfc_cruise=req.sfc_cruise,
            sfc_loiter=req.sfc_loiter,
        )

        return JSONResponse({
            "W0_kg": round(result.W0_kg, 1),
            "We_kg": round(result.We_kg, 1),
            "Wf_kg": round(result.Wf_kg, 1),
            "W_payload_kg": round(result.W_payload_kg, 1),
            "W_crew_kg": round(result.W_crew_kg, 1),
            "We_fraction": round(result.We_fraction, 5),
            "Wf_fraction": round(result.Wf_fraction, 5),
            "iterations": result.iterations,
            "convergence_history": [round(w, 1) for w in result.convergence_history],
            "segment_fractions": {
                k: round(v, 6) for k, v in result.segment_fractions.items()
            },
            "ld_cruise": round(result.ld_cruise, 2),
            "ld_loiter": round(result.ld_loiter, 2),
            "sfc_cruise": round(result.sfc_cruise, 4),
            "sfc_loiter": round(result.sfc_loiter, 4),
        })

    except (ValueError, RuntimeError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    except Exception as e:
        return JSONResponse(
            {"error": f"Internal error: {str(e)}"},
            status_code=500,
        )


@app.post("/api/constraints")
async def api_constraints(req: ConstraintRequest):
    """Run constraint analysis (Raymer Ch 5 T/W vs W/S)."""
    try:
        ca = ConstraintAnalysis(
            CD0=req.CD0,
            K=req.K,
            CLmax_clean=req.CLmax_clean,
            CLmax_takeoff=req.CLmax_takeoff,
            CLmax_landing=req.CLmax_landing,
            cruise_mach=req.cruise_mach,
            cruise_altitude_m=req.cruise_alt_m,
            takeoff_distance_m=req.takeoff_distance_m,
            landing_distance_m=req.landing_distance_m,
            climb_gradient_oei=req.climb_gradient_oei,
            service_ceiling_m=req.service_ceiling_m,
            n_engines=req.n_engines,
            thrust_lapse_cruise=req.thrust_lapse_cruise,
        )

        plot = ca.plot_data()
        ws_opt, tw_opt = ca.find_design_point()
        landing_ws_max = ca.landing_constraint()

        return JSONResponse({
            "traces": plot["traces"],
            "design_point": {"ws": round(ws_opt, 1), "tw": round(tw_opt, 5)},
            "landing_ws_max": round(landing_ws_max, 1),
            "plot_data": plot,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/drag_polar")
async def api_drag_polar(req: DragPolarRequest):
    """Compute drag polar curves for clean, takeoff, and landing configs."""
    try:
        dp = DragPolar(
            CD0=req.CD0,
            K=req.K,
            CD0_takeoff_increment=req.CD0_takeoff_increment,
            CD0_landing_increment=req.CD0_landing_increment,
        )

        cl_arr = np.arange(req.cl_min, req.cl_max + req.cl_step / 2.0, req.cl_step)

        cd_clean = [dp.cd(cl) for cl in cl_arr]
        cd_takeoff = [dp.cd_takeoff(cl) for cl in cl_arr]
        cd_landing = [dp.cd_landing(cl) for cl in cl_arr]

        ld_clean = []
        for cl in cl_arr:
            if cl > 0:
                ld_clean.append(dp.ld_at(cl))
            else:
                ld_clean.append(0.0)

        max_ld = dp.max_ld()
        cl_max_ld = dp.cl_for_max_ld()
        cl_min_power = dp.cl_for_min_power()

        return JSONResponse({
            "CL": cl_arr.tolist(),
            "CD_clean": cd_clean,
            "CD_takeoff": cd_takeoff,
            "CD_landing": cd_landing,
            "LD_clean": ld_clean,
            "max_ld": round(max_ld, 2),
            "cl_max_ld": round(cl_max_ld, 4),
            "cl_min_power": round(cl_min_power, 4),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/weights")
async def api_weights(req: WeightsRequest):
    """Compute detailed weight breakdown (Raymer Ch 15)."""
    try:
        tw = TransportWeights(
            W0_kg=req.W0_kg,
            Wf_kg=req.Wf_kg,
            AR=req.AR,
            Sref_m2=req.Sref_m2,
            sweep_quarter_rad=math.radians(req.sweep_quarter_deg),
            taper_ratio=req.taper_ratio,
            t_c_root=req.t_c_root,
            Nz=req.Nz,
            fuselage_length_m=req.fuselage_length_m,
            fuselage_width_m=req.fuselage_width_m,
            fuselage_depth_m=req.fuselage_depth_m,
            Sht_m2=req.Sht_m2,
            Svt_m2=req.Svt_m2,
            sweep_ht_rad=math.radians(req.sweep_ht_deg),
            sweep_vt_rad=math.radians(req.sweep_vt_deg),
            AR_ht=req.AR_ht,
            AR_vt=req.AR_vt,
            taper_ht=req.taper_ht,
            taper_vt=req.taper_vt,
            t_c_ht=req.t_c_ht,
            t_c_vt=req.t_c_vt,
            n_engines=req.n_engines,
            engine_weight_each_kg=req.engine_weight_each_kg,
            thrust_each_N=req.thrust_each_N,
            n_crew=req.n_crew,
            n_pax=req.n_pax,
            weight_growth_factor=req.weight_growth_factor,
        )

        ws = tw.weight_statement()

        result = {}
        for k, v in ws.items():
            result[k] = {
                "kg": round(v, 1),
                "lb": round(kg_to_lb(v), 1),
                "pct": round(v / req.W0_kg * 100, 2) if req.W0_kg > 0 else 0.0,
            }

        return JSONResponse({
            "weight_statement": result,
            "We_total_kg": round(ws["TOTAL_EMPTY"], 1),
            "We_total_lb": round(kg_to_lb(ws["TOTAL_EMPTY"]), 1),
            "We_fraction": round(ws["TOTAL_EMPTY"] / req.W0_kg, 4) if req.W0_kg > 0 else 0.0,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


@app.post("/api/performance/takeoff_landing")
async def api_takeoff_landing(req: TakeoffLandingRequest):
    """Compute takeoff and landing distances using fundamental equations."""
    try:
        W = req.W_kg * G0
        S = req.S_m2
        T = req.T_N
        atm = atmosphere(req.altitude_m)
        rho = atm["rho"]

        # ---- TAKEOFF ---- (Raymer Ch 17)
        ws = W / S
        tw = T / W

        V_stall_to = math.sqrt(2.0 * ws / (rho * req.CLmax_takeoff))
        V_LOF = 1.1 * V_stall_to
        V_R = 1.05 * V_stall_to

        CL_ground = 0.0
        CD_ground = req.CD0 + req.K * CL_ground**2 + 0.015
        q_avg = 0.5 * rho * (0.7 * V_LOF)**2
        D_avg = q_avg * S * CD_ground
        L_avg = q_avg * S * CL_ground
        F_friction = req.mu_ground * (W - L_avg)
        F_net = T - D_avg - F_friction

        if F_net <= 0:
            return JSONResponse(
                {"error": "Insufficient thrust for takeoff (T < D + friction)"},
                status_code=400,
            )

        S_ground = (W / G0) * V_LOF**2 / (2.0 * F_net)

        t_rotation = 3.0
        S_rotation = V_LOF * t_rotation

        CL_lof = req.CLmax_takeoff / 1.21
        CD_lof = req.CD0 + req.K * CL_lof**2 + 0.005
        gamma_climb_sin = tw - CD_lof / CL_lof
        gamma_climb_sin = max(min(gamma_climb_sin, 0.20), 0.01)
        gamma_climb = math.asin(gamma_climb_sin)

        n_transition = 1.15
        R_transition = V_LOF**2 / (G0 * (n_transition - 1.0))
        S_transition = R_transition * math.sin(gamma_climb)
        h_transition = R_transition * (1.0 - math.cos(gamma_climb))

        h_screen = 10.7
        h_climb = max(h_screen - h_transition, 0.0)
        if gamma_climb > 0:
            S_climb = h_climb / math.tan(gamma_climb) if gamma_climb > 0.01 else 0.0
        else:
            S_climb = 0.0

        S_takeoff_total = S_ground + S_rotation + S_transition + S_climb

        # ---- BFL ----
        ws_psf = ws * 0.020885434
        sigma = rho / 1.225
        TOP = ws_psf / (sigma * req.CLmax_takeoff * tw)
        BFL_ft = 37.5 * TOP
        BFL = BFL_ft * 0.3048

        # ---- LANDING ----
        V_stall_land = math.sqrt(2.0 * ws / (rho * req.CLmax_landing))
        V_approach = 1.3 * V_stall_land
        V_TD = 1.15 * V_stall_land

        gamma_approach = 3.0
        gamma_app_rad = math.radians(gamma_approach)
        h_obstacle = 15.24
        S_approach = h_obstacle / math.tan(gamma_app_rad)

        R_flare = V_approach**2 / (G0 * 0.2)
        h_flare = R_flare * (1.0 - math.cos(gamma_app_rad))
        S_flare = R_flare * math.sin(gamma_app_rad)

        t_free = 2.0
        S_free = V_TD * t_free

        CD_land = req.CD0 + req.CD0 * 3.5 + req.K * 0.1**2
        V_avg_brake = 0.7 * V_TD
        q_brake = 0.5 * rho * V_avg_brake**2
        D_brake = q_brake * S * CD_land
        a_brake = G0 * req.mu_braking + D_brake / (W / G0)

        if a_brake > 0:
            S_brake = V_TD**2 / (2.0 * a_brake)
        else:
            S_brake = 3000.0

        S_landing_total = S_approach + S_flare + S_free + S_brake

        return JSONResponse({
            "takeoff": {
                "ground_roll_m": round(S_ground, 1),
                "rotation_m": round(S_rotation, 1),
                "transition_m": round(S_transition, 1),
                "climb_to_35ft_m": round(S_climb, 1),
                "total_m": round(S_takeoff_total, 1),
                "total_ft": round(m_to_ft(S_takeoff_total), 0),
                "V_stall_ms": round(V_stall_to, 1),
                "V_LOF_ms": round(V_LOF, 1),
                "V_R_ms": round(V_R, 1),
                "V_stall_kt": round(V_stall_to / 0.514444, 1),
            },
            "landing": {
                "approach_m": round(S_approach, 1),
                "flare_m": round(S_flare, 1),
                "free_roll_m": round(S_free, 1),
                "braking_m": round(S_brake, 1),
                "total_m": round(S_landing_total, 1),
                "total_ft": round(m_to_ft(S_landing_total), 0),
                "V_approach_ms": round(V_approach, 1),
                "V_approach_kt": round(V_approach / 0.514444, 1),
                "V_TD_ms": round(V_TD, 1),
            },
            "BFL_m": round(BFL, 1),
            "BFL_ft": round(m_to_ft(BFL), 0),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Turn Performance (Raymer Eqs 17.52-17.54)
# ---------------------------------------------------------------------------

@app.post("/api/turn_performance")
async def api_turn_performance(req: TurnPerformanceRequest):
    """Turn performance analysis -- sustained and instantaneous turns."""
    try:
        W = req.W_kg * G0
        S = req.S_m2
        ws = W / S
        tw = req.T_N / W

        atm = atmosphere(req.altitude_m)
        rho = atm["rho"]

        V_arr = np.linspace(req.velocity_range_start, req.velocity_range_end, 200)

        V_corner = math.sqrt(2.0 * req.n_limit * W / (rho * S * req.CLmax))

        sustained_V = []
        sustained_rate = []
        sustained_n = []
        instant_V = []
        instant_rate = []
        instant_n = []

        max_sustained_rate = 0.0
        max_sustained_speed = 0.0

        for V in V_arr:
            q = 0.5 * rho * V**2

            term = (q / (req.K * ws)) * (tw - q * req.CD0 / ws)
            if term > 1.0:
                n_sust = math.sqrt(term)
            else:
                n_sust = 1.0

            if n_sust > 1.0:
                psi_dot_sust = G0 * math.sqrt(n_sust**2 - 1.0) / V
            else:
                psi_dot_sust = 0.0

            sustained_V.append(round(V, 1))
            sustained_rate.append(round(math.degrees(psi_dot_sust), 4))
            sustained_n.append(round(n_sust, 4))

            if psi_dot_sust > max_sustained_rate:
                max_sustained_rate = psi_dot_sust
                max_sustained_speed = V

            n_aero = q * req.CLmax / ws
            n_inst = min(n_aero, req.n_limit)
            n_inst = max(n_inst, 1.0)

            if n_inst > 1.0:
                psi_dot_inst = G0 * math.sqrt(n_inst**2 - 1.0) / V
            else:
                psi_dot_inst = 0.0

            instant_V.append(round(V, 1))
            instant_rate.append(round(math.degrees(psi_dot_inst), 4))
            instant_n.append(round(n_inst, 4))

        return JSONResponse({
            "sustained": {
                "V": sustained_V,
                "turn_rate_degs": sustained_rate,
                "load_factor": sustained_n,
            },
            "instantaneous": {
                "V": instant_V,
                "turn_rate_degs": instant_rate,
                "load_factor": instant_n,
            },
            "corner_speed_ms": round(V_corner, 2),
            "corner_speed_kt": round(V_corner / 0.514444, 1),
            "max_sustained_rate_degs": round(math.degrees(max_sustained_rate), 2),
            "max_sustained_speed_ms": round(max_sustained_speed, 1),
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Ps Diagram (Raymer Eq 17.89)
# ---------------------------------------------------------------------------

@app.post("/api/ps_diagram")
async def api_ps_diagram(req: PsDiagramRequest):
    """Specific excess power diagram."""
    try:
        W = req.W_kg * G0
        S = req.S_m2
        ws = W / S
        tw = req.T_N / W

        atm = atmosphere(req.altitude_m)
        rho = atm["rho"]
        a_sound = atm["a"]

        mach_arr = np.linspace(0.1, 1.2, 150)
        V_arr = mach_arr * a_sound

        ps_results: Dict[str, List[float]] = {}

        for n in req.load_factors:
            ps_line = []
            for i, V in enumerate(V_arr):
                q = 0.5 * rho * V**2
                ps = V * (tw - q * req.CD0 / ws - req.K * n**2 * ws / q)
                ps_line.append(round(ps, 2))
            key = str(int(n)) if n == int(n) else str(n)
            ps_results[key] = ps_line

        return JSONResponse({
            "mach": [round(m, 4) for m in mach_arr.tolist()],
            "ps": ps_results,
            "altitude_m": req.altitude_m,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# DAPCA IV Cost Estimation (Raymer Ch 18)
# ---------------------------------------------------------------------------

@app.post("/api/cost/dapca")
async def api_cost_dapca(req: DAPCACostRequest):
    """DAPCA IV cost estimation."""
    try:
        if HAS_DAPCA:
            inputs = DAPCAInputs(
                We_kg=req.We_kg,
                V_max_ms=req.V_max_ms,
                Q=req.Q,
                FTA=req.FTA,
                N_engines=req.N_engines,
                T_max_N=req.T_max_N,
                M_max_engine=req.M_max_engine,
                T_turbine_inlet_K=req.T_turbine_inlet_K,
                C_avionics=req.C_avionics,
                n_pax=req.n_pax,
                cargo_aircraft=req.cargo_aircraft,
                CPI_factor=req.CPI_factor,
            )
            result = compute_dapca(inputs)

            return JSONResponse({
                "hours": {
                    "engineering": round(result.H_E, 0),
                    "tooling": round(result.H_T, 0),
                    "manufacturing": round(result.H_M, 0),
                    "quality_control": round(result.H_Q, 0),
                },
                "costs": {
                    "engineering": round(result.C_engineering, 0),
                    "tooling": round(result.C_tooling, 0),
                    "manufacturing": round(result.C_manufacturing, 0),
                    "quality_control": round(result.C_QC, 0),
                    "development_support": round(result.C_development_support, 0),
                    "flight_test": round(result.C_flight_test, 0),
                    "materials": round(result.C_materials, 0),
                    "engines": round(result.C_engines_total, 0),
                    "avionics": round(result.C_avionics, 0),
                    "interior": round(result.C_interior, 0),
                },
                "totals": {
                    "RDTE": round(result.C_RDTE, 0),
                    "flyaway": round(result.C_flyaway_total, 0),
                    "program": round(result.C_program_total, 0),
                },
                "per_unit": {
                    "flyaway": round(result.C_flyaway_per_unit, 0),
                    "total": round(result.C_unit_cost, 0),
                },
            })
        else:
            # Inline DAPCA IV fallback
            W = req.We_kg * 0.62
            V_kmh = req.V_max_ms * 3.6
            Q = float(req.Q)
            FTA = float(req.FTA)

            H_E = 5.18 * W**0.777 * V_kmh**0.894 * Q**0.163
            H_T = 7.22 * W**0.777 * V_kmh**0.696 * Q**0.263
            H_M = 10.5 * W**0.82 * V_kmh**0.484 * Q**0.641
            H_Q = 0.133 * H_M

            C_D = 67.4 * W**0.630 * V_kmh**1.3
            C_F = 1947.0 * W**0.325 * V_kmh**0.822 * FTA**1.21
            C_mat = 31.2 * W**0.921 * V_kmh**0.621 * Q**0.799

            R_E, R_T, R_Q, R_M = 115.0, 118.0, 108.0, 98.0
            cpi = req.CPI_factor

            C_eng = H_E * R_E
            C_tool = H_T * R_T
            C_mfg = H_M * R_M
            C_qc = H_Q * R_Q

            C_RDTE = (C_eng + C_tool + C_D + C_F) * cpi
            C_flyaway = (C_mfg + C_qc + C_mat) * cpi
            C_total = C_RDTE + C_flyaway

            return JSONResponse({
                "hours": {
                    "engineering": round(H_E, 0),
                    "tooling": round(H_T, 0),
                    "manufacturing": round(H_M, 0),
                    "quality_control": round(H_Q, 0),
                },
                "costs": {
                    "engineering": round(C_eng * cpi, 0),
                    "tooling": round(C_tool * cpi, 0),
                    "manufacturing": round(C_mfg * cpi, 0),
                    "quality_control": round(C_qc * cpi, 0),
                    "development_support": round(C_D * cpi, 0),
                    "flight_test": round(C_F * cpi, 0),
                    "materials": round(C_mat * cpi, 0),
                    "engines": 0,
                    "avionics": 0,
                    "interior": 0,
                },
                "totals": {
                    "RDTE": round(C_RDTE, 0),
                    "flyaway": round(C_flyaway, 0),
                    "program": round(C_total, 0),
                },
                "per_unit": {
                    "flyaway": round(C_flyaway / req.Q, 0) if req.Q > 0 else 0,
                    "total": round(C_total / req.Q, 0) if req.Q > 0 else 0,
                },
            })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Operating Cost Estimation
# ---------------------------------------------------------------------------

@app.post("/api/cost/operating")
async def api_cost_operating(req: OperatingCostRequest):
    """Operating cost estimation for transport aircraft."""
    try:
        range_m = req.range_km * 1000.0
        block_time_hr = range_m / req.V_cruise_ms / 3600.0 + 0.5
        block_fuel_kg = req.Wf_mission_kg
        annual_hours = 3500
        annual_cycles = annual_hours / block_time_hr

        fuel_cost_per_hr = block_fuel_kg * req.fuel_price_per_kg / block_time_hr

        n_cabin_crew = max(1, req.n_pax // 50)
        pilot_cost_hr = 250.0 * 2
        cabin_crew_cost_hr = 80.0 * n_cabin_crew
        crew_cost_per_hr = pilot_cost_hr + cabin_crew_cost_hr

        airframe_maint_per_hr = 3.0 * (req.We_kg / 1000.0)
        engine_maint_per_hr = 150.0 * req.n_engines
        maintenance_per_hr = airframe_maint_per_hr + engine_maint_per_hr

        total_aircraft_cost = req.aircraft_cost + req.engine_cost_each * req.n_engines
        depreciation_per_hr = total_aircraft_cost * 0.90 / (20 * annual_hours)

        insurance_per_hr = total_aircraft_cost * 0.015 / annual_hours

        nav_fees_per_hr = 50.0 + 0.5 * req.W0_kg / 1000.0

        doc_per_hr = (fuel_cost_per_hr + crew_cost_per_hr + maintenance_per_hr
                      + depreciation_per_hr + insurance_per_hr + nav_fees_per_hr)

        doc_per_trip = doc_per_hr * block_time_hr
        fuel_per_trip = block_fuel_kg * req.fuel_price_per_kg

        trip_distance_nm = req.range_km * 0.539957
        seat_miles = req.n_pax * trip_distance_nm
        doc_per_seat_mile = doc_per_trip / seat_miles if seat_miles > 0 else 0
        casm = doc_per_trip / (req.n_pax * trip_distance_nm) * 100 if seat_miles > 0 else 0

        fuel_per_seat_km = block_fuel_kg / (req.n_pax * req.range_km) if req.n_pax > 0 and req.range_km > 0 else 0
        co2_per_seat_km = fuel_per_seat_km * 3.16

        annual_fuel = fuel_per_trip * annual_cycles
        annual_crew = crew_cost_per_hr * annual_hours
        annual_maint = maintenance_per_hr * annual_hours
        annual_deprec = depreciation_per_hr * annual_hours
        annual_insurance = insurance_per_hr * annual_hours
        annual_total = doc_per_hr * annual_hours

        return JSONResponse({
            "per_block_hour": {
                "fuel": round(fuel_cost_per_hr, 0),
                "crew": round(crew_cost_per_hr, 0),
                "maintenance": round(maintenance_per_hr, 0),
                "depreciation": round(depreciation_per_hr, 0),
                "insurance": round(insurance_per_hr, 0),
                "navigation": round(nav_fees_per_hr, 0),
                "total_DOC": round(doc_per_hr, 0),
            },
            "annual": {
                "fuel": round(annual_fuel, 0),
                "crew": round(annual_crew, 0),
                "maintenance": round(annual_maint, 0),
                "depreciation": round(annual_deprec, 0),
                "insurance": round(annual_insurance, 0),
                "total": round(annual_total, 0),
                "utilization_hours": annual_hours,
                "cycles": round(annual_cycles, 0),
            },
            "per_trip": {
                "DOC": round(doc_per_trip, 0),
                "fuel": round(fuel_per_trip, 0),
                "block_time_hr": round(block_time_hr, 2),
            },
            "economics": {
                "DOC_per_seat_mile": round(doc_per_seat_mile, 4),
                "CASM_cents": round(casm, 2),
                "fuel_per_seat_km_kg": round(fuel_per_seat_km, 4),
                "CO2_per_seat_km_kg": round(co2_per_seat_km, 4),
            },
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Climb Profile
# ---------------------------------------------------------------------------

@app.post("/api/performance/climb_profile")
async def api_climb_profile(req: ClimbProfileRequest):
    """Climb analysis: ROC vs altitude, service/absolute ceiling."""
    try:
        polar = DragPolar(CD0=req.CD0, K=req.K)
        T_SL = req.T_N

        alt_arr = np.linspace(req.alt_range_start, req.alt_range_end, 60)

        roc_arr = []
        v_best_arr = []
        service_ceil = None
        absolute_ceil = None

        for h in alt_arr:
            T = _thrust_lapse(T_SL, h)
            try:
                roc_max, v_best = max_rate_of_climb(
                    req.W_kg, req.S_m2, T, polar, h
                )
            except Exception:
                roc_max, v_best = 0.0, 0.0
            roc_arr.append(round(max(roc_max, 0.0), 2))
            v_best_arr.append(round(v_best, 1))

        roc_np = np.array(roc_arr)
        for i in range(len(roc_np) - 1):
            if roc_np[i] >= 0.508 and roc_np[i + 1] < 0.508:
                frac = (0.508 - roc_np[i]) / (roc_np[i + 1] - roc_np[i])
                service_ceil = float(alt_arr[i] + frac * (alt_arr[i + 1] - alt_arr[i]))
            if roc_np[i] > 0 and roc_np[i + 1] <= 0:
                frac = (0 - roc_np[i]) / (roc_np[i + 1] - roc_np[i])
                absolute_ceil = float(alt_arr[i] + frac * (alt_arr[i + 1] - alt_arr[i]))

        return JSONResponse({
            "altitude": [round(float(h), 0) for h in alt_arr],
            "ROC_max": roc_arr,
            "V_best": v_best_arr,
            "service_ceiling_m": round(service_ceil, 0) if service_ceil else None,
            "absolute_ceiling_m": round(absolute_ceil, 0) if absolute_ceil else None,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Range-Payload Diagram
# ---------------------------------------------------------------------------

@app.post("/api/performance/range_payload")
async def api_range_payload(req: RangePayloadRequest):
    """Range-payload diagram (Raymer Fig 6.1)."""
    try:
        polar = DragPolar(CD0=req.CD0, K=req.K)

        result = range_payload_diagram(
            W0_kg=req.W0_kg,
            We_kg=req.We_kg,
            Wf_max_kg=req.Wf_max_kg,
            sfc_per_hr=req.sfc_per_hr,
            velocity_ms=req.V_cruise_ms,
            polar=polar,
            W_crew_kg=req.W_crew_kg,
        )

        points = {}
        for label, pt in result["points"].items():
            points[label] = {
                "range_km": round(pt["range_m"] / 1000.0, 1),
                "payload_kg": round(pt["payload_kg"], 1),
            }

        curves_range = result["curves"]["range_m"]
        curves_payload = result["curves"]["payload_kg"]
        if hasattr(curves_range, "tolist"):
            curves_range = curves_range.tolist()
        if hasattr(curves_payload, "tolist"):
            curves_payload = curves_payload.tolist()

        return JSONResponse({
            "points": points,
            "curves": {
                "range_km": [round(r / 1000.0, 1) for r in curves_range],
                "payload_kg": [round(p, 1) for p in curves_payload],
            },
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Operating Envelope (Ps contours vs Mach and Altitude)
# ---------------------------------------------------------------------------

@app.post("/api/performance/envelope")
async def api_performance_envelope(req: EnvelopeRequest):
    """Operating envelope: Ps grid across Mach/altitude space."""
    try:
        W = req.W_kg * G0
        S = req.S_m2
        ws = W / S
        T_SL = req.T_N

        mach_arr = np.linspace(0.1, 1.0, 50)
        alt_arr = np.linspace(0, 15000, 40)

        ps_grid = []
        stall_line = []

        for alt in alt_arr:
            atm = atmosphere(float(alt))
            rho = atm["rho"]
            a = atm["a"]
            T = _thrust_lapse(T_SL, float(alt))
            tw = T / W

            ps_row = []
            for mach in mach_arr:
                V = float(mach) * a
                q = 0.5 * rho * V**2
                ps = V * (tw - q * req.CD0 / ws - req.K * ws / q)
                ps_row.append(round(ps, 1))
            ps_grid.append(ps_row)

            V_stall = math.sqrt(2.0 * ws / (rho * req.CLmax))
            M_stall = V_stall / a
            stall_line.append(round(M_stall, 4))

        q_max = 35000
        q_limit_line = []
        for alt in alt_arr:
            atm = atmosphere(float(alt))
            rho = atm["rho"]
            a = atm["a"]
            V_qlim = math.sqrt(2.0 * q_max / rho)
            M_qlim = V_qlim / a
            q_limit_line.append(round(min(M_qlim, 1.2), 4))

        return JSONResponse({
            "mach": [round(float(m), 4) for m in mach_arr],
            "altitude": [round(float(a), 0) for a in alt_arr],
            "ps_grid": ps_grid,
            "stall_line": stall_line,
            "q_limit_line": q_limit_line,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Aircraft SVG Visualization (NEW)
# ---------------------------------------------------------------------------

@app.post("/api/geometry/aircraft_svg")
async def api_aircraft_svg(req: AircraftSVGRequest):
    """Generate three-view SVG drawings of the aircraft."""
    try:
        top_svg = _generate_top_view_svg(req)
        side_svg = _generate_side_view_svg(req)
        front_svg = _generate_front_view_svg(req)

        return JSONResponse({
            "top_view_svg": top_svg,
            "side_view_svg": side_svg,
            "front_view_svg": front_svg,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Geometry from Sizing (NEW)
# ---------------------------------------------------------------------------

@app.post("/api/geometry/from_sizing")
async def api_geometry_from_sizing(req: GeometryFromSizingRequest):
    """Compute aircraft geometry from sizing results using Raymer rules."""
    try:
        S = req.S_m2
        AR = req.AR
        taper = req.taper_ratio
        sweep_deg = req.sweep_quarter_deg
        fuse_len = req.fuselage_length_m
        fuse_w = req.fuselage_width_m

        # Wing geometry
        span = math.sqrt(AR * S)
        root_chord = 2.0 * S / (span * (1.0 + taper))
        tip_chord = taper * root_chord
        MAC = (2.0 / 3.0) * root_chord * (1.0 + taper + taper**2) / (1.0 + taper)

        # Tail sizing from volume coefficients (Raymer Table 6.4)
        c_ht = 1.00  # horizontal tail volume coefficient (transport)
        c_vt = 0.09  # vertical tail volume coefficient (transport)
        l_ht = 0.45 * fuse_len  # tail arm (horizontal)
        l_vt = 0.45 * fuse_len  # tail arm (vertical)

        Sht = c_ht * MAC * S / l_ht
        Svt = c_vt * span * S / l_vt

        # Horizontal tail geometry
        AR_ht = 4.0
        taper_ht = 0.4
        tail_span = math.sqrt(AR_ht * Sht)
        tail_root_chord = 2.0 * Sht / (tail_span * (1.0 + taper_ht))
        tail_tip_chord = taper_ht * tail_root_chord
        tail_sweep_deg = sweep_deg + 5.0  # tail typically more swept

        # Vertical tail geometry
        AR_vt = 1.5
        taper_vt = 0.4
        vtail_height = math.sqrt(AR_vt * Svt)
        vtail_root_chord = 2.0 * Svt / (vtail_height * (1.0 + taper_vt))
        vtail_tip_chord = taper_vt * vtail_root_chord
        vtail_sweep_deg = sweep_deg + 10.0

        # Engine sizing (rough estimate from W0)
        # Typical high-BPR turbofan: diameter ~1.5-2.5m for 100-300kN class
        T_required = req.W0_kg * 9.81 * 0.30  # assume T/W ~ 0.30
        T_per_engine = T_required / req.n_engines
        # Fan diameter ~ (T / 40000)^0.5 * 1.5 (rough correlation)
        engine_diameter = max(1.0, min(3.5, (T_per_engine / 40000)**0.5 * 1.5))
        engine_length = engine_diameter * 2.0

        return JSONResponse({
            "span_m": round(span, 2),
            "root_chord_m": round(root_chord, 2),
            "tip_chord_m": round(tip_chord, 2),
            "MAC_m": round(MAC, 2),
            "sweep_quarter_deg": round(sweep_deg, 1),
            "fuselage_length_m": round(fuse_len, 2),
            "fuselage_width_m": round(fuse_w, 2),
            "tail_span_m": round(tail_span, 2),
            "tail_root_chord_m": round(tail_root_chord, 2),
            "tail_tip_chord_m": round(tail_tip_chord, 2),
            "tail_sweep_deg": round(tail_sweep_deg, 1),
            "Sht_m2": round(Sht, 2),
            "vtail_height_m": round(vtail_height, 2),
            "vtail_root_chord_m": round(vtail_root_chord, 2),
            "vtail_tip_chord_m": round(vtail_tip_chord, 2),
            "vtail_sweep_deg": round(vtail_sweep_deg, 1),
            "Svt_m2": round(Svt, 2),
            "engine_diameter_m": round(engine_diameter, 2),
            "engine_length_m": round(engine_length, 2),
            "n_engines": req.n_engines,
            "wing_position_fraction": 0.40,
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

@app.get("/api/presets/{preset_name}")
async def api_preset(preset_name: str):
    """Return preset aircraft configuration data."""
    name = preset_name.lower().strip()
    if name not in PRESETS:
        return JSONResponse(
            {"error": f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}"},
            status_code=400,
        )
    return JSONResponse(PRESETS[name])


@app.get("/api/presets")
async def api_presets_list():
    """List all available presets."""
    return JSONResponse({k: v["name"] for k, v in PRESETS.items()})


# ---------------------------------------------------------------------------
# Run with uvicorn if executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)
