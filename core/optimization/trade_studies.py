"""
Parametric Trade Studies
Reference: Raymer Ch 19.5, "Aircraft Design: A Conceptual Approach", 6th Edition

Provides functions for single-variable and multi-variable parametric
sweeps to study the sensitivity of aircraft weight, performance, and
cost to design parameter changes.

Key concepts (Raymer Ch 19.5):
  - Parametric variation of one design variable at a time while holding
    others constant reveals sensitivity and identifies optimum values.
  - Growth factor (Ch 19.5.1): measures how much W0 increases per unit
    of unexpected "dead weight" added to the aircraft. Typically 2-5
    for transports (i.e. 1 kg of dead weight grows to 2-5 kg of W0).
  - Sensitivity = (%Delta W0) / (%Delta parameter) at the baseline.

All internal calculations use SI units (kg, m, s, Pa).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
import math
import copy

from ..atmosphere import true_airspeed, density_at, G0
from ..mission import MissionProfile
from ..sizing.initial_sizing import size_aircraft
from ..aerodynamics.induced_drag import oswald_efficiency_swept, induced_drag_factor
from ..aerodynamics.drag_polar import DragPolar


@dataclass
class TradeStudyResult:
    """Results from a parametric trade study.

    Attributes:
        parameter_name: human-readable name of the swept parameter.
        parameter_values: 1-D numpy array of parameter values tested.
        parameter_unit: unit string for the parameter (e.g. "", "deg", "km").
        W0_values: sized TOGW (kg) at each parameter value. NaN if infeasible.
        We_values: empty weight (kg) at each value.
        Wf_values: fuel weight (kg) at each value.
        LD_values: cruise L/D at each value.
        range_values: optional range data (km).
        bfl_values: optional balanced field length data (m).
        cost_values: optional unit cost data ($).
        baseline_index: index in parameter_values of the baseline design.
    """
    parameter_name: str = ""
    parameter_values: np.ndarray = None
    parameter_unit: str = ""

    W0_values: np.ndarray = None
    We_values: np.ndarray = None
    Wf_values: np.ndarray = None
    LD_values: np.ndarray = None

    range_values: np.ndarray = None
    bfl_values: np.ndarray = None
    cost_values: np.ndarray = None

    baseline_index: int = 0

    def sensitivity(self) -> float:
        """Compute sensitivity: %Delta_W0 / %Delta_parameter around the baseline.

        Raymer Ch 19.5.1: The sensitivity measures how much TOGW changes
        relative to a change in a design parameter. Computed as a central
        difference around the baseline point.

        Returns:
            Dimensionless sensitivity ratio. Returns 0.0 if the baseline
            is at an edge of the array or if data is insufficient.
        """
        if (self.parameter_values is None or self.W0_values is None
                or len(self.parameter_values) < 3):
            return 0.0

        i = self.baseline_index
        if i <= 0 or i >= len(self.parameter_values) - 1:
            return 0.0

        # Central difference
        p_base = self.parameter_values[i]
        w_base = self.W0_values[i]

        if abs(p_base) < 1e-10 or abs(w_base) < 1e-10:
            return 0.0
        if not (np.isfinite(self.W0_values[i - 1])
                and np.isfinite(self.W0_values[i + 1])):
            return 0.0

        dp = (self.parameter_values[i + 1] - self.parameter_values[i - 1]) / p_base
        dw = (self.W0_values[i + 1] - self.W0_values[i - 1]) / w_base

        if abs(dp) < 1e-10:
            return 0.0

        return dw / dp

    def growth_factor(self) -> float:
        """Compute the growth factor: Delta_W0 / Delta_parameter (absolute).

        For dead-weight trades, this gives kg_W0 per kg_dead_weight.

        Returns:
            Absolute growth factor in units of W0_unit / parameter_unit.
            Returns 0.0 if data is insufficient.
        """
        if (self.parameter_values is None or self.W0_values is None
                or len(self.parameter_values) < 3):
            return 0.0

        i = self.baseline_index
        if i <= 0 or i >= len(self.parameter_values) - 1:
            return 0.0

        if not (np.isfinite(self.W0_values[i - 1])
                and np.isfinite(self.W0_values[i + 1])):
            return 0.0

        dp = self.parameter_values[i + 1] - self.parameter_values[i - 1]
        dw = self.W0_values[i + 1] - self.W0_values[i - 1]

        if abs(dp) < 1e-10:
            return 0.0

        return dw / dp

    def summary(self) -> str:
        """Generate a formatted summary of the trade study.

        Returns:
            Multi-line string with key results.
        """
        lines = [
            f"Trade Study: {self.parameter_name}",
            "-" * 50,
        ]

        if self.W0_values is not None and np.any(np.isfinite(self.W0_values)):
            valid = np.isfinite(self.W0_values)
            lines.append(f"  Range: {self.parameter_values[valid][0]:.3g} - "
                         f"{self.parameter_values[valid][-1]:.3g} {self.parameter_unit}")
            lines.append(f"  W0 range: {np.nanmin(self.W0_values):,.0f} - "
                         f"{np.nanmax(self.W0_values):,.0f} kg")

            i = self.baseline_index
            if 0 <= i < len(self.W0_values) and np.isfinite(self.W0_values[i]):
                lines.append(f"  Baseline: {self.parameter_values[i]:.3g} "
                             f"{self.parameter_unit} -> W0 = "
                             f"{self.W0_values[i]:,.0f} kg")
            lines.append(f"  Sensitivity: {self.sensitivity():.4f}")

        lines.append("-" * 50)
        return "\n".join(lines)


def _safe_size(mission: MissionProfile, ld_cruise: float, ld_loiter: float,
               sfc_cruise: float, sfc_loiter: float) -> Optional[dict]:
    """Attempt to size an aircraft, returning None if infeasible.

    Args:
        mission: MissionProfile to size.
        ld_cruise: cruise L/D.
        ld_loiter: loiter L/D.
        sfc_cruise: cruise SFC (1/hr).
        sfc_loiter: loiter SFC (1/hr).

    Returns:
        Dict with W0, We, Wf keys (all in kg), or None on failure.
    """
    try:
        result = size_aircraft(
            mission=mission,
            ld_cruise=ld_cruise,
            ld_loiter=ld_loiter,
            sfc_cruise=sfc_cruise,
            sfc_loiter=sfc_loiter,
        )
        return {
            'W0': result.W0_kg,
            'We': result.We_kg,
            'Wf': result.Wf_kg,
        }
    except (ValueError, RuntimeError):
        return None


def aspect_ratio_trade(mission: MissionProfile, AR_range: np.ndarray = None,
                        sweep_LE_deg: float = 35.0, sfc_cruise: float = 0.55,
                        sfc_loiter: float = 0.45, cruise_mach: float = 0.85,
                        cruise_alt_m: float = 10668.0,
                        CD0: float = 0.018) -> TradeStudyResult:
    """Aspect ratio trade study.

    Raymer Ch 19.5, Table 19.1: AR is a primary design trade variable.

    Varying AR affects:
      - Induced drag: K = 1/(pi*AR*e)  ->  better L/D at higher AR
      - Wing weight: increases roughly as AR^0.4 to AR^0.5  ->  heavier structure
      - Net effect: an optimum AR balances aerodynamic benefit vs weight penalty.

    The L/D is recomputed at each AR using the Oswald efficiency and the
    cruise flight condition. The aircraft is then re-sized with the new L/D.

    Args:
        mission: base mission profile.
        AR_range: array of AR values to test. Default np.linspace(6, 12, 13).
        sweep_LE_deg: leading-edge sweep in degrees. Default 35.
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        cruise_mach: cruise Mach number. Default 0.85.
        cruise_alt_m: cruise altitude (m). Default 10668.
        CD0: baseline zero-lift drag coefficient. Default 0.018.

    Returns:
        TradeStudyResult with AR sweep data.
    """
    if AR_range is None:
        AR_range = np.linspace(6.0, 12.0, 13)

    AR_arr = np.asarray(AR_range, dtype=float)
    n = len(AR_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, np.nan)

    # Cruise conditions for CL calculation
    rho = density_at(cruise_alt_m)
    V = true_airspeed(cruise_mach, cruise_alt_m)
    q = 0.5 * rho * V ** 2

    sweep_LE_rad = math.radians(sweep_LE_deg)

    # Find baseline index (closest to AR=9)
    baseline_AR = 9.0
    baseline_idx = int(np.argmin(np.abs(AR_arr - baseline_AR)))

    for i, AR in enumerate(AR_arr):
        if AR <= 0:
            continue

        # Oswald efficiency and K depend on AR
        e = oswald_efficiency_swept(AR, sweep_LE_rad)
        if e <= 0.0:
            e = 0.05  # floor
        K = induced_drag_factor(AR, e)

        # L/D at cruise: need a representative W/S.
        # Use the cruise CL from the baseline sizing.
        # For a typical transport at M 0.85 / 35 kft, W/S ~ 6000 Pa.
        ws_est = 6000.0  # Pa (representative)
        CL_cruise = ws_est / q
        CD = CD0 + K * CL_cruise ** 2
        if CD <= 0.0:
            continue
        ld_cruise = CL_cruise / CD

        # Max L/D for loiter
        ld_max = 1.0 / (2.0 * math.sqrt(CD0 * K))
        ld_loiter = ld_max

        LD[i] = ld_cruise

        sizing = _safe_size(mission, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Aspect Ratio",
        parameter_values=AR_arr,
        parameter_unit="",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def sweep_trade(mission: MissionProfile, sweep_range_deg: np.ndarray = None,
                AR: float = 9.0, sfc_cruise: float = 0.55,
                sfc_loiter: float = 0.45,
                cruise_mach: float = 0.85,
                cruise_alt_m: float = 10668.0,
                CD0: float = 0.018) -> TradeStudyResult:
    """Wing sweep trade study.

    Raymer Ch 19.5: Sweep affects Oswald efficiency (Eq 12.50), wave
    drag margin, and wing weight. Higher sweep reduces e, worsening
    induced drag, but provides better transonic performance.

    For each sweep angle, the Oswald efficiency is recomputed, a new
    L/D is estimated, and the aircraft is re-sized.

    Args:
        mission: base mission profile.
        sweep_range_deg: array of LE sweep angles in degrees.
            Default np.linspace(15, 45, 13).
        AR: wing aspect ratio. Default 9.0.
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        cruise_mach: cruise Mach number. Default 0.85.
        cruise_alt_m: cruise altitude (m). Default 10668.
        CD0: baseline zero-lift drag coefficient. Default 0.018.

    Returns:
        TradeStudyResult with sweep angle sweep data.
    """
    if sweep_range_deg is None:
        sweep_range_deg = np.linspace(15.0, 45.0, 13)

    sw_arr = np.asarray(sweep_range_deg, dtype=float)
    n = len(sw_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, np.nan)

    rho = density_at(cruise_alt_m)
    V = true_airspeed(cruise_mach, cruise_alt_m)
    q = 0.5 * rho * V ** 2
    ws_est = 6000.0  # representative W/S

    # Baseline index: closest to 35 deg
    baseline_idx = int(np.argmin(np.abs(sw_arr - 35.0)))

    for i, sw_deg in enumerate(sw_arr):
        sw_rad = math.radians(sw_deg)

        e = oswald_efficiency_swept(AR, sw_rad)
        if e <= 0.0:
            e = 0.05
        K = induced_drag_factor(AR, e)

        CL_cruise = ws_est / q
        CD = CD0 + K * CL_cruise ** 2
        if CD <= 0.0:
            continue
        ld_cruise = CL_cruise / CD

        ld_max = 1.0 / (2.0 * math.sqrt(CD0 * K))
        ld_loiter = ld_max
        LD[i] = ld_cruise

        sizing = _safe_size(mission, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Wing Sweep (LE)",
        parameter_values=sw_arr,
        parameter_unit="deg",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def range_trade(mission_factory: Callable, range_values_km: np.ndarray = None,
                sfc_cruise: float = 0.55, sfc_loiter: float = 0.45,
                ld_cruise: float = 17.0,
                ld_loiter: float = 18.0) -> TradeStudyResult:
    """Range trade study -- sensitivity of W0 to design range.

    Raymer Ch 19.5: Requirements trade. The Breguet range equation
    shows exponential growth of fuel fraction with range:
        W_f/W_0 = 1 - exp(-R * C / (V * L/D))

    This means W0 grows faster than linearly with range, a critical
    consideration when evaluating range requirements.

    Args:
        mission_factory: callable(range_km) -> MissionProfile.
        range_values_km: array of range values in km.
            Default np.linspace(2000, 15000, 14).
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        ld_cruise: cruise L/D. Default 17.0.
        ld_loiter: loiter L/D. Default 18.0.

    Returns:
        TradeStudyResult with range sweep data.
    """
    if range_values_km is None:
        range_values_km = np.linspace(2000.0, 15000.0, 14)

    r_arr = np.asarray(range_values_km, dtype=float)
    n = len(r_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, ld_cruise)

    # Baseline: closest to 5500 km (medium-haul transport)
    baseline_idx = int(np.argmin(np.abs(r_arr - 5500.0)))

    for i, range_km in enumerate(r_arr):
        if range_km <= 0:
            continue

        mission = mission_factory(range_km)

        sizing = _safe_size(mission, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Design Range",
        parameter_values=r_arr,
        parameter_unit="km",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def payload_trade(base_mission: MissionProfile,
                  pax_range: np.ndarray = None,
                  sfc_cruise: float = 0.55, sfc_loiter: float = 0.45,
                  ld_cruise: float = 17.0,
                  ld_loiter: float = 18.0) -> TradeStudyResult:
    """Payload (passengers) trade study.

    Raymer Ch 19.5: Payload requirements trade. Linear increase in
    payload produces roughly linear increase in W0, but with
    amplification due to the fuel fraction (more weight needs more fuel,
    which needs more structure, etc.).

    The growth factor (Delta_W0 / Delta_W_payload) is typically 1.5-3.0
    for transports.

    Args:
        base_mission: base MissionProfile (passengers will be varied).
        pax_range: array of passenger counts. Default np.arange(100, 401, 25).
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        ld_cruise: cruise L/D. Default 17.0.
        ld_loiter: loiter L/D. Default 18.0.

    Returns:
        TradeStudyResult with passenger count sweep data.
    """
    if pax_range is None:
        pax_range = np.arange(100, 401, 25, dtype=float)

    pax_arr = np.asarray(pax_range, dtype=float)
    n = len(pax_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, ld_cruise)

    # Baseline: closest to original passenger count
    baseline_pax = base_mission.passengers if base_mission.passengers > 0 else 180
    baseline_idx = int(np.argmin(np.abs(pax_arr - baseline_pax)))

    for i, pax in enumerate(pax_arr):
        pax_int = int(round(pax))
        if pax_int <= 0:
            continue

        # Deep copy and modify passenger count
        mission = copy.deepcopy(base_mission)
        mission.passengers = pax_int

        sizing = _safe_size(mission, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Passengers",
        parameter_values=pax_arr,
        parameter_unit="pax",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def sfc_trade(mission: MissionProfile, sfc_range: np.ndarray = None,
              ld_cruise: float = 17.0,
              ld_loiter: float = 18.0) -> TradeStudyResult:
    """SFC trade study -- sensitivity to engine efficiency.

    Raymer Ch 19.5: Engine technology trade. SFC directly affects the
    fuel fraction via the Breguet equations:
        Wi/Wi-1 = exp(-R * C / (V * L/D))    (Raymer Eq 6.11)

    Lower SFC reduces fuel fraction, which compounds through the sizing
    loop to reduce W0 significantly. Shows the leverage of improved
    engine technology.

    Both cruise and loiter SFC are varied proportionally (the ratio
    between them is preserved).

    Args:
        mission: base mission profile.
        sfc_range: array of cruise SFC values (1/hr).
            Default np.linspace(0.35, 0.75, 17).
        ld_cruise: cruise L/D. Default 17.0.
        ld_loiter: loiter L/D. Default 18.0.

    Returns:
        TradeStudyResult with SFC sweep data.
    """
    if sfc_range is None:
        sfc_range = np.linspace(0.35, 0.75, 17)

    sfc_arr = np.asarray(sfc_range, dtype=float)
    n = len(sfc_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, ld_cruise)

    # Baseline: closest to 0.55
    baseline_idx = int(np.argmin(np.abs(sfc_arr - 0.55)))

    # Ratio of loiter to cruise SFC (preserve this ratio)
    loiter_ratio = 0.45 / 0.55  # ~0.818

    for i, sfc_c in enumerate(sfc_arr):
        if sfc_c <= 0.0:
            continue

        sfc_l = sfc_c * loiter_ratio

        sizing = _safe_size(mission, ld_cruise, ld_loiter, sfc_c, sfc_l)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="SFC (cruise)",
        parameter_values=sfc_arr,
        parameter_unit="1/hr",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def dead_weight_sensitivity(mission: MissionProfile,
                             dead_weight_range_kg: np.ndarray = None,
                             sfc_cruise: float = 0.55, sfc_loiter: float = 0.45,
                             ld_cruise: float = 17.0,
                             ld_loiter: float = 18.0) -> TradeStudyResult:
    """Dead weight growth sensitivity study.

    Raymer Ch 19.5.1 (Growth Factor):
    "Dead weight" represents unexpected weight growth -- heavier structure,
    additional avionics, required modifications, etc. The growth factor
    shows how much W0 increases per unit of dead weight added:

        Growth Factor = Delta_W0 / Delta_W_dead

    Typically 2-5 for jet transports. This means 1 kg of unanticipated
    weight growth increases takeoff weight by 2-5 kg due to the
    compounding effect of needing more fuel and structure.

    The dead weight is added to the payload_weight_kg of the mission,
    which forces the sizing loop to converge at a higher W0.

    Args:
        mission: base mission profile.
        dead_weight_range_kg: array of dead weight values (kg).
            Default np.linspace(0, 5000, 21).
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        ld_cruise: cruise L/D. Default 17.0.
        ld_loiter: loiter L/D. Default 18.0.

    Returns:
        TradeStudyResult with dead-weight sweep data.
        The growth_factor() method on the result gives the growth factor.
    """
    if dead_weight_range_kg is None:
        # Include negative values so baseline (0) is not at the edge.
        # This enables proper central-difference sensitivity computation.
        dead_weight_range_kg = np.linspace(-2000.0, 5000.0, 21)

    dw_arr = np.asarray(dead_weight_range_kg, dtype=float)
    n = len(dw_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)
    LD = np.full(n, ld_cruise)

    # Baseline: zero dead weight (or closest).
    # The range should include values below zero so baseline is interior.
    baseline_idx = int(np.argmin(np.abs(dw_arr)))

    base_payload = mission.payload_weight_kg

    for i, dw in enumerate(dw_arr):
        m = copy.deepcopy(mission)
        m.payload_weight_kg = base_payload + dw

        sizing = _safe_size(m, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Dead Weight",
        parameter_values=dw_arr,
        parameter_unit="kg",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=LD,
        baseline_index=baseline_idx,
    )


def ld_trade(mission: MissionProfile, ld_range: np.ndarray = None,
             sfc_cruise: float = 0.55,
             sfc_loiter: float = 0.45) -> TradeStudyResult:
    """L/D trade study -- sensitivity to aerodynamic efficiency.

    Raymer Ch 19.5: Technology trade. L/D directly affects fuel fraction
    via the Breguet equations. Higher L/D reduces fuel burn, which
    compounds through the sizing loop.

    Both cruise and loiter L/D are varied. Loiter L/D is set to
    cruise L/D + 1.0 (Raymer approximation for jets).

    Args:
        mission: base mission profile.
        ld_range: array of cruise L/D values.
            Default np.linspace(12, 22, 21).
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.

    Returns:
        TradeStudyResult with L/D sweep data.
    """
    if ld_range is None:
        ld_range = np.linspace(12.0, 22.0, 21)

    ld_arr = np.asarray(ld_range, dtype=float)
    n = len(ld_arr)

    W0 = np.full(n, np.nan)
    We = np.full(n, np.nan)
    Wf = np.full(n, np.nan)

    # Baseline: closest to 17.0
    baseline_idx = int(np.argmin(np.abs(ld_arr - 17.0)))

    for i, ld_c in enumerate(ld_arr):
        if ld_c <= 0.0:
            continue

        ld_l = ld_c + 1.0  # Raymer: loiter L/D slightly higher for jets

        sizing = _safe_size(mission, ld_c, ld_l, sfc_cruise, sfc_loiter)
        if sizing is not None:
            W0[i] = sizing['W0']
            We[i] = sizing['We']
            Wf[i] = sizing['Wf']

    return TradeStudyResult(
        parameter_name="Cruise L/D",
        parameter_values=ld_arr,
        parameter_unit="",
        W0_values=W0,
        We_values=We,
        Wf_values=Wf,
        LD_values=ld_arr.copy(),
        baseline_index=baseline_idx,
    )


def multi_trade_summary(mission: MissionProfile,
                         cruise_mach: float = 0.85,
                         cruise_alt_m: float = 10668.0,
                         sfc_cruise: float = 0.55,
                         sfc_loiter: float = 0.45,
                         ld_cruise: float = 17.0,
                         ld_loiter: float = 18.0,
                         sweep_LE_deg: float = 35.0,
                         CD0: float = 0.018) -> dict:
    """Run all standard trade studies and return a summary.

    Raymer Ch 19.5: A complete parametric survey sweeps the key design
    variables to understand their relative importance. This function
    runs AR, sweep, range, payload, SFC, L/D, and dead-weight trades,
    then ranks them by sensitivity.

    Args:
        mission: base mission profile.
        cruise_mach: cruise Mach number. Default 0.85.
        cruise_alt_m: cruise altitude (m). Default 10668.
        sfc_cruise: cruise SFC (1/hr). Default 0.55.
        sfc_loiter: loiter SFC (1/hr). Default 0.45.
        ld_cruise: cruise L/D. Default 17.0.
        ld_loiter: loiter L/D. Default 18.0.
        sweep_LE_deg: leading-edge sweep (degrees). Default 35.
        CD0: baseline CD0. Default 0.018.

    Returns:
        Dict with:
            trades: {name: TradeStudyResult}
            sensitivities: {name: float} -- %Delta_W0 / %Delta_parameter
            growth_factor: dead weight growth factor (Delta_W0/Delta_W_dead)
            ranking: list of (name, abs_sensitivity) sorted descending
    """
    trades = {}

    # Aspect ratio trade
    ar_result = aspect_ratio_trade(
        mission, sweep_LE_deg=sweep_LE_deg, sfc_cruise=sfc_cruise,
        sfc_loiter=sfc_loiter, cruise_mach=cruise_mach,
        cruise_alt_m=cruise_alt_m, CD0=CD0,
    )
    trades['aspect_ratio'] = ar_result

    # Sweep trade
    sw_result = sweep_trade(
        mission, AR=9.0, sfc_cruise=sfc_cruise, sfc_loiter=sfc_loiter,
        cruise_mach=cruise_mach, cruise_alt_m=cruise_alt_m, CD0=CD0,
    )
    trades['sweep'] = sw_result

    # Range trade (need a factory)
    def _range_factory(range_km):
        return MissionProfile.transport_default(
            range_km=range_km,
            passengers=mission.passengers,
            cruise_mach=cruise_mach,
            cruise_alt_m=cruise_alt_m,
        )

    range_result = range_trade(
        _range_factory, sfc_cruise=sfc_cruise, sfc_loiter=sfc_loiter,
        ld_cruise=ld_cruise, ld_loiter=ld_loiter,
    )
    trades['range'] = range_result

    # Payload trade
    pax_result = payload_trade(
        mission, sfc_cruise=sfc_cruise, sfc_loiter=sfc_loiter,
        ld_cruise=ld_cruise, ld_loiter=ld_loiter,
    )
    trades['payload'] = pax_result

    # SFC trade
    sfc_result = sfc_trade(
        mission, ld_cruise=ld_cruise, ld_loiter=ld_loiter,
    )
    trades['sfc'] = sfc_result

    # L/D trade
    ld_result = ld_trade(
        mission, sfc_cruise=sfc_cruise, sfc_loiter=sfc_loiter,
    )
    trades['ld'] = ld_result

    # Dead weight
    dw_result = dead_weight_sensitivity(
        mission, sfc_cruise=sfc_cruise, sfc_loiter=sfc_loiter,
        ld_cruise=ld_cruise, ld_loiter=ld_loiter,
    )
    trades['dead_weight'] = dw_result

    # Collect sensitivities
    sensitivities = {}
    for name, tr in trades.items():
        sensitivities[name] = tr.sensitivity()

    # Growth factor from dead weight study
    gf = dw_result.growth_factor()

    # Rank by absolute sensitivity
    ranking = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        'trades': trades,
        'sensitivities': sensitivities,
        'growth_factor': gf,
        'ranking': ranking,
    }
