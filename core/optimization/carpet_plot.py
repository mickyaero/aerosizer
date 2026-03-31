"""
Sizing Matrix and Carpet Plot Generation
Reference: Raymer Ch 19.4 (Classic Optimization -- Sizing Matrix and Carpet Plots)

Generates parametric sizing data by varying T/W and W/S, computing
sized takeoff weight for each combination, and overlaying performance
constraint lines to find the optimal design point.

"The sizing matrix plot, sometimes called the carpet plot, is the
granddaddy of all trade studies." -- Raymer, p. 726

Process (Raymer Fig 19.5-19.6):
  1. Select a range of T/W and W/S values around a baseline.
  2. For each (T/W, W/S) pair, estimate L/D from aerodynamic model,
     then size the aircraft using the Breguet-based fuel fraction method.
  3. Record W0, We, Wf for each design point.
  4. Overlay performance constraint boundaries (BFL, landing, Ps, ceiling).
  5. The minimum-weight design satisfying ALL constraints is the optimum.

All internal calculations use SI units (kg, m, s, Pa).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable
import math
import copy

from ..atmosphere import true_airspeed, density_at, dynamic_pressure, G0, ft_to_m
from ..mission import MissionProfile, AircraftType
from ..sizing.initial_sizing import size_aircraft, SizingResult
from ..aerodynamics.drag_polar import DragPolar
from ..aerodynamics.induced_drag import oswald_efficiency_swept, induced_drag_factor


@dataclass
class CarpetPlotConfig:
    """Configuration for carpet plot generation.

    Attributes:
        mission: base mission profile for sizing.
        tw_baseline: baseline T/W ratio (dimensionless).
        ws_baseline: baseline W/S (Pa).
        tw_variations: explicit array of T/W values to test. If None,
            auto-generated from baseline +/- variation_pct.
        ws_variations: explicit array of W/S values to test (Pa). If None,
            auto-generated from baseline +/- variation_pct.
        n_tw: number of T/W grid points if auto-generating. Default 5.
        n_ws: number of W/S grid points if auto-generating. Default 5.
        variation_pct: fractional variation from baseline. Default 0.20 = +/-20%.

        bfl_max_m: maximum balanced field length (m).
        landing_dist_max_m: maximum landing distance (m).
        ps_requirement: tuple of (altitude_m, mach, load_factor, ps_min_ms).
            E.g. (9144.0, 0.9, 5.0, 0.0) for Ps >= 0 at M0.9, 30 kft, 5 g.
        ceiling_min_m: minimum service ceiling (m). Default 12192 m (40 000 ft).
        cruise_mach: cruise Mach number for aerodynamic estimation.
        cruise_alt_m: cruise altitude (m) for aerodynamic estimation.

        sfc_cruise: thrust-specific fuel consumption in 1/hr for cruise.
        sfc_loiter: thrust-specific fuel consumption in 1/hr for loiter.
        AR: wing aspect ratio (for drag polar estimation).
        sweep_LE_deg: leading-edge sweep angle (degrees).
        n_engines: number of engines (for BFL calculation). Default 2.
        CLmax_takeoff: max CL in takeoff configuration. Default 2.0.
        CLmax_landing: max CL in landing configuration. Default 2.4.
        CD0_base: baseline zero-lift drag coefficient. Default 0.018.
        landing_weight_fraction: W_landing / W0. Default 0.85.
    """
    mission: MissionProfile = None
    tw_baseline: float = 0.30
    ws_baseline: float = 6000.0  # Pa
    tw_variations: np.ndarray = None
    ws_variations: np.ndarray = None
    n_tw: int = 5
    n_ws: int = 5
    variation_pct: float = 0.20

    bfl_max_m: float = 2500.0
    landing_dist_max_m: float = 1800.0
    ps_requirement: tuple = None  # (alt_m, mach, n, ps_min_ms)
    ceiling_min_m: float = 12192.0  # 40,000 ft
    cruise_mach: float = 0.85
    cruise_alt_m: float = 10668.0

    sfc_cruise: float = 0.55
    sfc_loiter: float = 0.45
    AR: float = 9.0
    sweep_LE_deg: float = 35.0
    n_engines: int = 2
    CLmax_takeoff: float = 2.0
    CLmax_landing: float = 2.4
    CD0_base: float = 0.018
    landing_weight_fraction: float = 0.85


@dataclass
class CarpetPlotResult:
    """Results from carpet plot sizing matrix.

    Attributes:
        tw_values: 1-D array of T/W values used.
        ws_values: 1-D array of W/S values (Pa) used.
        W0_matrix: sized TOGW (kg), shape (n_tw, n_ws). NaN for infeasible.
        We_matrix: empty weight (kg), shape (n_tw, n_ws).
        Wf_matrix: fuel weight (kg), shape (n_tw, n_ws).
        ld_matrix: cruise L/D at each design point.
        constraints: dict of constraint boundary data. Keys are constraint
            names, values are dicts with 'ws' and 'tw' arrays.
        tw_optimal: optimal T/W at minimum feasible W0.
        ws_optimal: optimal W/S (Pa) at minimum feasible W0.
        W0_optimal: minimum feasible TOGW (kg).
        feasible_mask: boolean matrix, True where design meets all constraints.
        carpet_lines_tw: list of dicts for constant-T/W carpet lines.
            Each dict has 'tw', 'ws_values', 'W0_values'.
        carpet_lines_ws: list of dicts for constant-W/S carpet lines.
            Each dict has 'ws', 'tw_values', 'W0_values'.
    """
    tw_values: np.ndarray = None
    ws_values: np.ndarray = None

    W0_matrix: np.ndarray = None
    We_matrix: np.ndarray = None
    Wf_matrix: np.ndarray = None

    ld_matrix: np.ndarray = None

    constraints: dict = field(default_factory=dict)

    tw_optimal: float = 0.0
    ws_optimal: float = 0.0
    W0_optimal: float = 0.0
    feasible_mask: np.ndarray = None

    carpet_lines_tw: list = field(default_factory=list)
    carpet_lines_ws: list = field(default_factory=list)


def estimate_ld_for_design_point(tw: float, ws: float, AR: float,
                                  sweep_LE_deg: float, cruise_mach: float,
                                  cruise_alt_m: float,
                                  CD0: float = 0.018) -> float:
    """Estimate cruise L/D for a given design point.

    Raymer Ch 12 and Ch 19.4: For parametric trade studies, the drag polar
    is estimated from the aspect ratio and sweep using:
        e  = f(AR, sweep_LE)           -- Raymer Eq 12.50
        K  = 1 / (pi * AR * e)        -- Raymer Eq 12.48
        CL = (W/S) / q_cruise         -- from level-flight condition
        CD = CD0 + K * CL^2           -- Raymer Eq 12.1
        L/D = CL / CD

    The CD0 can be refined based on wetted-area correlations but is
    held at a typical transport value for trade studies.

    Args:
        tw: thrust-to-weight ratio (not used directly for L/D, included
            for interface consistency).
        ws: wing loading W/S in Pa.
        AR: wing aspect ratio.
        sweep_LE_deg: leading-edge sweep in degrees.
        cruise_mach: cruise Mach number.
        cruise_alt_m: cruise altitude in meters.
        CD0: zero-lift drag coefficient. Default 0.018.

    Returns:
        Estimated cruise L/D. Returns 0.0 if computation is degenerate.
    """
    sweep_LE_rad = math.radians(sweep_LE_deg)
    e = oswald_efficiency_swept(AR, sweep_LE_rad)

    # Guard against non-physical Oswald efficiency
    if e <= 0.0:
        e = 0.1  # floor to prevent division by zero

    K = induced_drag_factor(AR, e)

    # Cruise conditions
    rho = density_at(cruise_alt_m)
    V = true_airspeed(cruise_mach, cruise_alt_m)
    q = 0.5 * rho * V ** 2

    if q <= 0.0:
        return 0.0

    # CL from level flight: W = L = q * S * CL  =>  CL = (W/S) / q
    CL_cruise = ws / q

    if CL_cruise <= 0.0:
        return 0.0

    # Drag polar
    CD = CD0 + K * CL_cruise ** 2
    if CD <= 0.0:
        return 0.0

    LD = CL_cruise / CD
    return LD


def _compute_bfl_constraint(config: CarpetPlotConfig,
                             ws_range: np.ndarray) -> dict:
    """Compute the BFL constraint line: for each W/S, find the minimum T/W
    that satisfies the balanced field length requirement.

    Uses Raymer Eq 17.113 (empirical BFL correlation) solved for T/W.

    Raymer Eq 17.113:
        BFL = (0.863 / (1 + 2.3*G)) *
              (W_S / (rho*g*CL_climb) + h_obs) *
              (1 / (T_W - U) + 2.7) +
              655 / sqrt(sigma)

    Solved for T/W:
        T/W = U + 1 / ((BFL_max - 655/sqrt(sigma)) *
              (1+2.3*G) / (0.863 * (WS/(rho*g*CL_climb) + h_obs)) - 2.7)

    Args:
        config: carpet plot configuration.
        ws_range: W/S values in Pa.

    Returns:
        Dict with 'ws' and 'tw' arrays representing the constraint boundary.
    """
    from ..atmosphere import RHO0

    rho = density_at(0.0)  # sea level for takeoff
    sigma = rho / RHO0

    # FAR 25 minimum climb gradient (OEI, 2nd segment)
    gamma_min_map = {2: 0.024, 3: 0.027, 4: 0.030}
    gamma_min = gamma_min_map.get(config.n_engines, 0.024)

    # At V2, CL_climb = CLmax_TO / 1.44 (since V2 = 1.2 * Vs)
    CL_climb = config.CLmax_takeoff / 1.44

    # Estimate OEI climb gradient (depends on T/W -- we iterate)
    # Use simplified approach: assume G = 0.024 (2-engine minimum)
    G = gamma_min

    # Ground friction term
    U = 0.01 * config.CLmax_takeoff + 0.02

    h_obs = 10.7  # FAR 25: 35 ft obstacle

    BFL_max = config.bfl_max_m

    # Term from the BFL equation rearranged
    sigma_term = 655.0 / math.sqrt(sigma)
    remainder = BFL_max - sigma_term

    tw_values = np.full_like(ws_range, np.nan, dtype=float)

    for i, ws in enumerate(ws_range):
        # W/S in Pa; need to convert to N/m^2 (already in Pa = N/m^2)
        WS_term = ws / (rho * G0 * CL_climb) + h_obs
        factor = 0.863 / (1.0 + 2.3 * G)

        # BFL = factor * WS_term * (1/(TW - U) + 2.7) + sigma_term
        # remainder = factor * WS_term * (1/(TW-U) + 2.7)
        # remainder / (factor * WS_term) = 1/(TW-U) + 2.7
        # 1/(TW-U) = remainder / (factor * WS_term) - 2.7
        denom = factor * WS_term
        if denom <= 0.0 or remainder <= 0.0:
            continue

        inv_tw_u = remainder / denom - 2.7

        if inv_tw_u <= 0.0:
            # Constraint cannot be met at any T/W
            tw_values[i] = np.inf
            continue

        tw = U + 1.0 / inv_tw_u
        if tw > 0.0:
            tw_values[i] = tw

    # Remove NaN entries
    valid = np.isfinite(tw_values)
    return {'ws': ws_range[valid], 'tw': tw_values[valid]}


def _compute_landing_constraint(config: CarpetPlotConfig) -> float:
    """Compute the maximum W/S from landing distance constraint.

    Raymer Ch 5 / Ch 17: Landing field length determines the maximum
    approach speed, which constrains W/S via the stall-speed relationship.

    Using Raymer Fig 5.5 approximation:
        W/S_landing (psf) = S_L (ft) * sigma * CLmax_L / 80

    Returns:
        Maximum takeoff W/S in Pa.
    """
    from ..atmosphere import m_to_ft, RHO0

    S_L_ft = m_to_ft(config.landing_dist_max_m)
    sigma = 1.0  # sea-level landing (conservative)

    ws_landing_psf = S_L_ft * sigma * config.CLmax_landing / 80.0

    # Convert psf to Pa: 1 psf = 47.8803 Pa
    ws_landing_pa = ws_landing_psf / 0.020885434

    # Convert from landing to takeoff W/S
    ws_takeoff_pa = ws_landing_pa / config.landing_weight_fraction

    return ws_takeoff_pa


def _compute_ceiling_constraint(config: CarpetPlotConfig,
                                 ws_range: np.ndarray) -> dict:
    """Compute the service ceiling constraint.

    At the service ceiling, rate of climb = 100 ft/min = 0.508 m/s.
    Raymer Eq 5.8:
        T/W_SL = [q*CD0/(beta*W/S) + K*beta*(W/S)/q + ROC/V] / alpha

    Args:
        config: carpet plot configuration.
        ws_range: W/S values in Pa.

    Returns:
        Dict with 'ws' and 'tw' arrays.
    """
    from ..atmosphere import atmosphere as _atm

    ROC = 0.508  # m/s (100 ft/min)

    atm = _atm(config.ceiling_min_m)
    rho_ceil = atm['rho']
    sigma_ceil = atm['sigma']

    sweep_LE_rad = math.radians(config.sweep_LE_deg)
    e = oswald_efficiency_swept(config.AR, sweep_LE_rad)
    if e <= 0.0:
        e = 0.1
    K = induced_drag_factor(config.AR, e)
    CD0 = config.CD0_base

    # Thrust lapse at ceiling: alpha ~ sigma^0.6
    alpha = sigma_ceil ** 0.6
    beta = 0.95  # weight fraction at top of climb

    # CL for best ROC ~ CL at min drag
    CL_star = math.sqrt(CD0 / K)

    tw_values = np.zeros_like(ws_range, dtype=float)

    for i, ws in enumerate(ws_range):
        # V from CL: V = sqrt(2 * beta * W/S / (rho * CL))
        V = math.sqrt(2.0 * beta * ws / (rho_ceil * CL_star))
        if V <= 0.0:
            tw_values[i] = np.inf
            continue

        q = 0.5 * rho_ceil * V ** 2

        tw = (q * CD0 / (beta * ws)
              + K * beta * ws / q
              + ROC / V) / alpha
        tw_values[i] = tw

    return {'ws': ws_range, 'tw': tw_values}


def _compute_cruise_constraint(config: CarpetPlotConfig,
                                ws_range: np.ndarray) -> dict:
    """Compute the cruise speed constraint.

    At cruise, thrust = drag. Raymer Eq 5.6:
        T_SL/W_TO = [q*CD0/(beta*W/S) + K*beta*(W/S)/q] / alpha

    Args:
        config: carpet plot configuration.
        ws_range: W/S values in Pa.

    Returns:
        Dict with 'ws' and 'tw' arrays.
    """
    rho = density_at(config.cruise_alt_m)
    V = true_airspeed(config.cruise_mach, config.cruise_alt_m)
    q = 0.5 * rho * V ** 2

    sweep_LE_rad = math.radians(config.sweep_LE_deg)
    e = oswald_efficiency_swept(config.AR, sweep_LE_rad)
    if e <= 0.0:
        e = 0.1
    K = induced_drag_factor(config.AR, e)
    CD0 = config.CD0_base

    # Thrust lapse at cruise
    from ..atmosphere import atmosphere as _atm
    atm = _atm(config.cruise_alt_m)
    alpha = atm['sigma'] ** 0.6
    beta = 0.95  # cruise weight fraction

    tw_values = (q * CD0 / (beta * ws_range)
                 + K * beta * ws_range / q) / alpha

    return {'ws': ws_range, 'tw': tw_values}


def _compute_ps_constraint(config: CarpetPlotConfig,
                            ws_range: np.ndarray) -> dict:
    """Compute the specific excess power constraint.

    Raymer Eq 17.89:
        Ps = V * [T/W - q*CD0/(W/S) - K*n^2*(W/S)/q]

    Setting Ps >= Ps_min and solving for T/W:
        T/W >= Ps_min/V + q*CD0/(W/S) + K*n^2*(W/S)/q

    Adjusted to sea-level T/W via thrust lapse alpha and weight fraction beta.

    Args:
        config: carpet plot configuration.
        ws_range: W/S values in Pa.

    Returns:
        Dict with 'ws' and 'tw' arrays, or empty dict if no Ps requirement.
    """
    if config.ps_requirement is None:
        return {}

    alt_m, mach, n_load, ps_min = config.ps_requirement

    rho = density_at(alt_m)
    V = true_airspeed(mach, alt_m)
    q = 0.5 * rho * V ** 2

    sweep_LE_rad = math.radians(config.sweep_LE_deg)
    e = oswald_efficiency_swept(config.AR, sweep_LE_rad)
    if e <= 0.0:
        e = 0.1
    K = induced_drag_factor(config.AR, e)
    CD0 = config.CD0_base

    from ..atmosphere import atmosphere as _atm
    atm = _atm(alt_m)
    alpha = atm['sigma'] ** 0.6
    beta = 0.95

    # T/W at condition: T/(beta*W) >= Ps_min/V + q*CD0/(beta*W/S)
    #                                  + K*n^2*beta*(W/S)/q
    # Convert to sea-level T/W_TO:
    #   T_SL/W_TO = [Ps_min/V + q*CD0/(beta*W/S) + K*n^2*beta*(W/S)/q] / alpha
    tw_values = (ps_min / V
                 + q * CD0 / (beta * ws_range)
                 + K * n_load ** 2 * beta * ws_range / q) / alpha

    return {'ws': ws_range, 'tw': tw_values}


def generate_carpet_plot(config: CarpetPlotConfig) -> CarpetPlotResult:
    """Generate a complete carpet plot sizing matrix.

    Raymer Ch 19.4: Parametrically vary T/W and W/S, size the aircraft
    for each combination, evaluate performance constraints, and find
    the minimum-weight design that meets all requirements.

    Process:
      1. Generate T/W and W/S arrays (+/- variation_pct from baseline).
      2. For each (T/W, W/S) combination:
         a. Estimate L/D from AR, sweep, and design point.
         b. Size the aircraft using the mission profile.
         c. Record W0, We, Wf.
      3. Compute performance constraint boundary lines.
      4. Find the minimum W0 that satisfies all constraints.
      5. Format data for both sizing matrix and carpet plot visualisation.

    Args:
        config: CarpetPlotConfig with all parameters.

    Returns:
        CarpetPlotResult with full sizing matrix and constraint data.

    Raises:
        ValueError: if no mission profile is provided.
    """
    if config.mission is None:
        raise ValueError("CarpetPlotConfig.mission must be provided.")

    result = CarpetPlotResult()

    # ------------------------------------------------------------------
    # 1. Generate parameter arrays
    # ------------------------------------------------------------------
    if config.tw_variations is not None:
        tw_arr = np.asarray(config.tw_variations, dtype=float)
    else:
        tw_lo = config.tw_baseline * (1.0 - config.variation_pct)
        tw_hi = config.tw_baseline * (1.0 + config.variation_pct)
        tw_arr = np.linspace(tw_lo, tw_hi, config.n_tw)

    if config.ws_variations is not None:
        ws_arr = np.asarray(config.ws_variations, dtype=float)
    else:
        ws_lo = config.ws_baseline * (1.0 - config.variation_pct)
        ws_hi = config.ws_baseline * (1.0 + config.variation_pct)
        ws_arr = np.linspace(ws_lo, ws_hi, config.n_ws)

    result.tw_values = tw_arr
    result.ws_values = ws_arr

    n_tw = len(tw_arr)
    n_ws = len(ws_arr)

    W0_mat = np.full((n_tw, n_ws), np.nan)
    We_mat = np.full((n_tw, n_ws), np.nan)
    Wf_mat = np.full((n_tw, n_ws), np.nan)
    LD_mat = np.full((n_tw, n_ws), np.nan)

    # ------------------------------------------------------------------
    # 2. Size aircraft for each (T/W, W/S) combination
    # ------------------------------------------------------------------
    # The sizing loop uses the Breguet equation. T/W does not directly
    # enter the fuel fraction; it affects performance constraints.
    # W/S affects L/D via cruise CL = (W/S)/q.
    # We estimate L/D at each W/S, then size using that L/D.

    for j, ws in enumerate(ws_arr):
        # Estimate L/D for this W/S
        ld_cruise = estimate_ld_for_design_point(
            tw=config.tw_baseline,  # L/D is independent of T/W
            ws=ws,
            AR=config.AR,
            sweep_LE_deg=config.sweep_LE_deg,
            cruise_mach=config.cruise_mach,
            cruise_alt_m=config.cruise_alt_m,
            CD0=config.CD0_base,
        )

        if ld_cruise <= 0.0:
            # Degenerate -- skip this column
            continue

        # Loiter L/D: at lower speed, CL is higher, closer to max L/D.
        # Raymer: loiter L/D ~ L/D_max for jets.
        # Approximate max L/D for this AR / sweep / CD0.
        sweep_LE_rad = math.radians(config.sweep_LE_deg)
        e = oswald_efficiency_swept(config.AR, sweep_LE_rad)
        if e <= 0.0:
            e = 0.1
        K = induced_drag_factor(config.AR, e)
        ld_max = 1.0 / (2.0 * math.sqrt(config.CD0_base * K))
        ld_loiter = ld_max  # Raymer: loiter near max L/D for jets

        for i, tw in enumerate(tw_arr):
            LD_mat[i, j] = ld_cruise

            try:
                sizing = size_aircraft(
                    mission=config.mission,
                    ld_cruise=ld_cruise,
                    ld_loiter=ld_loiter,
                    sfc_cruise=config.sfc_cruise,
                    sfc_loiter=config.sfc_loiter,
                )
                W0_mat[i, j] = sizing.W0_kg
                We_mat[i, j] = sizing.We_kg
                Wf_mat[i, j] = sizing.Wf_kg
            except (ValueError, RuntimeError):
                # Infeasible design point -- leave as NaN
                pass

    result.W0_matrix = W0_mat
    result.We_matrix = We_mat
    result.Wf_matrix = Wf_mat
    result.ld_matrix = LD_mat

    # ------------------------------------------------------------------
    # 3. Compute constraint boundary lines
    # ------------------------------------------------------------------
    # Use a fine W/S grid for smooth constraint lines
    ws_fine = np.linspace(ws_arr[0] * 0.8, ws_arr[-1] * 1.2, 200)

    constraints = {}

    # BFL / takeoff constraint
    bfl_constraint = _compute_bfl_constraint(config, ws_fine)
    if len(bfl_constraint.get('ws', [])) > 0:
        constraints['takeoff_bfl'] = bfl_constraint

    # Landing constraint (vertical line = max W/S)
    ws_landing_max = _compute_landing_constraint(config)
    constraints['landing'] = {'ws_max': ws_landing_max}

    # Service ceiling constraint
    ceiling_constraint = _compute_ceiling_constraint(config, ws_fine)
    constraints['ceiling'] = ceiling_constraint

    # Cruise constraint
    cruise_constraint = _compute_cruise_constraint(config, ws_fine)
    constraints['cruise'] = cruise_constraint

    # Ps constraint (optional)
    ps_constraint = _compute_ps_constraint(config, ws_fine)
    if ps_constraint:
        constraints['specific_excess_power'] = ps_constraint

    result.constraints = constraints

    # ------------------------------------------------------------------
    # 4. Determine feasibility and find optimal design
    # ------------------------------------------------------------------
    feasible = np.ones((n_tw, n_ws), dtype=bool)

    # Mark NaN (infeasible sizing) as not feasible
    feasible[np.isnan(W0_mat)] = False

    for i, tw in enumerate(tw_arr):
        for j, ws in enumerate(ws_arr):
            if not feasible[i, j]:
                continue

            # Landing: W/S must be <= ws_landing_max
            if ws > ws_landing_max:
                feasible[i, j] = False
                continue

            # BFL: T/W must be >= BFL constraint at this W/S
            if 'takeoff_bfl' in constraints:
                bfl_data = constraints['takeoff_bfl']
                if len(bfl_data['ws']) > 0:
                    tw_bfl_req = np.interp(ws, bfl_data['ws'], bfl_data['tw'],
                                           left=np.nan, right=np.nan)
                    if np.isfinite(tw_bfl_req) and tw < tw_bfl_req:
                        feasible[i, j] = False
                        continue

            # Ceiling: T/W must be >= ceiling constraint at this W/S
            ceil_data = constraints['ceiling']
            tw_ceil_req = np.interp(ws, ceil_data['ws'], ceil_data['tw'],
                                    left=np.nan, right=np.nan)
            if np.isfinite(tw_ceil_req) and tw < tw_ceil_req:
                feasible[i, j] = False
                continue

            # Cruise: T/W must be >= cruise constraint at this W/S
            cr_data = constraints['cruise']
            tw_cr_req = np.interp(ws, cr_data['ws'], cr_data['tw'],
                                  left=np.nan, right=np.nan)
            if np.isfinite(tw_cr_req) and tw < tw_cr_req:
                feasible[i, j] = False
                continue

            # Ps constraint
            if 'specific_excess_power' in constraints:
                ps_data = constraints['specific_excess_power']
                tw_ps_req = np.interp(ws, ps_data['ws'], ps_data['tw'],
                                      left=np.nan, right=np.nan)
                if np.isfinite(tw_ps_req) and tw < tw_ps_req:
                    feasible[i, j] = False
                    continue

    result.feasible_mask = feasible

    # Find minimum W0 among feasible points
    tw_opt, ws_opt, W0_opt = find_optimal_design(result)
    result.tw_optimal = tw_opt
    result.ws_optimal = ws_opt
    result.W0_optimal = W0_opt

    # ------------------------------------------------------------------
    # 5. Build carpet plot line data (Raymer Fig 19.5-19.6)
    # ------------------------------------------------------------------
    # Lines of constant T/W: plot W0 vs W/S for each T/W
    for i, tw in enumerate(tw_arr):
        ws_vals = []
        w0_vals = []
        for j, ws in enumerate(ws_arr):
            if np.isfinite(W0_mat[i, j]):
                ws_vals.append(float(ws))
                w0_vals.append(float(W0_mat[i, j]))
        if ws_vals:
            result.carpet_lines_tw.append({
                'tw': float(tw),
                'ws_values': np.array(ws_vals),
                'W0_values': np.array(w0_vals),
            })

    # Lines of constant W/S: plot W0 vs T/W for each W/S
    for j, ws in enumerate(ws_arr):
        tw_vals = []
        w0_vals = []
        for i, tw in enumerate(tw_arr):
            if np.isfinite(W0_mat[i, j]):
                tw_vals.append(float(tw))
                w0_vals.append(float(W0_mat[i, j]))
        if tw_vals:
            result.carpet_lines_ws.append({
                'ws': float(ws),
                'tw_values': np.array(tw_vals),
                'W0_values': np.array(w0_vals),
            })

    return result


def find_optimal_design(result: CarpetPlotResult) -> tuple:
    """Find the minimum-weight design meeting all constraints.

    Raymer p. 721: "The desired solution is the lightest aircraft that
    meets all of the requirements."

    Scans the feasible region of the sizing matrix for the cell with
    minimum W0. If no feasible design exists, returns (NaN, NaN, NaN).

    Args:
        result: CarpetPlotResult with W0_matrix, feasible_mask, and
            the tw_values/ws_values arrays populated.

    Returns:
        Tuple (tw_opt, ws_opt, W0_opt). Returns (NaN, NaN, NaN)
        if no feasible design point exists.
    """
    if result.feasible_mask is None or not np.any(result.feasible_mask):
        return (float('nan'), float('nan'), float('nan'))

    # Mask infeasible points with inf for argmin
    W0_feasible = np.where(result.feasible_mask, result.W0_matrix, np.inf)

    # Replace NaN with inf as well
    W0_feasible = np.where(np.isfinite(W0_feasible), W0_feasible, np.inf)

    if np.all(np.isinf(W0_feasible)):
        return (float('nan'), float('nan'), float('nan'))

    idx = np.unravel_index(np.argmin(W0_feasible), W0_feasible.shape)
    i_opt, j_opt = idx

    tw_opt = float(result.tw_values[i_opt])
    ws_opt = float(result.ws_values[j_opt])
    W0_opt = float(result.W0_matrix[i_opt, j_opt])

    return (tw_opt, ws_opt, W0_opt)


def carpet_plot_summary(result: CarpetPlotResult) -> str:
    """Generate a human-readable summary of carpet plot results.

    Args:
        result: CarpetPlotResult from generate_carpet_plot.

    Returns:
        Formatted string summary.
    """
    lines = [
        "=" * 60,
        "  CARPET PLOT / SIZING MATRIX RESULTS",
        "  Raymer Ch 19.4",
        "=" * 60,
    ]

    if result.tw_values is not None:
        lines.append(f"  T/W range  : {result.tw_values[0]:.3f} - "
                     f"{result.tw_values[-1]:.3f}  ({len(result.tw_values)} pts)")
    if result.ws_values is not None:
        lines.append(f"  W/S range  : {result.ws_values[0]:.0f} - "
                     f"{result.ws_values[-1]:.0f} Pa  ({len(result.ws_values)} pts)")

    if result.W0_matrix is not None:
        valid = np.isfinite(result.W0_matrix)
        if np.any(valid):
            lines.append(f"  W0 range   : {np.nanmin(result.W0_matrix):,.0f} - "
                         f"{np.nanmax(result.W0_matrix):,.0f} kg")
        n_feasible = np.sum(result.feasible_mask) if result.feasible_mask is not None else 0
        n_total = result.W0_matrix.size
        lines.append(f"  Feasible   : {n_feasible} / {n_total} design points")

    lines.append("-" * 60)

    if np.isfinite(result.W0_optimal):
        lines.append(f"  OPTIMAL DESIGN POINT:")
        lines.append(f"    T/W      = {result.tw_optimal:.4f}")
        lines.append(f"    W/S      = {result.ws_optimal:.0f} Pa "
                     f"({result.ws_optimal / G0:.0f} kg/m^2)")
        lines.append(f"    W0       = {result.W0_optimal:,.0f} kg")
    else:
        lines.append("  NO FEASIBLE DESIGN FOUND")

    lines.append("")
    lines.append("  Constraints active:")
    for name, data in result.constraints.items():
        if name == 'landing':
            lines.append(f"    Landing W/S max = {data['ws_max']:.0f} Pa")
        else:
            lines.append(f"    {name}")

    lines.append("=" * 60)
    return "\n".join(lines)
