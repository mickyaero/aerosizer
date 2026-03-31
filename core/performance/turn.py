"""
Turn Performance and Energy-Maneuverability Methods
Reference: Raymer Ch 17.4 (Turning Flight), Ch 17.6 (Energy-Maneuverability)
           "Aircraft Design: A Conceptual Approach", 6th Edition

Provides sustained and instantaneous turn performance, specific excess power (Ps),
energy height, corner speed, and plot data generators for turn-rate envelopes,
Ps vs Mach, Ps contours, and Ps vs turn-rate diagrams.
All inputs and outputs in SI units (m, kg, s, N, rad) unless noted in docstrings.
"""

import math

import numpy as np

from ..atmosphere import (
    density_at,
    dynamic_pressure,
    speed_of_sound_at,
    G0,
)
from ..aerodynamics.drag_polar import DragPolar


# ============================================================================ #
# LEVEL TURNING FLIGHT (Raymer 17.4)
# ============================================================================ #

def turn_rate(velocity_ms: float, load_factor: float) -> float:
    """Level turn rate.

    Raymer Eq 17.52:
        psi_dot = g * sqrt(n^2 - 1) / V   (rad/s)

    In a level, coordinated turn the centripetal acceleration is
    provided by the horizontal component of lift.  The vertical
    component balances weight, giving n = 1/cos(phi).

    Args:
        velocity_ms: true airspeed in m/s (must be > 0).
        load_factor: load factor n (dimensionless, must be >= 1).

    Returns:
        Turn rate in rad/s.  Returns 0.0 for n < 1 or V <= 0
        (no turn possible).
    """
    if load_factor < 1.0 or velocity_ms <= 0.0:
        return 0.0
    return G0 * math.sqrt(load_factor ** 2 - 1.0) / velocity_ms


def turn_radius(velocity_ms: float, load_factor: float) -> float:
    """Turn radius in a level, coordinated turn.

    Raymer Eq 17.79:
        R = V^2 / (g * sqrt(n^2 - 1))   (meters)

    Args:
        velocity_ms: true airspeed in m/s (must be > 0).
        load_factor: load factor n (must be > 1 for a finite radius).

    Returns:
        Turn radius in meters.  Returns float('inf') for n <= 1
        (straight flight) or V <= 0.
    """
    if load_factor <= 1.0 or velocity_ms <= 0.0:
        return float("inf")
    return velocity_ms ** 2 / (G0 * math.sqrt(load_factor ** 2 - 1.0))


def bank_angle(load_factor: float) -> float:
    """Bank angle for a level, coordinated turn.

    In a level turn, L*cos(phi) = W, so n = 1/cos(phi),
    hence phi = arccos(1/n).

    Args:
        load_factor: load factor n (must be >= 1).

    Returns:
        Bank angle phi in radians.  Returns 0.0 for n < 1.
    """
    if load_factor < 1.0:
        return 0.0
    # Clamp argument to [-1, 1] for safety at exactly n = 1
    arg = min(1.0, 1.0 / load_factor)
    return math.acos(arg)


# ============================================================================ #
# SUSTAINED TURN (Raymer 17.4.2)
# ============================================================================ #

def sustained_load_factor(tw_ratio: float, ld_ratio: float) -> float:
    """Maximum sustained load factor (simplified).

    Raymer Eq 17.53:
        n = (T/W) * (L/D)

    This is the load factor at which thrust exactly equals drag in a
    level turn, so the aircraft can sustain the turn indefinitely
    without losing energy.

    Args:
        tw_ratio: thrust-to-weight ratio T/W (dimensionless).
        ld_ratio: lift-to-drag ratio L/D (dimensionless).

    Returns:
        Sustained load factor (dimensionless).  Returns 0.0 if either
        input is non-positive.
    """
    if tw_ratio <= 0.0 or ld_ratio <= 0.0:
        return 0.0
    return tw_ratio * ld_ratio


def sustained_load_factor_detailed(W_kg: float, S_m2: float, T_N: float,
                                   polar: DragPolar, altitude_m: float,
                                   velocity_ms: float) -> float:
    """Sustained load factor from detailed aerodynamics.

    Raymer Eq 17.54:
        n = sqrt( (q / (K * W/S)) * (T/W - q*CD0 / (W/S)) )

    where q = 0.5 * rho * V^2 and W/S is wing loading in N/m^2.

    The aircraft can sustain a turn at load factor n when thrust
    exactly balances drag.  The expression is derived by setting
    T = D and solving for n.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance (clean configuration).
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Sustained load factor (dimensionless).  Returns 0.0 if the
        aircraft cannot sustain even 1-g level flight at this condition.
    """
    if velocity_ms <= 0.0 or W_kg <= 0.0 or S_m2 <= 0.0:
        return 0.0

    W = W_kg * G0          # weight in Newtons
    WS = W / S_m2          # wing loading N/m^2
    q = dynamic_pressure(velocity_ms, altitude_m)

    # n^2 = (q / (K * WS)) * (T/W - q * CD0 / WS)
    term = (q / (polar.K * WS)) * (T_N / W - q * polar.CD0 / WS)

    if term <= 0.0:
        return 0.0
    return math.sqrt(term)


def sustained_turn_rate(W_kg: float, S_m2: float, T_N: float,
                        polar: DragPolar, altitude_m: float,
                        velocity_ms: float) -> float:
    """Sustained turn rate at a given flight condition.

    Combines Raymer Eq 17.54 (sustained n) with Eq 17.52 (turn rate):
        psi_dot = g * sqrt(n^2 - 1) / V

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Sustained turn rate in rad/s.  Returns 0.0 if the aircraft
        cannot sustain a turn at this condition.
    """
    n = sustained_load_factor_detailed(W_kg, S_m2, T_N, polar,
                                       altitude_m, velocity_ms)
    return turn_rate(velocity_ms, n)


def max_sustained_load_factor_speed(W_kg: float, S_m2: float, T_N: float,
                                    polar: DragPolar,
                                    altitude_m: float) -> tuple:
    """Find speed for maximum sustained turn load factor.

    Raymer Eq 17.55: Maximum sustained n occurs when the aircraft
    flies at CL for maximum L/D, where CD0 = K*CL^2.  At this
    condition:
        n_max = (T/W) * (L/D)_max
        V_best = sqrt(2*n*W / (rho*S*CL_maxLD))

    Since n depends on V and V depends on n, we use the closed-form:
        V_best = sqrt( (2*W / (rho*S)) * sqrt(K/CD0) * n )
    But the simplest approach is:
        n_max = (T/W) * (L/D)_max                       (Eq 17.53)
        V_best = sqrt(2 * n_max * W / (rho * S * CL*))  (from L = nW)

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Tuple (n_max, V_best_ms) where n_max is the maximum sustained
        load factor and V_best_ms is the corresponding speed in m/s.
        Returns (0.0, 0.0) if the aircraft has insufficient thrust.
    """
    W = W_kg * G0
    rho = density_at(altitude_m)

    ld_max = polar.max_ld()
    tw = T_N / W
    n_max = tw * ld_max

    if n_max < 1.0:
        return (0.0, 0.0)

    # CL at max L/D
    CL_star = polar.cl_for_max_ld()

    # At the turning condition: L = n*W = q*S*CL_star
    # => q = n*W / (S*CL_star)
    # => 0.5*rho*V^2 = n*W / (S*CL_star)
    # => V = sqrt(2*n*W / (rho*S*CL_star))
    V_best = math.sqrt(2.0 * n_max * W / (rho * S_m2 * CL_star))

    return (n_max, V_best)


# ============================================================================ #
# INSTANTANEOUS TURN (Raymer 17.4.1)
# ============================================================================ #

def instantaneous_load_factor(W_kg: float, S_m2: float, CLmax: float,
                              altitude_m: float, velocity_ms: float) -> float:
    """Instantaneous load factor limited by CLmax.

    Raymer Section 17.4.1:
        n = q * CLmax / (W/S)

    This is the aerodynamic limit -- the maximum n the wing can
    generate at a given speed before stalling.  The actual n is
    further limited by the structural load limit.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        CLmax: maximum lift coefficient (clean or maneuvering).
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Instantaneous load factor (dimensionless).  Returns 0.0
        for invalid inputs.
    """
    if velocity_ms <= 0.0 or W_kg <= 0.0 or S_m2 <= 0.0 or CLmax <= 0.0:
        return 0.0

    W = W_kg * G0
    q = dynamic_pressure(velocity_ms, altitude_m)
    return q * S_m2 * CLmax / W


def corner_speed(W_kg: float, S_m2: float, CLmax: float, n_limit: float,
                 altitude_m: float = 0.0) -> float:
    """Corner speed -- intersection of stall and structural limits.

    Raymer p.653:
        V_corner = sqrt(2 * n_limit * W / (rho * S * CLmax))

    The corner speed is the minimum speed at which the aircraft can
    pull the structural load limit.  Below this speed the turn is
    stall-limited; above it the turn is structurally limited.  This
    is the speed for the tightest, fastest instantaneous turn.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        CLmax: maximum lift coefficient.
        n_limit: structural load factor limit (positive g).
        altitude_m: altitude in meters (geometric). Default sea level.

    Returns:
        Corner speed in m/s.

    Raises:
        ValueError: if any input is non-positive.
    """
    if W_kg <= 0.0 or S_m2 <= 0.0 or CLmax <= 0.0 or n_limit <= 0.0:
        raise ValueError("All inputs must be positive.")

    W = W_kg * G0
    rho = density_at(altitude_m)
    return math.sqrt(2.0 * n_limit * W / (rho * S_m2 * CLmax))


def instantaneous_turn_rate(W_kg: float, S_m2: float, CLmax: float,
                            n_limit: float, altitude_m: float,
                            velocity_ms: float) -> float:
    """Instantaneous turn rate (stall or structural limited).

    At any speed the instantaneous load factor is the lesser of:
        - Aerodynamic limit: n_aero = q*CLmax / (W/S)    (stall)
        - Structural limit:  n_struct = n_limit

    The turn rate then follows from Raymer Eq 17.52:
        psi_dot = g * sqrt(n^2 - 1) / V

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        CLmax: maximum lift coefficient.
        n_limit: structural load factor limit.
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Instantaneous turn rate in rad/s.  Returns 0.0 if the
        aircraft cannot turn at this condition.
    """
    n_aero = instantaneous_load_factor(W_kg, S_m2, CLmax,
                                       altitude_m, velocity_ms)
    n = min(n_aero, n_limit)
    return turn_rate(velocity_ms, n)


# ============================================================================ #
# SPECIFIC EXCESS POWER (Raymer 17.6)
# ============================================================================ #

def specific_excess_power(W_kg: float, S_m2: float, T_N: float,
                          polar: DragPolar, altitude_m: float,
                          velocity_ms: float,
                          load_factor: float = 1.0) -> float:
    """Specific excess power Ps.

    Raymer Eq 17.89:
        Ps = V * [T/W - q*CD0/(W/S) - K*n^2*(W/S)/q]

    Ps represents the rate at which the aircraft can increase its
    total energy (altitude + kinetic).  At Ps = 0 the aircraft is
    at the edge of its performance envelope.  Positive Ps means
    excess energy available for climbing, accelerating, or
    sustaining a higher-g turn.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.
        load_factor: load factor n. Default 1.0 (level flight).

    Returns:
        Specific excess power Ps in m/s.  Equivalent to rate of
        climb when n = 1 and flying at the speed for max ROC.
    """
    if velocity_ms <= 0.0 or W_kg <= 0.0 or S_m2 <= 0.0:
        return 0.0

    W = W_kg * G0
    WS = W / S_m2
    q = dynamic_pressure(velocity_ms, altitude_m)

    # Ps = V * [T/W - q*CD0/WS - K*n^2*WS/q]
    ps = velocity_ms * (T_N / W - q * polar.CD0 / WS
                        - polar.K * load_factor ** 2 * WS / q)
    return ps


def energy_height(altitude_m: float, velocity_ms: float) -> float:
    """Specific energy (energy height).

    Raymer Eq 17.85:
        he = h + V^2 / (2*g)

    The energy height represents the total mechanical energy per
    unit weight.  An aircraft at high speed and low altitude has
    the same energy height as one at low speed and high altitude.

    Args:
        altitude_m: geometric altitude in meters.
        velocity_ms: true airspeed in m/s.

    Returns:
        Energy height in meters.
    """
    return altitude_m + velocity_ms ** 2 / (2.0 * G0)


# ============================================================================ #
# PLOT DATA GENERATORS
# ============================================================================ #

def generate_turn_rate_envelope(W_kg: float, S_m2: float, T_N: float,
                                polar: DragPolar, altitude_m: float,
                                CLmax: float = 1.6, n_limit: float = 2.5,
                                v_range: np.ndarray = None) -> dict:
    """Generate turn rate vs velocity data for plotting.

    Produces both the sustained turn-rate curve (thrust-limited) and
    the instantaneous turn-rate curve (stall/structural-limited) over
    a range of velocities.  The envelope between these curves defines
    the aircraft's turn performance.

    Raymer Fig 17.8: Turn-rate envelope diagram.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        CLmax: maximum lift coefficient. Default 1.6.
        n_limit: structural load factor limit. Default 2.5.
        v_range: optional velocity array (m/s).  If None, auto-generates
            from 1.05 * Vs to 1.5 * V_corner with 150 points.

    Returns:
        dict with:
            V: velocity array (m/s)
            sustained_turn_rate: sustained turn rate (deg/s)
            instantaneous_turn_rate: instantaneous turn rate (deg/s)
            sustained_load_factor: sustained n array
            instantaneous_load_factor: instantaneous n array (capped at n_limit)
            corner_speed: V_corner (m/s)
            max_sustained_rate: peak sustained turn rate (deg/s)
            max_sustained_speed: speed at peak sustained rate (m/s)
    """
    RAD2DEG = 180.0 / np.pi
    W = W_kg * G0
    rho = density_at(altitude_m)

    # Stall speed in 1-g level flight
    V_stall = math.sqrt(2.0 * W / (rho * S_m2 * CLmax))

    # Corner speed
    V_corner = math.sqrt(2.0 * n_limit * W / (rho * S_m2 * CLmax))

    if v_range is None:
        V_low = V_stall * 1.05
        V_high = V_corner * 1.5
        # Ensure sensible range
        if V_high <= V_low:
            V_high = V_low * 2.0
        v_range = np.linspace(V_low, V_high, 150)

    v_arr = np.asarray(v_range, dtype=float)
    n_pts = len(v_arr)

    sust_rate = np.zeros(n_pts)
    inst_rate = np.zeros(n_pts)
    sust_n = np.zeros(n_pts)
    inst_n = np.zeros(n_pts)

    for i, V in enumerate(v_arr):
        if V <= 0.0:
            continue

        # Aerodynamic (stall) limit on n at this speed
        n_aero = instantaneous_load_factor(W_kg, S_m2, CLmax, altitude_m, V)

        # Sustained: thrust-limited, capped by structural and stall limits
        n_s_raw = sustained_load_factor_detailed(W_kg, S_m2, T_N, polar,
                                                 altitude_m, V)
        n_s = min(n_s_raw, n_limit, n_aero)
        sust_n[i] = n_s
        sust_rate[i] = turn_rate(V, n_s) * RAD2DEG

        # Instantaneous: min of aero limit and structural limit
        n_inst = min(n_aero, n_limit)
        inst_n[i] = n_inst
        inst_rate[i] = turn_rate(V, n_inst) * RAD2DEG

    # Find peak sustained turn rate
    if np.any(sust_rate > 0):
        idx_max = np.argmax(sust_rate)
        max_sust_rate = float(sust_rate[idx_max])
        max_sust_speed = float(v_arr[idx_max])
    else:
        max_sust_rate = 0.0
        max_sust_speed = 0.0

    return {
        "V": v_arr,
        "sustained_turn_rate": sust_rate,
        "instantaneous_turn_rate": inst_rate,
        "sustained_load_factor": sust_n,
        "instantaneous_load_factor": inst_n,
        "corner_speed": V_corner,
        "max_sustained_rate": max_sust_rate,
        "max_sustained_speed": max_sust_speed,
    }


def generate_ps_plot(W_kg: float, S_m2: float, T_available_func,
                     polar: DragPolar, altitude_m: float,
                     mach_range: np.ndarray = None,
                     load_factors: list = None) -> dict:
    """Generate Ps vs Mach number for different load factors.

    Raymer Fig 17.9: Shows how Ps decreases with increasing load
    factor, and how the Mach range for positive Ps shrinks.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: either a callable(mach, altitude_m) returning
            thrust in Newtons, or a scalar float for constant thrust.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        mach_range: optional Mach number array.  If None, uses
            np.linspace(0.1, 1.0, 100).
        load_factors: list of load factor values to compute.
            Default [1, 3, 5, 7].

    Returns:
        dict with:
            mach: Mach number array
            ps: dict mapping load_factor (float) -> Ps array (m/s)
            altitude_m: altitude used (m)
    """
    if mach_range is None:
        mach_range = np.linspace(0.1, 1.0, 100)

    if load_factors is None:
        load_factors = [1, 3, 5, 7]

    mach_arr = np.asarray(mach_range, dtype=float)
    a_sound = speed_of_sound_at(altitude_m)

    # Determine if T_available_func is a callable or a scalar
    if callable(T_available_func):
        thrust_func = T_available_func
    else:
        # Treat as constant thrust
        _T_const = float(T_available_func)
        thrust_func = lambda m, h: _T_const

    ps_dict = {}
    for n in load_factors:
        ps_arr = np.zeros_like(mach_arr)
        for i, M in enumerate(mach_arr):
            V = M * a_sound
            if V <= 0.0:
                continue
            T = thrust_func(M, altitude_m)
            ps_arr[i] = specific_excess_power(W_kg, S_m2, T, polar,
                                              altitude_m, V,
                                              load_factor=float(n))
        ps_dict[float(n)] = ps_arr

    return {
        "mach": mach_arr,
        "ps": ps_dict,
        "altitude_m": altitude_m,
    }


def generate_ps_contours(W_kg: float, S_m2: float, T_available_func,
                         polar: DragPolar,
                         mach_range: np.ndarray = None,
                         alt_range: np.ndarray = None,
                         load_factor: float = 1.0) -> dict:
    """Generate Ps contour data on a Mach-altitude chart.

    Raymer Fig 17.11: The Ps = 0 contour defines the flight envelope.
    Interior contours show lines of constant Ps.

    For each (Mach, altitude) grid point, computes Ps, producing a
    2-D array suitable for Plotly contour plots.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: either a callable(mach, altitude_m) returning
            thrust in Newtons, or a scalar float for constant thrust.
        polar: DragPolar instance.
        mach_range: optional 1-D Mach array.  Default np.linspace(0.1, 1.2, 80).
        alt_range: optional 1-D altitude array (m).  Default
            np.linspace(0, 15000, 60).
        load_factor: load factor n for the contour. Default 1.0.

    Returns:
        dict with:
            mach: 1-D Mach array
            altitude: 1-D altitude array (m)
            ps_grid: 2-D array [alt x mach] of Ps values (m/s)
            load_factor: n used
    """
    if mach_range is None:
        mach_range = np.linspace(0.1, 1.2, 80)
    if alt_range is None:
        alt_range = np.linspace(0, 15000, 60)

    mach_arr = np.asarray(mach_range, dtype=float)
    alt_arr = np.asarray(alt_range, dtype=float)

    if callable(T_available_func):
        thrust_func = T_available_func
    else:
        _T_const = float(T_available_func)
        thrust_func = lambda m, h: _T_const

    n_alt = len(alt_arr)
    n_mach = len(mach_arr)
    ps_grid = np.zeros((n_alt, n_mach))

    for j, h in enumerate(alt_arr):
        a_sound = speed_of_sound_at(h)
        for i, M in enumerate(mach_arr):
            V = M * a_sound
            if V <= 0.0:
                continue
            T = thrust_func(M, h)
            ps_grid[j, i] = specific_excess_power(
                W_kg, S_m2, T, polar, h, V, load_factor=load_factor
            )

    return {
        "mach": mach_arr,
        "altitude": alt_arr,
        "ps_grid": ps_grid,
        "load_factor": load_factor,
    }


def generate_ps_vs_turn_rate(W_kg: float, S_m2: float, T_N: float,
                             polar: DragPolar, altitude_m: float,
                             mach: float, CLmax: float = 1.6,
                             n_limit: float = 2.5) -> dict:
    """Generate Ps vs turn rate diagram.

    Raymer Fig 17.10: At a fixed altitude and Mach number, vary the
    load factor from 1.0 to n_limit.  For each n, compute the
    corresponding Ps and turn rate.  The point where Ps = 0 gives the
    maximum sustained turn rate; the point where n = n_limit or
    n = n_stall gives the maximum instantaneous turn rate.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        mach: Mach number at which to evaluate.
        CLmax: maximum lift coefficient. Default 1.6.
        n_limit: structural load factor limit. Default 2.5.

    Returns:
        dict with:
            turn_rate_degs: array of turn rates (deg/s)
            ps: array of Ps values (m/s)
            stall_limit_rate: max turn rate from CLmax (deg/s)
            structural_limit_rate: max turn rate from n_limit (deg/s)
    """
    RAD2DEG = 180.0 / np.pi
    a_sound = speed_of_sound_at(altitude_m)
    V = mach * a_sound

    if V <= 0.0:
        return {
            "turn_rate_degs": np.array([0.0]),
            "ps": np.array([0.0]),
            "stall_limit_rate": 0.0,
            "structural_limit_rate": 0.0,
        }

    # Aerodynamic (stall) limit on n at this speed
    n_stall = instantaneous_load_factor(W_kg, S_m2, CLmax, altitude_m, V)

    # The maximum possible n is the lesser of stall and structural limits
    n_max = min(n_stall, n_limit)

    # Ensure we have at least n = 1.0 (level flight)
    if n_max < 1.0:
        n_max = 1.0

    # Generate load factor array from 1 to n_max
    n_points = 100
    n_arr = np.linspace(1.0, n_max, n_points)

    tr_arr = np.zeros(n_points)   # turn rate deg/s
    ps_arr = np.zeros(n_points)   # Ps m/s

    for i, n in enumerate(n_arr):
        tr_arr[i] = turn_rate(V, n) * RAD2DEG
        ps_arr[i] = specific_excess_power(W_kg, S_m2, T_N, polar,
                                          altitude_m, V, load_factor=n)

    # Stall-limited turn rate
    stall_tr = turn_rate(V, min(n_stall, n_limit)) * RAD2DEG

    # Structural-limited turn rate (if achievable before stall)
    if n_limit <= n_stall:
        struct_tr = turn_rate(V, n_limit) * RAD2DEG
    else:
        # Cannot reach structural limit before stalling
        struct_tr = stall_tr

    return {
        "turn_rate_degs": tr_arr,
        "ps": ps_arr,
        "stall_limit_rate": float(stall_tr),
        "structural_limit_rate": float(struct_tr),
    }
