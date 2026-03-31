"""
Level Flight Performance Analysis
Reference: Raymer Ch 17.1-17.2, "Aircraft Design: A Conceptual Approach", 6th Edition

Provides stall speed, max speed, minimum drag speed, max range/endurance
speeds, thrust/power required, and curve generation for level flight analysis.
All inputs and outputs in SI units.
"""

import math

import numpy as np
from scipy.optimize import brentq

from ..atmosphere import density_at, speed_of_sound_at, G0
from ..aerodynamics.drag_polar import DragPolar


def stall_speed(W_kg: float, S_m2: float, CLmax: float,
                altitude_m: float = 0.0) -> float:
    """Stall speed in level flight.

    Raymer Eq 17.1:
        Vs = sqrt(2 * W / (rho * S * CLmax))

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        CLmax: maximum lift coefficient for the configuration.
        altitude_m: altitude in meters (geometric). Default sea level.

    Returns:
        Stall speed in m/s.

    Raises:
        ValueError: if CLmax <= 0 or S_m2 <= 0 or W_kg <= 0.
    """
    if W_kg <= 0:
        raise ValueError("W_kg must be positive.")
    if S_m2 <= 0:
        raise ValueError("S_m2 must be positive.")
    if CLmax <= 0:
        raise ValueError("CLmax must be positive.")

    rho = density_at(altitude_m)
    W = W_kg * G0  # weight in Newtons
    return math.sqrt(2.0 * W / (rho * S_m2 * CLmax))


def thrust_required(W_kg: float, S_m2: float, polar: DragPolar,
                    altitude_m: float, velocity_ms: float) -> float:
    """Thrust required for steady level flight at a given speed.

    Raymer Eq 17.9 (rearranged): In level flight L = W, so
        CL = W / (q * S)
        CD = CD0 + K * CL^2
        T_req = D = q * S * CD

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance (clean configuration).
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Thrust required in Newtons.

    Raises:
        ValueError: if velocity_ms <= 0.
    """
    if velocity_ms <= 0:
        raise ValueError("velocity_ms must be positive.")

    rho = density_at(altitude_m)
    W = W_kg * G0
    q = 0.5 * rho * velocity_ms ** 2
    CL = W / (q * S_m2)
    CD = polar.cd(CL)
    return q * S_m2 * CD


def power_required(W_kg: float, S_m2: float, polar: DragPolar,
                   altitude_m: float, velocity_ms: float) -> float:
    """Power required for steady level flight.

    Raymer Eq 17.25:
        P_req = T_req * V = D * V

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Power required in Watts.
    """
    T_req = thrust_required(W_kg, S_m2, polar, altitude_m, velocity_ms)
    return T_req * velocity_ms


def min_drag_speed(W_kg: float, S_m2: float, polar: DragPolar,
                   altitude_m: float) -> float:
    """Speed for minimum drag (maximum L/D).

    Raymer Eq 17.13:
        V_md = sqrt(2*W/(rho*S)) * (K/CD0)^(1/4)

    This is the speed where parasite drag equals induced drag,
    yielding maximum L/D.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Minimum drag speed in m/s.
    """
    rho = density_at(altitude_m)
    W = W_kg * G0
    return math.sqrt(2.0 * W / (rho * S_m2)) * (polar.K / polar.CD0) ** 0.25


def max_range_speed(W_kg: float, S_m2: float, polar: DragPolar,
                    altitude_m: float) -> float:
    """Speed for maximum range (jet aircraft).

    Raymer Eq 17.14: For jet aircraft, maximum range occurs at
    maximum L/D, which is the same as minimum drag speed.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Maximum range speed in m/s.
    """
    return min_drag_speed(W_kg, S_m2, polar, altitude_m)


def max_endurance_speed(W_kg: float, S_m2: float, polar: DragPolar,
                        altitude_m: float) -> float:
    """Speed for maximum endurance (jet aircraft).

    Raymer Eq 17.28: For jets, max endurance occurs at minimum power
    required, where CL = sqrt(3*CD0/K).

    The corresponding speed is:
        V_me = sqrt(2*W/(rho*S*CL_min_power))

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Maximum endurance speed in m/s.
    """
    rho = density_at(altitude_m)
    W = W_kg * G0
    CL_min_power = polar.cl_for_min_power()
    return math.sqrt(2.0 * W / (rho * S_m2 * CL_min_power))


def max_speed(W_kg: float, S_m2: float, T_N: float, polar: DragPolar,
              altitude_m: float) -> tuple:
    """Maximum speed in level flight where thrust equals drag.

    Raymer Eq 17.9: Solve T = D for the highest velocity.
        T = q * S * (CD0 + K * (W/(q*S))^2)
        T = 0.5*rho*V^2*S*CD0 + K*W^2/(0.5*rho*V^2*S)

    Uses numerical root-finding (Brent's method) on the function
    f(V) = T_available - T_required(V).

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Tuple of (V_max_ms, Mach_max).

    Raises:
        ValueError: if thrust is insufficient for level flight at any speed.
    """
    rho = density_at(altitude_m)
    a = speed_of_sound_at(altitude_m)
    W = W_kg * G0

    def excess_thrust(V):
        q = 0.5 * rho * V ** 2
        CL = W / (q * S_m2)
        CD = polar.cd(CL)
        D = q * S_m2 * CD
        return T_N - D

    # Find the minimum-drag speed as the starting reference
    V_md = min_drag_speed(W_kg, S_m2, polar, altitude_m)

    # Check that there is excess thrust at V_md (otherwise can't fly)
    if excess_thrust(V_md) < 0:
        raise ValueError(
            "Insufficient thrust for level flight. "
            f"T_available={T_N:.0f} N < T_required={thrust_required(W_kg, S_m2, polar, altitude_m, V_md):.0f} N "
            f"at V_md={V_md:.1f} m/s."
        )

    # Search for V_max above V_md where excess thrust goes to zero.
    # Upper bound: expand until excess thrust is negative.
    V_upper = V_md * 1.5
    max_iter = 50
    for _ in range(max_iter):
        if excess_thrust(V_upper) < 0:
            break
        V_upper *= 1.3
    else:
        # Thrust always exceeds drag up to very high speed -- cap at Mach 3
        V_upper = 3.0 * a

    # If excess thrust is still positive at V_upper, return V_upper as limit
    if excess_thrust(V_upper) >= 0:
        return (V_upper, V_upper / a)

    V_max = brentq(excess_thrust, V_md, V_upper, xtol=0.01)
    return (V_max, V_max / a)


def generate_tr_curve(W_kg: float, S_m2: float, polar: DragPolar,
                      altitude_m: float, v_range=None) -> dict:
    """Generate thrust-required vs velocity data for plotting.

    Raymer Fig 17.2: The classic thrust-required curve showing the
    'drag bucket' at minimum drag speed.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).
        v_range: optional array of velocities (m/s). If None, generates
            from 1.1*Vstall to 1.8*Vmd with 100 points.

    Returns:
        dict with numpy arrays:
            V: velocity array (m/s)
            T_req: thrust required array (N)
            CL: lift coefficient array
            CD: drag coefficient array
            LD: lift-to-drag ratio array
    """
    if v_range is None:
        # Default range: from near stall to well above min-drag speed
        # Use a reasonable CLmax estimate for the lower bound
        V_md = min_drag_speed(W_kg, S_m2, polar, altitude_m)
        V_low = V_md * 0.6
        V_high = V_md * 2.0
        v_range = np.linspace(V_low, V_high, 100)

    v_range = np.asarray(v_range, dtype=float)
    rho = density_at(altitude_m)
    W = W_kg * G0

    q = 0.5 * rho * v_range ** 2
    CL = W / (q * S_m2)
    CD = polar.CD0 + polar.K * CL ** 2
    T_req = q * S_m2 * CD

    with np.errstate(divide="ignore", invalid="ignore"):
        LD = np.where(CD > 0, CL / CD, 0.0)

    return {
        "V": v_range,
        "T_req": T_req,
        "CL": CL,
        "CD": CD,
        "LD": LD,
    }
