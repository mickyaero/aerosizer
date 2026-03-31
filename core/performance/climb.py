"""
Climb Performance Analysis
Reference: Raymer Ch 17.3, "Aircraft Design: A Conceptual Approach", 6th Edition

Provides rate of climb, max ROC, climb gradient/angle, best angle-of-climb
speed, service/absolute ceiling, time to climb, and ROC curve generation.
All inputs and outputs in SI units.
"""

import math

import numpy as np
from scipy.optimize import minimize_scalar, brentq

from ..atmosphere import density_at, G0
from ..aerodynamics.drag_polar import DragPolar


def rate_of_climb(W_kg: float, S_m2: float, T_N: float, polar: DragPolar,
                  altitude_m: float, velocity_ms: float) -> float:
    """Rate of climb at a given speed and altitude.

    Raymer Eq 17.39:
        ROC = V * (T - D) / W

    where D is computed from the drag polar at the CL needed
    to support the weight in a shallow climb (approximation: L ~ W).

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance (clean configuration).
        altitude_m: altitude in meters (geometric).
        velocity_ms: true airspeed in m/s.

    Returns:
        Rate of climb in m/s. Negative means descending.

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
    D = q * S_m2 * CD
    return velocity_ms * (T_N - D) / W


def max_rate_of_climb(W_kg: float, S_m2: float, T_N: float, polar: DragPolar,
                      altitude_m: float) -> tuple:
    """Find velocity for maximum rate of climb and the corresponding ROC.

    Numerically maximizes ROC(V) using bounded scalar optimization.
    Raymer Section 17.3.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Tuple of (ROC_max_ms, V_best_ms).
    """
    rho = density_at(altitude_m)
    W = W_kg * G0

    # Search bounds: from a reasonable low speed to well above min-drag speed
    V_md = math.sqrt(2.0 * W / (rho * S_m2)) * (polar.K / polar.CD0) ** 0.25
    V_low = V_md * 0.4
    V_high = V_md * 2.5

    def neg_roc(V):
        q = 0.5 * rho * V ** 2
        CL = W / (q * S_m2)
        CD = polar.cd(CL)
        D = q * S_m2 * CD
        return -(V * (T_N - D) / W)

    result = minimize_scalar(neg_roc, bounds=(V_low, V_high), method="bounded")
    V_best = result.x
    ROC_max = -result.fun

    return (ROC_max, V_best)


def climb_gradient(W_kg: float, T_N: float, D_N: float) -> float:
    """Climb gradient (dimensionless).

    Raymer Eq 17.38:
        G = (T - D) / W

    Args:
        W_kg: aircraft weight in kg.
        T_N: thrust in Newtons.
        D_N: drag in Newtons.

    Returns:
        Climb gradient (dimensionless). Positive = climbing.
    """
    W = W_kg * G0
    return (T_N - D_N) / W


def climb_angle(W_kg: float, T_N: float, D_N: float) -> float:
    """Climb angle (flight path angle gamma).

    Raymer Eq 17.38:
        gamma = arcsin((T - D) / W)

    Args:
        W_kg: aircraft weight in kg.
        T_N: thrust in Newtons.
        D_N: drag in Newtons.

    Returns:
        Climb angle in radians. Positive = climbing.
    """
    G = climb_gradient(W_kg, T_N, D_N)
    # Clamp to [-1, 1] for arcsin safety
    G_clamped = max(-1.0, min(1.0, G))
    return math.asin(G_clamped)


def best_angle_of_climb_speed(W_kg: float, S_m2: float, T_N: float,
                              polar: DragPolar, altitude_m: float) -> float:
    """Speed for steepest climb (maximum climb angle / gradient).

    Raymer Section 17.3: Maximum gamma occurs at the speed where
    (T - D) / W is maximized. Since T is constant (jet), this is
    the speed for minimum drag.

    For jets with constant thrust, Vx = V_min_drag. Numerically
    optimized for generality.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: available thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: altitude in meters (geometric).

    Returns:
        Best angle-of-climb speed in m/s.
    """
    rho = density_at(altitude_m)
    W = W_kg * G0

    V_md = math.sqrt(2.0 * W / (rho * S_m2)) * (polar.K / polar.CD0) ** 0.25
    V_low = V_md * 0.4
    V_high = V_md * 2.5

    def neg_gradient(V):
        q = 0.5 * rho * V ** 2
        CL = W / (q * S_m2)
        CD = polar.cd(CL)
        D = q * S_m2 * CD
        return -(T_N - D) / W

    result = minimize_scalar(neg_gradient, bounds=(V_low, V_high), method="bounded")
    return result.x


def service_ceiling(W_kg: float, S_m2: float, T_available_func,
                    polar: DragPolar,
                    ROC_min_ms: float = 0.508) -> float:
    """Service ceiling: altitude where max ROC equals ROC_min.

    Raymer defines service ceiling as the altitude where:
    - ROC = 100 fpm (0.508 m/s) for jets
    - ROC = 500 fpm (2.54 m/s) for propeller aircraft

    Uses bisection search between sea level and 25 km.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: callable(altitude_m) -> thrust in Newtons.
        polar: DragPolar instance.
        ROC_min_ms: minimum ROC threshold in m/s. Default 0.508 (100 fpm).

    Returns:
        Service ceiling altitude in meters.

    Raises:
        ValueError: if ROC at sea level is already below ROC_min.
    """
    h_low = 0.0
    h_high = 25000.0  # 25 km upper bound

    def roc_residual(h):
        T = T_available_func(h)
        roc_max, _ = max_rate_of_climb(W_kg, S_m2, T, polar, h)
        return roc_max - ROC_min_ms

    # Check that aircraft can achieve ROC_min at sea level
    if roc_residual(h_low) < 0:
        raise ValueError(
            f"Max ROC at sea level is below {ROC_min_ms:.3f} m/s. "
            "Aircraft cannot reach service ceiling criteria even at sea level."
        )

    # If aircraft can still climb at h_high, return h_high as lower bound
    if roc_residual(h_high) > 0:
        return h_high

    return brentq(roc_residual, h_low, h_high, xtol=1.0)


def absolute_ceiling(W_kg: float, S_m2: float, T_available_func,
                     polar: DragPolar) -> float:
    """Absolute ceiling: altitude where max ROC = 0.

    Same method as service_ceiling with ROC_min = 0.
    Raymer Section 17.3.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: callable(altitude_m) -> thrust in Newtons.
        polar: DragPolar instance.

    Returns:
        Absolute ceiling altitude in meters.
    """
    return service_ceiling(W_kg, S_m2, T_available_func, polar,
                           ROC_min_ms=0.0)


def time_to_climb(W_kg: float, S_m2: float, T_available_func,
                  polar: DragPolar, h1_m: float, h2_m: float,
                  n_steps: int = 50) -> float:
    """Time to climb from h1 to h2.

    Raymer Eq 17.50: Numerically integrates dt = dh / ROC using
    the trapezoidal rule.

    At each altitude step, the max ROC is computed with the
    thrust available at that altitude.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: callable(altitude_m) -> thrust in Newtons.
        polar: DragPolar instance.
        h1_m: starting altitude in meters.
        h2_m: ending altitude in meters.
        n_steps: number of integration steps. Default 50.

    Returns:
        Time to climb in seconds.

    Raises:
        ValueError: if h2_m <= h1_m or if ROC is non-positive at any step.
    """
    if h2_m <= h1_m:
        raise ValueError("h2_m must be greater than h1_m.")

    altitudes = np.linspace(h1_m, h2_m, n_steps + 1)
    dh = altitudes[1] - altitudes[0]

    # Compute 1/ROC at each altitude (inverse of ROC for integration)
    inv_roc = np.zeros(n_steps + 1)
    for i, h in enumerate(altitudes):
        T = T_available_func(h)
        roc_max, _ = max_rate_of_climb(W_kg, S_m2, T, polar, h)
        if roc_max <= 0:
            raise ValueError(
                f"ROC is non-positive ({roc_max:.3f} m/s) at altitude "
                f"{h:.0f} m. Cannot complete climb."
            )
        inv_roc[i] = 1.0 / roc_max

    # Trapezoidal integration: t = integral(dh/ROC)
    time_s = np.trapezoid(inv_roc, altitudes)
    return float(time_s)


def generate_roc_curve(W_kg: float, S_m2: float, T_available_func,
                       polar: DragPolar, alt_range=None) -> dict:
    """Generate max rate-of-climb vs altitude data for plotting.

    Raymer Fig 17.10: Shows how max ROC decreases with altitude,
    reaching zero at the absolute ceiling.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_available_func: callable(altitude_m) -> thrust in Newtons.
        polar: DragPolar instance.
        alt_range: optional array of altitudes (m). If None, generates
            from 0 to 15000 m with 50 points.

    Returns:
        dict with numpy arrays:
            altitude: altitude array (m)
            ROC_max: maximum ROC array (m/s)
            V_best: best climb speed array (m/s)
    """
    if alt_range is None:
        alt_range = np.linspace(0, 15000, 50)

    alt_range = np.asarray(alt_range, dtype=float)
    roc_arr = np.zeros_like(alt_range)
    v_best_arr = np.zeros_like(alt_range)

    for i, h in enumerate(alt_range):
        T = T_available_func(h)
        roc_max, v_best = max_rate_of_climb(W_kg, S_m2, T, polar, h)
        roc_arr[i] = max(roc_max, 0.0)
        v_best_arr[i] = v_best

    return {
        "altitude": alt_range,
        "ROC_max": roc_arr,
        "V_best": v_best_arr,
    }
