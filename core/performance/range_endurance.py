"""
Range and Endurance Analysis
Reference: Raymer Ch 6 & 17.2, "Aircraft Design: A Conceptual Approach", 6th Edition

Provides Breguet range/endurance for jets and props, cruise-climb range,
specific air range, fuel-for-range, and range-payload diagram generation.
All inputs and outputs in SI units.
"""

import math

import numpy as np

from ..atmosphere import density_at, G0
from ..aerodynamics.drag_polar import DragPolar


def breguet_range_jet(W_initial_kg: float, W_final_kg: float,
                      sfc_per_hr: float, velocity_ms: float,
                      ld_ratio: float) -> float:
    """Breguet range equation for jet aircraft.

    Raymer Eq 6.11 / 17.16:
        R = (V / C) * (L/D) * ln(Wi / Wf)

    where C is the thrust-specific fuel consumption.

    Args:
        W_initial_kg: initial weight in kg (including fuel).
        W_final_kg: final weight in kg (fuel burned = Wi - Wf).
        sfc_per_hr: thrust-specific fuel consumption in 1/hr
            (e.g., 0.6 1/hr for modern turbofan).
        velocity_ms: cruise true airspeed in m/s.
        ld_ratio: lift-to-drag ratio during cruise.

    Returns:
        Range in meters.

    Raises:
        ValueError: if weights or SFC are non-positive, or Wf >= Wi.
    """
    if W_initial_kg <= 0 or W_final_kg <= 0:
        raise ValueError("Weights must be positive.")
    if W_final_kg >= W_initial_kg:
        raise ValueError("W_final_kg must be less than W_initial_kg.")
    if sfc_per_hr <= 0:
        raise ValueError("SFC must be positive.")
    if ld_ratio <= 0:
        raise ValueError("L/D ratio must be positive.")

    C = sfc_per_hr / 3600.0  # convert 1/hr to 1/s
    return (velocity_ms / C) * ld_ratio * math.log(W_initial_kg / W_final_kg)


def breguet_endurance_jet(W_initial_kg: float, W_final_kg: float,
                          sfc_per_hr: float, ld_ratio: float) -> float:
    """Breguet endurance equation for jet aircraft.

    Raymer Eq 6.13 / 17.21:
        E = (1 / C) * (L/D) * ln(Wi / Wf)

    Args:
        W_initial_kg: initial weight in kg.
        W_final_kg: final weight in kg.
        sfc_per_hr: thrust-specific fuel consumption in 1/hr.
        ld_ratio: lift-to-drag ratio during cruise.

    Returns:
        Endurance in seconds.

    Raises:
        ValueError: if weights or SFC are non-positive, or Wf >= Wi.
    """
    if W_initial_kg <= 0 or W_final_kg <= 0:
        raise ValueError("Weights must be positive.")
    if W_final_kg >= W_initial_kg:
        raise ValueError("W_final_kg must be less than W_initial_kg.")
    if sfc_per_hr <= 0:
        raise ValueError("SFC must be positive.")
    if ld_ratio <= 0:
        raise ValueError("L/D ratio must be positive.")

    C = sfc_per_hr / 3600.0  # convert 1/hr to 1/s
    return (1.0 / C) * ld_ratio * math.log(W_initial_kg / W_final_kg)


def breguet_range_prop(W_initial_kg: float, W_final_kg: float,
                       sfc_prop_per_hr: float, ld_ratio: float) -> float:
    """Breguet range equation for propeller aircraft.

    Raymer Eq 6.12:
        R = (eta_p / C_bhp) * (L/D) * ln(Wi / Wf)

    For propeller aircraft, the Breguet range does not depend on
    velocity explicitly (it is embedded in the prop efficiency and
    power-specific fuel consumption).

    Here sfc_prop_per_hr is C_bhp in lb/(hp*hr). The formula converts
    internally: C_bhp [lb/(hp*hr)] -> SI via
        C_SI = C_bhp * (4.44822 N/lbf) / (745.7 W/hp * 3600 s/hr)
    Then R = (1/C_SI) * (L/D) * ln(Wi/Wf) [using weight ratio, not mass].

    Actually, Raymer's prop Breguet is:
        R = (eta_p / Cbhp) * (L/D) * ln(Wi/Wf)
    where eta_p/Cbhp has units of m/N when Cbhp is in (N/W/s).

    For simplicity, this function expects sfc_prop_per_hr as the
    power-specific fuel consumption in kg/(W*s) * 3600 (i.e., kg/(W*hr)),
    which can be derived from Cbhp. The user provides eta_p/Cbhp combined
    as a single efficiency parameter.

    Simplified form: sfc_prop_per_hr is in 1/hr units (fuel weight flow
    per unit power, normalized). This is equivalent to C_bhp when
    eta_p is folded in.

    For practical use:
        sfc_prop_per_hr = C_bhp_lb_per_hp_hr / eta_prop
        converted to consistent units.

    The standard form used here:
        R = (1 / c_p) * (L/D) * ln(Wi/Wf)
    where c_p is in 1/m (fuel weight per unit energy = per unit distance
    contribution). Input sfc_prop_per_hr is in 1/m * 3600 * velocity...

    To keep this clean and consistent with Raymer's notation, we use:
        R = (eta_p / c_bhp) * (L/D) * ln(Wi/Wf)
    where the user passes eta_p_over_cbhp in m/N (or equivalently, the
    combined parameter sfc_prop_per_hr in (N/W/s) = 1/m, inverted and in /hr).

    SIMPLIFIED API: sfc_prop_per_hr is in 1/m, expressed as an hourly rate.
        Internally: R = (3600 / sfc_prop_per_hr) * (L/D) * ln(Wi/Wf)

    Actually, let's use the standard Raymer formulation directly:
        sfc_prop_per_hr = Cbhp in lb/(hp*hr)
        R (in meters) computed using proper unit conversion.

    Args:
        W_initial_kg: initial weight in kg.
        W_final_kg: final weight in kg.
        sfc_prop_per_hr: Cbhp in lb/(hp*hr). Typical values: 0.4-0.6.
            This is the brake-specific fuel consumption including prop
            efficiency (i.e., user should divide by eta_p if eta_p
            is not already included).
        ld_ratio: lift-to-drag ratio.

    Returns:
        Range in meters.

    Raises:
        ValueError: if weights or SFC are non-positive, or Wf >= Wi.
    """
    if W_initial_kg <= 0 or W_final_kg <= 0:
        raise ValueError("Weights must be positive.")
    if W_final_kg >= W_initial_kg:
        raise ValueError("W_final_kg must be less than W_initial_kg.")
    if sfc_prop_per_hr <= 0:
        raise ValueError("SFC must be positive.")
    if ld_ratio <= 0:
        raise ValueError("L/D ratio must be positive.")

    # Convert Cbhp from lb/(hp*hr) to N/(W*s):
    #   1 lb = 4.44822 N
    #   1 hp = 745.7 W
    #   1 hr = 3600 s
    # Cbhp_SI [N/(W*s)] = Cbhp [lb/(hp*hr)] * 4.44822 / (745.7 * 3600)
    cbhp_si = sfc_prop_per_hr * 4.44822 / (745.7 * 3600.0)

    # R = (1 / Cbhp_SI) * (L/D) * ln(Wi/Wf)
    # Note: Wi and Wf here are weight ratios, and since we use kg masses
    # the ratio is the same as the weight ratio.
    return (1.0 / cbhp_si) * ld_ratio * math.log(W_initial_kg / W_final_kg)


def cruise_climb_range(W_initial_kg: float, W_final_kg: float,
                       sfc_per_hr: float, altitude_start_m: float,
                       S_m2: float, polar: DragPolar) -> float:
    """Cruise-climb range for jet aircraft.

    Raymer Eq 17.22: Aircraft climbs as it burns fuel, maintaining
    constant CL (and thus constant L/D):
        R = (2/C) * sqrt(2/(rho*S)) * (sqrt(CL)/CD) * (sqrt(Wi) - sqrt(Wf))

    where Wi and Wf are weights in Newtons, CL is the optimum CL for
    max L/D, and rho is the density at the start altitude.

    Args:
        W_initial_kg: initial weight in kg.
        W_final_kg: final weight in kg.
        sfc_per_hr: thrust-specific fuel consumption in 1/hr.
        altitude_start_m: starting cruise altitude in meters.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.

    Returns:
        Range in meters.

    Raises:
        ValueError: if weights or SFC are non-positive, or Wf >= Wi.
    """
    if W_initial_kg <= 0 or W_final_kg <= 0:
        raise ValueError("Weights must be positive.")
    if W_final_kg >= W_initial_kg:
        raise ValueError("W_final_kg must be less than W_initial_kg.")
    if sfc_per_hr <= 0:
        raise ValueError("SFC must be positive.")

    C = sfc_per_hr / 3600.0  # 1/s
    rho = density_at(altitude_start_m)

    CL_opt = polar.cl_for_max_ld()
    CD_opt = polar.cd(CL_opt)

    Wi = W_initial_kg * G0  # Newtons
    Wf = W_final_kg * G0

    return (2.0 / C) * math.sqrt(2.0 / (rho * S_m2)) * \
           (math.sqrt(CL_opt) / CD_opt) * \
           (math.sqrt(Wi) - math.sqrt(Wf))


def specific_air_range(W_kg: float, S_m2: float, sfc_per_hr: float,
                       polar: DragPolar, altitude_m: float,
                       velocity_ms: float) -> float:
    """Specific air range: distance per unit fuel weight burned.

    SAR = V / (C * T_req) = V * (L/D) / (C * W)

    Raymer Section 17.2: SAR is a key metric for cruise efficiency.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        sfc_per_hr: thrust-specific fuel consumption in 1/hr.
        polar: DragPolar instance.
        altitude_m: altitude in meters.
        velocity_ms: true airspeed in m/s.

    Returns:
        Specific air range in m/N (meters per Newton of fuel weight burned).
    """
    C = sfc_per_hr / 3600.0  # 1/s
    W = W_kg * G0
    rho = density_at(altitude_m)
    q = 0.5 * rho * velocity_ms ** 2
    CL = W / (q * S_m2)
    CD = polar.cd(CL)
    LD = CL / CD
    return velocity_ms * LD / (C * W)


def fuel_for_range(range_m: float, sfc_per_hr: float, velocity_ms: float,
                   ld_ratio: float, W_initial_kg: float) -> float:
    """Fuel required for a given range (jet aircraft).

    Inverts the Breguet range equation (Raymer Eq 6.11):
        R = (V/C) * (L/D) * ln(Wi/Wf)
        Wi/Wf = exp(R * C / (V * L/D))
        Wf = Wi / exp(...)
        W_fuel = Wi - Wf

    Args:
        range_m: desired range in meters.
        sfc_per_hr: thrust-specific fuel consumption in 1/hr.
        velocity_ms: cruise true airspeed in m/s.
        ld_ratio: lift-to-drag ratio during cruise.
        W_initial_kg: initial aircraft weight in kg.

    Returns:
        Fuel weight required in kg.

    Raises:
        ValueError: if any input is non-positive.
    """
    if range_m <= 0:
        raise ValueError("range_m must be positive.")
    if sfc_per_hr <= 0:
        raise ValueError("SFC must be positive.")
    if velocity_ms <= 0:
        raise ValueError("velocity_ms must be positive.")
    if ld_ratio <= 0:
        raise ValueError("L/D ratio must be positive.")
    if W_initial_kg <= 0:
        raise ValueError("W_initial_kg must be positive.")

    C = sfc_per_hr / 3600.0
    exponent = range_m * C / (velocity_ms * ld_ratio)
    W_final_kg = W_initial_kg / math.exp(exponent)
    return W_initial_kg - W_final_kg


def range_payload_diagram(W0_kg: float, We_kg: float, Wf_max_kg: float,
                          sfc_per_hr: float, velocity_ms: float,
                          polar: DragPolar,
                          W_crew_kg: float = 0.0) -> dict:
    """Generate range-payload diagram data.

    Raymer Fig 6.1: The classic range-payload diagram has four key points:

    Point A: Maximum payload, fuel limited by MTOW.
        Payload = W0 - We - W_crew - Wf_available
        Wf_available = W0 - We - W_crew - W_payload_max
        (if MTOW constrains fuel)

    Point B: Maximum payload + maximum fuel (at MTOW).
        Both payload and fuel are at their max values that sum to MTOW.

    Point C: Maximum fuel, reduced payload.
        Wf = Wf_max, Payload = W0 - We - W_crew - Wf_max

    Point D: Ferry range (zero payload).
        Wf = min(Wf_max, W0 - We - W_crew), Payload = 0

    Args:
        W0_kg: maximum takeoff weight in kg.
        We_kg: operating empty weight in kg.
        Wf_max_kg: maximum fuel weight in kg (tank capacity).
        sfc_per_hr: thrust-specific fuel consumption in 1/hr.
        velocity_ms: cruise true airspeed in m/s.
        polar: DragPolar instance.
        W_crew_kg: crew weight in kg. Default 0.

    Returns:
        dict with:
            points: dict of {A, B, C, D} each with {range_m, payload_kg}
            curves: dict with range_m and payload_kg arrays for plotting
    """
    LD = polar.max_ld()

    W_useful = W0_kg - We_kg - W_crew_kg  # max useful load (fuel + payload)
    W_payload_max = W_useful  # theoretical max payload (zero fuel)

    # Point A: max fuel + max payload limited by MTOW
    # If max_fuel + max_payload > W_useful, then fuel is limited
    if Wf_max_kg >= W_useful:
        # Fuel tank can hold more than useful load: payload = 0 at max fuel
        Wf_A = W_useful
        Wp_A = 0.0
    else:
        Wp_A = W_useful - Wf_max_kg  # fill fuel tank, rest is payload
        Wf_A = Wf_max_kg

    # Actually, the standard diagram works as follows:
    # Point A: max payload, limited fuel
    #   Wp_A = W_payload_max = W_useful (zero fuel) -- this is the y-intercept
    #   Range_A = 0 (no fuel)
    #   But typically Point A has SOME fuel; let's use standard definition:

    # Standard Range-Payload diagram:
    # Point A (0 range): max payload, zero fuel -> range = 0
    # Point B (design range): max payload with fuel to fill MTOW
    #   Wp_B = max practical payload
    #   Wf_B = W0 - We - W_crew - Wp_B
    # Point C: max fuel, reduced payload
    #   Wf_C = Wf_max, Wp_C = W0 - We - W_crew - Wf_max
    # Point D: ferry (zero payload, max fuel)
    #   Wp_D = 0, Wf_D = min(Wf_max, W_useful)

    # Redefine for clarity:
    # A: (range=0, payload=W_payload_max) -- y-intercept
    # B: max payload + fuel to fill remaining MTOW
    Wp_B = max(W_useful - Wf_max_kg, 0.0)
    if Wp_B > 0:
        # Typical case: max payload constrained, fill rest with fuel
        Wf_B = W_useful - Wp_B
    else:
        # Fuel tank capacity exceeds useful load
        Wp_B = 0.0
        Wf_B = W_useful

    # For Point A, we use the same payload as B but with zero fuel
    Wp_A = Wp_B
    Wf_A = 0.0
    # But that gives zero range. Typically Point A = Point B's payload,
    # with full fuel. Let me use the standard 4-point diagram:

    # Revised standard definition:
    # Point A: max structural payload, with fuel = MTOW - OEW - max_payload
    # This assumes a max structural payload limit. Without one, use Wp = W_useful - Wf
    # For simplicity, assume max structural payload = W_useful - Wf_max
    # (i.e., max payload when tanks are full)

    # Let's use a cleaner formulation:
    Wp_max_structural = W_useful  # theoretical max (zero fuel)

    # Point B: max payload with fuel limited by MTOW
    # If we carry max payload, available fuel = W_useful - Wp_max_structural = 0
    # That's useless. So let's define max payload as a proportion or use
    # the common approach: max payload = useful load - max fuel
    Wp_max = max(W_useful - Wf_max_kg, 0.0)

    # Point A (harmonic range): max payload, zero range (y-axis intercept)
    # Actually this is just the starting point of the diagram
    R_A = 0.0
    P_A = Wp_max

    # Point B: max payload, max fuel that fits with that payload
    Wf_B = min(Wf_max_kg, W_useful - Wp_max)
    Wp_B = Wp_max
    if Wf_B > 0:
        Wi_B = W0_kg
        Wf_final_B = Wi_B - Wf_B
        R_B = breguet_range_jet(Wi_B, Wf_final_B, sfc_per_hr, velocity_ms, LD)
    else:
        R_B = 0.0

    # Point C: max fuel, reduced payload to stay at MTOW
    Wf_C = Wf_max_kg
    Wp_C = max(W_useful - Wf_max_kg, 0.0)
    Wi_C = We_kg + W_crew_kg + Wp_C + Wf_C
    Wf_final_C = Wi_C - Wf_C
    if Wf_C > 0 and Wi_C > Wf_final_C:
        R_C = breguet_range_jet(Wi_C, Wf_final_C, sfc_per_hr, velocity_ms, LD)
    else:
        R_C = R_B

    # Point D: ferry range (zero payload, max fuel)
    Wp_D = 0.0
    Wf_D = min(Wf_max_kg, W_useful)
    Wi_D = We_kg + W_crew_kg + Wf_D
    Wf_final_D = Wi_D - Wf_D
    if Wf_D > 0:
        R_D = breguet_range_jet(Wi_D, Wf_final_D, sfc_per_hr, velocity_ms, LD)
    else:
        R_D = 0.0

    # Generate smooth curve segments for plotting
    # Segment A-B: constant payload, increasing range (fuel increases)
    n_pts = 20
    if R_B > R_A:
        fuel_AB = np.linspace(0, Wf_B, n_pts)
        range_AB = np.zeros(n_pts)
        payload_AB = np.full(n_pts, Wp_max)
        for i, wf in enumerate(fuel_AB):
            if wf > 0:
                Wi = We_kg + W_crew_kg + Wp_max + wf
                Wfin = Wi - wf
                range_AB[i] = breguet_range_jet(Wi, Wfin, sfc_per_hr,
                                                velocity_ms, LD)
    else:
        range_AB = np.array([0.0])
        payload_AB = np.array([Wp_max])

    # Segment B-C: if B and C are different (they may be the same)
    # Between B and C, fuel stays at max, payload decreases
    if R_C > R_B and Wp_B > Wp_C:
        payload_BC = np.linspace(Wp_B, Wp_C, n_pts)
        range_BC = np.zeros(n_pts)
        for i, wp in enumerate(payload_BC):
            Wi = We_kg + W_crew_kg + wp + Wf_max_kg
            Wfin = Wi - Wf_max_kg
            range_BC[i] = breguet_range_jet(Wi, Wfin, sfc_per_hr,
                                            velocity_ms, LD)
    else:
        range_BC = np.array([R_B])
        payload_BC = np.array([Wp_B])

    # Segment C-D: fuel at max, payload decreases to zero
    if Wp_C > 0:
        payload_CD = np.linspace(Wp_C, 0.0, n_pts)
        range_CD = np.zeros(n_pts)
        for i, wp in enumerate(payload_CD):
            wf = min(Wf_max_kg, W_useful - wp) if W_useful > wp else 0
            if wf > 0:
                Wi = We_kg + W_crew_kg + wp + wf
                Wfin = Wi - wf
                range_CD[i] = breguet_range_jet(Wi, Wfin, sfc_per_hr,
                                                velocity_ms, LD)
        else:
            pass  # already handled
    else:
        range_CD = np.array([R_C])
        payload_CD = np.array([0.0])

    # Combine all segments
    range_all = np.concatenate([range_AB, range_BC[1:], range_CD[1:]])
    payload_all = np.concatenate([payload_AB, payload_BC[1:], payload_CD[1:]])

    return {
        "points": {
            "A": {"range_m": R_A, "payload_kg": P_A},
            "B": {"range_m": R_B, "payload_kg": Wp_B},
            "C": {"range_m": R_C, "payload_kg": Wp_C},
            "D": {"range_m": R_D, "payload_kg": Wp_D},
        },
        "curves": {
            "range_m": range_all,
            "payload_kg": payload_all,
        },
    }
