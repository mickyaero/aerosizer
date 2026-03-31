"""
Takeoff and Landing Performance Analysis
Reference: Raymer Ch 17.8-17.9, "Aircraft Design: A Conceptual Approach", 6th Edition

Provides ground roll, rotation, transition, climb-to-obstacle distances,
total takeoff distance, balanced field length, landing distance, and
comprehensive takeoff/landing summary generation.
All inputs and outputs in SI units.
"""

import math

from ..atmosphere import density_at, G0, RHO0
from ..aerodynamics.drag_polar import DragPolar


# Default CLmax values for configurations (Raymer Table 5.3)
_CLMAX_CLEAN = 1.6
_CLMAX_TAKEOFF = 2.0
_CLMAX_LANDING = 2.4


def _stall_speed(W_N: float, S_m2: float, CLmax: float,
                 rho: float) -> float:
    """Internal helper: stall speed from weight in Newtons and density."""
    return math.sqrt(2.0 * W_N / (rho * S_m2 * CLmax))


def ground_roll_distance(W_kg: float, S_m2: float, T_N: float,
                         polar: DragPolar, altitude_m: float = 0.0,
                         mu: float = 0.03,
                         CLground: float = 0.1) -> float:
    """Takeoff ground roll distance.

    Raymer Eqs 17.100-17.104: Uses the KT/KA integration method.
        SG = (1/(2*g*KA)) * ln((KT + KA*VTO^2) / (KT + KA*Vi^2))

    where:
        KT = T/W - mu
        KA = (rho / (2*W/S)) * (mu*CLground - CD0_TO - K_TO*CLground^2)
        VTO = 1.1 * Vstall (FAR 25 requirement)
        Vi = 0 (starting from rest)

    Note: KA is negative (drag > friction relief from lift), making the
    logarithm term work correctly for deceleration of the net force.

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: total takeoff thrust in Newtons.
        polar: DragPolar instance (uses takeoff configuration).
        altitude_m: field altitude in meters. Default 0 (sea level).
        mu: ground rolling friction coefficient. Default 0.03 (dry concrete).
        CLground: ground-roll lift coefficient. Default 0.1 (typical,
            wheels on ground limit AOA).

    Returns:
        Ground roll distance in meters.
    """
    rho = density_at(altitude_m)
    W = W_kg * G0
    WS = W / S_m2  # wing loading in N/m^2

    # Takeoff configuration drag at ground CL
    CD_ground = polar.cd_takeoff(CLground)

    # Stall speed in takeoff config
    Vs = _stall_speed(W, S_m2, _CLMAX_TAKEOFF, rho)
    VTO = 1.1 * Vs  # FAR 25: liftoff at 1.1 * Vstall

    # KT and KA parameters (Raymer Eq 17.101-17.102)
    KT = T_N / W - mu
    KA = (rho / (2.0 * WS)) * (mu * CLground - CD_ground)

    # Ground roll: SG = (1/(2*g*KA)) * ln((KT + KA*VTO^2) / (KT))
    # Starting from rest (Vi = 0):
    if abs(KA) < 1e-12:
        # Degenerate case: constant acceleration
        # a = g * KT, SG = VTO^2 / (2*a)
        a = G0 * KT
        if a <= 0:
            return float("inf")
        return VTO ** 2 / (2.0 * a)

    numerator = KT + KA * VTO ** 2
    denominator = KT  # KT + KA * 0^2

    if numerator <= 0 or denominator <= 0:
        # Cannot achieve takeoff speed (insufficient thrust)
        return float("inf")

    SG = (1.0 / (2.0 * G0 * KA)) * math.log(numerator / denominator)
    return abs(SG)


def rotation_distance(V_TO_ms: float) -> float:
    """Rotation distance during takeoff.

    Raymer approximation: rotation takes approximately 3 seconds.
        S_R = V_TO * t_rotation

    where t_rotation ~ 3 seconds.

    Args:
        V_TO_ms: takeoff (liftoff) speed in m/s.

    Returns:
        Rotation distance in meters.
    """
    t_rotation = 3.0  # seconds (Raymer approximation)
    return V_TO_ms * t_rotation


def transition_distance(W_kg: float, S_m2: float, polar: DragPolar,
                        altitude_m: float = 0.0) -> tuple:
    """Transition distance (circular arc from ground to climb).

    Raymer Eqs 17.105-17.110: The transition is modeled as a
    circular arc at constant speed V_TR = 1.15 * Vstall.

    Load factor n ~ 1.2 during transition:
        R = V_TR^2 / (g * (n - 1)) = V_TR^2 / (0.2 * g)
        h_TR = R * (1 - cos(gamma_climb))
        S_TR = R * sin(gamma_climb)

    For small gamma: h_TR ~ R * gamma^2/2, S_TR ~ R * gamma

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: field altitude in meters. Default 0.

    Returns:
        Tuple of (S_TR_m, h_TR_m): transition distance and height gained.
    """
    rho = density_at(altitude_m)
    W = W_kg * G0

    # Transition speed: 1.15 * Vstall (takeoff config)
    Vs = _stall_speed(W, S_m2, _CLMAX_TAKEOFF, rho)
    V_TR = 1.15 * Vs

    # Transition arc radius (n = 1.2 assumed)
    n_load = 1.2
    R = V_TR ** 2 / (G0 * (n_load - 1.0))

    # Estimate climb angle from thrust and drag at V_TR
    q = 0.5 * rho * V_TR ** 2
    CL_TR = W / (q * S_m2)  # approximately, since n ~ 1
    CD_TR = polar.cd_takeoff(CL_TR)
    D_TR = q * S_m2 * CD_TR

    # We need T for climb angle, but T isn't passed here.
    # Use a typical T/W ~ 0.3 for transport aircraft as default estimate.
    # Actually, compute based on available info: assume gamma is small
    # and use the geometry. For a proper calculation, use total_takeoff_distance.
    # Here we return the geometry for a typical climb gradient.
    # Use small-angle approximation with a representative gamma.
    # Raymer suggests gamma_climb ~ 5-15 degrees for jet transport.
    # For now, estimate: if no thrust info, use 0.05 rad (~3 deg) as default.
    # This function is a building block; total_takeoff_distance() handles it properly.

    # Approximate gamma from typical all-engine T/W and L/D
    # gamma = arcsin((T-D)/W) -- but we don't have T here
    # Use a reasonable default
    gamma = 0.10  # ~ 5.7 degrees, reasonable for jet transport initial climb

    h_TR = R * (1.0 - math.cos(gamma))
    S_TR = R * math.sin(gamma)

    return (S_TR, h_TR)


def climb_to_obstacle(h_obstacle_m: float, h_transition_m: float,
                      gamma_climb_rad: float) -> float:
    """Horizontal distance during climb to clear obstacle.

    Raymer Eq 17.112:
        S_c = (h_obstacle - h_TR) / tan(gamma_climb)

    If the transition height exceeds the obstacle height, the
    obstacle is already cleared during transition and S_c = 0.

    Args:
        h_obstacle_m: obstacle height in meters. 10.7 m (35 ft) for
            FAR 25, 15.2 m (50 ft) for military.
        h_transition_m: height gained during transition in meters.
        gamma_climb_rad: climb angle in radians.

    Returns:
        Climb distance in meters. Returns 0 if obstacle is cleared
        during transition.
    """
    if h_transition_m >= h_obstacle_m:
        return 0.0

    if gamma_climb_rad <= 0:
        return float("inf")

    dh = h_obstacle_m - h_transition_m
    return dh / math.tan(gamma_climb_rad)


def total_takeoff_distance(W_kg: float, S_m2: float, T_N: float,
                           polar: DragPolar, altitude_m: float = 0.0,
                           h_obstacle_m: float = 10.7,
                           mu: float = 0.03) -> dict:
    """Total takeoff distance over an obstacle.

    Raymer Section 17.8: Total = ground roll + rotation + transition + climb.
        S_total = SG + SR + STR + SC

    FAR 25 obstacle height: 10.7 m (35 ft).
    Military: 15.2 m (50 ft).

    Args:
        W_kg: aircraft weight in kg.
        S_m2: wing reference area in m^2.
        T_N: total takeoff thrust in Newtons.
        polar: DragPolar instance.
        altitude_m: field altitude in meters. Default 0.
        h_obstacle_m: obstacle height in meters. Default 10.7 m (35 ft, FAR 25).
        mu: ground rolling friction coefficient. Default 0.03.

    Returns:
        dict with:
            SG: ground roll distance (m)
            SR: rotation distance (m)
            STR: transition distance (m)
            SC: climb distance (m)
            total: total takeoff distance (m)
            VTO: liftoff speed (m/s)
            V_TR: transition speed (m/s)
            gamma_climb: climb angle (rad)
            h_TR: height gained in transition (m)
    """
    rho = density_at(altitude_m)
    W = W_kg * G0

    # Ground roll
    SG = ground_roll_distance(W_kg, S_m2, T_N, polar, altitude_m, mu)

    # Liftoff speed
    Vs = _stall_speed(W, S_m2, _CLMAX_TAKEOFF, rho)
    VTO = 1.1 * Vs

    # Rotation distance
    SR = rotation_distance(VTO)

    # Transition: circular arc
    V_TR = 1.15 * Vs
    n_load = 1.2
    R = V_TR ** 2 / (G0 * (n_load - 1.0))

    # Climb angle from T and D at climb speed (V_climb ~ 1.2 * Vs)
    V_climb = 1.2 * Vs
    q_climb = 0.5 * rho * V_climb ** 2
    CL_climb = W / (q_climb * S_m2)
    CD_climb = polar.cd_takeoff(CL_climb)
    D_climb = q_climb * S_m2 * CD_climb

    sin_gamma = (T_N - D_climb) / W
    sin_gamma = max(-1.0, min(1.0, sin_gamma))
    gamma_climb = math.asin(sin_gamma)

    # Transition distance and height
    if gamma_climb > 0:
        h_TR = R * (1.0 - math.cos(gamma_climb))
        STR = R * math.sin(gamma_climb)
    else:
        h_TR = 0.0
        STR = 0.0

    # Climb to obstacle
    SC = climb_to_obstacle(h_obstacle_m, h_TR, gamma_climb)

    total = SG + SR + STR + SC

    return {
        "SG": SG,
        "SR": SR,
        "STR": STR,
        "SC": SC,
        "total": total,
        "VTO": VTO,
        "V_TR": V_TR,
        "gamma_climb": gamma_climb,
        "h_TR": h_TR,
    }


def balanced_field_length(W_kg: float, S_m2: float, T_each_N: float,
                          n_engines: int, polar: DragPolar,
                          altitude_m: float = 0.0) -> float:
    """Balanced field length (BFL) for multi-engine aircraft.

    Raymer Eq 17.113 (empirical correlation):
        BFL = 0.863 / (1 + 2.3*G) *
              (W/S / (rho*g*CLclimb) + h_obs) *
              (1 / (T/W - U) + 2.7) +
              655 / sqrt(rho/rho0)

    where:
        G = gamma_climb_OEI - gamma_min_required
        CLclimb = CL at 1.2*Vstall (V2 speed)
        T = thrust with one engine inoperative
        U = 0.01*CLmax_TO + 0.02 (ground friction approximation)
        gamma_min = 0.024 for 2-engine, 0.027 for 3-engine, 0.030 for 4-engine

    This is the field length where accelerate-go distance equals
    accelerate-stop distance with one engine failed at V1.

    Args:
        W_kg: maximum takeoff weight in kg.
        S_m2: wing reference area in m^2.
        T_each_N: thrust per engine in Newtons.
        n_engines: number of engines (2, 3, or 4).
        polar: DragPolar instance.
        altitude_m: field altitude in meters. Default 0.

    Returns:
        Balanced field length in meters.

    Raises:
        ValueError: if n_engines is not 2, 3, or 4.
    """
    if n_engines not in (2, 3, 4):
        raise ValueError("n_engines must be 2, 3, or 4.")

    rho = density_at(altitude_m)
    W = W_kg * G0
    WS = W / S_m2

    # Minimum required climb gradient with OEI (FAR 25.121)
    gamma_min_map = {2: 0.024, 3: 0.027, 4: 0.030}
    gamma_min = gamma_min_map[n_engines]

    # OEI thrust
    T_OEI = T_each_N * (n_engines - 1)

    # Climb CL at V2 = 1.2 * Vstall
    Vs = _stall_speed(W, S_m2, _CLMAX_TAKEOFF, rho)
    V2 = 1.2 * Vs
    q_V2 = 0.5 * rho * V2 ** 2
    CL_climb = W / (q_V2 * S_m2)

    # OEI climb gradient
    CD_climb = polar.cd_takeoff(CL_climb)
    D_OEI = q_V2 * S_m2 * CD_climb
    gamma_OEI = (T_OEI - D_OEI) / W

    # G = climb gradient margin
    G = gamma_OEI - gamma_min

    # U = friction / drag term during ground roll
    U = 0.01 * _CLMAX_TAKEOFF + 0.02

    # Total T/W (all engines) for ground roll portion
    T_total = T_each_N * n_engines
    TW = T_total / W

    # Obstacle height for FAR 25: 10.7 m (35 ft)
    h_obs = 10.7

    # Raymer Eq 17.113
    sigma = rho / RHO0
    term1 = 0.863 / (1.0 + 2.3 * G)
    term2 = WS / (rho * G0 * CL_climb) + h_obs
    term3 = 1.0 / (TW - U) + 2.7
    term4 = 655.0 / math.sqrt(sigma)

    BFL = term1 * term2 * term3 + term4
    return BFL


def landing_distance(W_landing_kg: float, S_m2: float, polar: DragPolar,
                     altitude_m: float = 0.0,
                     h_obstacle_m: float = 15.24,
                     mu_braking: float = 0.4) -> dict:
    """Total landing distance over an obstacle.

    Raymer Ch 17.9: Landing consists of approach, flare, free roll, and
    ground roll (braking) segments.

    Approach: Va = 1.3 * Vstall (landing config)
        Sa = (h_obs - h_flare) / tan(gamma_approach)
        gamma_approach ~ 3 degrees (standard glideslope)

    Flare: Circular arc from approach angle to level flight.
        Vf = 1.23 * Vstall, n ~ 1.2
        R = Vf^2 / (g*(n-1))
        Sf = R * sin(gamma_approach)
        hf = R * (1 - cos(gamma_approach))

    Free roll: ~3 seconds at touchdown speed VTD = 1.15 * Vstall.

    Ground roll (braking): From VTD to 0.
        Sgr = VTD^2 / (2*g*mu_braking) (simplified, no reverse thrust)

    FAR field length = total / 0.6 for commercial operations.

    Args:
        W_landing_kg: landing weight in kg.
        S_m2: wing reference area in m^2.
        polar: DragPolar instance.
        altitude_m: field altitude in meters. Default 0.
        h_obstacle_m: obstacle height in meters. Default 15.24 m (50 ft).
        mu_braking: braking friction coefficient. Default 0.4 (dry, no reverse thrust).

    Returns:
        dict with:
            Sa: approach distance (m)
            Sf: flare distance (m)
            Sfr: free roll distance (m)
            Sgr: ground roll (braking) distance (m)
            total: total landing distance (m)
            FAR_field_length: FAR landing field length = total / 0.6 (m)
            Va: approach speed (m/s)
            VTD: touchdown speed (m/s)
    """
    rho = density_at(altitude_m)
    W = W_landing_kg * G0

    # Stall speed in landing configuration
    Vs = _stall_speed(W, S_m2, _CLMAX_LANDING, rho)

    # Approach speed: 1.3 * Vstall (FAR 25)
    Va = 1.3 * Vs

    # Approach angle: standard 3-degree glideslope
    gamma_approach = math.radians(3.0)

    # Flare: circular arc
    Vf = 1.23 * Vs
    n_flare = 1.2
    R_flare = Vf ** 2 / (G0 * (n_flare - 1.0))

    hf = R_flare * (1.0 - math.cos(gamma_approach))
    Sf = R_flare * math.sin(gamma_approach)

    # Approach distance (from obstacle to start of flare)
    h_approach = h_obstacle_m - hf
    if h_approach > 0:
        Sa = h_approach / math.tan(gamma_approach)
    else:
        Sa = 0.0

    # Free roll: ~3 seconds at touchdown speed
    VTD = 1.15 * Vs
    t_free = 3.0  # seconds
    Sfr = VTD * t_free

    # Ground roll (braking): decelerate from VTD to 0
    # Sgr = VTD^2 / (2 * g * mu_braking)
    # This is simplified (no aerodynamic drag or reverse thrust credit)
    if mu_braking > 0:
        Sgr = VTD ** 2 / (2.0 * G0 * mu_braking)
    else:
        Sgr = float("inf")

    total = Sa + Sf + Sfr + Sgr

    # FAR landing field length: actual distance / 0.6
    FAR_field = total / 0.6

    return {
        "Sa": Sa,
        "Sf": Sf,
        "Sfr": Sfr,
        "Sgr": Sgr,
        "total": total,
        "FAR_field_length": FAR_field,
        "Va": Va,
        "VTD": VTD,
    }


def generate_takeoff_landing_summary(W_kg: float, S_m2: float, T_N: float,
                                     n_engines: int, polar: DragPolar,
                                     altitude_m: float = 0.0,
                                     W_landing_fraction: float = 0.85) -> dict:
    """Complete takeoff and landing analysis.

    Combines all takeoff and landing calculations into a comprehensive
    summary for design evaluation.

    Args:
        W_kg: maximum takeoff weight in kg.
        S_m2: wing reference area in m^2.
        T_N: total takeoff thrust in Newtons.
        n_engines: number of engines (2, 3, or 4).
        polar: DragPolar instance.
        altitude_m: field altitude in meters. Default 0.
        W_landing_fraction: landing weight as fraction of MTOW. Default 0.85.

    Returns:
        dict with:
            takeoff: total_takeoff_distance results
            landing: landing_distance results
            balanced_field_length: BFL in meters
            stall_speeds: dict of Vs_clean, Vs_takeoff, Vs_landing (m/s)
            reference_speeds: dict of V1, VR, V2, Va, VTD (m/s)
    """
    rho = density_at(altitude_m)
    W = W_kg * G0
    W_land = W_kg * W_landing_fraction
    W_land_N = W_land * G0

    # Stall speeds
    Vs_clean = _stall_speed(W, S_m2, _CLMAX_CLEAN, rho)
    Vs_takeoff = _stall_speed(W, S_m2, _CLMAX_TAKEOFF, rho)
    Vs_landing = _stall_speed(W_land_N, S_m2, _CLMAX_LANDING, rho)

    # Takeoff analysis
    to_result = total_takeoff_distance(W_kg, S_m2, T_N, polar,
                                       altitude_m)

    # Landing analysis
    ld_result = landing_distance(W_land, S_m2, polar, altitude_m)

    # Balanced field length
    T_each = T_N / n_engines
    BFL = balanced_field_length(W_kg, S_m2, T_each, n_engines, polar,
                                altitude_m)

    # Reference speeds
    V1 = 0.95 * to_result["VTO"]  # decision speed ~ 0.95 * VR (approximation)
    VR = to_result["VTO"]  # rotation speed ~ liftoff speed
    V2 = 1.2 * Vs_takeoff  # takeoff safety speed

    return {
        "takeoff": to_result,
        "landing": ld_result,
        "balanced_field_length": BFL,
        "stall_speeds": {
            "Vs_clean": Vs_clean,
            "Vs_takeoff": Vs_takeoff,
            "Vs_landing": Vs_landing,
        },
        "reference_speeds": {
            "V1": V1,
            "VR": VR,
            "V2": V2,
            "Va": ld_result["Va"],
            "VTD": ld_result["VTD"],
        },
    }
