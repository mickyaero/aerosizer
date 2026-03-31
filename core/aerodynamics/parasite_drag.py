"""
Parasite Drag Estimation via Component Buildup Method
Reference: Raymer Ch 12, "Aircraft Design: A Conceptual Approach", 6th Edition

Implements the component buildup method (Raymer Eq 12.24) where total
parasite drag is the sum of individual component contributions:
    CD0 = sum(Cf_c * FF_c * Q_c * Swet_c) / Sref + CD_misc + CD_L&P
"""

import math
from typing import Optional

from ..atmosphere import atmosphere


# --------------------------------------------------------------------------- #
# Default interference factors (Raymer Table 12.1)
# --------------------------------------------------------------------------- #
DEFAULT_Q = {
    "wing": 1.0,
    "fuselage": 1.0,
    "nacelle": 1.3,
    "tail": 1.05,
}


# --------------------------------------------------------------------------- #
# Core equations
# --------------------------------------------------------------------------- #

def skin_friction_coeff(
    Re: float,
    mach: float = 0.0,
    laminar_fraction: float = 0.0,
) -> float:
    """Flat-plate skin friction coefficient with compressibility correction.

    Turbulent: Raymer Eq 12.27
        Cf_turb = 0.455 / (log10(Re))^2.58 / (1 + 0.144*M^2)^0.65

    Laminar: Blasius solution
        Cf_lam = 1.328 / sqrt(Re)

    Mixed flow uses a weighted average based on *laminar_fraction*.

    Args:
        Re: Reynolds number (must be > 0).
        mach: Freestream Mach number.
        laminar_fraction: fraction of wetted area with laminar flow [0, 1].
            0.0 means fully turbulent (default, conservative).

    Returns:
        Skin friction coefficient Cf (dimensionless).

    Raises:
        ValueError: if Re <= 0 or laminar_fraction is out of [0, 1].
    """
    if Re <= 0:
        raise ValueError(f"Reynolds number must be positive, got {Re}")
    if not 0.0 <= laminar_fraction <= 1.0:
        raise ValueError(
            f"laminar_fraction must be in [0, 1], got {laminar_fraction}"
        )

    log_Re = math.log10(Re)

    # Turbulent (Raymer Eq 12.27)
    Cf_turb = 0.455 / (log_Re ** 2.58) / (1.0 + 0.144 * mach ** 2) ** 0.65

    if laminar_fraction == 0.0:
        return Cf_turb

    # Laminar (Blasius)
    Cf_lam = 1.328 / math.sqrt(Re)

    if laminar_fraction == 1.0:
        return Cf_lam

    # Mixed
    return laminar_fraction * Cf_lam + (1.0 - laminar_fraction) * Cf_turb


def form_factor_wing(
    x_c_max: float,
    t_c: float,
    mach: float,
    sweep_m: float,
) -> float:
    """Wing/tail form factor.

    Raymer Eq 12.30:
        FF = [1 + 0.6/(x/c)_m * (t/c) + 100*(t/c)^4]
             * [1.34 * M^0.18 * cos(sweep_m)^0.28]

    Args:
        x_c_max: chordwise location of maximum thickness (fraction of chord,
            typically 0.3-0.4 for modern airfoils).
        t_c: thickness-to-chord ratio.
        mach: freestream Mach number.
        sweep_m: sweep angle of the maximum-thickness line in radians.

    Returns:
        Form factor FF (dimensionless).

    Raises:
        ValueError: if x_c_max <= 0.
    """
    if x_c_max <= 0:
        raise ValueError(f"x_c_max must be positive, got {x_c_max}")

    thickness_term = 1.0 + 0.6 / x_c_max * t_c + 100.0 * t_c ** 4
    compressibility_term = 1.34 * mach ** 0.18 * math.cos(sweep_m) ** 0.28

    return thickness_term * compressibility_term


def form_factor_fuselage(fineness_ratio: float) -> float:
    """Fuselage form factor.

    Raymer Eq 12.31:
        FF = 1 + 60/f^3 + f/400

    where f = l/d (length / diameter), the fineness ratio.

    Args:
        fineness_ratio: fuselage fineness ratio l/d (typically 6-10 for transports).

    Returns:
        Form factor FF (dimensionless).

    Raises:
        ValueError: if fineness_ratio <= 0.
    """
    if fineness_ratio <= 0:
        raise ValueError(
            f"fineness_ratio must be positive, got {fineness_ratio}"
        )
    f = fineness_ratio
    return 1.0 + 60.0 / f ** 3 + f / 400.0


def form_factor_nacelle(fineness_ratio: float) -> float:
    """Nacelle form factor.

    Raymer Eq 12.32:
        FF = 1 + 0.35/f

    where f = l/d (fineness ratio of the nacelle).

    Args:
        fineness_ratio: nacelle fineness ratio l/d (typically 3-5).

    Returns:
        Form factor FF (dimensionless).

    Raises:
        ValueError: if fineness_ratio <= 0.
    """
    if fineness_ratio <= 0:
        raise ValueError(
            f"fineness_ratio must be positive, got {fineness_ratio}"
        )
    return 1.0 + 0.35 / fineness_ratio


def reynolds_number(
    velocity: float,
    length: float,
    altitude_m: float,
) -> float:
    """Compute Reynolds number at given flight conditions.

    Re = rho * V * L / mu

    Args:
        velocity: true airspeed in m/s.
        length: reference length (chord for wings, body length for fuselage) in m.
        altitude_m: geometric altitude in meters.

    Returns:
        Reynolds number (dimensionless).
    """
    atm = atmosphere(altitude_m)
    rho = atm["rho"]
    mu = atm["mu"]
    return rho * velocity * length / mu


def component_drag(
    Cf: float,
    FF: float,
    Q: float,
    Swet: float,
    Sref: float,
) -> float:
    """Zero-lift drag contribution of a single component.

    Raymer Eq 12.24:
        CD0_comp = Cf * FF * Q * Swet / Sref

    Args:
        Cf: skin friction coefficient.
        FF: form factor.
        Q: interference factor.
        Swet: component wetted area (m^2).
        Sref: reference wing area (m^2).

    Returns:
        Component contribution to CD0 (dimensionless).
    """
    return Cf * FF * Q * Swet / Sref


def parasite_drag_buildup(
    components: list[dict],
    Sref: float,
    altitude_m: float,
    velocity_ms: float,
    laminar_fraction: float = 0.0,
) -> dict:
    """Total parasite drag via Raymer component buildup method (Eq 12.24).

    Each component dict must contain:
        name (str): component identifier
        length (float): reference length in m (chord or body length)
        Swet (float): wetted area in m^2
        component_type (str): one of 'wing', 'fuselage', 'nacelle', 'tail'

    Geometry parameters by type:
        wing/tail: t_c, x_c_max, sweep_m
        fuselage/nacelle: fineness_ratio

    Optional per-component overrides:
        Q (float): interference factor (overrides default from Raymer Table 12.1)
        laminar_fraction (float): per-component laminar fraction

    Args:
        components: list of component specification dicts.
        Sref: reference wing area (m^2).
        altitude_m: cruise altitude in meters.
        velocity_ms: true airspeed in m/s.
        laminar_fraction: default laminar fraction for all components.

    Returns:
        dict with:
            CD0 (float): total clean parasite drag coefficient
            breakdown (dict): per-component {name: {Cf, FF, Q, Swet, CD0_comp}}
    """
    atm = atmosphere(altitude_m)
    mach = velocity_ms / atm["a"]

    breakdown = {}
    CD0_total = 0.0

    for comp in components:
        name = comp["name"]
        length = comp["length"]
        Swet = comp["Swet"]
        ctype = comp["component_type"]

        # Reynolds number for this component
        Re = reynolds_number(velocity_ms, length, altitude_m)

        # Skin friction coefficient
        lf = comp.get("laminar_fraction", laminar_fraction)
        Cf = skin_friction_coeff(Re, mach, lf)

        # Form factor
        if ctype in ("wing", "tail"):
            t_c = comp["t_c"]
            x_c_max = comp["x_c_max"]
            sweep_m = comp["sweep_m"]
            FF = form_factor_wing(x_c_max, t_c, mach, sweep_m)
        elif ctype == "fuselage":
            FF = form_factor_fuselage(comp["fineness_ratio"])
        elif ctype == "nacelle":
            FF = form_factor_nacelle(comp["fineness_ratio"])
        else:
            raise ValueError(
                f"Unknown component_type '{ctype}' for component '{name}'. "
                f"Must be one of: wing, fuselage, nacelle, tail"
            )

        # Interference factor
        Q = comp.get("Q", DEFAULT_Q.get(ctype, 1.0))

        # Component drag contribution
        CD0_comp = component_drag(Cf, FF, Q, Swet, Sref)
        CD0_total += CD0_comp

        breakdown[name] = {
            "Cf": Cf,
            "FF": FF,
            "Q": Q,
            "Re": Re,
            "Swet": Swet,
            "CD0_comp": CD0_comp,
        }

    return {"CD0": CD0_total, "breakdown": breakdown}


def leakage_protuberance_drag(
    CD0_clean: float,
    fraction: float = 0.03,
) -> float:
    """Estimate drag increment due to leakage and protuberance effects.

    Raymer Section 12.5 suggests adding 2-5% of the clean CD0
    to account for leakage, protuberance, antenna, and miscellaneous drag.
    A default of 3% is typical for modern transports.

    Args:
        CD0_clean: clean parasite drag coefficient (from component buildup).
        fraction: leakage/protuberance fraction [0, 1]. Default 0.03 (3%).

    Returns:
        CD_LP: leakage and protuberance drag increment.
    """
    return CD0_clean * fraction
