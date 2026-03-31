"""
Induced Drag and Oswald Efficiency Estimation
Reference: Raymer Ch 12, "Aircraft Design: A Conceptual Approach", 6th Edition

Induced drag arises from the generation of lift and is proportional to CL^2.
The Oswald efficiency factor e accounts for the non-elliptical lift distribution
and other departures from the theoretical minimum induced drag.
"""

import math


def oswald_efficiency_straight(AR: float) -> float:
    """Oswald span efficiency for straight (unswept) wings.

    Raymer Eq 12.49:
        e = 1.78*(1 - 0.045*AR^0.68) - 0.64

    Typical range: 0.7 - 0.85 for straight-wing aircraft.

    Args:
        AR: wing aspect ratio (span^2 / Sref).

    Returns:
        Oswald span efficiency factor e (dimensionless).

    Raises:
        ValueError: if AR <= 0.
    """
    if AR <= 0:
        raise ValueError(f"Aspect ratio must be positive, got {AR}")
    return 1.78 * (1.0 - 0.045 * AR ** 0.68) - 0.64


def oswald_efficiency_swept(AR: float, sweep_LE_rad: float) -> float:
    """Oswald span efficiency for swept wings.

    Raymer Eq 12.50:
        e = 4.61*(1 - 0.045*AR^0.68)*cos(sweep_LE)^0.15 - 3.1

    Typical range: 0.75 - 0.85 for swept-wing transports.

    Args:
        AR: wing aspect ratio.
        sweep_LE_rad: leading-edge sweep angle in radians.

    Returns:
        Oswald span efficiency factor e (dimensionless).

    Raises:
        ValueError: if AR <= 0.
    """
    if AR <= 0:
        raise ValueError(f"Aspect ratio must be positive, got {AR}")
    return (
        4.61 * (1.0 - 0.045 * AR ** 0.68)
        * math.cos(sweep_LE_rad) ** 0.15
        - 3.1
    )


def induced_drag_factor(AR: float, e: float) -> float:
    """Induced drag factor K.

    Raymer Eq 12.48:
        K = 1 / (pi * AR * e)

    This is the coefficient in CDi = K * CL^2.

    Args:
        AR: wing aspect ratio.
        e: Oswald span efficiency factor.

    Returns:
        Induced drag factor K (dimensionless).

    Raises:
        ValueError: if AR or e is non-positive.
    """
    if AR <= 0:
        raise ValueError(f"Aspect ratio must be positive, got {AR}")
    if e <= 0:
        raise ValueError(f"Oswald efficiency must be positive, got {e}")
    return 1.0 / (math.pi * AR * e)


def induced_drag_coeff(CL: float, K: float) -> float:
    """Induced drag coefficient.

    Raymer Eq 12.48:
        CDi = K * CL^2

    Args:
        CL: lift coefficient.
        K: induced drag factor (from :func:`induced_drag_factor`).

    Returns:
        Induced drag coefficient CDi (dimensionless).
    """
    return K * CL ** 2
