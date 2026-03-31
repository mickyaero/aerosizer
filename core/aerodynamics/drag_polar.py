"""
Complete Drag Polar Model
Reference: Raymer Ch 12, Ch 17, "Aircraft Design: A Conceptual Approach", 6th Edition

The drag polar relates CL to CD via the classic parabolic approximation:
    CD = CD0 + K * CL^2

This module provides the DragPolar class for clean, takeoff, and landing
configurations, plus optimum-CL calculations for max L/D and min power.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .parasite_drag import parasite_drag_buildup, leakage_protuberance_drag
from .induced_drag import (
    oswald_efficiency_swept,
    induced_drag_factor,
)


@dataclass
class DragPolar:
    """Parabolic drag polar for an aircraft configuration.

    Attributes:
        CD0: zero-lift (parasite) drag coefficient.
        K: induced drag factor (1 / pi*AR*e).
        CD0_takeoff_increment: additional CD0 for takeoff config (flaps + gear).
            Raymer Table 12.2 suggests ~0.015-0.025 total.
        CD0_landing_increment: additional CD0 for landing config (full flaps + gear).
            Raymer Table 12.2 suggests ~0.065-0.075 total.
        K_takeoff: induced drag factor in takeoff config. Flaps reduce effective
            e slightly. Defaults to K * 1.0 (no change) if not specified.
        K_landing: induced drag factor in landing config. Defaults to K * 1.0.
    """
    CD0: float = 0.020
    K: float = 0.04
    CD0_takeoff_increment: float = 0.02
    CD0_landing_increment: float = 0.07
    K_takeoff: Optional[float] = None
    K_landing: Optional[float] = None

    def __post_init__(self):
        if self.K_takeoff is None:
            self.K_takeoff = self.K
        if self.K_landing is None:
            self.K_landing = self.K

    # ------------------------------------------------------------------ #
    # Clean configuration
    # ------------------------------------------------------------------ #

    def cd(self, cl: float) -> float:
        """Total drag coefficient in clean configuration.

        Raymer Eq 12.1:
            CD = CD0 + K * CL^2

        Args:
            cl: lift coefficient.

        Returns:
            Total drag coefficient CD.
        """
        return self.CD0 + self.K * cl ** 2

    # ------------------------------------------------------------------ #
    # Takeoff / landing configurations
    # ------------------------------------------------------------------ #

    def cd_takeoff(self, cl: float) -> float:
        """Drag coefficient in takeoff configuration (partial flaps + gear).

        Raymer Table 12.2:
            CD = (CD0 + dCD0_TO) + K_TO * CL^2

        Args:
            cl: lift coefficient.

        Returns:
            Total drag coefficient in takeoff configuration.
        """
        return (self.CD0 + self.CD0_takeoff_increment) + self.K_takeoff * cl ** 2

    def cd_landing(self, cl: float) -> float:
        """Drag coefficient in landing configuration (full flaps + gear).

        Raymer Table 12.2:
            CD = (CD0 + dCD0_L) + K_L * CL^2

        Args:
            cl: lift coefficient.

        Returns:
            Total drag coefficient in landing configuration.
        """
        return (self.CD0 + self.CD0_landing_increment) + self.K_landing * cl ** 2

    # ------------------------------------------------------------------ #
    # Optimum lift coefficients
    # ------------------------------------------------------------------ #

    def cl_for_max_ld(self) -> float:
        """Lift coefficient for maximum lift-to-drag ratio (clean).

        Raymer Eq 17.14:
            CL_maxL/D = sqrt(CD0 / K)

        This is the CL where parasite drag equals induced drag, yielding
        the minimum total drag coefficient for a given CL.

        Returns:
            CL for max L/D.
        """
        return math.sqrt(self.CD0 / self.K)

    def max_ld(self) -> float:
        """Maximum lift-to-drag ratio (clean configuration).

        Raymer Eq 17.15:
            (L/D)_max = 1 / (2 * sqrt(CD0 * K))

        Returns:
            Maximum L/D ratio.
        """
        return 1.0 / (2.0 * math.sqrt(self.CD0 * self.K))

    def cl_for_max_ld_cruise(self) -> float:
        """Lift coefficient for maximum range (jet cruise).

        For jet aircraft, maximum range occurs at maximum L/D, so this
        is identical to :meth:`cl_for_max_ld`. (Raymer Eq 17.14)

        For propeller aircraft, max range occurs at a different condition
        (max CL/CD is at CL for min drag, but max CL^0.5/CD for Breguet).

        Returns:
            CL for best jet cruise range.
        """
        return self.cl_for_max_ld()

    def cl_for_min_power(self) -> float:
        """Lift coefficient for minimum power required (maximum endurance for jets).

        Raymer Eq 17.28:
            CL_min_power = sqrt(3 * CD0 / K)

        At this CL, induced drag is three times the parasite drag. This
        condition minimizes the power required (drag * velocity), which
        maximizes endurance for jet aircraft or range for propeller aircraft.

        Returns:
            CL for minimum power required.
        """
        return math.sqrt(3.0 * self.CD0 / self.K)

    def ld_at(self, cl: float) -> float:
        """Lift-to-drag ratio at a given CL (clean configuration).

        L/D = CL / CD = CL / (CD0 + K * CL^2)

        Args:
            cl: lift coefficient. Must be > 0 for a meaningful result.

        Returns:
            L/D ratio at the given CL.

        Raises:
            ValueError: if cl is zero (division by zero in CD).
        """
        cd_val = self.cd(cl)
        if cd_val == 0.0:
            raise ValueError("CD is zero; cannot compute L/D.")
        return cl / cd_val

    def generate_polar(
        self,
        cl_range: Optional[np.ndarray] = None,
    ) -> dict:
        """Generate arrays of CL, CD, and L/D over a range of lift coefficients.

        Args:
            cl_range: array of CL values. If None, uses np.linspace(0.0, 1.5, 100).

        Returns:
            dict with:
                CL: numpy array of lift coefficients
                CD: numpy array of drag coefficients
                LD: numpy array of lift-to-drag ratios (L/D)
        """
        if cl_range is None:
            cl_range = np.linspace(0.0, 1.5, 100)

        cl_arr = np.asarray(cl_range, dtype=float)
        cd_arr = self.CD0 + self.K * cl_arr ** 2

        # Avoid division by zero for CL=0
        with np.errstate(divide="ignore", invalid="ignore"):
            ld_arr = np.where(cd_arr > 0, cl_arr / cd_arr, 0.0)

        return {"CL": cl_arr, "CD": cd_arr, "LD": ld_arr}


# --------------------------------------------------------------------------- #
# Factory function
# --------------------------------------------------------------------------- #

def create_transport_polar(
    Sref: float,
    altitude_m: float,
    mach: float,
    components: Optional[list[dict]] = None,
    AR: float = 9.0,
    sweep_LE_rad: float = 0.55,
    CD0_override: Optional[float] = None,
    leakage_fraction: float = 0.03,
) -> DragPolar:
    """Create a drag polar for a transport-category aircraft.

    If *components* are provided, uses the Raymer component buildup method
    (Ch 12) to compute CD0. Otherwise falls back to a typical clean CD0
    value of 0.015-0.020.

    Args:
        Sref: reference wing area (m^2).
        altitude_m: cruise altitude (m, geometric).
        mach: cruise Mach number.
        components: list of component dicts for buildup (see
            :func:`parasite_drag.parasite_drag_buildup`). Optional.
        AR: wing aspect ratio (used for Oswald efficiency). Default 9.0.
        sweep_LE_rad: leading-edge sweep in radians. Default 0.55 (~31.5 deg).
        CD0_override: if provided, skip buildup and use this CD0 directly.
        leakage_fraction: leakage & protuberance fraction. Default 0.03.

    Returns:
        DragPolar instance.
    """
    from ..atmosphere import atmosphere as _atm

    atm = _atm(altitude_m)
    velocity_ms = mach * atm["a"]

    # --- CD0 ---
    if CD0_override is not None:
        CD0 = CD0_override
    elif components is not None:
        result = parasite_drag_buildup(
            components, Sref, altitude_m, velocity_ms
        )
        CD0_clean = result["CD0"]
        CD0 = CD0_clean + leakage_protuberance_drag(CD0_clean, leakage_fraction)
    else:
        # Typical modern transport (Raymer Table 12.3)
        CD0 = 0.018

    # --- Induced drag factor K ---
    e = oswald_efficiency_swept(AR, sweep_LE_rad)
    K = induced_drag_factor(AR, e)

    return DragPolar(CD0=CD0, K=K)
