"""
Constraint Analysis -- T/W vs W/S Diagram
Reference: Raymer, "Aircraft Design: A Conceptual Approach", 6th Edition
           Chapter 5 (Thrust-to-Weight and Wing Loading), Ch 17 (Performance)

Produces the classic constraint (carpet) plot that identifies the feasible
design space and optimal design point (minimum T/W that satisfies all
performance requirements).

All inputs are SI.  Methods return T/W (dimensionless) and W/S in Pa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from ..atmosphere import (
    atmosphere,
    G0,
    RHO0,
    P0,
    m_to_ft,
    ft_to_m,
    m2_to_ft2,
    ft2_to_m2,
    true_airspeed,
)


@dataclass
class ConstraintAnalysis:
    """Raymer Ch 5 constraint analysis for jet transport aircraft.

    All aerodynamic and performance parameters needed to compute
    the T/W vs W/S constraint lines.

    Attributes:
        CD0               : zero-lift drag coefficient
        K                 : induced drag factor (1 / pi*e*AR)
        CLmax_clean       : max lift coefficient, clean configuration
        CLmax_takeoff     : max CL with takeoff flaps
        CLmax_landing     : max CL with landing flaps
        cruise_mach       : design cruise Mach number
        cruise_altitude_m : design cruise altitude [m]
        takeoff_distance_m: balanced field length for takeoff [m]
        landing_distance_m: landing field length [m]
        climb_gradient_oei: required OEI climb gradient (FAR 25 2nd segment)
        service_ceiling_m : service ceiling altitude [m]
        n_engines         : number of engines
        thrust_lapse_cruise: T_cruise / T_SL ratio at cruise condition
        bypass_ratio      : engine bypass ratio (for lapse estimation)
    """

    # Aerodynamic parameters
    CD0: float = 0.020
    K: float = 0.04
    CLmax_clean: float = 1.6
    CLmax_takeoff: float = 2.0
    CLmax_landing: float = 2.4

    # Requirements
    cruise_mach: float = 0.85
    cruise_altitude_m: float = 10668.0   # 35,000 ft
    takeoff_distance_m: float = 2500.0   # balanced field length
    landing_distance_m: float = 1800.0
    climb_gradient_oei: float = 0.024    # FAR 25 2nd segment: 2.4% for 2-engine
    service_ceiling_m: float = 12192.0   # 40,000 ft
    n_engines: int = 2

    # Propulsion
    thrust_lapse_cruise: float = 0.25
    bypass_ratio: float = 8.0

    # Weight fractions
    _beta_cruise: float = 0.95   # W_cruise / W_TO
    _beta_landing: float = 0.85  # W_landing / W_TO

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #

    def _cruise_conditions(self) -> dict:
        """Return cruise atmosphere, velocity, and dynamic pressure."""
        atm = atmosphere(self.cruise_altitude_m)
        V = true_airspeed(self.cruise_mach, self.cruise_altitude_m)
        q = 0.5 * atm["rho"] * V ** 2
        return {"atm": atm, "V": V, "q": q, "rho": atm["rho"], "sigma": atm["sigma"]}

    def _ceiling_conditions(self) -> dict:
        """Return atmosphere at service ceiling."""
        atm = atmosphere(self.service_ceiling_m)
        return {"atm": atm, "rho": atm["rho"], "sigma": atm["sigma"]}

    def _thrust_lapse_at(self, altitude_m: float, mach: float = 0.0) -> float:
        """Estimate thrust lapse ratio alpha = T_avail / T_SL.

        Uses a simple density-ratio model:
            alpha ~ sigma^0.6  (typical high-BPR turbofan)

        If a specific cruise lapse was provided and we are at cruise
        conditions, use that directly.
        """
        atm = atmosphere(altitude_m)
        sigma = atm["sigma"]
        # Simple lapse model: alpha = sigma^n, n ~ 0.6 for high-BPR
        alpha = sigma ** 0.6
        return alpha

    def _max_ld(self) -> float:
        """Maximum lift-to-drag ratio from parabolic polar.

        (L/D)_max = 1 / (2 * sqrt(CD0 * K))   -- Raymer Eq 3.9
        """
        return 1.0 / (2.0 * math.sqrt(self.CD0 * self.K))

    def _cl_for_max_ld(self) -> float:
        """CL at maximum L/D.

        CL* = sqrt(CD0 / K)
        """
        return math.sqrt(self.CD0 / self.K)

    def _default_ws_range(self) -> np.ndarray:
        """Default W/S range in Pa (roughly 100-800 kg/m^2 * g)."""
        return np.linspace(1000.0, 9000.0, 500)

    # --------------------------------------------------------------------- #
    #  Constraint methods                                                    #
    # --------------------------------------------------------------------- #

    def cruise_constraint(self, ws_array: np.ndarray) -> np.ndarray:
        """Cruise speed constraint -- Raymer Eq 5.6.

        At cruise, thrust must equal drag:
            T = D = q * S * (CD0 + K * CL^2)
        where CL = (beta * W/S) / q.

        Converting to T/W at sea-level (takeoff):
            T_SL/W_TO = [q*CD0 / (beta * W/S) + K * beta * (W/S) / q] / alpha

        Args:
            ws_array: wing loading array W/S [Pa]

        Returns:
            T/W array (sea-level, takeoff conditions)
        """
        cc = self._cruise_conditions()
        q = cc["q"]
        alpha = self.thrust_lapse_cruise  # use user-specified cruise lapse
        beta = self._beta_cruise

        tw = (
            q * self.CD0 / (beta * ws_array)
            + self.K * beta * ws_array / q
        ) / alpha

        return tw

    def takeoff_constraint(self, ws_array: np.ndarray) -> np.ndarray:
        """Takeoff field length constraint -- Raymer Ch 5 / Ch 17.

        For jet transports, the balanced field length (BFL) relates to
        the takeoff parameter (TOP):
            TOP = BFL / 37.5    (BFL in ft, Raymer Fig 5.4)

        And TOP = (W/S) / (sigma * CLto * (T/W))

        Solving for T/W:
            T/W = (W/S) / (sigma * CLto * TOP)

        At sea level, sigma = 1.0.

        Args:
            ws_array: wing loading array W/S [Pa]

        Returns:
            T/W array (sea-level static)
        """
        BFL_ft = m_to_ft(self.takeoff_distance_m)
        TOP = BFL_ft / 37.5  # takeoff parameter (Raymer Fig 5.4 for jets)

        # Convert W/S from Pa to psf for the TOP formula
        ws_psf = ws_array * 0.020885434  # 1 Pa = 0.020885 psf

        # sigma at sea level = 1.0 for standard takeoff
        sigma = 1.0

        tw = ws_psf / (sigma * self.CLmax_takeoff * TOP)

        return tw

    def landing_constraint(self) -> float:
        """Landing field length constraint -- Raymer Eq 5.11 / Ch 17.

        The approach speed is:
            Va = 1.3 * Vstall = 1.3 * sqrt(2 * W_L / (rho * S * CLmax_L))

        Landing distance:
            Sl = (Va^2) / (2 * g * 0.3)   (approx, FAR 25 with ~0.3g decel)

        Or using Raymer's simplified method (Fig 5.5):
            W/S|_landing = (Sl / 80) * sigma * CLmax_L * rho0 * g   [Imperial]

        Converting for landing weight fraction:
            W/S|_TO = (W/S|_landing) / beta_landing

        Returns:
            Maximum allowable takeoff W/S [Pa]
        """
        # Raymer method: Sa = Sl * 0.6 (for FAR 25, landing distance includes
        # 1/0.6 factor over air distance).
        # Approach speed: Va = sqrt(Sl / 0.3) * ... simplified.
        #
        # Direct method from stall speed:
        # Sl = 80 * (W/S)_landing / (sigma * CLmax_L)  [Imperial, psf, ft]
        # => (W/S)_landing [psf] = Sl [ft] * sigma * CLmax_L / 80

        Sl_ft = m_to_ft(self.landing_distance_m)
        sigma = 1.0  # sea-level landing (conservative)

        ws_landing_psf = Sl_ft * sigma * self.CLmax_landing / 80.0

        # Convert psf to Pa: 1 psf = 47.8803 Pa
        ws_landing_pa = ws_landing_psf / 0.020885434

        # Adjust from landing to takeoff wing loading
        ws_takeoff_pa = ws_landing_pa / self._beta_landing

        return ws_takeoff_pa

    def climb_gradient_constraint(self, ws_array: np.ndarray) -> np.ndarray:
        """OEI climb gradient constraint -- FAR 25 Second Segment.

        Reference: Raymer Eq 5.7, FAR 25.121(b).

        With one engine inoperative, the aircraft must maintain a minimum
        climb gradient G (e.g., 2.4% for twin-engine).

        The required T/W (all engines, sea-level) is:
            T/W = (N / (N-1)) * (G + 1 / (L/D)_climb)

        where N = number of engines.
        L/D at climb is evaluated at the takeoff CL (near CLmax_TO / 1.44
        for V2 = 1.2 * Vstall, so CL_climb ~ CLmax_TO / 1.44).

        This constraint is independent of W/S (horizontal line), but we
        evaluate it at each W/S for consistency.  The L/D depends weakly
        on CL which depends on W/S, so we compute it properly.

        Args:
            ws_array: wing loading array W/S [Pa]

        Returns:
            T/W array (sea-level, all engines)
        """
        N = self.n_engines
        G = self.climb_gradient_oei

        # At V2 = 1.2 * Vstall, CL = CLmax_TO / 1.44
        CL_climb = self.CLmax_takeoff / 1.44
        CD_climb = self.CD0 + self.K * CL_climb ** 2
        # Additional OEI drag: windmilling + asymmetric ~ 0.005 delta-CD0
        CD_climb += 0.005
        LD_climb = CL_climb / CD_climb

        tw = (N / (N - 1.0)) * (G + 1.0 / LD_climb)

        # Return as constant array (same shape as ws_array)
        return np.full_like(ws_array, tw, dtype=float)

    def ceiling_constraint(self, ws_array: np.ndarray) -> np.ndarray:
        """Service ceiling constraint -- Raymer Ch 5, Eq 5.8.

        At service ceiling, the rate of climb = 100 ft/min (0.508 m/s).

        T = D + W * (ROC / V)

        => T/W = q*CD0/(W/S) + K*(W/S)/q + ROC/V

        Adjusted to sea-level:
            T_SL/W_TO = [q*CD0/(beta*W/S) + K*beta*(W/S)/q + ROC/V] / alpha

        where alpha is thrust lapse at ceiling, beta ~ 0.95 (still near TOC weight).

        Args:
            ws_array: wing loading array W/S [Pa]

        Returns:
            T/W array (sea-level, takeoff conditions)
        """
        ROC = 0.508  # 100 ft/min in m/s

        cc = self._ceiling_conditions()
        rho = cc["rho"]
        alpha = self._thrust_lapse_at(self.service_ceiling_m)
        beta = self._beta_cruise  # near top of climb, similar to cruise weight

        # Fly at speed for best ROC ~ speed for minimum drag
        # CL at min drag: CL* = sqrt(CD0/K)
        CL_star = self._cl_for_max_ld()

        # Velocity from CL: V = sqrt(2 * beta * (W/S) / (rho * CL))
        # Since V depends on W/S, compute for each element
        V = np.sqrt(2.0 * beta * ws_array / (rho * CL_star))
        q = 0.5 * rho * V ** 2

        tw = (
            q * self.CD0 / (beta * ws_array)
            + self.K * beta * ws_array / q
            + ROC / V
        ) / alpha

        return tw

    def stall_constraint(self, stall_speed_ms: Optional[float] = None) -> Optional[float]:
        """Stall speed constraint on W/S.

        W/S_max = 0.5 * rho_0 * Vs^2 * CLmax_clean

        Args:
            stall_speed_ms: stall speed requirement [m/s].
                            If None, no constraint is applied.

        Returns:
            Maximum W/S [Pa] from stall, or None if no stall speed specified.
        """
        if stall_speed_ms is None:
            return None
        return 0.5 * RHO0 * stall_speed_ms ** 2 * self.CLmax_clean

    # --------------------------------------------------------------------- #
    #  Aggregate methods                                                     #
    # --------------------------------------------------------------------- #

    def compute_all(
        self,
        ws_range: Optional[np.ndarray] = None,
        stall_speed_ms: Optional[float] = None,
    ) -> Dict:
        """Compute all constraint lines.

        Args:
            ws_range: W/S array [Pa].  If None, a default range is used.
            stall_speed_ms: stall speed [m/s] for stall constraint (optional).

        Returns:
            dict with keys:
                "ws"           : the W/S array [Pa]
                "cruise"       : {"ws": arr, "tw": arr}
                "takeoff"      : {"ws": arr, "tw": arr}
                "climb"        : {"ws": arr, "tw": arr}
                "ceiling"      : {"ws": arr, "tw": arr}
                "landing_ws_max" : float, max W/S from landing [Pa]
                "stall_ws_max"   : float or None
        """
        if ws_range is None:
            ws_range = self._default_ws_range()

        ws = ws_range.copy()

        result = {
            "ws": ws,
            "cruise": {"ws": ws, "tw": self.cruise_constraint(ws)},
            "takeoff": {"ws": ws, "tw": self.takeoff_constraint(ws)},
            "climb": {"ws": ws, "tw": self.climb_gradient_constraint(ws)},
            "ceiling": {"ws": ws, "tw": self.ceiling_constraint(ws)},
            "landing_ws_max": self.landing_constraint(),
            "stall_ws_max": self.stall_constraint(stall_speed_ms),
        }

        return result

    def find_design_point(
        self,
        ws_range: Optional[np.ndarray] = None,
        stall_speed_ms: Optional[float] = None,
    ) -> Tuple[float, float]:
        """Find the optimal design point (minimum T/W satisfying all constraints).

        The design point sits at the intersection of the most critical
        constraints, giving the lowest T/W within the feasible region.

        Args:
            ws_range: W/S array [Pa].  If None, a default range is used.
            stall_speed_ms: optional stall speed requirement [m/s].

        Returns:
            (W/S_opt [Pa], T/W_opt [dimensionless])
        """
        if ws_range is None:
            ws_range = self._default_ws_range()

        data = self.compute_all(ws_range, stall_speed_ms)
        ws = data["ws"]

        # For each W/S, find the maximum T/W required across all constraints
        tw_cruise = data["cruise"]["tw"]
        tw_takeoff = data["takeoff"]["tw"]
        tw_climb = data["climb"]["tw"]
        tw_ceiling = data["ceiling"]["tw"]

        # Stack all T/W constraint lines and take the envelope (max at each W/S)
        tw_envelope = np.maximum.reduce([tw_cruise, tw_takeoff, tw_climb, tw_ceiling])

        # Apply W/S upper-bound constraints (landing, stall)
        ws_max = data["landing_ws_max"]
        if data["stall_ws_max"] is not None:
            ws_max = min(ws_max, data["stall_ws_max"])

        # Mask out W/S values beyond the maximum
        valid = ws <= ws_max

        if not np.any(valid):
            # All points exceed the W/S limit; return at the limit
            ws_opt = ws_max
            # Evaluate T/W at ws_max
            ws_single = np.array([ws_max])
            tw_at_limit = max(
                self.cruise_constraint(ws_single)[0],
                self.takeoff_constraint(ws_single)[0],
                self.climb_gradient_constraint(ws_single)[0],
                self.ceiling_constraint(ws_single)[0],
            )
            return (ws_opt, float(tw_at_limit))

        tw_feasible = tw_envelope.copy()
        tw_feasible[~valid] = np.inf

        # Find minimum T/W in the feasible region
        idx = np.argmin(tw_feasible)
        ws_opt = float(ws[idx])
        tw_opt = float(tw_feasible[idx])

        return (ws_opt, tw_opt)

    def plot_data(
        self,
        ws_range: Optional[np.ndarray] = None,
        stall_speed_ms: Optional[float] = None,
    ) -> Dict:
        """Return all data needed for Plotly visualisation.

        Args:
            ws_range: W/S array [Pa].  If None, a default range is used.
            stall_speed_ms: optional stall speed [m/s].

        Returns:
            dict with:
                "traces": list of dicts, each with
                    "name", "ws" (array), "tw" (array), "type" ("line" or "vline")
                "design_point": {"ws": float, "tw": float}
                "landing_ws_max": float
                "stall_ws_max": float or None
                "feasible_region": {"ws": arr, "tw_lower": arr, "tw_upper": float}
        """
        if ws_range is None:
            ws_range = self._default_ws_range()

        data = self.compute_all(ws_range, stall_speed_ms)
        ws_opt, tw_opt = self.find_design_point(ws_range, stall_speed_ms)

        traces = [
            {
                "name": "Cruise",
                "ws": data["cruise"]["ws"].tolist(),
                "tw": data["cruise"]["tw"].tolist(),
                "type": "line",
            },
            {
                "name": "Takeoff (BFL)",
                "ws": data["takeoff"]["ws"].tolist(),
                "tw": data["takeoff"]["tw"].tolist(),
                "type": "line",
            },
            {
                "name": "OEI Climb (FAR 25)",
                "ws": data["climb"]["ws"].tolist(),
                "tw": data["climb"]["tw"].tolist(),
                "type": "line",
            },
            {
                "name": "Service Ceiling",
                "ws": data["ceiling"]["ws"].tolist(),
                "tw": data["ceiling"]["tw"].tolist(),
                "type": "line",
            },
            {
                "name": "Landing (max W/S)",
                "ws": data["landing_ws_max"],
                "tw": None,
                "type": "vline",
            },
        ]

        if data["stall_ws_max"] is not None:
            traces.append({
                "name": "Stall (max W/S)",
                "ws": data["stall_ws_max"],
                "tw": None,
                "type": "vline",
            })

        result = {
            "traces": traces,
            "design_point": {"ws": ws_opt, "tw": tw_opt},
            "landing_ws_max": data["landing_ws_max"],
            "stall_ws_max": data["stall_ws_max"],
        }

        return result

    def summary(self) -> str:
        """Return a formatted summary of the constraint analysis."""
        ws_opt, tw_opt = self.find_design_point()
        landing_ws = self.landing_constraint()

        lines = []
        lines.append("=" * 55)
        lines.append(f"{'CONSTRAINT ANALYSIS SUMMARY':^55}")
        lines.append(f"{'(Raymer Ch 5 T/W vs W/S)':^55}")
        lines.append("=" * 55)
        lines.append("")
        lines.append("INPUTS:")
        lines.append(f"  CD0              = {self.CD0:.4f}")
        lines.append(f"  K                = {self.K:.4f}")
        lines.append(f"  CLmax (clean)    = {self.CLmax_clean:.2f}")
        lines.append(f"  CLmax (TO)       = {self.CLmax_takeoff:.2f}")
        lines.append(f"  CLmax (landing)  = {self.CLmax_landing:.2f}")
        lines.append(f"  Cruise Mach      = {self.cruise_mach:.3f}")
        lines.append(f"  Cruise alt       = {self.cruise_altitude_m:.0f} m ({m_to_ft(self.cruise_altitude_m):.0f} ft)")
        lines.append(f"  BFL (TO)         = {self.takeoff_distance_m:.0f} m ({m_to_ft(self.takeoff_distance_m):.0f} ft)")
        lines.append(f"  Landing dist     = {self.landing_distance_m:.0f} m ({m_to_ft(self.landing_distance_m):.0f} ft)")
        lines.append(f"  OEI gradient     = {self.climb_gradient_oei:.3f}")
        lines.append(f"  Service ceiling  = {self.service_ceiling_m:.0f} m ({m_to_ft(self.service_ceiling_m):.0f} ft)")
        lines.append(f"  N engines        = {self.n_engines}")
        lines.append(f"  Thrust lapse     = {self.thrust_lapse_cruise:.3f}")
        lines.append("")
        lines.append("RESULTS:")
        lines.append(f"  Landing W/S max  = {landing_ws:.0f} Pa ({landing_ws / G0:.0f} kg/m2)")
        lines.append(f"  Design W/S       = {ws_opt:.0f} Pa ({ws_opt / G0:.0f} kg/m2)")
        lines.append(f"  Design T/W       = {tw_opt:.4f}")
        lines.append(f"  Max L/D          = {self._max_ld():.1f}")
        lines.append("=" * 55)

        return "\n".join(lines)
