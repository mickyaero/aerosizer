"""
Initial Aircraft Sizing Module
Reference: Raymer, "Aircraft Design: A Conceptual Approach", 6th Ed (2018)
           Chapters 3 and 6

Implements the iterative takeoff gross weight (W0) sizing method:
  W0 = W_crew + W_payload + W_fuel + W_empty

where W_fuel and W_empty are functions of W0 itself, requiring iteration.

The fuel weight is derived from the mission fuel fraction (Mff) computed
via the Breguet range/endurance equations. The empty weight is derived
from statistical regressions of historical aircraft data (Raymer Table 3.1).

All internal calculations use SI units (kg, m, s, N).
The empty weight statistical equation requires W0 in lb; conversion
is handled internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..atmosphere import true_airspeed, kg_to_lb, lb_to_kg
from ..mission import (
    MissionProfile,
    MissionSegment,
    SegmentType,
    AircraftType,
    EMPTY_WEIGHT_FRACTION,
)


# ---------------------------------------------------------------------------
# Typical L/D and SFC values  (Raymer Table 3.3 / Table 3.4)
# ---------------------------------------------------------------------------

_TYPICAL_LD_MAX: dict[AircraftType, float] = {
    AircraftType.JET_TRANSPORT: 17.0,
    AircraftType.MILITARY_CARGO: 17.0,
    AircraftType.BUSINESS_JET: 13.0,
    AircraftType.JET_FIGHTER: 7.5,
    AircraftType.JET_TRAINER: 10.0,
    AircraftType.TURBOPROP: 16.0,
    AircraftType.GA_SINGLE: 12.0,
    AircraftType.GA_TWIN: 12.0,
    AircraftType.FLYING_BOAT: 14.0,
    AircraftType.UAV: 18.0,
}

_TYPICAL_SFC: dict[AircraftType, dict[str, float]] = {
    # SFC values in 1/hr  (Raymer Table 3.3 / Table 3.4)
    AircraftType.JET_TRANSPORT: {"cruise": 0.55, "loiter": 0.45},
    AircraftType.MILITARY_CARGO: {"cruise": 0.55, "loiter": 0.45},
    AircraftType.BUSINESS_JET:   {"cruise": 0.60, "loiter": 0.50},
    AircraftType.JET_FIGHTER:    {"cruise": 0.90, "loiter": 0.80},
    AircraftType.JET_TRAINER:    {"cruise": 0.70, "loiter": 0.60},
    AircraftType.TURBOPROP:      {"cruise": 0.50, "loiter": 0.40},
    AircraftType.GA_SINGLE:      {"cruise": 0.50, "loiter": 0.40},
    AircraftType.GA_TWIN:        {"cruise": 0.50, "loiter": 0.40},
    AircraftType.FLYING_BOAT:    {"cruise": 0.55, "loiter": 0.45},
    AircraftType.UAV:            {"cruise": 0.50, "loiter": 0.40},
}


# ---------------------------------------------------------------------------
# SizingResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SizingResult:
    """Result of the initial weight sizing iteration.

    All weights are in kg.

    Attributes:
        W0_kg: takeoff gross weight
        We_kg: empty weight
        Wf_kg: fuel weight (including reserves)
        W_payload_kg: payload weight
        W_crew_kg: crew weight
        We_fraction: We/W0
        Wf_fraction: Wf/W0
        iterations: number of iterations to converge
        convergence_history: list of W0 guesses (kg) per iteration
        segment_fractions: dict mapping segment label to its Wi/Wi-1
        ld_cruise: L/D used for cruise
        ld_loiter: L/D used for loiter
        sfc_cruise: SFC used for cruise (1/hr)
        sfc_loiter: SFC used for loiter (1/hr)
    """
    W0_kg: float
    We_kg: float
    Wf_kg: float
    W_payload_kg: float
    W_crew_kg: float
    We_fraction: float
    Wf_fraction: float
    iterations: int
    convergence_history: list[float] = field(default_factory=list)
    segment_fractions: dict[str, float] = field(default_factory=dict)
    ld_cruise: float = 0.0
    ld_loiter: float = 0.0
    sfc_cruise: float = 0.0
    sfc_loiter: float = 0.0

    def summary(self) -> str:
        """Return a formatted summary string of the sizing result."""
        lines = [
            "=" * 60,
            "  INITIAL SIZING RESULT",
            "=" * 60,
            f"  Takeoff Gross Weight (W0) : {self.W0_kg:>12,.1f} kg  "
            f"({kg_to_lb(self.W0_kg):>12,.0f} lb)",
            f"  Empty Weight         (We) : {self.We_kg:>12,.1f} kg  "
            f"({kg_to_lb(self.We_kg):>12,.0f} lb)",
            f"  Fuel Weight          (Wf) : {self.Wf_kg:>12,.1f} kg  "
            f"({kg_to_lb(self.Wf_kg):>12,.0f} lb)",
            f"  Payload Weight            : {self.W_payload_kg:>12,.1f} kg  "
            f"({kg_to_lb(self.W_payload_kg):>12,.0f} lb)",
            f"  Crew Weight               : {self.W_crew_kg:>12,.1f} kg  "
            f"({kg_to_lb(self.W_crew_kg):>12,.0f} lb)",
            "-" * 60,
            f"  We/W0 : {self.We_fraction:.4f}",
            f"  Wf/W0 : {self.Wf_fraction:.4f}",
            "-" * 60,
            f"  L/D cruise : {self.ld_cruise:.2f}    L/D loiter : {self.ld_loiter:.2f}",
            f"  SFC cruise : {self.sfc_cruise:.3f} 1/hr  "
            f"SFC loiter : {self.sfc_loiter:.3f} 1/hr",
            "-" * 60,
            f"  Converged in {self.iterations} iterations",
        ]

        if self.segment_fractions:
            lines.append("-" * 60)
            lines.append("  Mission Segment Weight Fractions:")
            for label, wf in self.segment_fractions.items():
                lines.append(f"    {label:<35s} Wi/Wi-1 = {wf:.5f}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def empty_weight_fraction(W0_kg: float, aircraft_type: AircraftType) -> float:
    """Compute the statistical empty weight fraction We/W0.

    Raymer Table 3.1 (Eq. 3.4):
        We/W0 = A * W0_lb^C

    where W0 must be in pounds.

    Args:
        W0_kg: takeoff gross weight in kg
        aircraft_type: type of aircraft (selects A, C coefficients)

    Returns:
        We/W0 (dimensionless fraction, typically 0.4-0.6)
    """
    if aircraft_type not in EMPTY_WEIGHT_FRACTION:
        raise ValueError(
            f"No empty weight coefficients for aircraft type: {aircraft_type}"
        )

    coeffs = EMPTY_WEIGHT_FRACTION[aircraft_type]
    A = coeffs["A"]
    C = coeffs["C"]

    W0_lb = kg_to_lb(W0_kg)
    return A * W0_lb ** C


def cruise_weight_fraction(
    range_km: float,
    sfc: float,
    velocity_ms: float,
    ld_ratio: float,
) -> float:
    """Compute the cruise segment weight fraction using the Breguet range equation.

    Raymer Eq 6.11:
        Wi/Wi-1 = exp( -R * C / (V * L/D) )

    where R is range (m), C is SFC (1/s), V is velocity (m/s),
    and L/D is the lift-to-drag ratio.

    Args:
        range_km: cruise range in km
        sfc: specific fuel consumption in 1/hr (converted to 1/s internally)
        velocity_ms: cruise true airspeed in m/s
        ld_ratio: cruise lift-to-drag ratio

    Returns:
        Wi/Wi-1 weight fraction for the cruise segment (< 1.0)
    """
    R = range_km * 1000.0               # km -> m
    C = sfc / 3600.0                    # 1/hr -> 1/s
    exponent = -R * C / (velocity_ms * ld_ratio)
    return math.exp(exponent)


def loiter_weight_fraction(
    endurance_hr: float,
    sfc: float,
    ld_ratio: float,
) -> float:
    """Compute the loiter segment weight fraction using the Breguet endurance equation.

    Raymer Eq 6.13:
        Wi/Wi-1 = exp( -E * C / (L/D) )

    where E is endurance (s), C is SFC (1/s), and L/D is the
    lift-to-drag ratio.

    Args:
        endurance_hr: loiter endurance in hours
        sfc: specific fuel consumption in 1/hr (converted to 1/s internally)
        ld_ratio: loiter lift-to-drag ratio

    Returns:
        Wi/Wi-1 weight fraction for the loiter segment (< 1.0)
    """
    E = endurance_hr * 3600.0           # hr -> s
    C = sfc / 3600.0                    # 1/hr -> 1/s
    exponent = -E * C / ld_ratio
    return math.exp(exponent)


def estimate_ld_max(aircraft_type: AircraftType) -> float:
    """Return a typical maximum lift-to-drag ratio for the aircraft type.

    Raymer Table 3.3: Typical L/D_max values for conceptual design.
        Jet transport    : 15-18  (use 17)
        Business jet     : 12-14  (use 13)
        Military cargo   : 15-18  (use 17)
        Jet fighter      : 6-9    (use 7.5)
        Jet trainer      : 8-12   (use 10)
        Turboprop        : 14-18  (use 16)
        GA single        : 10-14  (use 12)
        GA twin          : 10-14  (use 12)
        Flying boat      : 12-16  (use 14)
        UAV              : 16-20  (use 18)

    For cruise, use L/D_max. For jet loiter, L/D_max is also
    near-optimal (unlike props where loiter L/D differs).

    Args:
        aircraft_type: type of aircraft

    Returns:
        estimated maximum L/D (dimensionless)
    """
    if aircraft_type not in _TYPICAL_LD_MAX:
        raise ValueError(
            f"No typical L/D data for aircraft type: {aircraft_type}"
        )
    return _TYPICAL_LD_MAX[aircraft_type]


def estimate_sfc(
    aircraft_type: AircraftType,
    flight_phase: str = "cruise",
) -> float:
    """Return a typical specific fuel consumption for the aircraft type and phase.

    Raymer Table 3.3 / Table 3.4. Values in 1/hr (thrust-specific fuel
    consumption for jets).

    Typical SFC values:
        Jet transport cruise : 0.5-0.6 1/hr  (use 0.55)
        Jet transport loiter : 0.4-0.5 1/hr  (use 0.45)
        Business jet cruise  : 0.5-0.7 1/hr  (use 0.60)
        Jet fighter cruise   : 0.8-1.0 1/hr  (use 0.90)

    Args:
        aircraft_type: type of aircraft
        flight_phase: "cruise" or "loiter"

    Returns:
        SFC in 1/hr
    """
    if aircraft_type not in _TYPICAL_SFC:
        raise ValueError(
            f"No typical SFC data for aircraft type: {aircraft_type}"
        )

    phase = flight_phase.lower()
    if phase not in ("cruise", "loiter"):
        raise ValueError(
            f"flight_phase must be 'cruise' or 'loiter', got: {flight_phase!r}"
        )

    return _TYPICAL_SFC[aircraft_type][phase]


def compute_fuel_fraction(
    mission: MissionProfile,
    ld_cruise: float,
    ld_loiter: float,
    sfc_cruise: float,
    sfc_loiter: float,
) -> dict:
    """Walk the mission profile and compute the overall mission fuel fraction.

    For each segment:
    - If the segment has a fixed weight_fraction, use it directly.
    - If the segment is CRUISE with range_km, compute via Breguet range (Eq 6.11).
    - If the segment is LOITER with endurance_hr, compute via Breguet endurance (Eq 6.13).
    - Otherwise fall back to the segment's weight_fraction (set from defaults).

    The overall mission fuel fraction is:
        Mff = product of all Wi/Wi-1

    The fuel weight fraction including reserves is:
        Wf/W0 = (1 + reserve_fraction) * (1 - Mff)

    Args:
        mission: the MissionProfile
        ld_cruise: L/D to use for cruise segments
        ld_loiter: L/D to use for loiter segments
        sfc_cruise: SFC for cruise in 1/hr
        sfc_loiter: SFC for loiter in 1/hr

    Returns:
        dict with keys:
            Mff: overall mission fuel fraction (product of all segment fractions)
            Wf_over_W0: fuel weight fraction including reserves
            segment_fractions: dict mapping segment label to Wi/Wi-1
    """
    segment_fractions: dict[str, float] = {}
    Mff = 1.0

    for seg in mission.segments:
        wf: Optional[float] = None

        if seg.segment_type == SegmentType.CRUISE and seg.range_km is not None:
            # Compute cruise velocity from Mach + altitude, or use speed_ms
            if seg.speed_ms is not None:
                V = seg.speed_ms
            elif seg.mach is not None:
                V = true_airspeed(seg.mach, seg.altitude_m)
            else:
                raise ValueError(
                    f"Cruise segment '{seg.label}' needs either mach or speed_ms"
                )

            ld = seg.ld_ratio if seg.ld_ratio is not None else ld_cruise
            sfc = seg.sfc if seg.sfc is not None else sfc_cruise
            wf = cruise_weight_fraction(seg.range_km, sfc, V, ld)

        elif seg.segment_type == SegmentType.LOITER and seg.endurance_hr is not None:
            ld = seg.ld_ratio if seg.ld_ratio is not None else ld_loiter
            sfc = seg.sfc if seg.sfc is not None else sfc_loiter
            wf = loiter_weight_fraction(seg.endurance_hr, sfc, ld)

        elif seg.weight_fraction is not None:
            wf = seg.weight_fraction

        else:
            raise ValueError(
                f"Segment '{seg.label}' ({seg.segment_type.value}) has no "
                f"weight_fraction, range_km, or endurance_hr defined"
            )

        segment_fractions[seg.label] = wf
        Mff *= wf

    reserve = mission.fuel_reserve_fraction + mission.trapped_fuel_fraction
    Wf_over_W0 = (1.0 + reserve) * (1.0 - Mff)

    return {
        "Mff": Mff,
        "Wf_over_W0": Wf_over_W0,
        "segment_fractions": segment_fractions,
    }


def size_aircraft(
    mission: MissionProfile,
    ld_cruise: Optional[float] = None,
    ld_loiter: Optional[float] = None,
    sfc_cruise: Optional[float] = None,
    sfc_loiter: Optional[float] = None,
    max_iterations: int = 100,
    tolerance: float = 0.5,
) -> SizingResult:
    """Iteratively solve for takeoff gross weight W0.

    This is the core sizing loop from Raymer Ch 3 / Ch 6:

    1. Guess W0.
    2. Compute We/W0 from the statistical equation (Raymer Table 3.1).
    3. Compute Wf/W0 from the mission fuel fraction (Breguet equations).
    4. Calculate W0_new = (W_crew + W_payload) / (1 - We/W0 - Wf/W0).
    5. If |W0_new - W0_guess| < tolerance, converged.
    6. Otherwise average: W0_guess = (W0_new + W0_guess) / 2.

    Default aerodynamic and propulsion parameters are filled from
    statistical data if not supplied.

    Args:
        mission: complete mission profile
        ld_cruise: cruise L/D ratio (default: estimated from aircraft type)
        ld_loiter: loiter L/D ratio (default: estimated from aircraft type)
        sfc_cruise: cruise SFC in 1/hr (default: estimated from aircraft type)
        sfc_loiter: loiter SFC in 1/hr (default: estimated from aircraft type)
        max_iterations: maximum number of iterations
        tolerance: convergence tolerance in kg

    Returns:
        SizingResult dataclass with all sizing outputs

    Raises:
        ValueError: if the mission is infeasible (We/W0 + Wf/W0 >= 1)
        RuntimeError: if the iteration does not converge
    """
    ac_type = mission.aircraft_type

    # Fill defaults from statistical data
    if ld_cruise is None:
        ld_cruise = estimate_ld_max(ac_type)
    if ld_loiter is None:
        # For jets, loiter L/D is near L/D_max; for widebodies, slightly
        # higher due to lower speed.  Use L/D_max + 1 as a simple estimate.
        ld_loiter = estimate_ld_max(ac_type) + 1.0
    if sfc_cruise is None:
        sfc_cruise = estimate_sfc(ac_type, "cruise")
    if sfc_loiter is None:
        sfc_loiter = estimate_sfc(ac_type, "loiter")

    W_payload = mission.total_payload_kg
    W_crew = mission.crew_weight_kg
    W_fixed = W_payload + W_crew

    if W_fixed <= 0.0:
        raise ValueError(
            "Mission must have positive crew + payload weight for sizing"
        )

    # Compute fuel fraction (independent of W0)
    fuel_result = compute_fuel_fraction(
        mission, ld_cruise, ld_loiter, sfc_cruise, sfc_loiter
    )
    Wf_over_W0 = fuel_result["Wf_over_W0"]

    # Initial guess: W0 ~ W_fixed / (1 - 0.50 - Wf/W0)
    # Use We/W0 ~ 0.50 as a starting point
    initial_we_frac = 0.50
    denom = 1.0 - initial_we_frac - Wf_over_W0
    if denom <= 0.0:
        # Very fuel-heavy mission; try a more generous guess
        denom = 0.05
    W0_guess = W_fixed / denom

    convergence_history: list[float] = []

    for iteration in range(1, max_iterations + 1):
        convergence_history.append(W0_guess)

        We_frac = empty_weight_fraction(W0_guess, ac_type)

        available = 1.0 - We_frac - Wf_over_W0
        if available <= 0.0:
            raise ValueError(
                f"Infeasible design: We/W0 ({We_frac:.4f}) + "
                f"Wf/W0 ({Wf_over_W0:.4f}) = {We_frac + Wf_over_W0:.4f} >= 1.0. "
                f"The mission requires more fuel and empty weight than the "
                f"aircraft can carry. Try reducing range, payload, or "
                f"improving L/D or SFC."
            )

        W0_calc = W_fixed / available

        if abs(W0_calc - W0_guess) < tolerance:
            # Converged
            We_kg = We_frac * W0_calc
            Wf_kg = Wf_over_W0 * W0_calc
            convergence_history.append(W0_calc)

            return SizingResult(
                W0_kg=W0_calc,
                We_kg=We_kg,
                Wf_kg=Wf_kg,
                W_payload_kg=W_payload,
                W_crew_kg=W_crew,
                We_fraction=We_frac,
                Wf_fraction=Wf_over_W0,
                iterations=iteration,
                convergence_history=convergence_history,
                segment_fractions=fuel_result["segment_fractions"],
                ld_cruise=ld_cruise,
                ld_loiter=ld_loiter,
                sfc_cruise=sfc_cruise,
                sfc_loiter=sfc_loiter,
            )

        # Damped update: average of guess and calculation
        W0_guess = (W0_calc + W0_guess) / 2.0

    raise RuntimeError(
        f"Sizing did not converge after {max_iterations} iterations. "
        f"Last W0 guess: {W0_guess:.1f} kg, last W0 calc: {W0_calc:.1f} kg, "
        f"delta: {abs(W0_calc - W0_guess):.1f} kg"
    )
