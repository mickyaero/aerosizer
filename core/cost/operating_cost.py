"""
Aircraft Operating Cost Estimation
Reference: Raymer Ch 18.5-18.7, "Aircraft Design: A Conceptual Approach", 6th Ed.

Estimates Direct Operating Cost (DOC), crew costs, fuel costs, maintenance
costs, depreciation, insurance, and airline economics metrics such as
cost per available seat-mile (CASM) and CO2 per seat-km.

All inputs in SI units -- conversions happen internally where required
by the Raymer equations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..atmosphere import kg_to_lb, ms_to_kt, km_to_nm


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CO2_PER_KG_FUEL = 3.16      # kg CO2 emitted per kg Jet-A burned
_DEFAULT_LABOR_RATE = 98.0    # $/hr manufacturing labour (2012)
_RESALE_FRACTION = 0.10       # 10 % residual value after depreciation
_AIRFRAME_DEPR_YEARS = 12.0   # straight-line depreciation period (airframe)
_ENGINE_DEPR_YEARS = 4.0      # engine overhaul / depreciation cycle
_INSURANCE_RATE = 0.02         # 2 % of aircraft value per year
_LANDING_FEE_PER_1000KG = 5.0 # $ per 1000 kg landing weight
_GROUND_TIME_HR = 0.25         # 15 min ground manoeuvre (Raymer p. 698)
_AIR_MANOEUVRE_HR = 0.10       # 6 min air manoeuvre allowance
_DOMESTIC_FARE_PER_SM = 0.14   # average domestic fare $/seat-mile
_INTL_FARE_PER_SM = 0.08       # average international fare $/seat-mile


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class OperatingCostInputs:
    """Inputs for operating cost estimation.

    All values in SI units unless noted otherwise.

    Attributes:
        W0_kg:               maximum takeoff weight (kg).
        We_kg:               operating empty weight (kg).
        Wf_mission_kg:       mission fuel burn (kg).
        V_cruise_ms:         cruise true airspeed (m/s).
        range_km:            design range (km).
        n_pax:               passenger count.
        n_engines:           number of engines.
        aircraft_cost:       purchase price of aircraft ($).
        engine_cost_each:    cost per engine ($).
        fuel_price_per_kg:   fuel price ($/kg Jet-A).  ~$0.80/kg is ~$3/gal.
        flight_hours_per_year: annual utilisation (block hours).
        crew_type:           ``"two_man"`` or ``"three_man"`` cockpit.
        CPI_factor:          CPI multiplier from 2012 to present.
        MMH_FH:              maintenance man-hours per flight hour
                             (Raymer Table 18.1; ~10 for large transport).
        labor_rate:          maintenance labour rate ($/hr, 2012).
        fare_type:           ``"domestic"`` or ``"international"`` -- used
                             for break-even load-factor calculation.
    """

    W0_kg: float
    We_kg: float
    Wf_mission_kg: float
    V_cruise_ms: float
    range_km: float
    n_pax: int = 0
    n_engines: int = 2
    aircraft_cost: float = 0.0
    engine_cost_each: float = 0.0
    fuel_price_per_kg: float = 0.80
    flight_hours_per_year: float = 3500.0
    crew_type: str = "two_man"
    CPI_factor: float = 1.35
    MMH_FH: float = 10.0
    labor_rate: float = _DEFAULT_LABOR_RATE
    fare_type: str = "domestic"


@dataclass
class OperatingCostResult:
    """Operating cost estimation results.

    Dollar amounts are CPI-adjusted to the year implied by the input
    ``CPI_factor``.
    """

    # Per block-hour costs ($)
    crew_cost_per_bh: float = 0.0
    fuel_cost_per_bh: float = 0.0
    maintenance_labor_per_bh: float = 0.0
    maintenance_material_per_bh: float = 0.0
    depreciation_per_bh: float = 0.0
    insurance_per_bh: float = 0.0
    landing_fees_per_bh: float = 0.0

    # Annual costs ($)
    annual_fuel: float = 0.0
    annual_crew: float = 0.0
    annual_maintenance: float = 0.0
    annual_depreciation: float = 0.0
    annual_insurance: float = 0.0
    annual_DOC: float = 0.0

    # Per-trip costs ($)
    fuel_per_trip: float = 0.0
    total_per_trip: float = 0.0

    # Airline economics
    DOC_per_seat_mile: float = 0.0
    DOC_per_seat_km: float = 0.0
    CASM: float = 0.0
    fuel_per_seat_km: float = 0.0
    CO2_per_seat_km: float = 0.0

    # Break-even
    breakeven_load_factor: float = 0.0

    # Utilisation
    block_hours_per_year: float = 0.0
    block_time_hr: float = 0.0
    cycles_per_year: float = 0.0

    def summary(self) -> str:
        """Format operating cost results as a readable string."""
        lines = [
            "=" * 64,
            "  OPERATING COST ESTIMATE",
            "=" * 64,
            "",
            "  UTILISATION",
            f"    Block time / trip   : {self.block_time_hr:>10.2f} hr",
            f"    Cycles / year       : {self.cycles_per_year:>10.0f}",
            f"    Block hours / year  : {self.block_hours_per_year:>10.0f}",
            "",
            "  PER BLOCK-HOUR ($)",
            f"    Crew               : ${self.crew_cost_per_bh:>12,.0f}",
            f"    Fuel               : ${self.fuel_cost_per_bh:>12,.0f}",
            f"    Maintenance labour : ${self.maintenance_labor_per_bh:>12,.0f}",
            f"    Maintenance matl   : ${self.maintenance_material_per_bh:>12,.0f}",
            f"    Depreciation       : ${self.depreciation_per_bh:>12,.0f}",
            f"    Insurance          : ${self.insurance_per_bh:>12,.0f}",
            f"    Landing fees       : ${self.landing_fees_per_bh:>12,.0f}",
            "",
            "  ANNUAL COSTS ($)",
            f"    Fuel               : ${self.annual_fuel:>14,.0f}",
            f"    Crew               : ${self.annual_crew:>14,.0f}",
            f"    Maintenance        : ${self.annual_maintenance:>14,.0f}",
            f"    Depreciation       : ${self.annual_depreciation:>14,.0f}",
            f"    Insurance          : ${self.annual_insurance:>14,.0f}",
            f"    Total DOC          : ${self.annual_DOC:>14,.0f}",
            "",
            "  PER-TRIP ($)",
            f"    Fuel               : ${self.fuel_per_trip:>12,.0f}",
            f"    Total              : ${self.total_per_trip:>12,.0f}",
            "",
            "  AIRLINE ECONOMICS",
            f"    DOC / seat-mile    : ${self.DOC_per_seat_mile:>10.4f}",
            f"    DOC / seat-km      : ${self.DOC_per_seat_km:>10.4f}",
            f"    CASM               : ${self.CASM:>10.4f}",
            f"    Fuel / seat-km     : {self.fuel_per_seat_km:>10.4f} kg",
            f"    CO2  / seat-km     : {self.CO2_per_seat_km:>10.4f} kg",
            f"    Break-even LF      : {self.breakeven_load_factor * 100:>10.1f} %",
            "=" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Crew cost CERs (Raymer Eqs 18.10-18.11, mks)
# ---------------------------------------------------------------------------

def _crew_cost_per_block_hour(V_cruise_ms: float, W0_kg: float,
                              crew_type: str, CPI_factor: float) -> float:
    """Estimate crew cost per block hour (CPI-adjusted $).

    Raymer Eqs 18.10-18.11 (mks, 6th Ed.):
        Two-man:   C_crew = 74.5 * (Vc * W0 / 10^5)^0.3 + 168.8   (Eq 18.10)
        Three-man: C_crew = 100  * (Vc * W0 / 10^5)^0.3 + 237.2   (Eq 18.11)

    Where Vc in km/h, W0 in kg.  Result in 2012 $, scaled by CPI.
    """
    V_kmh = V_cruise_ms * 3.6
    x = V_kmh * W0_kg / 1.0e5

    if crew_type == "three_man":
        cost = 100.0 * x ** 0.3 + 237.2          # Eq 18.11
    else:
        cost = 74.5 * x ** 0.3 + 168.8           # Eq 18.10

    return cost * CPI_factor


# ---------------------------------------------------------------------------
# Maintenance material CERs (Raymer Eqs 18.12-18.13, mks)
# ---------------------------------------------------------------------------

def _maintenance_material_per_fh(aircraft_cost: float,
                                 engine_cost_each: float,
                                 n_engines: int,
                                 CPI_factor: float) -> float:
    """Maintenance material cost per flight hour (CPI-adjusted $).

    Raymer Eq 18.12 (mks, 6th Ed.):
        C_mat/FH = 3.3*(Ca/10^6) + 14.2
                   + [58*(Ce/10^6) - 26.1] * Ne

    Where Ca = aircraft cost less engines, Ce = cost per engine.
    All costs in 2012 $.
    """
    Ca = aircraft_cost - engine_cost_each * n_engines
    Ca_M = Ca / 1.0e6
    Ce_M = engine_cost_each / 1.0e6

    cost = 3.3 * Ca_M + 14.2 + (58.0 * Ce_M - 26.1) * n_engines   # Eq 18.12
    return max(cost, 0.0) * CPI_factor


def _maintenance_material_per_cycle(aircraft_cost: float,
                                    engine_cost_each: float,
                                    n_engines: int,
                                    CPI_factor: float) -> float:
    """Maintenance material cost per flight cycle (CPI-adjusted $).

    Raymer Eq 18.13 (mks, 6th Ed.):
        C_mat/cycle = 4.0*(Ca/10^6) + 9.3
                      + [7.5*(Ce/10^6) + 5.6] * Ne
    """
    Ca = aircraft_cost - engine_cost_each * n_engines
    Ca_M = Ca / 1.0e6
    Ce_M = engine_cost_each / 1.0e6

    cost = 4.0 * Ca_M + 9.3 + (7.5 * Ce_M + 5.6) * n_engines      # Eq 18.13
    return max(cost, 0.0) * CPI_factor


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_operating_costs(inputs: OperatingCostInputs) -> OperatingCostResult:
    """Compute aircraft direct operating costs.

    Implements Raymer Ch 18.5-18.7 cost estimating relationships.  The
    computation covers:

    1. **Block time**: mission flight time plus ground (15 min) and air
       manoeuvre (6 min) allowances (Raymer p. 698).
    2. **Crew cost**: Eqs 18.10/18.11 (two- or three-man crew).
    3. **Fuel cost**: mission fuel burn times fuel price.
    4. **Maintenance**: labour (MMH/FH * rate) + material (Eqs 18.12-18.13).
    5. **Depreciation**: airframe over 12 yr (10 % residual), engines over
       4 yr (Raymer p. 700).
    6. **Insurance**: 2 % of aircraft value per year (Raymer p. 701).
    7. **Landing fees**: proportional to landing weight.
    8. **Airline metrics**: DOC/seat-mile, CASM, fuel/seat-km, CO2/seat-km,
       break-even load factor.

    Args:
        inputs: ``OperatingCostInputs`` dataclass.

    Returns:
        ``OperatingCostResult`` with per-block-hour, annual, per-trip,
        and airline economics metrics.
    """
    r = OperatingCostResult()

    # ------------------------------------------------------------------ #
    # Block time and utilisation
    # ------------------------------------------------------------------ #
    range_m = inputs.range_km * 1000.0
    if inputs.V_cruise_ms <= 0:
        raise ValueError("V_cruise_ms must be positive.")

    flight_time_hr = (range_m / inputs.V_cruise_ms) / 3600.0
    r.block_time_hr = flight_time_hr + _GROUND_TIME_HR + _AIR_MANOEUVRE_HR

    r.block_hours_per_year = inputs.flight_hours_per_year
    if r.block_time_hr > 0:
        r.cycles_per_year = inputs.flight_hours_per_year / r.block_time_hr
    else:
        r.cycles_per_year = 0.0

    # ------------------------------------------------------------------ #
    # 1. Crew cost (Raymer Eqs 18.10-18.11)
    # ------------------------------------------------------------------ #
    r.crew_cost_per_bh = _crew_cost_per_block_hour(
        inputs.V_cruise_ms, inputs.W0_kg,
        inputs.crew_type, inputs.CPI_factor,
    )

    # ------------------------------------------------------------------ #
    # 2. Fuel cost
    # ------------------------------------------------------------------ #
    fuel_per_trip = inputs.Wf_mission_kg * inputs.fuel_price_per_kg
    r.fuel_per_trip = fuel_per_trip
    if r.block_time_hr > 0:
        r.fuel_cost_per_bh = fuel_per_trip / r.block_time_hr

    # ------------------------------------------------------------------ #
    # 3. Maintenance
    # ------------------------------------------------------------------ #
    # Labour
    maint_labor_rate = inputs.labor_rate * inputs.CPI_factor
    r.maintenance_labor_per_bh = inputs.MMH_FH * maint_labor_rate

    # Material (per FH + per cycle amortised to per BH)
    mat_per_fh = _maintenance_material_per_fh(
        inputs.aircraft_cost, inputs.engine_cost_each,
        inputs.n_engines, inputs.CPI_factor,
    )
    mat_per_cycle = _maintenance_material_per_cycle(
        inputs.aircraft_cost, inputs.engine_cost_each,
        inputs.n_engines, inputs.CPI_factor,
    )
    # Convert per-cycle cost to per block-hour by dividing by block time
    if r.block_time_hr > 0:
        mat_per_bh = mat_per_fh + mat_per_cycle / r.block_time_hr
    else:
        mat_per_bh = mat_per_fh
    r.maintenance_material_per_bh = mat_per_bh

    # ------------------------------------------------------------------ #
    # 4. Depreciation
    # ------------------------------------------------------------------ #
    # Airframe cost = aircraft cost minus engines
    airframe_cost = inputs.aircraft_cost - inputs.engine_cost_each * inputs.n_engines
    airframe_cost = max(airframe_cost, 0.0)

    annual_airframe_depr = airframe_cost * (1.0 - _RESALE_FRACTION) / _AIRFRAME_DEPR_YEARS
    annual_engine_depr = (inputs.engine_cost_each * inputs.n_engines) / _ENGINE_DEPR_YEARS
    annual_depr = annual_airframe_depr + annual_engine_depr

    if inputs.flight_hours_per_year > 0:
        r.depreciation_per_bh = annual_depr / inputs.flight_hours_per_year

    # ------------------------------------------------------------------ #
    # 5. Insurance
    # ------------------------------------------------------------------ #
    annual_insurance = _INSURANCE_RATE * inputs.aircraft_cost
    if inputs.flight_hours_per_year > 0:
        r.insurance_per_bh = annual_insurance / inputs.flight_hours_per_year

    # ------------------------------------------------------------------ #
    # 6. Landing fees
    # ------------------------------------------------------------------ #
    # Landing weight approximately = W0 - 0.5 * Wf_mission (mid-mission avg
    # is conservative; using full W0 for simplicity is common for fee calcs).
    landing_weight_kg = inputs.W0_kg - 0.5 * inputs.Wf_mission_kg
    fee_per_landing = _LANDING_FEE_PER_1000KG * (landing_weight_kg / 1000.0)
    # Two landings (departure + arrival) per cycle is typical for fees;
    # Raymer uses one landing per cycle.
    if r.block_time_hr > 0:
        r.landing_fees_per_bh = fee_per_landing / r.block_time_hr

    # ------------------------------------------------------------------ #
    # Annual costs
    # ------------------------------------------------------------------ #
    bh = inputs.flight_hours_per_year

    r.annual_fuel = r.fuel_cost_per_bh * bh
    r.annual_crew = r.crew_cost_per_bh * bh
    r.annual_maintenance = (r.maintenance_labor_per_bh + r.maintenance_material_per_bh) * bh
    r.annual_depreciation = annual_depr
    r.annual_insurance = annual_insurance

    r.annual_DOC = (
        r.annual_fuel
        + r.annual_crew
        + r.annual_maintenance
        + r.annual_depreciation
        + r.annual_insurance
        + r.landing_fees_per_bh * bh
    )

    # ------------------------------------------------------------------ #
    # Per-trip total
    # ------------------------------------------------------------------ #
    if r.cycles_per_year > 0:
        r.total_per_trip = r.annual_DOC / r.cycles_per_year

    # ------------------------------------------------------------------ #
    # Airline economics
    # ------------------------------------------------------------------ #
    range_nm = km_to_nm(inputs.range_km)

    if inputs.n_pax > 0 and r.cycles_per_year > 0:
        total_seat_miles = inputs.n_pax * range_nm * r.cycles_per_year
        total_seat_km = inputs.n_pax * inputs.range_km * r.cycles_per_year

        if total_seat_miles > 0:
            r.DOC_per_seat_mile = r.annual_DOC / total_seat_miles
            r.CASM = r.DOC_per_seat_mile

        if total_seat_km > 0:
            r.DOC_per_seat_km = r.annual_DOC / total_seat_km

        # Fuel efficiency
        if inputs.range_km > 0:
            r.fuel_per_seat_km = inputs.Wf_mission_kg / (inputs.n_pax * inputs.range_km)
            r.CO2_per_seat_km = r.fuel_per_seat_km * _CO2_PER_KG_FUEL

    # ------------------------------------------------------------------ #
    # Break-even load factor
    # ------------------------------------------------------------------ #
    if inputs.fare_type == "international":
        fare_per_sm = _INTL_FARE_PER_SM
    else:
        fare_per_sm = _DOMESTIC_FARE_PER_SM

    # Adjust fare for CPI (base fares are roughly 2012 era)
    fare_per_sm *= inputs.CPI_factor

    if fare_per_sm > 0 and r.CASM > 0:
        r.breakeven_load_factor = r.CASM / fare_per_sm
        # Clamp to [0, 1] -- values > 1 mean the route is unprofitable
        r.breakeven_load_factor = min(r.breakeven_load_factor, 1.5)
        r.breakeven_load_factor = max(r.breakeven_load_factor, 0.0)

    return r
