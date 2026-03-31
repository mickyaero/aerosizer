"""
Modified DAPCA IV Cost Model
Reference: Raymer Ch 18, Eqs 18.1-18.9 (6th Edition, 2018)

Estimates RDT&E and production costs for aircraft programs.
All costs in 2012 US dollars unless adjusted by CPI factor.

The RAND DAPCA IV (Development and Procurement Costs of Aircraft)
model uses statistical cost estimating relationships (CERs) derived
from historical military and commercial aircraft programmes.

All inputs in SI units -- converted internally to the mks form used
by Raymer's equations (kg and km/h).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from ..atmosphere import kg_to_lb, ms_to_kt, n_to_lbf


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DAPCAInputs:
    """Inputs for DAPCA IV cost model.

    All inputs in SI units -- converted internally for the equations.
    Weight is the airframe unit weight (AMPR weight); for transport
    aircraft Raymer (p. 692) recommends AMPR ~= 0.62 * We.

    Attributes:
        We_kg:            empty weight (kg).  Internally multiplied by
                          ``ampr_factor`` to obtain AMPR weight.
        V_max_ms:         maximum velocity (m/s).
        Q:                production quantity (lesser of 5-yr production
                          or total programme).
        FTA:              flight-test aircraft (typically 2-6).
        N_engines:        engines per aircraft.
        T_max_N:          max thrust per engine (N) -- used in engine CER.
        M_max_engine:     engine maximum Mach number.
        T_turbine_inlet_K: turbine inlet temperature (K).
        C_avionics:       avionics cost ($) -- estimated separately.
        n_pax:            passenger count (for interior cost).
        R_E:              engineering wrap rate ($/hr, 2012).
        R_T:              tooling wrap rate ($/hr, 2012).
        R_Q:              quality-control wrap rate ($/hr, 2012).
        R_M:              manufacturing wrap rate ($/hr, 2012).
        cargo_aircraft:   if True QC hours = 0.076 * H_M, else 0.133 * H_M.
        CPI_factor:       consumer-price-index multiplier from 2012 to present.
        material_factor:  1.0 Al, 1.1-1.8 composites, 1.5-2.0 steel/Ti.
        stealth_factor:   1.0 conventional, ~1.2 modern stealth.
        commercial_factor: DAPCA overpredicts for civil; 0.9 typical.
        turbofan_factor:  ~1.175 for turbofan vs turbojet engine cost.
        ampr_factor:      fraction of We used as AMPR weight (default 0.62
                          for transports per Raymer p. 692).
        interior_cost_per_pax: interior cost per passenger (2012 $).
                          Raymer: $3 500 jet transport, $1 700 regional,
                          $850 GA.  Default is $3 500.
    """

    We_kg: float
    V_max_ms: float
    Q: int
    FTA: int = 4
    N_engines: int = 2
    T_max_N: float = 0.0
    M_max_engine: float = 0.0
    T_turbine_inlet_K: float = 0.0
    C_avionics: float = 0.0
    n_pax: int = 0

    # Wrap rates (2012 $)
    R_E: float = 115.0     # engineering $/hr
    R_T: float = 118.0     # tooling $/hr
    R_Q: float = 108.0     # quality control $/hr
    R_M: float = 98.0      # manufacturing $/hr

    # Adjustment factors
    cargo_aircraft: bool = False
    CPI_factor: float = 1.35
    material_factor: float = 1.0
    stealth_factor: float = 1.0
    commercial_factor: float = 0.9
    turbofan_factor: float = 1.175
    ampr_factor: float = 0.62
    interior_cost_per_pax: float = 3500.0


@dataclass
class DAPCACostResult:
    """DAPCA IV cost estimation results.

    All dollar amounts are in the year specified by the CPI factor
    applied during the computation (default: ~2024 dollars).
    """

    # Hours
    H_E: float = 0.0      # engineering hours (total programme)
    H_T: float = 0.0      # tooling hours
    H_M: float = 0.0      # manufacturing hours
    H_Q: float = 0.0      # quality-control hours

    # Costs ($)
    C_engineering: float = 0.0
    C_tooling: float = 0.0
    C_manufacturing: float = 0.0
    C_QC: float = 0.0
    C_development_support: float = 0.0
    C_flight_test: float = 0.0
    C_materials: float = 0.0
    C_engines_total: float = 0.0
    C_avionics: float = 0.0
    C_interior: float = 0.0

    # Totals
    C_RDTE: float = 0.0
    C_flyaway_total: float = 0.0
    C_flyaway_per_unit: float = 0.0
    C_program_total: float = 0.0
    C_unit_cost: float = 0.0

    # Production quantity (echoed for convenience)
    Q: int = 0

    def summary(self) -> str:
        """Format cost results as a readable multi-line string."""
        lines = [
            "=" * 64,
            "  DAPCA IV COST ESTIMATE",
            "=" * 64,
            "",
            "  HOURS",
            f"    Engineering        : {self.H_E:>14,.0f} hrs",
            f"    Tooling            : {self.H_T:>14,.0f} hrs",
            f"    Manufacturing      : {self.H_M:>14,.0f} hrs",
            f"    Quality Control    : {self.H_Q:>14,.0f} hrs",
            "",
            "  RDT&E COSTS",
            f"    Engineering        : ${self.C_engineering:>14,.0f}",
            f"    Tooling            : ${self.C_tooling:>14,.0f}",
            f"    Dev. Support       : ${self.C_development_support:>14,.0f}",
            f"    Flight Test        : ${self.C_flight_test:>14,.0f}",
            f"    Total RDT&E        : ${self.C_RDTE:>14,.0f}",
            "",
            "  PRODUCTION COSTS",
            f"    Manufacturing      : ${self.C_manufacturing:>14,.0f}",
            f"    Quality Control    : ${self.C_QC:>14,.0f}",
            f"    Materials          : ${self.C_materials:>14,.0f}",
            f"    Engines (total)    : ${self.C_engines_total:>14,.0f}",
            f"    Avionics           : ${self.C_avionics:>14,.0f}",
            f"    Interior           : ${self.C_interior:>14,.0f}",
            f"    Total Flyaway      : ${self.C_flyaway_total:>14,.0f}",
            f"    Per-Unit Flyaway   : ${self.C_flyaway_per_unit:>14,.0f}",
            "",
            "  PROGRAMME TOTALS",
            f"    Programme Total    : ${self.C_program_total:>14,.0f}",
            f"    Unit Cost (Q={self.Q:>3d}) : ${self.C_unit_cost:>14,.0f}",
            "=" * 64,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Engine cost CER  (Raymer Eq 18.8, mks)
# ---------------------------------------------------------------------------

def _engine_cost_each(T_max_N: float, M_max: float,
                      T_turbine_inlet_K: float,
                      turbofan_factor: float) -> float:
    """Estimate the cost of one engine (2012 $).

    Raymer Eq 18.8 (mks form, 6th Ed.):
        C_eng = 3112 * (9.66*T_max + 243.25*M_max
                        + 1.74*T_ti - 2228)

    Where T_max in kN, T_ti in K.

    For turbofan engines multiply by ``turbofan_factor`` (~1.175) because
    the original CER was calibrated on turbojet data (Raymer p. 694).

    Returns 0 if thrust is zero (no engine cost to estimate).
    """
    if T_max_N <= 0.0:
        return 0.0

    T_max_kN = T_max_N / 1000.0  # convert N -> kN

    C_eng = 3112.0 * (9.66 * T_max_kN
                       + 243.25 * M_max
                       + 1.74 * T_turbine_inlet_K
                       - 2228.0)

    # Floor at zero -- the CER can go negative for very small engines;
    # in that case the user should supply a known engine price.
    C_eng = max(C_eng, 0.0)

    return C_eng * turbofan_factor


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------

def compute_dapca(inputs: DAPCAInputs) -> DAPCACostResult:
    """Run the full DAPCA IV cost model.

    Implements Raymer Eqs 18.1-18.9 (mks units, 6th Edition).

    The mks equations use weight in **kg** and velocity in **km/h**.
    AMPR weight (airframe unit weight) is approximated as
    ``ampr_factor * We`` (default 0.62 for transports, Raymer p. 692).

    Equations (2012 $):
        H_E  = 5.18  * W^0.777 * V^0.894 * Q^0.163         (Eq 18.1)
        H_T  = 7.22  * W^0.777 * V^0.696 * Q^0.263         (Eq 18.2)
        H_M  = 10.5  * W^0.82  * V^0.484 * Q^0.641         (Eq 18.3)
        H_Q  = f_qc  * H_M                                  (Eq 18.4)
             where f_qc = 0.076 (cargo) or 0.133 (other)
        C_D  = 67.4  * W^0.630 * V^1.3                      (Eq 18.5)
        C_F  = 1947  * W^0.325 * V^0.822 * FTA^1.21         (Eq 18.6)
        C_mat = 31.2 * W^0.921 * V^0.621 * Q^0.799          (Eq 18.7)
        C_eng = per _engine_cost_each()                      (Eq 18.8)

    RDT&E + Flyaway (Eq 18.9):
        Total = H_E*R_E + H_T*R_T + H_M*R_M + H_Q*R_Q
                + C_D + C_F + C_mat
                + C_eng * N_eng * (Q + FTA)
                + C_avionics + C_interior

    Adjustment factors are applied multiplicatively:
        - material_factor  : on all hours (H_E, H_T, H_M, H_Q)
        - stealth_factor   : on total programme cost
        - commercial_factor: on total programme cost
        - CPI_factor       : on all dollar amounts

    Args:
        inputs: ``DAPCAInputs`` dataclass with all programme parameters.

    Returns:
        ``DAPCACostResult`` with hours, cost breakdowns, and totals.
    """
    r = DAPCACostResult()
    r.Q = inputs.Q

    # ------------------------------------------------------------------ #
    # Unit conversions
    # ------------------------------------------------------------------ #
    W = inputs.We_kg * inputs.ampr_factor              # AMPR weight (kg)
    V_kmh = inputs.V_max_ms * 3.6                      # m/s -> km/h

    Q = float(inputs.Q)
    FTA = float(inputs.FTA)

    # ------------------------------------------------------------------ #
    # Hours  (Raymer Eqs 18.1-18.4, mks)
    # ------------------------------------------------------------------ #
    # Apply material_factor to all hour estimates (composites / advanced
    # materials increase labour hours).
    mf = inputs.material_factor

    r.H_E = 5.18 * W ** 0.777 * V_kmh ** 0.894 * Q ** 0.163 * mf   # Eq 18.1
    r.H_T = 7.22 * W ** 0.777 * V_kmh ** 0.696 * Q ** 0.263 * mf   # Eq 18.2
    r.H_M = 10.5 * W ** 0.82  * V_kmh ** 0.484 * Q ** 0.641 * mf   # Eq 18.3

    # Eq 18.4 -- QC factor
    qc_factor = 0.076 if inputs.cargo_aircraft else 0.133
    r.H_Q = qc_factor * r.H_M                                       # Eq 18.4

    # ------------------------------------------------------------------ #
    # Dollar costs (2012 $, before CPI adjustment)
    # ------------------------------------------------------------------ #
    # Development support cost (Eq 18.5)
    C_D = 67.4 * W ** 0.630 * V_kmh ** 1.3                          # Eq 18.5

    # Flight test operations cost (Eq 18.6)
    C_F = 1947.0 * W ** 0.325 * V_kmh ** 0.822 * FTA ** 1.21       # Eq 18.6

    # Manufacturing materials cost (Eq 18.7)
    C_mat = 31.2 * W ** 0.921 * V_kmh ** 0.621 * Q ** 0.799        # Eq 18.7

    # Engine cost (Eq 18.8)
    C_eng_each = _engine_cost_each(
        inputs.T_max_N, inputs.M_max_engine,
        inputs.T_turbine_inlet_K, inputs.turbofan_factor,
    )
    # Total engine cost covers production aircraft + flight test aircraft
    N_total_engines = inputs.N_engines * (inputs.Q + inputs.FTA)
    C_eng_total = C_eng_each * N_total_engines

    # Interior cost (Raymer p. 695)
    C_interior = inputs.interior_cost_per_pax * inputs.n_pax * inputs.Q

    # Avionics total for programme
    C_avionics_total = inputs.C_avionics * (inputs.Q + inputs.FTA)

    # ------------------------------------------------------------------ #
    # Labour costs (hours * rates)
    # ------------------------------------------------------------------ #
    C_engineering = r.H_E * inputs.R_E
    C_tooling     = r.H_T * inputs.R_T
    C_mfg         = r.H_M * inputs.R_M
    C_qc          = r.H_Q * inputs.R_Q

    # ------------------------------------------------------------------ #
    # RDT&E vs production split (Raymer p. 690)
    # ------------------------------------------------------------------ #
    # RDT&E includes engineering, tooling, development support, flight test.
    # Production (flyaway) includes manufacturing, QC, materials, engines,
    # avionics, and interior.
    #
    # Note: in Raymer Eq 18.9, all terms are lumped into one total; we split
    # them here for insight.

    C_RDTE = C_engineering + C_tooling + C_D + C_F
    C_flyaway = C_mfg + C_qc + C_mat + C_eng_total + C_avionics_total + C_interior
    C_total = C_RDTE + C_flyaway

    # ------------------------------------------------------------------ #
    # Apply programme-level factors
    # ------------------------------------------------------------------ #
    factor = inputs.stealth_factor * inputs.commercial_factor * inputs.CPI_factor

    # Store CPI-adjusted results
    r.C_engineering         = C_engineering * inputs.CPI_factor
    r.C_tooling             = C_tooling * inputs.CPI_factor
    r.C_manufacturing       = C_mfg * inputs.CPI_factor
    r.C_QC                  = C_qc * inputs.CPI_factor
    r.C_development_support = C_D * inputs.CPI_factor
    r.C_flight_test         = C_F * inputs.CPI_factor
    r.C_materials           = C_mat * inputs.CPI_factor
    r.C_engines_total       = C_eng_total * inputs.CPI_factor
    r.C_avionics            = C_avionics_total * inputs.CPI_factor
    r.C_interior            = C_interior * inputs.CPI_factor

    r.C_RDTE           = C_RDTE * factor
    r.C_flyaway_total  = C_flyaway * factor
    r.C_program_total  = C_total * factor

    if inputs.Q > 0:
        r.C_flyaway_per_unit = r.C_flyaway_total / inputs.Q
        r.C_unit_cost        = r.C_program_total / inputs.Q

    return r
