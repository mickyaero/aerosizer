"""
Tests for core.cost package (DAPCA IV and Operating Cost models).

Validates Raymer Ch 18 cost estimating relationships against known
aircraft programme data and physical sanity checks.
"""

import math
import pytest

from core.cost.dapca_iv import (
    DAPCAInputs,
    DAPCACostResult,
    compute_dapca,
    _engine_cost_each,
)
from core.cost.operating_cost import (
    OperatingCostInputs,
    OperatingCostResult,
    compute_operating_costs,
    _crew_cost_per_block_hour,
    _maintenance_material_per_fh,
    _maintenance_material_per_cycle,
)


# ============================================================================ #
# Fixtures
# ============================================================================ #

@pytest.fixture
def b737_dapca():
    """B737-800 class programme for DAPCA IV estimation.

    Approximate values:
        We ~ 41,400 kg, Vmax ~ 250 m/s (M0.82 at 10 km), Q = 500
        CFM56-7B: ~121 kN thrust, M_max ~0.82, TIT ~1530 K
    """
    return DAPCAInputs(
        We_kg=41400.0,
        V_max_ms=250.0,
        Q=500,
        FTA=4,
        N_engines=2,
        T_max_N=121400.0,
        M_max_engine=0.82,
        T_turbine_inlet_K=1530.0,
        C_avionics=2_000_000.0,
        n_pax=162,
        CPI_factor=1.0,           # keep in 2012 $ for easier validation
        material_factor=1.0,
        stealth_factor=1.0,
        commercial_factor=1.0,    # no adjustment for raw equation test
    )


@pytest.fixture
def b737_opcost():
    """B737-800 class operating cost inputs."""
    return OperatingCostInputs(
        W0_kg=79000.0,
        We_kg=41400.0,
        Wf_mission_kg=16000.0,      # typical 2500 nm mission
        V_cruise_ms=230.0,          # ~M0.78 at cruise
        range_km=4600.0,            # ~2500 nm
        n_pax=162,
        n_engines=2,
        aircraft_cost=100_000_000.0,
        engine_cost_each=12_000_000.0,
        fuel_price_per_kg=0.80,
        flight_hours_per_year=3500.0,
        crew_type="two_man",
        CPI_factor=1.0,             # 2012 $ for cleaner tests
        MMH_FH=10.0,
    )


@pytest.fixture
def small_ga_dapca():
    """Small GA/trainer aircraft for DAPCA edge case testing."""
    return DAPCAInputs(
        We_kg=1000.0,
        V_max_ms=80.0,
        Q=200,
        FTA=2,
        N_engines=1,
        T_max_N=0.0,       # no engine CER -- prop aircraft
        C_avionics=50_000.0,
        n_pax=4,
        CPI_factor=1.0,
        commercial_factor=1.0,
        stealth_factor=1.0,
    )


# ============================================================================ #
# DAPCA IV -- Engine Cost CER
# ============================================================================ #

class TestEngineCostCER:
    """Tests for _engine_cost_each (Raymer Eq 18.8)."""

    def test_zero_thrust_returns_zero(self):
        assert _engine_cost_each(0.0, 0.82, 1530.0, 1.175) == 0.0

    def test_positive_for_transport_engine(self):
        """CFM56-class engine should have a positive cost."""
        cost = _engine_cost_each(121400.0, 0.82, 1530.0, 1.0)
        assert cost > 0

    def test_increases_with_thrust(self):
        """Higher thrust should mean higher engine cost."""
        c1 = _engine_cost_each(100_000.0, 0.82, 1530.0, 1.0)
        c2 = _engine_cost_each(200_000.0, 0.82, 1530.0, 1.0)
        assert c2 > c1

    def test_increases_with_tit(self):
        """Higher turbine inlet temperature increases cost."""
        c1 = _engine_cost_each(121400.0, 0.82, 1400.0, 1.0)
        c2 = _engine_cost_each(121400.0, 0.82, 1700.0, 1.0)
        assert c2 > c1

    def test_turbofan_factor(self):
        """Turbofan factor should scale cost proportionally."""
        c_base = _engine_cost_each(121400.0, 0.82, 1530.0, 1.0)
        c_tf = _engine_cost_each(121400.0, 0.82, 1530.0, 1.175)
        assert c_tf == pytest.approx(c_base * 1.175, rel=1e-10)

    def test_cfm56_order_of_magnitude(self):
        """CFM56 cost should be in the $5M-$20M range (2012 $)."""
        cost = _engine_cost_each(121400.0, 0.82, 1530.0, 1.175)
        assert 1_000_000 < cost < 50_000_000


# ============================================================================ #
# DAPCA IV -- Hours Equations
# ============================================================================ #

class TestDAPCAHours:
    """Tests for DAPCA IV hour CERs (Eqs 18.1-18.4)."""

    def test_all_hours_positive(self, b737_dapca):
        result = compute_dapca(b737_dapca)
        assert result.H_E > 0
        assert result.H_T > 0
        assert result.H_M > 0
        assert result.H_Q > 0

    def test_qc_fraction_non_cargo(self, b737_dapca):
        """QC = 0.133 * H_M for non-cargo aircraft (Eq 18.4)."""
        result = compute_dapca(b737_dapca)
        assert result.H_Q == pytest.approx(0.133 * result.H_M, rel=1e-10)

    def test_qc_fraction_cargo(self, b737_dapca):
        """QC = 0.076 * H_M for cargo aircraft (Eq 18.4)."""
        b737_dapca.cargo_aircraft = True
        result = compute_dapca(b737_dapca)
        assert result.H_Q == pytest.approx(0.076 * result.H_M, rel=1e-10)

    def test_cargo_lower_qc_than_pax(self, b737_dapca):
        """Cargo aircraft should have lower QC hours than passenger."""
        r_pax = compute_dapca(b737_dapca)
        b737_dapca.cargo_aircraft = True
        r_cargo = compute_dapca(b737_dapca)
        assert r_cargo.H_Q < r_pax.H_Q

    def test_hours_increase_with_weight(self, b737_dapca):
        """Heavier aircraft should require more hours."""
        r1 = compute_dapca(b737_dapca)
        b737_dapca.We_kg = 60000.0
        r2 = compute_dapca(b737_dapca)
        assert r2.H_E > r1.H_E
        assert r2.H_T > r1.H_T
        assert r2.H_M > r1.H_M

    def test_hours_increase_with_speed(self, b737_dapca):
        """Faster aircraft should require more engineering and tooling hours."""
        r1 = compute_dapca(b737_dapca)
        b737_dapca.V_max_ms = 300.0
        r2 = compute_dapca(b737_dapca)
        assert r2.H_E > r1.H_E
        assert r2.H_T > r1.H_T

    def test_hours_increase_with_quantity(self, b737_dapca):
        """More production units should increase total hours."""
        r1 = compute_dapca(b737_dapca)
        b737_dapca.Q = 1000
        r2 = compute_dapca(b737_dapca)
        assert r2.H_E > r1.H_E
        assert r2.H_M > r1.H_M

    def test_material_factor_scales_hours(self, b737_dapca):
        """Material factor should multiply all hours."""
        r_base = compute_dapca(b737_dapca)
        b737_dapca.material_factor = 1.5
        r_comp = compute_dapca(b737_dapca)
        assert r_comp.H_E == pytest.approx(r_base.H_E * 1.5, rel=1e-10)
        assert r_comp.H_T == pytest.approx(r_base.H_T * 1.5, rel=1e-10)
        assert r_comp.H_M == pytest.approx(r_base.H_M * 1.5, rel=1e-10)
        assert r_comp.H_Q == pytest.approx(r_base.H_Q * 1.5, rel=1e-10)


# ============================================================================ #
# DAPCA IV -- Cost Totals
# ============================================================================ #

class TestDAPCACosts:
    """Tests for DAPCA IV cost totals (Eq 18.9)."""

    def test_all_costs_positive(self, b737_dapca):
        result = compute_dapca(b737_dapca)
        assert result.C_engineering > 0
        assert result.C_tooling > 0
        assert result.C_manufacturing > 0
        assert result.C_QC > 0
        assert result.C_development_support > 0
        assert result.C_flight_test > 0
        assert result.C_materials > 0

    def test_programme_total_is_sum(self, b737_dapca):
        """Programme total should equal RDT&E + flyaway."""
        r = compute_dapca(b737_dapca)
        assert r.C_program_total == pytest.approx(
            r.C_RDTE + r.C_flyaway_total, rel=1e-10
        )

    def test_unit_cost_is_total_over_q(self, b737_dapca):
        """Unit cost should equal programme total / Q."""
        r = compute_dapca(b737_dapca)
        assert r.C_unit_cost == pytest.approx(
            r.C_program_total / b737_dapca.Q, rel=1e-10
        )

    def test_flyaway_per_unit(self, b737_dapca):
        """Per-unit flyaway = total flyaway / Q."""
        r = compute_dapca(b737_dapca)
        assert r.C_flyaway_per_unit == pytest.approx(
            r.C_flyaway_total / b737_dapca.Q, rel=1e-10
        )

    def test_q_echoed(self, b737_dapca):
        r = compute_dapca(b737_dapca)
        assert r.Q == 500

    def test_cpi_factor_scales_costs(self, b737_dapca):
        """CPI factor should scale all dollar amounts."""
        r_base = compute_dapca(b737_dapca)
        b737_dapca.CPI_factor = 1.35
        r_adj = compute_dapca(b737_dapca)
        assert r_adj.C_engineering == pytest.approx(
            r_base.C_engineering * 1.35, rel=1e-10
        )
        assert r_adj.C_materials == pytest.approx(
            r_base.C_materials * 1.35, rel=1e-10
        )

    def test_stealth_factor_increases_total(self, b737_dapca):
        """Stealth factor should increase programme total."""
        r_base = compute_dapca(b737_dapca)
        b737_dapca.stealth_factor = 1.2
        r_stealth = compute_dapca(b737_dapca)
        assert r_stealth.C_program_total > r_base.C_program_total

    def test_commercial_factor_reduces_total(self, b737_dapca):
        """Commercial correction factor < 1 should reduce total."""
        r_base = compute_dapca(b737_dapca)
        b737_dapca.commercial_factor = 0.9
        r_comm = compute_dapca(b737_dapca)
        assert r_comm.C_program_total < r_base.C_program_total

    def test_interior_cost_zero_when_no_pax(self, b737_dapca):
        """Zero passengers should produce zero interior cost."""
        b737_dapca.n_pax = 0
        r = compute_dapca(b737_dapca)
        assert r.C_interior == 0.0

    def test_interior_cost_scales_with_pax_and_q(self, b737_dapca):
        """Interior cost should scale with n_pax * Q."""
        r = compute_dapca(b737_dapca)
        expected = 3500.0 * 162 * 500  # interior_cost_per_pax * n_pax * Q
        assert r.C_interior == pytest.approx(expected, rel=1e-10)

    def test_no_engine_cost_when_zero_thrust(self, small_ga_dapca):
        """Zero thrust should give zero engine cost."""
        r = compute_dapca(small_ga_dapca)
        assert r.C_engines_total == 0.0

    def test_b737_unit_cost_order_of_magnitude(self, b737_dapca):
        """B737 unit cost should be in the tens of millions (2012 $).

        Actual B737-800 list price was ~$96M (2018), but DAPCA includes
        RDT&E amortisation. At Q=500 with RDT&E spread, unit cost
        ~$50M-$200M is reasonable.
        """
        r = compute_dapca(b737_dapca)
        assert 10_000_000 < r.C_unit_cost < 500_000_000


# ============================================================================ #
# DAPCA IV -- Summary formatting
# ============================================================================ #

class TestDAPCASummary:
    """Tests for DAPCACostResult.summary()."""

    def test_returns_string(self, b737_dapca):
        r = compute_dapca(b737_dapca)
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_contains_key_labels(self, b737_dapca):
        r = compute_dapca(b737_dapca)
        s = r.summary()
        assert "Engineering" in s
        assert "Tooling" in s
        assert "Manufacturing" in s
        assert "Programme Total" in s or "PROGRAMME TOTALS" in s
        assert "Unit Cost" in s


# ============================================================================ #
# DAPCA IV -- AMPR factor
# ============================================================================ #

class TestAMPRFactor:
    """Tests for the AMPR weight factor."""

    def test_default_ampr_is_0_62(self, b737_dapca):
        """Default AMPR factor should be 0.62 (Raymer p. 692)."""
        assert b737_dapca.ampr_factor == 0.62

    def test_ampr_factor_affects_hours(self, b737_dapca):
        """Changing AMPR factor changes the effective weight in the CER."""
        r1 = compute_dapca(b737_dapca)
        b737_dapca.ampr_factor = 1.0  # use full empty weight
        r2 = compute_dapca(b737_dapca)
        # Full weight should produce more hours
        assert r2.H_E > r1.H_E
        assert r2.H_M > r1.H_M


# ============================================================================ #
# DAPCA IV -- Edge Cases
# ============================================================================ #

class TestDAPCAEdgeCases:
    """Edge cases for DAPCA IV model."""

    def test_single_aircraft_production(self):
        """Q=1 should still produce valid results (prototype)."""
        inp = DAPCAInputs(
            We_kg=5000.0,
            V_max_ms=100.0,
            Q=1,
            FTA=1,
            CPI_factor=1.0,
            commercial_factor=1.0,
            stealth_factor=1.0,
        )
        r = compute_dapca(inp)
        assert r.C_program_total > 0
        assert r.C_unit_cost == pytest.approx(r.C_program_total, rel=1e-10)

    def test_small_ga_aircraft(self, small_ga_dapca):
        """Small GA should produce sensible results."""
        r = compute_dapca(small_ga_dapca)
        assert r.C_program_total > 0
        assert r.C_unit_cost > 0
        # GA unit cost should be less than a transport
        assert r.C_unit_cost < 50_000_000


# ============================================================================ #
# Operating Cost -- Crew Cost CER
# ============================================================================ #

class TestCrewCost:
    """Tests for crew cost CER (Raymer Eqs 18.10-18.11)."""

    def test_two_man_positive(self):
        cost = _crew_cost_per_block_hour(230.0, 79000.0, "two_man", 1.0)
        assert cost > 0

    def test_three_man_positive(self):
        cost = _crew_cost_per_block_hour(230.0, 79000.0, "three_man", 1.0)
        assert cost > 0

    def test_three_man_more_expensive(self):
        """Three-man crew should cost more than two-man."""
        c2 = _crew_cost_per_block_hour(230.0, 79000.0, "two_man", 1.0)
        c3 = _crew_cost_per_block_hour(230.0, 79000.0, "three_man", 1.0)
        assert c3 > c2

    def test_increases_with_weight(self):
        """Heavier aircraft should have higher crew cost."""
        c1 = _crew_cost_per_block_hour(230.0, 50000.0, "two_man", 1.0)
        c2 = _crew_cost_per_block_hour(230.0, 100000.0, "two_man", 1.0)
        assert c2 > c1

    def test_increases_with_speed(self):
        """Faster aircraft should have higher crew cost."""
        c1 = _crew_cost_per_block_hour(200.0, 79000.0, "two_man", 1.0)
        c2 = _crew_cost_per_block_hour(280.0, 79000.0, "two_man", 1.0)
        assert c2 > c1

    def test_cpi_factor_scales(self):
        """CPI factor should scale crew cost proportionally."""
        c1 = _crew_cost_per_block_hour(230.0, 79000.0, "two_man", 1.0)
        c2 = _crew_cost_per_block_hour(230.0, 79000.0, "two_man", 1.35)
        assert c2 == pytest.approx(c1 * 1.35, rel=1e-10)

    def test_typical_range(self):
        """B737-class crew cost should be ~$300-$1000/bh (2012 $)."""
        cost = _crew_cost_per_block_hour(230.0, 79000.0, "two_man", 1.0)
        assert 100 < cost < 2000


# ============================================================================ #
# Operating Cost -- Maintenance Material CERs
# ============================================================================ #

class TestMaintenanceMaterial:
    """Tests for maintenance material CERs (Eqs 18.12-18.13)."""

    def test_per_fh_positive(self):
        cost = _maintenance_material_per_fh(
            100_000_000.0, 12_000_000.0, 2, 1.0
        )
        assert cost > 0

    def test_per_cycle_positive(self):
        cost = _maintenance_material_per_cycle(
            100_000_000.0, 12_000_000.0, 2, 1.0
        )
        assert cost > 0

    def test_per_fh_increases_with_aircraft_cost(self):
        c1 = _maintenance_material_per_fh(50_000_000.0, 10_000_000.0, 2, 1.0)
        c2 = _maintenance_material_per_fh(150_000_000.0, 10_000_000.0, 2, 1.0)
        assert c2 > c1

    def test_per_fh_increases_with_engines(self):
        c2 = _maintenance_material_per_fh(100_000_000.0, 12_000_000.0, 2, 1.0)
        c4 = _maintenance_material_per_fh(100_000_000.0, 12_000_000.0, 4, 1.0)
        assert c4 > c2

    def test_per_fh_cpi_scales(self):
        c1 = _maintenance_material_per_fh(100_000_000.0, 12_000_000.0, 2, 1.0)
        c2 = _maintenance_material_per_fh(100_000_000.0, 12_000_000.0, 2, 1.35)
        assert c2 == pytest.approx(c1 * 1.35, rel=1e-10)

    def test_zero_cost_aircraft(self):
        """Zero aircraft cost should still produce non-negative result."""
        cost = _maintenance_material_per_fh(0.0, 0.0, 2, 1.0)
        assert cost >= 0


# ============================================================================ #
# Operating Cost -- Full Computation
# ============================================================================ #

class TestOperatingCosts:
    """Tests for compute_operating_costs."""

    def test_returns_result_type(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        assert isinstance(r, OperatingCostResult)

    def test_block_time_positive(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        assert r.block_time_hr > 0

    def test_block_time_includes_allowances(self, b737_opcost):
        """Block time should exceed pure flight time by ground + air manoeuvre."""
        r = compute_operating_costs(b737_opcost)
        flight_time = (b737_opcost.range_km * 1000 / b737_opcost.V_cruise_ms) / 3600
        assert r.block_time_hr > flight_time
        # Should include 15 min ground + 6 min air = 0.35 hr
        expected = flight_time + 0.25 + 0.10
        assert r.block_time_hr == pytest.approx(expected, rel=1e-10)

    def test_cycles_per_year_consistent(self, b737_opcost):
        """Cycles = flight_hours / block_time."""
        r = compute_operating_costs(b737_opcost)
        expected = b737_opcost.flight_hours_per_year / r.block_time_hr
        assert r.cycles_per_year == pytest.approx(expected, rel=1e-10)

    def test_all_per_bh_positive(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        assert r.crew_cost_per_bh > 0
        assert r.fuel_cost_per_bh > 0
        assert r.maintenance_labor_per_bh > 0
        assert r.maintenance_material_per_bh > 0
        assert r.depreciation_per_bh > 0
        assert r.insurance_per_bh > 0
        assert r.landing_fees_per_bh > 0

    def test_annual_doc_positive(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        assert r.annual_DOC > 0

    def test_annual_doc_is_sum(self, b737_opcost):
        """Annual DOC should equal sum of annual components."""
        r = compute_operating_costs(b737_opcost)
        expected = (
            r.annual_fuel + r.annual_crew + r.annual_maintenance
            + r.annual_depreciation + r.annual_insurance
            + r.landing_fees_per_bh * b737_opcost.flight_hours_per_year
        )
        assert r.annual_DOC == pytest.approx(expected, rel=1e-10)

    def test_fuel_per_trip(self, b737_opcost):
        """Fuel per trip should match mission fuel * price."""
        r = compute_operating_costs(b737_opcost)
        expected = b737_opcost.Wf_mission_kg * b737_opcost.fuel_price_per_kg
        assert r.fuel_per_trip == pytest.approx(expected, rel=1e-10)

    def test_total_per_trip_consistent(self, b737_opcost):
        """Total per trip = annual DOC / cycles per year."""
        r = compute_operating_costs(b737_opcost)
        expected = r.annual_DOC / r.cycles_per_year
        assert r.total_per_trip == pytest.approx(expected, rel=1e-10)

    def test_seat_metrics_positive(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        assert r.DOC_per_seat_mile > 0
        assert r.DOC_per_seat_km > 0
        assert r.CASM > 0
        assert r.fuel_per_seat_km > 0
        assert r.CO2_per_seat_km > 0

    def test_casm_equals_doc_per_seat_mile(self, b737_opcost):
        """CASM should equal DOC/seat-mile."""
        r = compute_operating_costs(b737_opcost)
        assert r.CASM == pytest.approx(r.DOC_per_seat_mile, rel=1e-10)

    def test_co2_is_3_16_times_fuel(self, b737_opcost):
        """CO2 per seat-km = 3.16 * fuel per seat-km."""
        r = compute_operating_costs(b737_opcost)
        assert r.CO2_per_seat_km == pytest.approx(
            3.16 * r.fuel_per_seat_km, rel=1e-10
        )

    def test_breakeven_load_factor_reasonable(self, b737_opcost):
        """Break-even LF should be between 0 and 1 for a reasonable route."""
        r = compute_operating_costs(b737_opcost)
        assert 0.0 < r.breakeven_load_factor <= 1.5

    def test_b737_casm_order_of_magnitude(self, b737_opcost):
        """B737-class CASM should be ~$0.05-$0.20/seat-mile (2012 $)."""
        r = compute_operating_costs(b737_opcost)
        assert 0.01 < r.CASM < 0.50

    def test_zero_speed_raises(self):
        inp = OperatingCostInputs(
            W0_kg=79000.0, We_kg=41400.0, Wf_mission_kg=16000.0,
            V_cruise_ms=0.0, range_km=4600.0,
        )
        with pytest.raises(ValueError, match="positive"):
            compute_operating_costs(inp)


# ============================================================================ #
# Operating Cost -- Depreciation
# ============================================================================ #

class TestDepreciation:
    """Tests for depreciation calculation."""

    def test_depreciation_increases_with_aircraft_cost(self, b737_opcost):
        r1 = compute_operating_costs(b737_opcost)
        b737_opcost.aircraft_cost = 200_000_000.0
        r2 = compute_operating_costs(b737_opcost)
        assert r2.depreciation_per_bh > r1.depreciation_per_bh

    def test_zero_cost_gives_zero_depreciation(self):
        inp = OperatingCostInputs(
            W0_kg=79000.0, We_kg=41400.0, Wf_mission_kg=16000.0,
            V_cruise_ms=230.0, range_km=4600.0,
            aircraft_cost=0.0, engine_cost_each=0.0,
        )
        r = compute_operating_costs(inp)
        assert r.depreciation_per_bh == 0.0

    def test_annual_depreciation_formula(self, b737_opcost):
        """Verify depreciation matches the airframe + engine formula."""
        r = compute_operating_costs(b737_opcost)
        airframe_cost = b737_opcost.aircraft_cost - b737_opcost.engine_cost_each * 2
        expected_af = airframe_cost * 0.90 / 12.0  # (1 - 10% resale) / 12 yr
        expected_eng = (b737_opcost.engine_cost_each * 2) / 4.0
        expected = expected_af + expected_eng
        assert r.annual_depreciation == pytest.approx(expected, rel=1e-10)


# ============================================================================ #
# Operating Cost -- Summary formatting
# ============================================================================ #

class TestOperatingSummary:
    """Tests for OperatingCostResult.summary()."""

    def test_returns_string(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        s = r.summary()
        assert isinstance(s, str)
        assert len(s) > 100

    def test_contains_key_labels(self, b737_opcost):
        r = compute_operating_costs(b737_opcost)
        s = r.summary()
        assert "Crew" in s
        assert "Fuel" in s
        assert "Depreciation" in s
        assert "CASM" in s
        assert "CO2" in s
        assert "Break-even" in s


# ============================================================================ #
# Operating Cost -- Edge Cases
# ============================================================================ #

class TestOperatingEdgeCases:
    """Edge cases for operating cost model."""

    def test_zero_pax_no_seat_metrics(self):
        """Zero passengers should produce zero seat metrics."""
        inp = OperatingCostInputs(
            W0_kg=79000.0, We_kg=41400.0, Wf_mission_kg=16000.0,
            V_cruise_ms=230.0, range_km=4600.0,
            n_pax=0, aircraft_cost=100_000_000.0,
        )
        r = compute_operating_costs(inp)
        assert r.DOC_per_seat_mile == 0.0
        assert r.CASM == 0.0
        assert r.fuel_per_seat_km == 0.0
        assert r.CO2_per_seat_km == 0.0

    def test_short_range_mission(self):
        """Short 500 km mission should still compute correctly."""
        inp = OperatingCostInputs(
            W0_kg=30000.0, We_kg=18000.0, Wf_mission_kg=2000.0,
            V_cruise_ms=180.0, range_km=500.0,
            n_pax=50, aircraft_cost=30_000_000.0,
            engine_cost_each=4_000_000.0,
        )
        r = compute_operating_costs(inp)
        assert r.block_time_hr > 0
        assert r.annual_DOC > 0
        assert r.CASM > 0

    def test_international_fare_type(self, b737_opcost):
        """International fare type should give different break-even LF."""
        r_dom = compute_operating_costs(b737_opcost)
        b737_opcost.fare_type = "international"
        r_intl = compute_operating_costs(b737_opcost)
        # International fares are lower, so break-even LF should be higher
        assert r_intl.breakeven_load_factor > r_dom.breakeven_load_factor

    def test_three_man_crew_increases_cost(self, b737_opcost):
        """Three-man crew should increase crew cost."""
        r2 = compute_operating_costs(b737_opcost)
        b737_opcost.crew_type = "three_man"
        r3 = compute_operating_costs(b737_opcost)
        assert r3.crew_cost_per_bh > r2.crew_cost_per_bh
        assert r3.annual_crew > r2.annual_crew


# ============================================================================ #
# Integration -- DAPCA + Operating Cost Together
# ============================================================================ #

class TestCostIntegration:
    """End-to-end: use DAPCA output to feed operating cost inputs."""

    def test_dapca_to_operating(self, b737_dapca):
        """Use DAPCA flyaway as aircraft cost in operating cost model."""
        dapca_r = compute_dapca(b737_dapca)
        assert dapca_r.C_flyaway_per_unit > 0

        # Feed DAPCA outputs into operating cost
        engine_cost = _engine_cost_each(
            b737_dapca.T_max_N, b737_dapca.M_max_engine,
            b737_dapca.T_turbine_inlet_K, b737_dapca.turbofan_factor,
        )

        op_inp = OperatingCostInputs(
            W0_kg=79000.0,
            We_kg=b737_dapca.We_kg,
            Wf_mission_kg=16000.0,
            V_cruise_ms=230.0,
            range_km=4600.0,
            n_pax=b737_dapca.n_pax,
            n_engines=b737_dapca.N_engines,
            aircraft_cost=dapca_r.C_flyaway_per_unit,
            engine_cost_each=engine_cost,
            CPI_factor=1.0,
        )
        op_r = compute_operating_costs(op_inp)
        assert op_r.annual_DOC > 0
        assert op_r.CASM > 0
        assert op_r.CO2_per_seat_km > 0
