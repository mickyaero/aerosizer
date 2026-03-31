"""
Tests for core.weights.statistical_weights module.

Validates Raymer Ch 15 statistical weight equations for transport aircraft
against known aircraft data (B737-800 class) and physical sanity checks.
"""

import math
import pytest

from core.weights.statistical_weights import TransportWeights
from core.atmosphere import kg_to_lb, lb_to_kg


# ============================================================================ #
# Fixtures: representative aircraft configurations
# ============================================================================ #

@pytest.fixture
def b737_like():
    """B737-800 class transport configuration.

    Approximate values:
        MTOW ~ 79,000 kg, Fuel ~ 21,000 kg, Sref ~ 124.6 m^2
        AR ~ 9.4, sweep ~ 25 deg, taper ~ 0.278, t/c ~ 0.128
        Fuselage: L ~ 38.0 m, W ~ 3.76 m, D ~ 3.76 m
        HT area ~ 32 m^2, VT area ~ 26 m^2
        Engines: 2x CFM56-7B (~2380 kg each, ~121 kN each)
        Pax: 162
    """
    return TransportWeights(
        W0_kg=79000.0,
        Wf_kg=21000.0,
        AR=9.44,
        Sref_m2=124.6,
        sweep_quarter_rad=math.radians(25.0),
        taper_ratio=0.278,
        t_c_root=0.128,
        Nz=3.75,
        fuselage_length_m=38.0,
        fuselage_width_m=3.76,
        fuselage_depth_m=3.76,
        Sht_m2=32.0,
        Svt_m2=26.0,
        sweep_ht_rad=math.radians(30.0),
        sweep_vt_rad=math.radians(35.0),
        AR_ht=4.0,
        AR_vt=1.5,
        taper_ht=0.4,
        taper_vt=0.4,
        t_c_ht=0.09,
        t_c_vt=0.09,
        n_engines=2,
        engine_weight_each_kg=2380.0,
        thrust_each_N=121400.0,
        Wl_kg=66360.0,
        n_crew=2,
        n_pax=162,
        cargo_floor_area_m2=25.0,
        pressurized=True,
        fuselage_Swet_m2=400.0,
    )


@pytest.fixture
def b787_like():
    """B787-8 class wide-body transport configuration.

    Approximate values:
        MTOW ~ 228,000 kg, Fuel ~ 101,000 kg, Sref ~ 377 m^2
        AR ~ 10.6, sweep ~ 32 deg, taper ~ 0.18, t/c ~ 0.11
        Fuselage: L ~ 57.0 m, W ~ 5.77 m, D ~ 5.97 m
        HT area ~ 60 m^2, VT area ~ 42 m^2
        Engines: 2x GEnx-1B (~5800 kg each, ~296 kN each)
        Pax: 242
    """
    return TransportWeights(
        W0_kg=228000.0,
        Wf_kg=101000.0,
        AR=10.58,
        Sref_m2=377.0,
        sweep_quarter_rad=math.radians(32.2),
        taper_ratio=0.18,
        t_c_root=0.11,
        Nz=3.75,
        fuselage_length_m=57.0,
        fuselage_width_m=5.77,
        fuselage_depth_m=5.97,
        Sht_m2=60.0,
        Svt_m2=42.0,
        sweep_ht_rad=math.radians(35.0),
        sweep_vt_rad=math.radians(40.0),
        AR_ht=4.5,
        AR_vt=1.7,
        taper_ht=0.35,
        taper_vt=0.35,
        t_c_ht=0.09,
        t_c_vt=0.09,
        n_engines=2,
        engine_weight_each_kg=5800.0,
        thrust_each_N=296000.0,
        Wl_kg=172000.0,
        n_crew=2,
        n_pax=242,
        cargo_floor_area_m2=80.0,
        pressurized=True,
        fuselage_Swet_m2=900.0,
    )


# ============================================================================ #
# Unit Tests -- Individual Components
# ============================================================================ #

class TestWingWeight:
    """Tests for wing_weight (Raymer Eq 15.25)."""

    def test_returns_positive(self, b737_like):
        w = b737_like.wing_weight()
        assert w > 0

    def test_b737_range(self, b737_like):
        """B737-800 wing weight is approximately 6000-9000 kg."""
        w = b737_like.wing_weight()
        assert 4000 < w < 12000

    def test_b787_heavier_than_b737(self, b737_like, b787_like):
        """Larger aircraft should have heavier wing."""
        assert b787_like.wing_weight() > b737_like.wing_weight()

    def test_increases_with_area(self, b737_like):
        """Larger wing area should produce heavier wing."""
        w1 = b737_like.wing_weight()
        b737_like.Sref_m2 = 150.0
        w2 = b737_like.wing_weight()
        assert w2 > w1

    def test_increases_with_w0(self, b737_like):
        """Higher MTOW should produce heavier wing."""
        w1 = b737_like.wing_weight()
        b737_like.W0_kg = 90000.0
        w2 = b737_like.wing_weight()
        assert w2 > w1

    def test_sweep_effect(self, b737_like):
        """More sweep at constant AR should reduce the term (A/cos^2)^0.6
        contribution but increase (100*tc/cos)^-0.3.  Overall, modest change."""
        w_25 = b737_like.wing_weight()
        b737_like.sweep_quarter_rad = math.radians(35.0)
        w_35 = b737_like.wing_weight()
        # Should differ, exact direction depends on balance of exponents
        assert w_35 != pytest.approx(w_25, rel=0.001)


class TestHorizontalTailWeight:
    """Tests for horizontal_tail_weight (Raymer Eq 15.26)."""

    def test_returns_positive(self, b737_like):
        w = b737_like.horizontal_tail_weight()
        assert w > 0

    def test_zero_area_returns_zero(self, b737_like):
        b737_like.Sht_m2 = 0.0
        assert b737_like.horizontal_tail_weight() == 0.0

    def test_reasonable_range(self, b737_like):
        """HT weight should be ~200-1500 kg for narrow-body."""
        w = b737_like.horizontal_tail_weight()
        assert 100 < w < 2000


class TestVerticalTailWeight:
    """Tests for vertical_tail_weight (Raymer Eq 15.27)."""

    def test_returns_positive(self, b737_like):
        w = b737_like.vertical_tail_weight()
        assert w > 0

    def test_zero_area_returns_zero(self, b737_like):
        b737_like.Svt_m2 = 0.0
        assert b737_like.vertical_tail_weight() == 0.0

    def test_reasonable_range(self, b737_like):
        """VT weight should be ~200-1200 kg for narrow-body."""
        w = b737_like.vertical_tail_weight()
        assert 100 < w < 1500


class TestFuselageWeight:
    """Tests for fuselage_weight (Raymer Eq 15.28)."""

    def test_returns_positive(self, b737_like):
        w = b737_like.fuselage_weight()
        assert w > 0

    def test_b737_range(self, b737_like):
        """B737-800 fuselage weight should be ~6000-12000 kg."""
        w = b737_like.fuselage_weight()
        assert 4000 < w < 15000

    def test_pressurised_heavier_than_unpressurised(self, b737_like):
        """Pressurised fuselage should be heavier."""
        w_press = b737_like.fuselage_weight()
        b737_like.pressurized = False
        w_unpress = b737_like.fuselage_weight()
        assert w_press > w_unpress


class TestLandingGearWeight:
    """Tests for main_gear_weight and nose_gear_weight (Eqs 15.29-15.30)."""

    def test_main_gear_positive(self, b737_like):
        assert b737_like.main_gear_weight() > 0

    def test_nose_gear_positive(self, b737_like):
        assert b737_like.nose_gear_weight() > 0

    def test_main_heavier_than_nose(self, b737_like):
        """Main gear should be significantly heavier than nose gear."""
        assert b737_like.main_gear_weight() > b737_like.nose_gear_weight()

    def test_main_gear_reasonable(self, b737_like):
        """Main gear should be ~1500-4500 kg for narrow-body."""
        w = b737_like.main_gear_weight()
        assert 800 < w < 6000

    def test_nose_gear_reasonable(self, b737_like):
        """Nose gear should be ~200-700 kg for narrow-body."""
        w = b737_like.nose_gear_weight()
        assert 100 < w < 1200

    def test_default_landing_weight(self, b737_like):
        """If Wl_kg = 0, should default to 0.85 * W0."""
        b737_like.Wl_kg = 0.0
        w_default = b737_like.main_gear_weight()
        b737_like.Wl_kg = 0.85 * b737_like.W0_kg
        w_explicit = b737_like.main_gear_weight()
        assert w_default == pytest.approx(w_explicit, rel=1e-6)


class TestInstalledEngineWeight:
    """Tests for installed_engine_weight (Raymer Eq 15.31)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.installed_engine_weight() > 0

    def test_zero_engine_weight(self, b737_like):
        b737_like.engine_weight_each_kg = 0.0
        assert b737_like.installed_engine_weight() == 0.0

    def test_installed_greater_than_bare(self, b737_like):
        """Installed weight should exceed bare engine weight (nacelles, etc.)."""
        bare = b737_like.engine_weight_each_kg * b737_like.n_engines
        installed = b737_like.installed_engine_weight()
        assert installed > bare

    def test_scales_with_n_engines(self, b737_like):
        """More engines should produce higher installed weight."""
        w2 = b737_like.installed_engine_weight()
        b737_like.n_engines = 4
        w4 = b737_like.installed_engine_weight()
        assert w4 > w2


class TestFuelSystemWeight:
    """Tests for fuel_system_weight (Raymer Eq 15.32)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.fuel_system_weight() > 0

    def test_reasonable_range(self, b737_like):
        """Fuel system weight should be ~100-600 kg for narrow-body."""
        w = b737_like.fuel_system_weight()
        assert 50 < w < 1000


class TestFlightControlsWeight:
    """Tests for flight_controls_weight (Raymer Eq 15.33)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.flight_controls_weight() > 0

    def test_reasonable_range(self, b737_like):
        """Flight controls should be ~500-3000 kg."""
        w = b737_like.flight_controls_weight()
        assert 200 < w < 5000


class TestHydraulicsWeight:
    """Tests for hydraulics_weight (Raymer Eq 15.36)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.hydraulics_weight() > 0

    def test_reasonable_range(self, b737_like):
        """Hydraulics should be ~200-800 kg for narrow-body."""
        w = b737_like.hydraulics_weight()
        assert 100 < w < 1500


class TestElectricalWeight:
    """Tests for electrical_weight (Raymer Eq 15.38)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.electrical_weight() > 0

    def test_reasonable_range(self, b737_like):
        """Electrical system should be ~500-2000 kg."""
        w = b737_like.electrical_weight()
        assert 200 < w < 3000


class TestAvionicsWeight:
    """Tests for avionics_weight (Raymer Eq 15.39)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.avionics_weight() > 0

    def test_reasonable_range(self, b737_like):
        """Avionics should be ~500-1500 kg."""
        w = b737_like.avionics_weight()
        assert 200 < w < 2000


class TestFurnishingsWeight:
    """Tests for furnishings_weight (Raymer Eq 15.41)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.furnishings_weight() > 0

    def test_scales_with_fuselage_size(self, b737_like, b787_like):
        """Larger fuselage should have heavier furnishings."""
        assert b787_like.furnishings_weight() > b737_like.furnishings_weight()


class TestAirConditioningWeight:
    """Tests for air_conditioning_weight (Raymer Eq 15.42)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.air_conditioning_weight() > 0

    def test_reasonable_range(self, b737_like):
        """A/C system should be ~300-2000 kg for narrow-body."""
        w = b737_like.air_conditioning_weight()
        assert 100 < w < 3000


class TestAntiIceWeight:
    """Tests for anti_ice_weight (Raymer Eq 15.43)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.anti_ice_weight() > 0

    def test_fraction_of_w0(self, b737_like):
        """Anti-ice ~ 0.2% of W0."""
        w = b737_like.anti_ice_weight()
        expected = 0.002 * b737_like.W0_kg
        assert w == pytest.approx(expected, rel=0.01)


class TestHandlingGearWeight:
    """Tests for handling_gear_weight (Raymer Eq 15.44)."""

    def test_returns_positive(self, b737_like):
        assert b737_like.handling_gear_weight() > 0

    def test_fraction_of_w0(self, b737_like):
        """Handling gear ~ 0.03% of W0."""
        w = b737_like.handling_gear_weight()
        expected = 0.0003 * b737_like.W0_kg
        assert w == pytest.approx(expected, rel=0.01)


# ============================================================================ #
# Integration / Weight Statement Tests
# ============================================================================ #

class TestWeightStatement:
    """Tests for weight_statement and empty_weight aggregate methods."""

    def test_returns_dict(self, b737_like):
        ws = b737_like.weight_statement()
        assert isinstance(ws, dict)

    def test_contains_all_components(self, b737_like):
        ws = b737_like.weight_statement()
        expected_keys = {
            "wing", "horizontal_tail", "vertical_tail", "fuselage",
            "main_gear", "nose_gear", "STRUCTURES_SUBTOTAL",
            "installed_engines", "fuel_system", "PROPULSION_SUBTOTAL",
            "flight_controls", "hydraulics", "electrical", "avionics",
            "instruments", "furnishings", "air_conditioning", "anti_ice",
            "handling_gear", "apu", "operational_items",
            "EQUIPMENT_SUBTOTAL", "TOTAL_EMPTY",
        }
        assert set(ws.keys()) == expected_keys

    def test_subtotals_sum_correctly(self, b737_like):
        ws = b737_like.weight_statement()
        struct = (ws["wing"] + ws["horizontal_tail"] + ws["vertical_tail"]
                  + ws["fuselage"] + ws["main_gear"] + ws["nose_gear"])
        assert ws["STRUCTURES_SUBTOTAL"] == pytest.approx(struct, rel=1e-10)

        prop = ws["installed_engines"] + ws["fuel_system"]
        assert ws["PROPULSION_SUBTOTAL"] == pytest.approx(prop, rel=1e-10)

        equip = (ws["flight_controls"] + ws["hydraulics"] + ws["electrical"]
                 + ws["avionics"] + ws["instruments"] + ws["furnishings"]
                 + ws["air_conditioning"] + ws["anti_ice"] + ws["handling_gear"]
                 + ws["apu"] + ws["operational_items"])
        assert ws["EQUIPMENT_SUBTOTAL"] == pytest.approx(equip, rel=1e-10)

    def test_total_equals_group_sums(self, b737_like):
        ws = b737_like.weight_statement()
        total = (ws["STRUCTURES_SUBTOTAL"] + ws["PROPULSION_SUBTOTAL"]
                 + ws["EQUIPMENT_SUBTOTAL"])
        assert ws["TOTAL_EMPTY"] == pytest.approx(total, rel=1e-10)

    def test_all_components_positive(self, b737_like):
        ws = b737_like.weight_statement()
        for key, val in ws.items():
            assert val >= 0, f"{key} should be non-negative, got {val}"

    def test_b737_empty_weight_fraction(self, b737_like):
        """OEW/MTOW for B737-800.

        Actual B737-800 OEW is ~41,413 kg, MTOW = 79,016 kg => 0.524.
        Raymer Ch 15 equations typically produce 75-90% of actual OEW.
        Raw equation output gives We/W0 ~ 0.40-0.50.
        With weight_growth_factor=1.0 (default), expect 0.35-0.60.
        """
        ws = b737_like.weight_statement()
        we_frac = ws["TOTAL_EMPTY"] / b737_like.W0_kg
        assert 0.35 < we_frac < 0.65

    def test_b737_with_growth_factor(self, b737_like):
        """With a 15% weight growth factor, should approach actual OEW."""
        b737_like.weight_growth_factor = 1.15
        ws = b737_like.weight_statement()
        we_frac = ws["TOTAL_EMPTY"] / b737_like.W0_kg
        assert 0.45 < we_frac < 0.65

    def test_b787_empty_weight_fraction(self, b787_like):
        """OEW/MTOW for B787-8.

        Actual B787-8 OEW is ~119,950 kg, MTOW = 228,000 kg => 0.526.
        Statistical equations give raw We/W0 ~ 0.35-0.55.
        """
        ws = b787_like.weight_statement()
        we_frac = ws["TOTAL_EMPTY"] / b787_like.W0_kg
        assert 0.30 < we_frac < 0.60

    def test_print_weight_statement(self, b737_like):
        """Print method should return a non-empty string."""
        s = b737_like.print_weight_statement()
        assert isinstance(s, str)
        assert len(s) > 100
        assert "TOTAL EMPTY WEIGHT" in s
        assert "STRUCTURES GROUP" in s
        assert "PROPULSION GROUP" in s
        assert "EQUIPMENT GROUP" in s


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_minimal_aircraft(self):
        """Minimal valid inputs should not crash."""
        tw = TransportWeights(
            W0_kg=5000.0,
            Wf_kg=1000.0,
            AR=6.0,
            Sref_m2=30.0,
            sweep_quarter_rad=0.0,
            taper_ratio=0.5,
            t_c_root=0.15,
        )
        ws = tw.weight_statement()
        assert ws["TOTAL_EMPTY"] > 0

    def test_zero_tail_areas(self):
        """Zero tail areas should produce zero tail weights without errors."""
        tw = TransportWeights(
            W0_kg=50000.0,
            Wf_kg=10000.0,
            AR=8.0,
            Sref_m2=100.0,
            sweep_quarter_rad=math.radians(25.0),
            taper_ratio=0.3,
            t_c_root=0.12,
            Sht_m2=0.0,
            Svt_m2=0.0,
        )
        assert tw.horizontal_tail_weight() == 0.0
        assert tw.vertical_tail_weight() == 0.0

    def test_no_engines(self):
        """Zero engine weight should give zero installed engine weight."""
        tw = TransportWeights(
            W0_kg=50000.0,
            Wf_kg=10000.0,
            AR=8.0,
            Sref_m2=100.0,
            sweep_quarter_rad=math.radians(25.0),
            taper_ratio=0.3,
            t_c_root=0.12,
            engine_weight_each_kg=0.0,
        )
        assert tw.installed_engine_weight() == 0.0

    def test_unpressurised(self):
        """Unpressurised aircraft should compute without error."""
        tw = TransportWeights(
            W0_kg=50000.0,
            Wf_kg=10000.0,
            AR=8.0,
            Sref_m2=100.0,
            sweep_quarter_rad=math.radians(25.0),
            taper_ratio=0.3,
            t_c_root=0.12,
            pressurized=False,
        )
        ws = tw.weight_statement()
        assert ws["TOTAL_EMPTY"] > 0
