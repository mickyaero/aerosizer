"""
Tests for core.aerodynamics package.

Validates parasite drag, induced drag, and drag polar calculations
against hand-computed values from Raymer equations and physical sanity checks.
"""

import math
import pytest
import numpy as np

from core.aerodynamics.parasite_drag import (
    skin_friction_coeff,
    form_factor_wing,
    form_factor_fuselage,
    form_factor_nacelle,
    reynolds_number,
    component_drag,
    parasite_drag_buildup,
    leakage_protuberance_drag,
)

from core.aerodynamics.induced_drag import (
    oswald_efficiency_straight,
    oswald_efficiency_swept,
    induced_drag_factor,
    induced_drag_coeff,
)

from core.aerodynamics.drag_polar import (
    DragPolar,
    create_transport_polar,
)


# ============================================================================ #
# Parasite Drag Tests
# ============================================================================ #

class TestSkinFrictionCoeff:
    """Tests for skin_friction_coeff (Raymer Eq 12.27)."""

    def test_turbulent_incompressible(self):
        """Fully turbulent at Mach 0 should match Eq 12.27 with M=0."""
        Re = 1e7
        Cf = skin_friction_coeff(Re, mach=0.0, laminar_fraction=0.0)
        # Hand calc: 0.455 / (log10(1e7))^2.58 / (1)^0.65
        #          = 0.455 / (7)^2.58
        #          = 0.455 / 7^2.58
        log_Re = math.log10(Re)  # 7.0
        expected = 0.455 / (log_Re ** 2.58)
        assert Cf == pytest.approx(expected, rel=1e-6)

    def test_turbulent_compressible(self):
        """Compressibility correction should reduce Cf."""
        Re = 1e7
        Cf_M0 = skin_friction_coeff(Re, mach=0.0)
        Cf_M08 = skin_friction_coeff(Re, mach=0.8)
        assert Cf_M08 < Cf_M0

    def test_fully_laminar(self):
        """Blasius solution: Cf = 1.328 / sqrt(Re)."""
        Re = 5e5
        Cf = skin_friction_coeff(Re, mach=0.0, laminar_fraction=1.0)
        expected = 1.328 / math.sqrt(Re)
        assert Cf == pytest.approx(expected, rel=1e-6)

    def test_mixed_flow(self):
        """Mixed Cf should be between laminar and turbulent."""
        Re = 1e7
        Cf_turb = skin_friction_coeff(Re, mach=0.0, laminar_fraction=0.0)
        Cf_lam = skin_friction_coeff(Re, mach=0.0, laminar_fraction=1.0)
        Cf_mix = skin_friction_coeff(Re, mach=0.0, laminar_fraction=0.5)
        assert Cf_lam < Cf_mix < Cf_turb

    def test_mixed_flow_exact(self):
        """50% laminar should be the exact midpoint."""
        Re = 1e7
        Cf_turb = skin_friction_coeff(Re, mach=0.0, laminar_fraction=0.0)
        Cf_lam = skin_friction_coeff(Re, mach=0.0, laminar_fraction=1.0)
        Cf_mix = skin_friction_coeff(Re, mach=0.0, laminar_fraction=0.5)
        expected = 0.5 * Cf_lam + 0.5 * Cf_turb
        assert Cf_mix == pytest.approx(expected, rel=1e-10)

    def test_negative_Re_raises(self):
        with pytest.raises(ValueError, match="positive"):
            skin_friction_coeff(-1.0)

    def test_zero_Re_raises(self):
        with pytest.raises(ValueError, match="positive"):
            skin_friction_coeff(0.0)

    def test_invalid_laminar_fraction_raises(self):
        with pytest.raises(ValueError, match="laminar_fraction"):
            skin_friction_coeff(1e6, laminar_fraction=1.5)

    def test_typical_transport_range(self):
        """At typical transport Re (~50M) and M 0.8, Cf should be ~0.002-0.003."""
        Cf = skin_friction_coeff(5e7, mach=0.8)
        assert 0.001 < Cf < 0.004


class TestFormFactors:
    """Tests for wing, fuselage, and nacelle form factors."""

    def test_wing_ff_basic(self):
        """Raymer Eq 12.30: verify structure."""
        # Typical values: x_c_max=0.3, t_c=0.12, M=0.8, sweep_m=25 deg
        FF = form_factor_wing(0.3, 0.12, 0.8, math.radians(25))
        # Should be in range 1.2-1.8 for typical transonic transport wing
        assert 1.1 < FF < 2.0

    def test_wing_ff_increases_with_thickness(self):
        """Thicker airfoil should have higher form factor."""
        FF_thin = form_factor_wing(0.3, 0.10, 0.8, math.radians(25))
        FF_thick = form_factor_wing(0.3, 0.15, 0.8, math.radians(25))
        assert FF_thick > FF_thin

    def test_wing_ff_zero_x_c_max_raises(self):
        with pytest.raises(ValueError, match="x_c_max"):
            form_factor_wing(0.0, 0.12, 0.8, math.radians(25))

    def test_fuselage_ff_basic(self):
        """Raymer Eq 12.31: typical transport f ~ 8."""
        FF = form_factor_fuselage(8.0)
        # FF = 1 + 60/512 + 8/400 = 1 + 0.1172 + 0.02 = 1.137
        expected = 1.0 + 60.0 / 8.0 ** 3 + 8.0 / 400.0
        assert FF == pytest.approx(expected, rel=1e-6)

    def test_fuselage_ff_decreases_with_fineness(self):
        """More slender fuselage (higher f) should have lower FF (to a point)."""
        FF_stubby = form_factor_fuselage(4.0)
        FF_slender = form_factor_fuselage(10.0)
        assert FF_slender < FF_stubby

    def test_fuselage_ff_zero_raises(self):
        with pytest.raises(ValueError, match="fineness_ratio"):
            form_factor_fuselage(0.0)

    def test_nacelle_ff_basic(self):
        """Raymer Eq 12.32: FF = 1 + 0.35/f."""
        FF = form_factor_nacelle(4.0)
        expected = 1.0 + 0.35 / 4.0
        assert FF == pytest.approx(expected, rel=1e-6)

    def test_nacelle_ff_zero_raises(self):
        with pytest.raises(ValueError, match="fineness_ratio"):
            form_factor_nacelle(0.0)


class TestReynoldsNumber:
    """Tests for reynolds_number."""

    def test_sea_level(self):
        """At sea level, known rho and mu should give predictable Re."""
        # Sea level: rho=1.225, mu ~1.789e-5
        Re = reynolds_number(100.0, 5.0, 0.0)
        # Re = 1.225 * 100 * 5 / 1.789e-5 ~ 34.2M
        assert 3e7 < Re < 4e7

    def test_increases_with_velocity(self):
        """Re should scale linearly with velocity."""
        Re1 = reynolds_number(100.0, 5.0, 0.0)
        Re2 = reynolds_number(200.0, 5.0, 0.0)
        assert Re2 == pytest.approx(2.0 * Re1, rel=1e-6)

    def test_increases_with_length(self):
        """Re should scale linearly with reference length."""
        Re1 = reynolds_number(100.0, 5.0, 0.0)
        Re2 = reynolds_number(100.0, 10.0, 0.0)
        assert Re2 == pytest.approx(2.0 * Re1, rel=1e-6)


class TestComponentDrag:
    """Tests for component_drag (Eq 12.24)."""

    def test_basic_computation(self):
        """CD0_comp = Cf * FF * Q * Swet / Sref."""
        result = component_drag(0.003, 1.3, 1.0, 200.0, 100.0)
        expected = 0.003 * 1.3 * 1.0 * 200.0 / 100.0
        assert result == pytest.approx(expected, rel=1e-10)

    def test_zero_Swet(self):
        """Zero wetted area should give zero drag."""
        assert component_drag(0.003, 1.3, 1.0, 0.0, 100.0) == 0.0


class TestParasiteDragBuildup:
    """Tests for parasite_drag_buildup."""

    def _make_simple_components(self):
        """B737-like simplified component list."""
        return [
            {
                "name": "wing",
                "length": 4.0,  # MAC
                "Swet": 210.0,
                "component_type": "wing",
                "t_c": 0.12,
                "x_c_max": 0.35,
                "sweep_m": math.radians(20),
            },
            {
                "name": "fuselage",
                "length": 33.4,
                "Swet": 380.0,
                "component_type": "fuselage",
                "fineness_ratio": 8.5,
            },
            {
                "name": "htail",
                "length": 2.5,
                "Swet": 60.0,
                "component_type": "tail",
                "t_c": 0.09,
                "x_c_max": 0.30,
                "sweep_m": math.radians(25),
            },
            {
                "name": "vtail",
                "length": 3.0,
                "Swet": 40.0,
                "component_type": "tail",
                "t_c": 0.09,
                "x_c_max": 0.30,
                "sweep_m": math.radians(30),
            },
            {
                "name": "nacelle_L",
                "length": 4.0,
                "Swet": 30.0,
                "component_type": "nacelle",
                "fineness_ratio": 3.5,
            },
            {
                "name": "nacelle_R",
                "length": 4.0,
                "Swet": 30.0,
                "component_type": "nacelle",
                "fineness_ratio": 3.5,
            },
        ]

    def test_returns_dict_structure(self):
        """Result should have CD0 and breakdown keys."""
        comps = self._make_simple_components()
        result = parasite_drag_buildup(comps, 124.6, 10000, 230.0)
        assert "CD0" in result
        assert "breakdown" in result
        assert len(result["breakdown"]) == 6

    def test_cd0_reasonable_range(self):
        """Transport aircraft CD0 should be ~0.015-0.030."""
        comps = self._make_simple_components()
        result = parasite_drag_buildup(comps, 124.6, 10000, 230.0)
        assert 0.010 < result["CD0"] < 0.040

    def test_breakdown_sums_to_total(self):
        """Sum of component CD0s should equal total."""
        comps = self._make_simple_components()
        result = parasite_drag_buildup(comps, 124.6, 10000, 230.0)
        total_from_breakdown = sum(
            v["CD0_comp"] for v in result["breakdown"].values()
        )
        assert total_from_breakdown == pytest.approx(result["CD0"], rel=1e-10)

    def test_unknown_component_type_raises(self):
        comps = [{"name": "x", "length": 1.0, "Swet": 10.0,
                  "component_type": "engine"}]
        with pytest.raises(ValueError, match="Unknown component_type"):
            parasite_drag_buildup(comps, 100.0, 0.0, 100.0)

    def test_custom_Q_override(self):
        """Per-component Q should override the default."""
        comps = [{
            "name": "wing",
            "length": 4.0,
            "Swet": 200.0,
            "component_type": "wing",
            "t_c": 0.12,
            "x_c_max": 0.35,
            "sweep_m": 0.0,
            "Q": 1.25,
        }]
        result = parasite_drag_buildup(comps, 100.0, 0.0, 100.0)
        assert result["breakdown"]["wing"]["Q"] == 1.25


class TestLeakageProtuberanceDrag:
    """Tests for leakage_protuberance_drag."""

    def test_default_3_percent(self):
        CD0 = 0.020
        CD_LP = leakage_protuberance_drag(CD0)
        assert CD_LP == pytest.approx(0.020 * 0.03, rel=1e-10)

    def test_custom_fraction(self):
        CD0 = 0.020
        CD_LP = leakage_protuberance_drag(CD0, fraction=0.05)
        assert CD_LP == pytest.approx(0.001, rel=1e-10)


# ============================================================================ #
# Induced Drag Tests
# ============================================================================ #

class TestOswaldEfficiency:
    """Tests for Oswald span efficiency estimations."""

    def test_straight_wing_typical(self):
        """AR=8 straight wing: e should be ~0.7-0.85."""
        e = oswald_efficiency_straight(8.0)
        assert 0.6 < e < 0.9

    def test_straight_wing_formula(self):
        """Verify Eq 12.49 at AR=10."""
        AR = 10.0
        e = oswald_efficiency_straight(AR)
        expected = 1.78 * (1.0 - 0.045 * AR ** 0.68) - 0.64
        assert e == pytest.approx(expected, rel=1e-10)

    def test_straight_wing_negative_AR_raises(self):
        with pytest.raises(ValueError, match="positive"):
            oswald_efficiency_straight(-1.0)

    def test_swept_wing_typical(self):
        """AR=9, sweep_LE=30 deg: Eq 12.50 gives e ~ 0.51 for this config.

        Note: Raymer Eq 12.50 tends to give lower e values at high AR+sweep
        combinations. Values in the 0.4-0.8 range are expected for typical
        swept-wing transports depending on AR and sweep.
        """
        e = oswald_efficiency_swept(9.0, math.radians(30))
        assert 0.3 < e < 0.95

    def test_swept_wing_formula(self):
        """Verify Eq 12.50 at AR=9, sweep=30 deg."""
        AR = 9.0
        sweep = math.radians(30)
        e = oswald_efficiency_swept(AR, sweep)
        expected = 4.61 * (1.0 - 0.045 * AR ** 0.68) * math.cos(sweep) ** 0.15 - 3.1
        assert e == pytest.approx(expected, rel=1e-10)

    def test_swept_lower_than_straight(self):
        """Sweep generally reduces e compared to unswept."""
        AR = 8.0
        e_str = oswald_efficiency_straight(AR)
        e_swp = oswald_efficiency_swept(AR, math.radians(30))
        assert e_swp < e_str


class TestInducedDrag:
    """Tests for induced_drag_factor and induced_drag_coeff."""

    def test_induced_drag_factor_formula(self):
        """K = 1/(pi * AR * e)."""
        AR, e = 9.0, 0.8
        K = induced_drag_factor(AR, e)
        expected = 1.0 / (math.pi * AR * e)
        assert K == pytest.approx(expected, rel=1e-10)

    def test_induced_drag_factor_invalid_AR(self):
        with pytest.raises(ValueError):
            induced_drag_factor(0.0, 0.8)

    def test_induced_drag_factor_invalid_e(self):
        with pytest.raises(ValueError):
            induced_drag_factor(9.0, 0.0)

    def test_induced_drag_coeff(self):
        """CDi = K * CL^2."""
        K, CL = 0.04, 0.5
        CDi = induced_drag_coeff(CL, K)
        assert CDi == pytest.approx(0.04 * 0.25, rel=1e-10)

    def test_induced_drag_zero_CL(self):
        """Zero lift should produce zero induced drag."""
        assert induced_drag_coeff(0.0, 0.05) == 0.0


# ============================================================================ #
# Drag Polar Tests
# ============================================================================ #

class TestDragPolar:
    """Tests for DragPolar class."""

    @pytest.fixture
    def polar(self):
        """Typical transport polar: CD0=0.020, K=0.045."""
        return DragPolar(CD0=0.020, K=0.045)

    def test_cd_clean(self, polar):
        """CD = CD0 + K*CL^2."""
        cd = polar.cd(0.5)
        expected = 0.020 + 0.045 * 0.25
        assert cd == pytest.approx(expected, rel=1e-10)

    def test_cd_at_zero_lift(self, polar):
        """At CL=0, CD should equal CD0."""
        assert polar.cd(0.0) == pytest.approx(polar.CD0, rel=1e-10)

    def test_cd_takeoff(self, polar):
        """Takeoff config adds increment."""
        cd = polar.cd_takeoff(0.5)
        expected = (0.020 + 0.02) + 0.045 * 0.25
        assert cd == pytest.approx(expected, rel=1e-10)

    def test_cd_landing(self, polar):
        """Landing config adds larger increment."""
        cd = polar.cd_landing(0.5)
        expected = (0.020 + 0.07) + 0.045 * 0.25
        assert cd == pytest.approx(expected, rel=1e-10)

    def test_cd_landing_greater_than_takeoff(self, polar):
        """Landing drag should exceed takeoff drag at same CL."""
        assert polar.cd_landing(0.8) > polar.cd_takeoff(0.8)

    def test_cl_for_max_ld(self, polar):
        """CL_maxLD = sqrt(CD0/K) -- Raymer Eq 17.14."""
        cl = polar.cl_for_max_ld()
        expected = math.sqrt(0.020 / 0.045)
        assert cl == pytest.approx(expected, rel=1e-10)

    def test_max_ld(self, polar):
        """(L/D)_max = 1/(2*sqrt(CD0*K)) -- Raymer Eq 17.15."""
        ld = polar.max_ld()
        expected = 1.0 / (2.0 * math.sqrt(0.020 * 0.045))
        assert ld == pytest.approx(expected, rel=1e-10)

    def test_max_ld_at_optimal_cl(self, polar):
        """L/D at CL_opt should equal (L/D)_max."""
        cl_opt = polar.cl_for_max_ld()
        ld_at_opt = polar.ld_at(cl_opt)
        assert ld_at_opt == pytest.approx(polar.max_ld(), rel=1e-6)

    def test_cl_for_max_ld_cruise_same_as_max_ld(self, polar):
        """For jets, max range CL = max L/D CL."""
        assert polar.cl_for_max_ld_cruise() == pytest.approx(
            polar.cl_for_max_ld(), rel=1e-10
        )

    def test_cl_for_min_power(self, polar):
        """CL_min_power = sqrt(3*CD0/K) -- Raymer Eq 17.28."""
        cl = polar.cl_for_min_power()
        expected = math.sqrt(3.0 * 0.020 / 0.045)
        assert cl == pytest.approx(expected, rel=1e-10)

    def test_min_power_cl_greater_than_max_ld_cl(self, polar):
        """CL for min power > CL for max L/D (since sqrt(3) > 1)."""
        assert polar.cl_for_min_power() > polar.cl_for_max_ld()

    def test_ld_at(self, polar):
        """L/D at CL=0.5."""
        ld = polar.ld_at(0.5)
        cd = 0.020 + 0.045 * 0.25
        expected = 0.5 / cd
        assert ld == pytest.approx(expected, rel=1e-10)

    def test_generate_polar_default(self, polar):
        """Default generate_polar should return 100-element arrays."""
        result = polar.generate_polar()
        assert len(result["CL"]) == 100
        assert len(result["CD"]) == 100
        assert len(result["LD"]) == 100
        # First element CL=0, LD=0
        assert result["CL"][0] == 0.0
        assert result["LD"][0] == 0.0

    def test_generate_polar_custom_range(self, polar):
        """Custom CL range should work."""
        cl_arr = np.linspace(0.1, 1.0, 50)
        result = polar.generate_polar(cl_arr)
        assert len(result["CL"]) == 50

    def test_generate_polar_cd_values(self, polar):
        """CD values should follow the parabolic equation."""
        cl_arr = np.array([0.0, 0.5, 1.0])
        result = polar.generate_polar(cl_arr)
        expected_cd = np.array([
            0.020,
            0.020 + 0.045 * 0.25,
            0.020 + 0.045 * 1.0,
        ])
        np.testing.assert_allclose(result["CD"], expected_cd, rtol=1e-10)

    def test_max_ld_realistic(self, polar):
        """Typical transport max L/D should be 15-20."""
        ld_max = polar.max_ld()
        assert 10.0 < ld_max < 25.0

    def test_K_takeoff_default(self):
        """If K_takeoff not specified, should equal K."""
        polar = DragPolar(CD0=0.02, K=0.045)
        assert polar.K_takeoff == polar.K

    def test_K_takeoff_custom(self):
        """Custom K_takeoff should be used."""
        polar = DragPolar(CD0=0.02, K=0.045, K_takeoff=0.05)
        assert polar.K_takeoff == 0.05


class TestCreateTransportPolar:
    """Tests for create_transport_polar factory function."""

    def test_default_no_components(self):
        """Without components, should use typical CD0 ~0.018."""
        polar = create_transport_polar(124.6, 10668.0, 0.85)
        assert polar.CD0 == pytest.approx(0.018, rel=1e-10)
        assert polar.K > 0

    def test_with_cd0_override(self):
        """CD0 override should be used directly."""
        polar = create_transport_polar(
            124.6, 10668.0, 0.85, CD0_override=0.025
        )
        assert polar.CD0 == pytest.approx(0.025, rel=1e-10)

    def test_with_components(self):
        """Buildup with components should produce reasonable CD0."""
        components = [
            {
                "name": "wing",
                "length": 4.0,
                "Swet": 210.0,
                "component_type": "wing",
                "t_c": 0.12,
                "x_c_max": 0.35,
                "sweep_m": math.radians(20),
            },
            {
                "name": "fuselage",
                "length": 33.4,
                "Swet": 380.0,
                "component_type": "fuselage",
                "fineness_ratio": 8.5,
            },
        ]
        polar = create_transport_polar(
            Sref=124.6,
            altitude_m=10668.0,
            mach=0.85,
            components=components,
        )
        assert 0.005 < polar.CD0 < 0.035
        assert polar.K > 0

    def test_returns_drag_polar_instance(self):
        polar = create_transport_polar(100.0, 10000.0, 0.80)
        assert isinstance(polar, DragPolar)


# ============================================================================ #
# Integration / End-to-End Tests
# ============================================================================ #

class TestEndToEnd:
    """Full pipeline integration tests."""

    def test_full_buildup_to_polar(self):
        """Run component buildup through to drag polar and verify L/D."""
        components = [
            {
                "name": "wing",
                "length": 4.2,
                "Swet": 220.0,
                "component_type": "wing",
                "t_c": 0.11,
                "x_c_max": 0.37,
                "sweep_m": math.radians(25),
            },
            {
                "name": "fuselage",
                "length": 38.0,
                "Swet": 420.0,
                "component_type": "fuselage",
                "fineness_ratio": 9.0,
            },
            {
                "name": "htail",
                "length": 2.2,
                "Swet": 55.0,
                "component_type": "tail",
                "t_c": 0.09,
                "x_c_max": 0.30,
                "sweep_m": math.radians(28),
            },
            {
                "name": "vtail",
                "length": 3.5,
                "Swet": 45.0,
                "component_type": "tail",
                "t_c": 0.09,
                "x_c_max": 0.30,
                "sweep_m": math.radians(35),
            },
            {
                "name": "nacelle_L",
                "length": 4.5,
                "Swet": 35.0,
                "component_type": "nacelle",
                "fineness_ratio": 3.8,
            },
            {
                "name": "nacelle_R",
                "length": 4.5,
                "Swet": 35.0,
                "component_type": "nacelle",
                "fineness_ratio": 3.8,
            },
        ]
        Sref = 130.0
        polar = create_transport_polar(
            Sref=Sref,
            altitude_m=10668.0,
            mach=0.85,
            components=components,
            AR=9.5,
            sweep_LE_rad=math.radians(32),
        )

        # CD0 should be transport-like
        assert 0.010 < polar.CD0 < 0.035

        # Max L/D should be transport-like (15-22)
        max_ld = polar.max_ld()
        assert 12.0 < max_ld < 25.0

        # CL for max L/D should be reasonable
        cl_opt = polar.cl_for_max_ld()
        assert 0.3 < cl_opt < 1.0

    def test_drag_increases_from_clean_to_landing(self):
        """Verify CD order: clean < takeoff < landing at any CL."""
        polar = DragPolar(CD0=0.020, K=0.045)
        cl = 0.6
        assert polar.cd(cl) < polar.cd_takeoff(cl) < polar.cd_landing(cl)
