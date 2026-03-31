"""
Tests for core.sizing.tw_ws_selection module.

Validates Raymer Ch 5 constraint analysis (T/W vs W/S diagram)
against physical sanity checks and typical transport values.
"""

import math
import pytest
import numpy as np

from core.sizing.tw_ws_selection import ConstraintAnalysis
from core.atmosphere import G0, RHO0, m_to_ft


# ============================================================================ #
# Fixtures
# ============================================================================ #

@pytest.fixture
def default_ca():
    """Default constraint analysis for a typical twin-engine transport."""
    return ConstraintAnalysis()


@pytest.fixture
def b737_ca():
    """B737-800 class constraint analysis.

    CD0 ~ 0.020, K ~ 0.045 (AR=9.44, e=0.75)
    Cruise M0.78 at 35,000 ft, BFL=2250 m, Landing=1650 m
    """
    return ConstraintAnalysis(
        CD0=0.020,
        K=0.045,
        CLmax_clean=1.5,
        CLmax_takeoff=1.8,
        CLmax_landing=2.2,
        cruise_mach=0.785,
        cruise_altitude_m=10668.0,
        takeoff_distance_m=2250.0,
        landing_distance_m=1650.0,
        climb_gradient_oei=0.024,
        service_ceiling_m=12497.0,  # 41,000 ft
        n_engines=2,
        thrust_lapse_cruise=0.24,
        bypass_ratio=5.0,
    )


@pytest.fixture
def ws_range():
    """Standard W/S range array [Pa]."""
    return np.linspace(2000.0, 8000.0, 300)


# ============================================================================ #
# Cruise Constraint Tests
# ============================================================================ #

class TestCruiseConstraint:
    """Tests for cruise_constraint (Raymer Eq 5.6)."""

    def test_returns_array(self, default_ca, ws_range):
        tw = default_ca.cruise_constraint(ws_range)
        assert isinstance(tw, np.ndarray)
        assert len(tw) == len(ws_range)

    def test_all_positive(self, default_ca, ws_range):
        tw = default_ca.cruise_constraint(ws_range)
        assert np.all(tw > 0)

    def test_u_shape(self, default_ca):
        """Cruise T/W should be U-shaped (min in the middle).

        At low W/S: parasite drag dominates (q*CD0/(W/S)).
        At high W/S: induced drag dominates (K*(W/S)/q).

        The optimal cruise W/S for M0.85 at 35kft is ~9000 Pa, so we
        use a wide range to capture the U-shape.
        """
        ws = np.linspace(2000.0, 16000.0, 500)
        tw = default_ca.cruise_constraint(ws)
        idx_min = np.argmin(tw)
        # Minimum should not be at the edges
        assert 10 < idx_min < 490

    def test_minimum_tw_reasonable(self, default_ca):
        """Minimum cruise T/W should be ~0.15-0.35 for a typical transport.

        With thrust_lapse_cruise=0.25 (T_cruise/T_SL), the minimum
        sea-level T/W for cruise is in the 0.20-0.30 range.
        """
        ws = np.linspace(2000.0, 16000.0, 500)
        tw = default_ca.cruise_constraint(ws)
        assert 0.10 < np.min(tw) < 0.40

    def test_sensitive_to_cd0(self, default_ca, ws_range):
        """Higher CD0 should increase T/W at all W/S."""
        tw1 = default_ca.cruise_constraint(ws_range).copy()
        default_ca.CD0 = 0.030
        tw2 = default_ca.cruise_constraint(ws_range)
        assert np.all(tw2 > tw1)

    def test_sensitive_to_thrust_lapse(self, default_ca, ws_range):
        """Lower thrust lapse (worse engine at altitude) -> higher T/W required."""
        tw1 = default_ca.cruise_constraint(ws_range).copy()
        default_ca.thrust_lapse_cruise = 0.15
        tw2 = default_ca.cruise_constraint(ws_range)
        assert np.all(tw2 > tw1)


# ============================================================================ #
# Takeoff Constraint Tests
# ============================================================================ #

class TestTakeoffConstraint:
    """Tests for takeoff_constraint (Raymer Ch 5, Ch 17 BFL)."""

    def test_returns_array(self, default_ca, ws_range):
        tw = default_ca.takeoff_constraint(ws_range)
        assert isinstance(tw, np.ndarray)
        assert len(tw) == len(ws_range)

    def test_all_positive(self, default_ca, ws_range):
        tw = default_ca.takeoff_constraint(ws_range)
        assert np.all(tw > 0)

    def test_increases_with_ws(self, default_ca, ws_range):
        """Higher wing loading requires more thrust for same takeoff distance."""
        tw = default_ca.takeoff_constraint(ws_range)
        assert np.all(np.diff(tw) > 0)

    def test_linear_relationship(self, default_ca, ws_range):
        """T/W should be linear with W/S for the takeoff constraint."""
        tw = default_ca.takeoff_constraint(ws_range)
        # Check linearity: fit a line and verify R^2 ~ 1.0
        coeffs = np.polyfit(ws_range, tw, 1)
        tw_fit = np.polyval(coeffs, ws_range)
        ss_res = np.sum((tw - tw_fit) ** 2)
        ss_tot = np.sum((tw - np.mean(tw)) ** 2)
        r_squared = 1.0 - ss_res / ss_tot
        assert r_squared > 0.999

    def test_shorter_bfl_needs_more_thrust(self, default_ca, ws_range):
        """Shorter takeoff distance requires more T/W."""
        tw_long = default_ca.takeoff_constraint(ws_range).copy()
        default_ca.takeoff_distance_m = 1500.0
        tw_short = default_ca.takeoff_constraint(ws_range)
        assert np.all(tw_short > tw_long)

    def test_higher_clmax_helps(self, default_ca, ws_range):
        """Higher CLmax_takeoff reduces T/W requirement."""
        tw1 = default_ca.takeoff_constraint(ws_range).copy()
        default_ca.CLmax_takeoff = 2.5
        tw2 = default_ca.takeoff_constraint(ws_range)
        assert np.all(tw2 < tw1)


# ============================================================================ #
# Landing Constraint Tests
# ============================================================================ #

class TestLandingConstraint:
    """Tests for landing_constraint (Raymer Ch 5)."""

    def test_returns_scalar(self, default_ca):
        ws_max = default_ca.landing_constraint()
        assert isinstance(ws_max, float)

    def test_positive(self, default_ca):
        assert default_ca.landing_constraint() > 0

    def test_reasonable_value(self, default_ca):
        """Landing W/S should be ~4000-8000 Pa for typical transport."""
        ws_max = default_ca.landing_constraint()
        assert 2000 < ws_max < 12000

    def test_longer_runway_allows_higher_ws(self, default_ca):
        """Longer landing distance allows higher wing loading."""
        ws1 = default_ca.landing_constraint()
        default_ca.landing_distance_m = 2500.0
        ws2 = default_ca.landing_constraint()
        assert ws2 > ws1

    def test_higher_clmax_allows_higher_ws(self, default_ca):
        """Higher CLmax_landing allows higher wing loading."""
        ws1 = default_ca.landing_constraint()
        default_ca.CLmax_landing = 3.0
        ws2 = default_ca.landing_constraint()
        assert ws2 > ws1


# ============================================================================ #
# Climb Gradient Constraint Tests
# ============================================================================ #

class TestClimbGradientConstraint:
    """Tests for climb_gradient_constraint (FAR 25 2nd segment OEI)."""

    def test_returns_array(self, default_ca, ws_range):
        tw = default_ca.climb_gradient_constraint(ws_range)
        assert isinstance(tw, np.ndarray)
        assert len(tw) == len(ws_range)

    def test_all_positive(self, default_ca, ws_range):
        tw = default_ca.climb_gradient_constraint(ws_range)
        assert np.all(tw > 0)

    def test_constant_value(self, default_ca, ws_range):
        """OEI climb constraint should be approximately constant (horizontal line).

        It is exactly constant since the CL_climb is fixed and does not depend on W/S.
        """
        tw = default_ca.climb_gradient_constraint(ws_range)
        assert np.allclose(tw, tw[0], rtol=1e-10)

    def test_reasonable_value(self, default_ca):
        """OEI climb T/W should be ~0.20-0.40 for twin-engine transport."""
        ws = np.array([5000.0])
        tw = default_ca.climb_gradient_constraint(ws)
        assert 0.10 < tw[0] < 0.60

    def test_twin_vs_quad(self, default_ca, ws_range):
        """Twin engine should require more T/W than quad for same gradient.

        N/(N-1) is 2.0 for twin vs 1.33 for quad.
        """
        tw_twin = default_ca.climb_gradient_constraint(ws_range).copy()
        default_ca.n_engines = 4
        tw_quad = default_ca.climb_gradient_constraint(ws_range)
        assert np.all(tw_twin > tw_quad)

    def test_steeper_gradient_needs_more_thrust(self, default_ca, ws_range):
        """Higher required gradient increases T/W."""
        tw1 = default_ca.climb_gradient_constraint(ws_range).copy()
        default_ca.climb_gradient_oei = 0.05
        tw2 = default_ca.climb_gradient_constraint(ws_range)
        assert np.all(tw2 > tw1)


# ============================================================================ #
# Ceiling Constraint Tests
# ============================================================================ #

class TestCeilingConstraint:
    """Tests for ceiling_constraint (Raymer Ch 5)."""

    def test_returns_array(self, default_ca, ws_range):
        tw = default_ca.ceiling_constraint(ws_range)
        assert isinstance(tw, np.ndarray)
        assert len(tw) == len(ws_range)

    def test_all_positive(self, default_ca, ws_range):
        tw = default_ca.ceiling_constraint(ws_range)
        assert np.all(tw > 0)

    def test_reasonable_values(self, default_ca):
        """Ceiling T/W should be in a reasonable range."""
        ws = np.linspace(3000.0, 7000.0, 100)
        tw = default_ca.ceiling_constraint(ws)
        assert np.all(tw > 0.01)
        assert np.all(tw < 2.0)

    def test_higher_ceiling_needs_more_thrust(self, default_ca, ws_range):
        """Higher service ceiling requires more T/W."""
        tw1 = default_ca.ceiling_constraint(ws_range).copy()
        default_ca.service_ceiling_m = 13716.0  # 45,000 ft
        tw2 = default_ca.ceiling_constraint(ws_range)
        assert np.all(tw2 > tw1)


# ============================================================================ #
# Stall Constraint Tests
# ============================================================================ #

class TestStallConstraint:
    """Tests for stall_constraint."""

    def test_returns_none_when_unspecified(self, default_ca):
        assert default_ca.stall_constraint() is None

    def test_returns_float_when_specified(self, default_ca):
        ws = default_ca.stall_constraint(stall_speed_ms=70.0)
        assert isinstance(ws, float)

    def test_formula_correct(self, default_ca):
        """W/S = 0.5 * rho0 * Vs^2 * CLmax."""
        Vs = 70.0
        ws = default_ca.stall_constraint(Vs)
        expected = 0.5 * RHO0 * Vs ** 2 * default_ca.CLmax_clean
        assert ws == pytest.approx(expected, rel=1e-10)

    def test_higher_stall_speed_allows_higher_ws(self, default_ca):
        """Higher stall speed permits higher wing loading."""
        ws1 = default_ca.stall_constraint(60.0)
        ws2 = default_ca.stall_constraint(80.0)
        assert ws2 > ws1


# ============================================================================ #
# Aggregate / Integration Tests
# ============================================================================ #

class TestComputeAll:
    """Tests for compute_all."""

    def test_returns_dict(self, default_ca):
        result = default_ca.compute_all()
        assert isinstance(result, dict)

    def test_has_all_keys(self, default_ca):
        result = default_ca.compute_all()
        expected_keys = {"ws", "cruise", "takeoff", "climb", "ceiling",
                         "landing_ws_max", "stall_ws_max"}
        assert set(result.keys()) == expected_keys

    def test_constraint_shapes_match(self, default_ca):
        result = default_ca.compute_all()
        n = len(result["ws"])
        assert len(result["cruise"]["tw"]) == n
        assert len(result["takeoff"]["tw"]) == n
        assert len(result["climb"]["tw"]) == n
        assert len(result["ceiling"]["tw"]) == n

    def test_custom_ws_range(self, default_ca):
        ws = np.linspace(3000.0, 6000.0, 50)
        result = default_ca.compute_all(ws_range=ws)
        assert len(result["ws"]) == 50

    def test_stall_ws_none_by_default(self, default_ca):
        result = default_ca.compute_all()
        assert result["stall_ws_max"] is None

    def test_stall_ws_set_when_given(self, default_ca):
        result = default_ca.compute_all(stall_speed_ms=70.0)
        assert result["stall_ws_max"] is not None
        assert result["stall_ws_max"] > 0


class TestFindDesignPoint:
    """Tests for find_design_point."""

    def test_returns_tuple(self, default_ca):
        ws, tw = default_ca.find_design_point()
        assert isinstance(ws, float)
        assert isinstance(tw, float)

    def test_positive_values(self, default_ca):
        ws, tw = default_ca.find_design_point()
        assert ws > 0
        assert tw > 0

    def test_ws_within_feasible_region(self, default_ca):
        """Design W/S should not exceed landing W/S limit."""
        ws, tw = default_ca.find_design_point()
        landing_ws = default_ca.landing_constraint()
        assert ws <= landing_ws * 1.01  # small tolerance

    def test_tw_reasonable_for_transport(self, default_ca):
        """Design T/W for transport should be ~0.25-0.40."""
        ws, tw = default_ca.find_design_point()
        assert 0.10 < tw < 0.60

    def test_ws_reasonable_for_transport(self, default_ca):
        """Design W/S for transport should be ~4000-7000 Pa (~400-700 kg/m^2 * g)."""
        ws, tw = default_ca.find_design_point()
        assert 2000 < ws < 9000

    def test_b737_class_design_point(self, b737_ca):
        """B737-800 design point should be in a reasonable range.

        B737-800: T/W ~ 0.31, W/S ~ 6100 Pa (623 kg/m^2).
        Statistical methods should get within a neighborhood.
        """
        ws, tw = b737_ca.find_design_point()
        assert 3000 < ws < 8000
        assert 0.15 < tw < 0.50


class TestPlotData:
    """Tests for plot_data."""

    def test_returns_dict(self, default_ca):
        data = default_ca.plot_data()
        assert isinstance(data, dict)

    def test_has_traces(self, default_ca):
        data = default_ca.plot_data()
        assert "traces" in data
        assert len(data["traces"]) >= 5  # cruise, takeoff, climb, ceiling, landing

    def test_has_design_point(self, default_ca):
        data = default_ca.plot_data()
        assert "design_point" in data
        assert "ws" in data["design_point"]
        assert "tw" in data["design_point"]

    def test_trace_types(self, default_ca):
        data = default_ca.plot_data()
        types = {t["type"] for t in data["traces"]}
        assert "line" in types
        assert "vline" in types

    def test_line_traces_have_arrays(self, default_ca):
        data = default_ca.plot_data()
        for trace in data["traces"]:
            if trace["type"] == "line":
                assert isinstance(trace["ws"], list)
                assert isinstance(trace["tw"], list)
                assert len(trace["ws"]) == len(trace["tw"])

    def test_vline_traces_have_scalar_ws(self, default_ca):
        data = default_ca.plot_data()
        for trace in data["traces"]:
            if trace["type"] == "vline":
                assert isinstance(trace["ws"], float)

    def test_stall_trace_added(self, default_ca):
        data = default_ca.plot_data(stall_speed_ms=65.0)
        names = {t["name"] for t in data["traces"]}
        assert "Stall (max W/S)" in names


class TestSummary:
    """Tests for summary method."""

    def test_returns_string(self, default_ca):
        s = default_ca.summary()
        assert isinstance(s, str)

    def test_contains_key_info(self, default_ca):
        s = default_ca.summary()
        assert "CONSTRAINT ANALYSIS" in s
        assert "Design W/S" in s
        assert "Design T/W" in s
        assert "Landing W/S max" in s

    def test_contains_inputs(self, default_ca):
        s = default_ca.summary()
        assert "CD0" in s
        assert "Cruise Mach" in s


# ============================================================================ #
# Internal Helper Tests
# ============================================================================ #

class TestInternalHelpers:
    """Tests for internal helper methods."""

    def test_max_ld(self, default_ca):
        """(L/D)_max = 1/(2*sqrt(CD0*K))."""
        ld = default_ca._max_ld()
        expected = 1.0 / (2.0 * math.sqrt(0.020 * 0.04))
        assert ld == pytest.approx(expected, rel=1e-10)

    def test_cl_for_max_ld(self, default_ca):
        """CL* = sqrt(CD0/K)."""
        cl = default_ca._cl_for_max_ld()
        expected = math.sqrt(0.020 / 0.04)
        assert cl == pytest.approx(expected, rel=1e-10)

    def test_max_ld_reasonable(self, default_ca):
        """Max L/D for transport should be ~15-20."""
        ld = default_ca._max_ld()
        assert 10 < ld < 25

    def test_thrust_lapse_decreases_with_altitude(self, default_ca):
        """Thrust lapse should decrease at higher altitude."""
        alpha_0 = default_ca._thrust_lapse_at(0.0)
        alpha_35k = default_ca._thrust_lapse_at(10668.0)
        assert alpha_35k < alpha_0

    def test_thrust_lapse_sea_level(self, default_ca):
        """At sea level, sigma=1, so alpha should be ~1.0."""
        alpha = default_ca._thrust_lapse_at(0.0)
        assert alpha == pytest.approx(1.0, rel=0.01)
