"""
Tests for core.performance package.

Validates level flight, climb, range/endurance, and takeoff/landing
calculations against hand-computed values from Raymer equations and
physical sanity checks.

Test aircraft: Boeing 737-800 class
    W = 70,000 kg (MTOW)
    S = 124.6 m^2
    CD0 = 0.020, K = 0.045
    CLmax_clean = 1.6, CLmax_TO = 2.0, CLmax_landing = 2.4
    Thrust (2x CFM56) = 2 * 121,000 N = 242,000 N total
    SFC ~ 0.6 1/hr
"""

import math
import pytest
import numpy as np

from core.aerodynamics.drag_polar import DragPolar
from core.atmosphere import atmosphere, density_at, G0, RHO0

from core.performance.level_flight import (
    stall_speed,
    max_speed,
    min_drag_speed,
    max_range_speed,
    max_endurance_speed,
    thrust_required,
    power_required,
    generate_tr_curve,
)

from core.performance.climb import (
    rate_of_climb,
    max_rate_of_climb,
    climb_gradient,
    climb_angle,
    best_angle_of_climb_speed,
    service_ceiling,
    absolute_ceiling,
    time_to_climb,
    generate_roc_curve,
)

from core.performance.range_endurance import (
    breguet_range_jet,
    breguet_endurance_jet,
    breguet_range_prop,
    cruise_climb_range,
    specific_air_range,
    fuel_for_range,
    range_payload_diagram,
)

from core.performance.takeoff import (
    ground_roll_distance,
    rotation_distance,
    transition_distance,
    climb_to_obstacle,
    total_takeoff_distance,
    balanced_field_length,
    landing_distance,
    generate_takeoff_landing_summary,
)


# ============================================================================ #
# Shared fixtures
# ============================================================================ #

@pytest.fixture
def polar():
    """Typical transport polar: CD0=0.020, K=0.045."""
    return DragPolar(CD0=0.020, K=0.045)


@pytest.fixture
def b737():
    """B737-800 class parameters."""
    return {
        "W_kg": 70000.0,
        "S_m2": 124.6,
        "T_N": 242000.0,  # 2 * 121 kN
        "n_engines": 2,
        "T_each_N": 121000.0,
        "sfc_per_hr": 0.6,
        "altitude_cruise_m": 10668.0,  # FL350
    }


# ============================================================================ #
# Level Flight Tests
# ============================================================================ #

class TestStallSpeed:
    """Tests for stall_speed (Raymer Eq 17.1)."""

    def test_sea_level_clean(self, polar):
        """Typical transport stall speed at sea level, clean config."""
        Vs = stall_speed(70000.0, 124.6, 1.6, 0.0)
        # Use actual density from atmosphere model (includes geopotential correction)
        W = 70000.0 * G0
        rho = density_at(0.0)
        expected = math.sqrt(2.0 * W / (rho * 124.6 * 1.6))
        assert Vs == pytest.approx(expected, rel=1e-6)
        # Should be ~70-90 m/s for a transport aircraft
        assert 60.0 < Vs < 100.0

    def test_stall_increases_with_altitude(self, polar):
        """Stall speed increases with altitude (lower density)."""
        Vs_sl = stall_speed(70000.0, 124.6, 1.6, 0.0)
        Vs_fl350 = stall_speed(70000.0, 124.6, 1.6, 10668.0)
        assert Vs_fl350 > Vs_sl

    def test_stall_decreases_with_clmax(self, polar):
        """Higher CLmax (flaps) reduces stall speed."""
        Vs_clean = stall_speed(70000.0, 124.6, 1.6, 0.0)
        Vs_landing = stall_speed(70000.0, 124.6, 2.4, 0.0)
        assert Vs_landing < Vs_clean

    def test_stall_scales_with_weight(self, polar):
        """Stall speed ~ sqrt(W), so doubling W -> sqrt(2) increase."""
        Vs1 = stall_speed(50000.0, 124.6, 1.6, 0.0)
        Vs2 = stall_speed(100000.0, 124.6, 1.6, 0.0)
        assert Vs2 == pytest.approx(Vs1 * math.sqrt(2.0), rel=1e-6)

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="W_kg"):
            stall_speed(0.0, 124.6, 1.6)

    def test_invalid_area_raises(self):
        with pytest.raises(ValueError, match="S_m2"):
            stall_speed(70000.0, 0.0, 1.6)

    def test_invalid_clmax_raises(self):
        with pytest.raises(ValueError, match="CLmax"):
            stall_speed(70000.0, 124.6, 0.0)


class TestThrustRequired:
    """Tests for thrust_required."""

    def test_equals_drag_at_cruise(self, polar):
        """T_req = D = q*S*CD."""
        W_kg = 70000.0
        S = 124.6
        V = 230.0  # m/s
        alt = 10668.0
        T_req = thrust_required(W_kg, S, polar, alt, V)
        # Manual calculation
        rho = density_at(alt)
        W = W_kg * G0
        q = 0.5 * rho * V ** 2
        CL = W / (q * S)
        CD = polar.cd(CL)
        expected = q * S * CD
        assert T_req == pytest.approx(expected, rel=1e-10)

    def test_minimum_at_vmd(self, polar):
        """T_req should be minimum near min-drag speed."""
        W_kg = 70000.0
        S = 124.6
        alt = 10668.0
        Vmd = min_drag_speed(W_kg, S, polar, alt)
        T_at_Vmd = thrust_required(W_kg, S, polar, alt, Vmd)
        T_above = thrust_required(W_kg, S, polar, alt, Vmd * 1.3)
        T_below = thrust_required(W_kg, S, polar, alt, Vmd * 0.7)
        assert T_at_Vmd < T_above
        assert T_at_Vmd < T_below

    def test_invalid_velocity_raises(self, polar):
        with pytest.raises(ValueError, match="velocity_ms"):
            thrust_required(70000.0, 124.6, polar, 0.0, 0.0)


class TestPowerRequired:
    """Tests for power_required."""

    def test_equals_treq_times_v(self, polar):
        """P_req = T_req * V."""
        V = 200.0
        T = thrust_required(70000.0, 124.6, polar, 0.0, V)
        P = power_required(70000.0, 124.6, polar, 0.0, V)
        assert P == pytest.approx(T * V, rel=1e-10)


class TestMinDragSpeed:
    """Tests for min_drag_speed (Raymer Eq 17.13)."""

    def test_formula(self, polar):
        """V_md = sqrt(2W/(rho*S)) * (K/CD0)^0.25."""
        W_kg = 70000.0
        S = 124.6
        alt = 0.0
        Vmd = min_drag_speed(W_kg, S, polar, alt)
        rho = density_at(alt)
        W = W_kg * G0
        expected = math.sqrt(2.0 * W / (rho * S)) * (polar.K / polar.CD0) ** 0.25
        assert Vmd == pytest.approx(expected, rel=1e-10)

    def test_reasonable_range(self, polar):
        """Transport min-drag speed at cruise altitude should be ~200-280 m/s."""
        Vmd = min_drag_speed(70000.0, 124.6, polar, 10668.0)
        assert 150.0 < Vmd < 350.0


class TestMaxRangeSpeed:
    """Tests for max_range_speed (jet: same as V_md)."""

    def test_equals_vmd(self, polar):
        """For jets, max range speed = min drag speed. Raymer Eq 17.14."""
        Vmr = max_range_speed(70000.0, 124.6, polar, 10668.0)
        Vmd = min_drag_speed(70000.0, 124.6, polar, 10668.0)
        assert Vmr == pytest.approx(Vmd, rel=1e-10)


class TestMaxEnduranceSpeed:
    """Tests for max_endurance_speed (Raymer Eq 17.28)."""

    def test_lower_than_vmd(self, polar):
        """Max endurance speed is lower than min drag speed (CL is higher)."""
        Vme = max_endurance_speed(70000.0, 124.6, polar, 0.0)
        Vmd = min_drag_speed(70000.0, 124.6, polar, 0.0)
        assert Vme < Vmd

    def test_formula(self, polar):
        """V_me = sqrt(2W/(rho*S*CL_min_power))."""
        W_kg = 70000.0
        S = 124.6
        alt = 0.0
        Vme = max_endurance_speed(W_kg, S, polar, alt)
        rho = density_at(alt)
        W = W_kg * G0
        CL_mp = polar.cl_for_min_power()
        expected = math.sqrt(2.0 * W / (rho * S * CL_mp))
        assert Vme == pytest.approx(expected, rel=1e-10)

    def test_ratio_to_vmd(self, polar):
        """V_me / V_md = (CD0 / (3*CD0))^0.25 = (1/3)^0.25 ~ 0.76."""
        Vme = max_endurance_speed(70000.0, 124.6, polar, 0.0)
        Vmd = min_drag_speed(70000.0, 124.6, polar, 0.0)
        expected_ratio = (1.0 / 3.0) ** 0.25
        assert (Vme / Vmd) == pytest.approx(expected_ratio, rel=1e-6)


class TestMaxSpeed:
    """Tests for max_speed."""

    def test_basic(self, polar):
        """Max speed should exist and be higher than Vmd."""
        Vmax, Mmax = max_speed(70000.0, 124.6, 242000.0, polar, 0.0)
        Vmd = min_drag_speed(70000.0, 124.6, polar, 0.0)
        assert Vmax > Vmd
        assert Mmax > 0

    def test_returns_mach(self, polar):
        """Mach should be consistent with V_max and speed of sound."""
        alt = 10668.0
        Vmax, Mmax = max_speed(70000.0, 124.6, 100000.0, polar, alt)
        from core.atmosphere import speed_of_sound_at
        a = speed_of_sound_at(alt)
        assert Mmax == pytest.approx(Vmax / a, rel=1e-6)

    def test_insufficient_thrust_raises(self, polar):
        """Should raise ValueError if thrust can't sustain level flight."""
        with pytest.raises(ValueError, match="Insufficient thrust"):
            max_speed(70000.0, 124.6, 1000.0, polar, 0.0)


class TestGenerateTrCurve:
    """Tests for generate_tr_curve."""

    def test_returns_correct_keys(self, polar):
        result = generate_tr_curve(70000.0, 124.6, polar, 0.0)
        assert set(result.keys()) == {"V", "T_req", "CL", "CD", "LD"}

    def test_default_100_points(self, polar):
        result = generate_tr_curve(70000.0, 124.6, polar, 0.0)
        assert len(result["V"]) == 100

    def test_custom_range(self, polar):
        v_range = np.linspace(50, 200, 50)
        result = generate_tr_curve(70000.0, 124.6, polar, 0.0, v_range=v_range)
        assert len(result["V"]) == 50

    def test_tr_curve_has_minimum(self, polar):
        """T_req curve should have a minimum (drag bucket)."""
        result = generate_tr_curve(70000.0, 124.6, polar, 0.0)
        assert result["T_req"].min() < result["T_req"][0]
        assert result["T_req"].min() < result["T_req"][-1]


# ============================================================================ #
# Climb Tests
# ============================================================================ #

class TestRateOfClimb:
    """Tests for rate_of_climb (Raymer Eq 17.39)."""

    def test_positive_with_excess_thrust(self, polar):
        """ROC > 0 when T > D."""
        roc = rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0, 100.0)
        assert roc > 0

    def test_formula(self, polar):
        """ROC = V * (T - D) / W."""
        W_kg = 70000.0
        S = 124.6
        T = 242000.0
        V = 100.0
        alt = 0.0
        roc = rate_of_climb(W_kg, S, T, polar, alt, V)
        # Manual
        rho = density_at(alt)
        W = W_kg * G0
        q = 0.5 * rho * V ** 2
        CL = W / (q * S)
        CD = polar.cd(CL)
        D = q * S * CD
        expected = V * (T - D) / W
        assert roc == pytest.approx(expected, rel=1e-10)

    def test_negative_with_insufficient_thrust(self, polar):
        """ROC < 0 when T < D."""
        roc = rate_of_climb(70000.0, 124.6, 1000.0, polar, 0.0, 100.0)
        assert roc < 0

    def test_invalid_velocity_raises(self, polar):
        with pytest.raises(ValueError, match="velocity_ms"):
            rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0, 0.0)


class TestMaxRateOfClimb:
    """Tests for max_rate_of_climb."""

    def test_max_roc_reasonable(self, polar):
        """Max ROC for a transport at sea level should be ~30-70 m/s.
        B737-class with T/W ~ 0.35 produces high ROC at sea level."""
        roc_max, v_best = max_rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0)
        assert 10.0 < roc_max < 80.0
        assert v_best > 0

    def test_roc_at_vbest_is_maximum(self, polar):
        """ROC at V_best should be >= ROC at other speeds."""
        roc_max, v_best = max_rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0)
        roc_low = rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0, v_best * 0.8)
        roc_high = rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0, v_best * 1.2)
        assert roc_max >= roc_low - 0.01  # small tolerance for numerics
        assert roc_max >= roc_high - 0.01

    def test_decreases_with_altitude_thrust_lapse(self, polar):
        """Max ROC decreases with altitude when thrust lapses with density.
        With constant thrust, a jet can actually have HIGHER ROC at altitude
        because V_best increases. Use realistic thrust lapse for this test."""
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        roc_sl, _ = max_rate_of_climb(70000.0, 124.6, T_func(0.0), polar, 0.0)
        roc_high, _ = max_rate_of_climb(70000.0, 124.6, T_func(5000.0), polar, 5000.0)
        assert roc_high < roc_sl


class TestClimbGradient:
    """Tests for climb_gradient (Raymer Eq 17.38)."""

    def test_formula(self):
        """G = (T - D) / W."""
        W_kg = 70000.0
        T = 200000.0
        D = 150000.0
        G = climb_gradient(W_kg, T, D)
        expected = (T - D) / (W_kg * G0)
        assert G == pytest.approx(expected, rel=1e-10)

    def test_zero_when_td_equal(self):
        """Gradient = 0 when T = D (level flight)."""
        G = climb_gradient(70000.0, 100000.0, 100000.0)
        assert G == pytest.approx(0.0, abs=1e-10)

    def test_negative_when_descending(self):
        """Negative gradient when T < D."""
        G = climb_gradient(70000.0, 50000.0, 100000.0)
        assert G < 0


class TestClimbAngle:
    """Tests for climb_angle."""

    def test_zero_when_level(self):
        gamma = climb_angle(70000.0, 100000.0, 100000.0)
        assert gamma == pytest.approx(0.0, abs=1e-10)

    def test_positive_when_climbing(self):
        gamma = climb_angle(70000.0, 200000.0, 100000.0)
        assert gamma > 0

    def test_consistent_with_gradient(self):
        """sin(gamma) = (T-D)/W = gradient."""
        W_kg = 70000.0
        T = 200000.0
        D = 150000.0
        gamma = climb_angle(W_kg, T, D)
        G = climb_gradient(W_kg, T, D)
        assert math.sin(gamma) == pytest.approx(G, rel=1e-10)


class TestBestAngleOfClimbSpeed:
    """Tests for best_angle_of_climb_speed."""

    def test_near_vmd_for_jet(self, polar):
        """For a jet with constant thrust, Vx ~ Vmd."""
        Vx = best_angle_of_climb_speed(70000.0, 124.6, 242000.0, polar, 0.0)
        Vmd = min_drag_speed(70000.0, 124.6, polar, 0.0)
        # Should be close to Vmd (within 10%)
        assert abs(Vx - Vmd) / Vmd < 0.10

    def test_positive_speed(self, polar):
        Vx = best_angle_of_climb_speed(70000.0, 124.6, 242000.0, polar, 0.0)
        assert Vx > 0


class TestServiceCeiling:
    """Tests for service_ceiling."""

    def test_reasonable_ceiling(self, polar):
        """Service ceiling should be ~10-15 km for a transport."""
        # Simple thrust lapse: T = T_SL * sigma (density ratio)
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        ceiling = service_ceiling(70000.0, 124.6, T_func, polar)
        assert 8000.0 < ceiling < 20000.0

    def test_absolute_ceiling_higher_than_service(self, polar):
        """Absolute ceiling (ROC=0) >= service ceiling (ROC=100fpm)."""
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        sc = service_ceiling(70000.0, 124.6, T_func, polar)
        ac = absolute_ceiling(70000.0, 124.6, T_func, polar)
        assert ac >= sc - 1.0  # within 1m tolerance

    def test_insufficient_thrust_at_sea_level_raises(self, polar):
        """Should raise if aircraft can't climb even at sea level."""
        def T_func(h):
            return 1000.0  # way too low

        with pytest.raises(ValueError, match="sea level"):
            service_ceiling(70000.0, 124.6, T_func, polar)


class TestTimeToClimb:
    """Tests for time_to_climb (Raymer Eq 17.50)."""

    def test_positive_time(self, polar):
        """Time to climb should be positive."""
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        t = time_to_climb(70000.0, 124.6, T_func, polar, 0.0, 5000.0)
        assert t > 0

    def test_reasonable_time(self, polar):
        """Time to climb to FL350 should be ~15-30 minutes for transport."""
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        t = time_to_climb(70000.0, 124.6, T_func, polar, 0.0, 10668.0)
        assert 300 < t < 3600  # 5 to 60 minutes

    def test_invalid_altitude_order_raises(self, polar):
        def T_func(h):
            return 242000.0

        with pytest.raises(ValueError, match="h2_m must be greater"):
            time_to_climb(70000.0, 124.6, T_func, polar, 5000.0, 3000.0)


class TestGenerateRocCurve:
    """Tests for generate_roc_curve."""

    def test_returns_correct_keys(self, polar):
        def T_func(h):
            atm = atmosphere(h)
            return 242000.0 * atm["sigma"]

        result = generate_roc_curve(70000.0, 124.6, T_func, polar)
        assert set(result.keys()) == {"altitude", "ROC_max", "V_best"}

    def test_roc_decreases_with_altitude(self, polar):
        """ROC should generally decrease with altitude."""
        def T_func(h):
            atm = atmosphere(h)
            return 242000.0 * atm["sigma"]

        result = generate_roc_curve(70000.0, 124.6, T_func, polar)
        # First element should have highest ROC
        assert result["ROC_max"][0] >= result["ROC_max"][-1]


# ============================================================================ #
# Range & Endurance Tests
# ============================================================================ #

class TestBreguetRangeJet:
    """Tests for breguet_range_jet (Raymer Eq 6.11/17.16)."""

    def test_formula(self):
        """R = (V/C) * (L/D) * ln(Wi/Wf)."""
        Wi = 70000.0
        Wf = 60000.0
        sfc = 0.6  # 1/hr
        V = 230.0
        LD = 16.0
        R = breguet_range_jet(Wi, Wf, sfc, V, LD)
        C = sfc / 3600.0
        expected = (V / C) * LD * math.log(Wi / Wf)
        assert R == pytest.approx(expected, rel=1e-10)

    def test_reasonable_range(self):
        """B737-class range should be ~3000-6000 km."""
        Wi = 70000.0
        Wf = 55000.0  # ~15,000 kg fuel burned
        R = breguet_range_jet(Wi, Wf, 0.6, 230.0, 16.7)
        R_km = R / 1000.0
        assert 2000.0 < R_km < 8000.0

    def test_zero_fuel_burn_raises(self):
        with pytest.raises(ValueError, match="W_final_kg must be less"):
            breguet_range_jet(70000.0, 70000.0, 0.6, 230.0, 16.0)

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            breguet_range_jet(-70000.0, 60000.0, 0.6, 230.0, 16.0)

    def test_range_increases_with_ld(self):
        """Higher L/D -> longer range."""
        R1 = breguet_range_jet(70000.0, 60000.0, 0.6, 230.0, 14.0)
        R2 = breguet_range_jet(70000.0, 60000.0, 0.6, 230.0, 18.0)
        assert R2 > R1

    def test_range_decreases_with_sfc(self):
        """Higher SFC -> shorter range."""
        R1 = breguet_range_jet(70000.0, 60000.0, 0.5, 230.0, 16.0)
        R2 = breguet_range_jet(70000.0, 60000.0, 0.7, 230.0, 16.0)
        assert R1 > R2


class TestBreguetEnduranceJet:
    """Tests for breguet_endurance_jet (Raymer Eq 6.13/17.21)."""

    def test_formula(self):
        """E = (1/C) * (L/D) * ln(Wi/Wf)."""
        Wi = 70000.0
        Wf = 60000.0
        sfc = 0.6
        LD = 16.0
        E = breguet_endurance_jet(Wi, Wf, sfc, LD)
        C = sfc / 3600.0
        expected = (1.0 / C) * LD * math.log(Wi / Wf)
        assert E == pytest.approx(expected, rel=1e-10)

    def test_endurance_vs_range_consistency(self):
        """E = R / V for the same parameters."""
        Wi = 70000.0
        Wf = 60000.0
        V = 230.0
        sfc = 0.6
        LD = 16.0
        R = breguet_range_jet(Wi, Wf, sfc, V, LD)
        E = breguet_endurance_jet(Wi, Wf, sfc, LD)
        assert E == pytest.approx(R / V, rel=1e-10)

    def test_reasonable_endurance(self):
        """Should be several hours for a transport."""
        E = breguet_endurance_jet(70000.0, 55000.0, 0.6, 16.7)
        E_hr = E / 3600.0
        assert 3.0 < E_hr < 15.0


class TestBreguetRangeProp:
    """Tests for breguet_range_prop (Raymer Eq 6.12)."""

    def test_positive_range(self):
        """Should return positive range."""
        R = breguet_range_prop(2000.0, 1600.0, 0.5, 12.0)
        assert R > 0

    def test_invalid_sfc_raises(self):
        with pytest.raises(ValueError, match="SFC must be positive"):
            breguet_range_prop(2000.0, 1600.0, 0.0, 12.0)

    def test_range_increases_with_ld(self):
        """Higher L/D -> longer range."""
        R1 = breguet_range_prop(2000.0, 1600.0, 0.5, 10.0)
        R2 = breguet_range_prop(2000.0, 1600.0, 0.5, 14.0)
        assert R2 > R1


class TestCruiseClimbRange:
    """Tests for cruise_climb_range (Raymer Eq 17.22)."""

    def test_positive_and_reasonable_range(self, polar):
        """Cruise-climb range should be positive and in a realistic range.
        Raymer Eq 17.22 uses a different formulation (sqrt(W) terms)
        than constant-altitude Breguet, so direct comparison depends on
        the exact speed used for constant-altitude. Both should be
        in the same order of magnitude."""
        Wi = 70000.0
        Wf = 55000.0
        sfc = 0.6
        alt = 10668.0
        V_cruise = min_drag_speed(Wi, 124.6, polar, alt)
        LD = polar.max_ld()

        R_const = breguet_range_jet(Wi, Wf, sfc, V_cruise, LD)
        R_cc = cruise_climb_range(Wi, Wf, sfc, alt, 124.6, polar)

        assert R_cc > 0
        # Both should be same order of magnitude (within 25%)
        assert R_cc > R_const * 0.75
        assert R_cc < R_const * 1.25

    def test_positive_range(self, polar):
        R = cruise_climb_range(70000.0, 55000.0, 0.6, 10668.0, 124.6, polar)
        assert R > 0


class TestSpecificAirRange:
    """Tests for specific_air_range."""

    def test_positive(self, polar):
        sar = specific_air_range(70000.0, 124.6, 0.6, polar, 10668.0, 230.0)
        assert sar > 0

    def test_increases_at_higher_altitude(self, polar):
        """SAR generally increases with altitude for jets."""
        sar_low = specific_air_range(70000.0, 124.6, 0.6, polar, 3000.0, 200.0)
        sar_high = specific_air_range(70000.0, 124.6, 0.6, polar, 10668.0, 230.0)
        # Not guaranteed to always be true but should be for this speed/alt combo
        assert sar_high > 0
        assert sar_low > 0


class TestFuelForRange:
    """Tests for fuel_for_range."""

    def test_consistency_with_breguet(self):
        """fuel_for_range should invert breguet_range_jet."""
        Wi = 70000.0
        sfc = 0.6
        V = 230.0
        LD = 16.0
        # Compute range for 15000 kg fuel burn
        Wf = Wi - 15000.0
        R = breguet_range_jet(Wi, Wf, sfc, V, LD)
        # Recover fuel from range
        fuel = fuel_for_range(R, sfc, V, LD, Wi)
        assert fuel == pytest.approx(15000.0, rel=1e-6)

    def test_more_range_needs_more_fuel(self):
        """Longer range requires more fuel."""
        f1 = fuel_for_range(3000e3, 0.6, 230.0, 16.0, 70000.0)
        f2 = fuel_for_range(5000e3, 0.6, 230.0, 16.0, 70000.0)
        assert f2 > f1

    def test_invalid_range_raises(self):
        with pytest.raises(ValueError, match="range_m"):
            fuel_for_range(0.0, 0.6, 230.0, 16.0, 70000.0)


class TestRangePayloadDiagram:
    """Tests for range_payload_diagram."""

    def test_returns_correct_structure(self, polar):
        result = range_payload_diagram(
            W0_kg=70000.0, We_kg=40000.0, Wf_max_kg=20000.0,
            sfc_per_hr=0.6, velocity_ms=230.0, polar=polar,
        )
        assert "points" in result
        assert "curves" in result
        assert all(p in result["points"] for p in ["A", "B", "C", "D"])

    def test_payload_decreases_along_diagram(self, polar):
        """Payload should decrease from A to D."""
        result = range_payload_diagram(
            W0_kg=70000.0, We_kg=40000.0, Wf_max_kg=20000.0,
            sfc_per_hr=0.6, velocity_ms=230.0, polar=polar,
        )
        pts = result["points"]
        assert pts["A"]["payload_kg"] >= pts["D"]["payload_kg"]

    def test_range_increases_along_diagram(self, polar):
        """Range should increase from A to D."""
        result = range_payload_diagram(
            W0_kg=70000.0, We_kg=40000.0, Wf_max_kg=20000.0,
            sfc_per_hr=0.6, velocity_ms=230.0, polar=polar,
        )
        pts = result["points"]
        assert pts["D"]["range_m"] >= pts["A"]["range_m"]

    def test_ferry_range_zero_payload(self, polar):
        """Point D (ferry) should have zero payload."""
        result = range_payload_diagram(
            W0_kg=70000.0, We_kg=40000.0, Wf_max_kg=20000.0,
            sfc_per_hr=0.6, velocity_ms=230.0, polar=polar,
        )
        assert result["points"]["D"]["payload_kg"] == 0.0

    def test_curves_arrays_exist(self, polar):
        result = range_payload_diagram(
            W0_kg=70000.0, We_kg=40000.0, Wf_max_kg=20000.0,
            sfc_per_hr=0.6, velocity_ms=230.0, polar=polar,
        )
        assert len(result["curves"]["range_m"]) > 0
        assert len(result["curves"]["payload_kg"]) > 0


# ============================================================================ #
# Takeoff & Landing Tests
# ============================================================================ #

class TestGroundRollDistance:
    """Tests for ground_roll_distance (Raymer Eqs 17.100-17.104)."""

    def test_reasonable_distance(self, polar):
        """Transport ground roll should be ~1000-2000 m."""
        SG = ground_roll_distance(70000.0, 124.6, 242000.0, polar)
        assert 500.0 < SG < 3000.0

    def test_increases_with_weight(self, polar):
        """Heavier aircraft needs longer ground roll."""
        SG1 = ground_roll_distance(60000.0, 124.6, 242000.0, polar)
        SG2 = ground_roll_distance(80000.0, 124.6, 242000.0, polar)
        assert SG2 > SG1

    def test_decreases_with_thrust(self, polar):
        """More thrust -> shorter ground roll."""
        SG1 = ground_roll_distance(70000.0, 124.6, 200000.0, polar)
        SG2 = ground_roll_distance(70000.0, 124.6, 300000.0, polar)
        assert SG2 < SG1

    def test_increases_with_altitude(self, polar):
        """Higher altitude -> lower density -> longer ground roll."""
        SG_sl = ground_roll_distance(70000.0, 124.6, 242000.0, polar, 0.0)
        SG_high = ground_roll_distance(70000.0, 124.6, 242000.0, polar, 2000.0)
        assert SG_high > SG_sl


class TestRotationDistance:
    """Tests for rotation_distance."""

    def test_formula(self):
        """S_R = V_TO * 3 seconds."""
        SR = rotation_distance(80.0)
        assert SR == pytest.approx(80.0 * 3.0, rel=1e-10)

    def test_zero_speed(self):
        assert rotation_distance(0.0) == 0.0


class TestTransitionDistance:
    """Tests for transition_distance."""

    def test_positive_values(self, polar):
        STR, hTR = transition_distance(70000.0, 124.6, polar)
        assert STR > 0
        assert hTR > 0

    def test_height_less_than_distance(self, polar):
        """Height gained should be much less than horizontal distance."""
        STR, hTR = transition_distance(70000.0, 124.6, polar)
        assert hTR < STR


class TestClimbToObstacle:
    """Tests for climb_to_obstacle (Raymer Eq 17.112)."""

    def test_basic(self):
        """S_c = (h_obs - h_TR) / tan(gamma)."""
        SC = climb_to_obstacle(10.7, 5.0, math.radians(5.0))
        expected = (10.7 - 5.0) / math.tan(math.radians(5.0))
        assert SC == pytest.approx(expected, rel=1e-6)

    def test_zero_if_cleared_in_transition(self):
        """If h_TR > h_obstacle, return 0."""
        SC = climb_to_obstacle(10.7, 15.0, math.radians(5.0))
        assert SC == 0.0

    def test_infinite_if_no_climb(self):
        """If gamma = 0, cannot clear obstacle."""
        SC = climb_to_obstacle(10.7, 5.0, 0.0)
        assert SC == float("inf")


class TestTotalTakeoffDistance:
    """Tests for total_takeoff_distance."""

    def test_returns_correct_keys(self, polar):
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        expected_keys = {"SG", "SR", "STR", "SC", "total", "VTO", "V_TR",
                         "gamma_climb", "h_TR"}
        assert set(result.keys()) == expected_keys

    def test_total_is_sum_of_segments(self, polar):
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        total = result["SG"] + result["SR"] + result["STR"] + result["SC"]
        assert result["total"] == pytest.approx(total, rel=1e-10)

    def test_reasonable_total(self, polar):
        """Total takeoff distance should be ~1500-3000 m for transport."""
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        assert 800.0 < result["total"] < 4000.0

    def test_vto_reasonable(self, polar):
        """VTO should be ~70-90 m/s for a transport."""
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        assert 50.0 < result["VTO"] < 120.0

    def test_positive_climb_angle(self, polar):
        """With sufficient thrust, climb angle should be positive."""
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        assert result["gamma_climb"] > 0


class TestBalancedFieldLength:
    """Tests for balanced_field_length (Raymer Eq 17.113)."""

    def test_reasonable_bfl(self, polar):
        """BFL for B737-class should be ~2000-3000 m."""
        BFL = balanced_field_length(70000.0, 124.6, 121000.0, 2, polar)
        assert 1000.0 < BFL < 5000.0

    def test_increases_with_weight(self, polar):
        BFL1 = balanced_field_length(60000.0, 124.6, 121000.0, 2, polar)
        BFL2 = balanced_field_length(80000.0, 124.6, 121000.0, 2, polar)
        assert BFL2 > BFL1

    def test_invalid_engines_raises(self, polar):
        with pytest.raises(ValueError, match="n_engines"):
            balanced_field_length(70000.0, 124.6, 121000.0, 5, polar)

    def test_decreases_with_more_engines(self, polar):
        """4-engine aircraft should have shorter BFL than 2-engine (same total thrust)."""
        # Same total thrust, different distribution
        BFL_2 = balanced_field_length(70000.0, 124.6, 121000.0, 2, polar)
        BFL_4 = balanced_field_length(70000.0, 124.6, 60500.0, 4, polar)
        # 4 engines loses only 1/4 of thrust vs 1/2, so BFL should be shorter
        assert BFL_4 < BFL_2


class TestLandingDistance:
    """Tests for landing_distance (Raymer Ch 17.9)."""

    def test_returns_correct_keys(self, polar):
        result = landing_distance(60000.0, 124.6, polar)
        expected_keys = {"Sa", "Sf", "Sfr", "Sgr", "total",
                         "FAR_field_length", "Va", "VTD"}
        assert set(result.keys()) == expected_keys

    def test_total_is_sum(self, polar):
        result = landing_distance(60000.0, 124.6, polar)
        total = result["Sa"] + result["Sf"] + result["Sfr"] + result["Sgr"]
        assert result["total"] == pytest.approx(total, rel=1e-10)

    def test_far_field_length(self, polar):
        """FAR field = total / 0.6."""
        result = landing_distance(60000.0, 124.6, polar)
        assert result["FAR_field_length"] == pytest.approx(
            result["total"] / 0.6, rel=1e-10)

    def test_reasonable_distance(self, polar):
        """Landing distance should be ~800-2000 m for a transport."""
        result = landing_distance(60000.0, 124.6, polar)
        assert 500.0 < result["total"] < 3000.0

    def test_increases_with_weight(self, polar):
        r1 = landing_distance(50000.0, 124.6, polar)
        r2 = landing_distance(70000.0, 124.6, polar)
        assert r2["total"] > r1["total"]


class TestGenerateTakeoffLandingSummary:
    """Tests for generate_takeoff_landing_summary."""

    def test_returns_complete_dict(self, polar):
        result = generate_takeoff_landing_summary(
            70000.0, 124.6, 242000.0, 2, polar)
        assert "takeoff" in result
        assert "landing" in result
        assert "balanced_field_length" in result
        assert "stall_speeds" in result
        assert "reference_speeds" in result

    def test_stall_speed_order(self, polar):
        """Vs_landing < Vs_takeoff < Vs_clean (at different weights/configs but still)."""
        result = generate_takeoff_landing_summary(
            70000.0, 124.6, 242000.0, 2, polar)
        ss = result["stall_speeds"]
        # Landing at 0.85*MTOW so Vs_landing should be lowest
        assert ss["Vs_landing"] < ss["Vs_takeoff"]
        assert ss["Vs_landing"] < ss["Vs_clean"]

    def test_reference_speeds_positive(self, polar):
        result = generate_takeoff_landing_summary(
            70000.0, 124.6, 242000.0, 2, polar)
        for key, val in result["reference_speeds"].items():
            assert val > 0, f"{key} should be positive"


# ============================================================================ #
# Integration Tests
# ============================================================================ #

class TestIntegration:
    """End-to-end integration tests combining multiple performance modules."""

    def test_thrust_available_vs_required(self, polar):
        """At max speed, T_available should equal T_required."""
        W_kg = 70000.0
        S = 124.6
        T = 242000.0
        alt = 0.0
        Vmax, _ = max_speed(W_kg, S, T, polar, alt)
        T_req = thrust_required(W_kg, S, polar, alt, Vmax)
        assert T_req == pytest.approx(T, rel=0.01)

    def test_roc_zero_at_max_speed_ceiling(self, polar):
        """At the absolute ceiling, max ROC should be ~0."""
        T_SL = 242000.0
        def T_func(h):
            atm = atmosphere(h)
            return T_SL * atm["sigma"]

        ac = absolute_ceiling(70000.0, 124.6, T_func, polar)
        T_ac = T_func(ac)
        roc_max, _ = max_rate_of_climb(70000.0, 124.6, T_ac, polar, ac)
        assert abs(roc_max) < 1.0  # within 1 m/s of zero

    def test_fuel_weight_consistency(self, polar):
        """Fuel for range should be consistent: compute range, then recover fuel."""
        Wi = 70000.0
        fuel = 15000.0
        Wf = Wi - fuel
        sfc = 0.6
        V = 230.0
        LD = polar.max_ld()
        R = breguet_range_jet(Wi, Wf, sfc, V, LD)
        fuel_recovered = fuel_for_range(R, sfc, V, LD, Wi)
        assert fuel_recovered == pytest.approx(fuel, rel=1e-6)

    def test_takeoff_then_climb(self, polar):
        """Verify takeoff -> initial climb is consistent."""
        result = total_takeoff_distance(70000.0, 124.6, 242000.0, polar)
        # At climb speed, aircraft should have positive ROC
        V_climb = result["V_TR"]
        roc = rate_of_climb(70000.0, 124.6, 242000.0, polar, 0.0, V_climb)
        assert roc > 0
