"""
Tests for core.performance.turn module.

Validates turn performance, energy-maneuverability, and plot data generators
against hand-computed values from Raymer equations and physical sanity checks.

Test aircraft:
    Transport (B737-800 class):
        W = 70,000 kg, S = 124.6 m^2
        CD0 = 0.020, K = 0.045
        CLmax = 1.6, T = 242,000 N, n_limit = 2.5

    Fighter (F-16 class):
        W = 12,000 kg, S = 27.87 m^2
        CD0 = 0.015, K = 0.06
        CLmax = 1.4, T = 127,000 N, n_limit = 9.0
"""

import math
import pytest
import numpy as np

from core.aerodynamics.drag_polar import DragPolar
from core.atmosphere import atmosphere, density_at, dynamic_pressure, speed_of_sound_at, G0, RHO0

from core.performance.turn import (
    turn_rate,
    turn_radius,
    bank_angle,
    sustained_load_factor,
    sustained_load_factor_detailed,
    sustained_turn_rate,
    max_sustained_load_factor_speed,
    instantaneous_load_factor,
    corner_speed,
    instantaneous_turn_rate,
    specific_excess_power,
    energy_height,
    generate_turn_rate_envelope,
    generate_ps_plot,
    generate_ps_contours,
    generate_ps_vs_turn_rate,
)


# ============================================================================ #
# Shared fixtures
# ============================================================================ #

@pytest.fixture
def polar():
    """Typical transport polar: CD0=0.020, K=0.045."""
    return DragPolar(CD0=0.020, K=0.045)


@pytest.fixture
def fighter_polar():
    """Fighter polar: CD0=0.015, K=0.06."""
    return DragPolar(CD0=0.015, K=0.06)


@pytest.fixture
def b737():
    """B737-800 class parameters."""
    return {
        "W_kg": 70000.0,
        "S_m2": 124.6,
        "T_N": 242000.0,
        "CLmax": 1.6,
        "n_limit": 2.5,
    }


@pytest.fixture
def f16():
    """F-16 class parameters."""
    return {
        "W_kg": 12000.0,
        "S_m2": 27.87,
        "T_N": 127000.0,
        "CLmax": 1.4,
        "n_limit": 9.0,
    }


# ============================================================================ #
# Level Turning Flight (Raymer 17.4)
# ============================================================================ #

class TestTurnRate:
    """Tests for turn_rate (Raymer Eq 17.52)."""

    def test_known_values(self):
        """Hand calc: g*sqrt(n^2-1)/V = 9.80665*sqrt(3)/100 = 0.16987 rad/s."""
        tr = turn_rate(100.0, 2.0)
        expected = G0 * math.sqrt(3.0) / 100.0
        assert abs(tr - expected) < 1e-6

    def test_n_equals_one_zero_rate(self):
        """At n=1 (level flight), turn rate should be zero."""
        assert turn_rate(100.0, 1.0) == 0.0

    def test_n_less_than_one_returns_zero(self):
        """Load factor < 1 cannot sustain level turn."""
        assert turn_rate(100.0, 0.5) == 0.0

    def test_zero_velocity_returns_zero(self):
        assert turn_rate(0.0, 2.0) == 0.0

    def test_negative_velocity_returns_zero(self):
        assert turn_rate(-50.0, 2.0) == 0.0

    def test_increases_with_load_factor(self):
        """Turn rate should increase with n at fixed V."""
        tr2 = turn_rate(100.0, 2.0)
        tr3 = turn_rate(100.0, 3.0)
        tr5 = turn_rate(100.0, 5.0)
        assert tr2 < tr3 < tr5

    def test_decreases_with_velocity(self):
        """Turn rate should decrease with V at fixed n."""
        tr100 = turn_rate(100.0, 3.0)
        tr200 = turn_rate(200.0, 3.0)
        assert tr100 > tr200

    def test_high_n_value(self):
        """Should work at n=9 (fighter)."""
        tr = turn_rate(200.0, 9.0)
        expected = G0 * math.sqrt(80.0) / 200.0
        assert abs(tr - expected) < 1e-6


class TestTurnRadius:
    """Tests for turn_radius (Raymer Eq 17.79)."""

    def test_known_values(self):
        """Hand calc: V^2/(g*sqrt(n^2-1)) = 10000/(9.80665*sqrt(3)) = 588.7 m."""
        r = turn_radius(100.0, 2.0)
        expected = 10000.0 / (G0 * math.sqrt(3.0))
        assert abs(r - expected) < 0.1

    def test_n_equals_one_infinite(self):
        """Straight flight has infinite turn radius."""
        assert turn_radius(100.0, 1.0) == float("inf")

    def test_n_less_than_one_infinite(self):
        assert turn_radius(100.0, 0.8) == float("inf")

    def test_zero_velocity_infinite(self):
        assert turn_radius(0.0, 2.0) == float("inf")

    def test_decreases_with_load_factor(self):
        """Higher n gives tighter (smaller) turn radius."""
        r2 = turn_radius(100.0, 2.0)
        r5 = turn_radius(100.0, 5.0)
        assert r5 < r2

    def test_increases_with_velocity(self):
        """Higher V gives larger turn radius."""
        r100 = turn_radius(100.0, 3.0)
        r200 = turn_radius(200.0, 3.0)
        assert r200 > r100

    def test_consistency_with_turn_rate(self):
        """R = V / psi_dot."""
        V = 150.0
        n = 3.0
        r = turn_radius(V, n)
        tr = turn_rate(V, n)
        assert abs(r - V / tr) < 0.01


class TestBankAngle:
    """Tests for bank_angle."""

    def test_n_two_gives_60_degrees(self):
        """phi = arccos(1/2) = 60 degrees."""
        phi = bank_angle(2.0)
        assert abs(math.degrees(phi) - 60.0) < 0.01

    def test_n_one_gives_zero(self):
        """Level flight, no bank."""
        assert bank_angle(1.0) == 0.0

    def test_n_less_than_one_returns_zero(self):
        assert bank_angle(0.5) == 0.0

    def test_high_n_near_90_degrees(self):
        """At n=10, phi should be close to 90 degrees."""
        phi = bank_angle(10.0)
        assert math.degrees(phi) > 80.0
        assert math.degrees(phi) < 90.0

    def test_n_sqrt2_gives_45_degrees(self):
        """phi = arccos(1/sqrt(2)) = 45 degrees."""
        phi = bank_angle(math.sqrt(2.0))
        assert abs(math.degrees(phi) - 45.0) < 0.01


# ============================================================================ #
# Sustained Turn (Raymer 17.4.2)
# ============================================================================ #

class TestSustainedLoadFactor:
    """Tests for sustained_load_factor (Raymer Eq 17.53)."""

    def test_known_values(self, polar):
        """n = (T/W) * (L/D)_max."""
        W = 70000.0 * G0
        T = 242000.0
        ld_max = polar.max_ld()
        n = sustained_load_factor(T / W, ld_max)
        expected = (T / W) * ld_max
        assert abs(n - expected) < 0.01
        assert n > 1.0  # Should be capable of turning

    def test_zero_tw_returns_zero(self):
        assert sustained_load_factor(0.0, 15.0) == 0.0

    def test_zero_ld_returns_zero(self):
        assert sustained_load_factor(0.3, 0.0) == 0.0

    def test_negative_tw_returns_zero(self):
        assert sustained_load_factor(-0.3, 15.0) == 0.0


class TestSustainedLoadFactorDetailed:
    """Tests for sustained_load_factor_detailed (Raymer Eq 17.54)."""

    def test_at_optimal_speed_matches_simplified(self, polar, b737):
        """At V for max L/D, detailed should match simplified formula."""
        W = b737["W_kg"] * G0
        n_simple = sustained_load_factor(b737["T_N"] / W, polar.max_ld())
        _, V_best = max_sustained_load_factor_speed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0
        )
        n_det = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V_best
        )
        assert abs(n_det - n_simple) < 0.05

    def test_decreases_at_off_optimal_speeds(self, polar, b737):
        """n should be lower at speeds far from optimal."""
        _, V_best = max_sustained_load_factor_speed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0
        )
        n_best = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V_best
        )
        n_slow = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V_best * 0.5
        )
        n_fast = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V_best * 1.5
        )
        assert n_best >= n_slow
        assert n_best >= n_fast

    def test_zero_velocity_returns_zero(self, polar, b737):
        n = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 0.0
        )
        assert n == 0.0

    def test_insufficient_thrust_returns_zero(self, polar, b737):
        """Very low thrust should yield n < 1 which returns 0."""
        n = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], 1000.0, polar, 0.0, 100.0
        )
        # With only 1000 N thrust, n will be very small
        assert n < 1.0


class TestSustainedTurnRate:
    """Tests for sustained_turn_rate."""

    def test_positive_at_reasonable_condition(self, polar, b737):
        tr = sustained_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0
        )
        assert tr > 0.0

    def test_zero_at_zero_velocity(self, polar, b737):
        tr = sustained_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 0.0
        )
        assert tr == 0.0


class TestMaxSustainedLoadFactorSpeed:
    """Tests for max_sustained_load_factor_speed (Raymer Eq 17.55)."""

    def test_returns_tuple(self, polar, b737):
        result = max_sustained_load_factor_speed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_n_max_positive(self, polar, b737):
        n_max, V_best = max_sustained_load_factor_speed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0
        )
        assert n_max > 1.0
        assert V_best > 0.0

    def test_insufficient_thrust_returns_zeros(self, polar):
        """If T/W * (L/D)max < 1, returns (0, 0)."""
        n_max, V_best = max_sustained_load_factor_speed(
            70000.0, 124.6, 1000.0, polar, 0.0
        )
        assert n_max == 0.0
        assert V_best == 0.0


# ============================================================================ #
# Instantaneous Turn (Raymer 17.4.1)
# ============================================================================ #

class TestInstantaneousLoadFactor:
    """Tests for instantaneous_load_factor (Raymer 17.4.1)."""

    def test_known_values(self, b737):
        """n = q*CLmax*S / W at sea level, V=100 m/s."""
        W = b737["W_kg"] * G0
        q = 0.5 * RHO0 * 100.0 ** 2
        expected = q * b737["S_m2"] * b737["CLmax"] / W
        n = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, 100.0
        )
        assert abs(n - expected) < 0.001

    def test_increases_with_speed(self, b737):
        n100 = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, 100.0
        )
        n200 = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, 200.0
        )
        assert n200 > n100

    def test_zero_velocity_returns_zero(self, b737):
        n = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, 0.0
        )
        assert n == 0.0


class TestCornerSpeed:
    """Tests for corner_speed (Raymer p.653)."""

    def test_known_formula(self, b737):
        """V_c = sqrt(2*n_limit*W / (rho*S*CLmax))."""
        W = b737["W_kg"] * G0
        expected = math.sqrt(2.0 * b737["n_limit"] * W /
                             (RHO0 * b737["S_m2"] * b737["CLmax"]))
        V_c = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        assert abs(V_c - expected) < 0.01

    def test_n_equals_n_limit_at_corner(self, b737):
        """Instantaneous n at corner speed should equal n_limit."""
        V_c = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        n = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, V_c
        )
        assert abs(n - b737["n_limit"]) < 0.01

    def test_increases_with_altitude(self, b737):
        """Corner speed should increase at altitude (lower density)."""
        V_sl = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], b737["n_limit"], 0.0
        )
        V_alt = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], b737["n_limit"], 5000.0
        )
        assert V_alt > V_sl

    def test_increases_with_weight(self, b737):
        V_light = corner_speed(40000.0, b737["S_m2"], b737["CLmax"],
                               b737["n_limit"], 0.0)
        V_heavy = corner_speed(b737["W_kg"], b737["S_m2"], b737["CLmax"],
                               b737["n_limit"], 0.0)
        assert V_heavy > V_light

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            corner_speed(0.0, 100.0, 1.6, 2.5)
        with pytest.raises(ValueError):
            corner_speed(70000.0, 0.0, 1.6, 2.5)
        with pytest.raises(ValueError):
            corner_speed(70000.0, 100.0, 0.0, 2.5)
        with pytest.raises(ValueError):
            corner_speed(70000.0, 100.0, 1.6, 0.0)


class TestInstantaneousTurnRate:
    """Tests for instantaneous_turn_rate."""

    def test_positive_above_stall(self, b737):
        V_c = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        itr = instantaneous_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0, V_c
        )
        assert itr > 0.0

    def test_stall_limited_below_corner(self, b737):
        """Below corner speed, n_aero < n_limit, so stall-limited."""
        V_c = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        V_low = V_c * 0.7
        n_aero = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, V_low
        )
        assert n_aero < b737["n_limit"]

    def test_structural_limited_above_corner(self, b737):
        """Above corner speed, n_aero > n_limit, so structural-limited."""
        V_c = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        V_high = V_c * 1.5
        n_aero = instantaneous_load_factor(
            b737["W_kg"], b737["S_m2"], b737["CLmax"], 0.0, V_high
        )
        assert n_aero > b737["n_limit"]
        # Turn rate should be computed at n_limit, not n_aero
        itr = instantaneous_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0, V_high
        )
        expected = turn_rate(V_high, b737["n_limit"])
        assert abs(itr - expected) < 1e-6


# ============================================================================ #
# Specific Excess Power (Raymer 17.6)
# ============================================================================ #

class TestSpecificExcessPower:
    """Tests for specific_excess_power (Raymer Eq 17.89)."""

    def test_hand_calculation(self, polar, b737):
        """Ps = V * [T/W - q*CD0/WS - K*n^2*WS/q]."""
        V = 200.0
        n = 1.0
        W = b737["W_kg"] * G0
        WS = W / b737["S_m2"]
        q = 0.5 * RHO0 * V ** 2
        expected = V * (b737["T_N"] / W - q * polar.CD0 / WS
                        - polar.K * n ** 2 * WS / q)
        ps = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V, n
        )
        assert abs(ps - expected) < 0.01

    def test_positive_at_1g(self, polar, b737):
        """At 1g and reasonable speed, Ps should be positive (can climb)."""
        ps = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0, 1.0
        )
        assert ps > 0.0

    def test_decreases_with_load_factor(self, polar, b737):
        """Ps should decrease as n increases (more drag from turning)."""
        ps1 = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0, 1.0
        )
        ps3 = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0, 3.0
        )
        ps5 = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0, 5.0
        )
        assert ps1 > ps3 > ps5

    def test_zero_velocity_returns_zero(self, polar, b737):
        ps = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 0.0, 1.0
        )
        assert ps == 0.0

    def test_default_n_is_one(self, polar, b737):
        """Default load_factor should be 1.0."""
        ps_default = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0
        )
        ps_explicit = specific_excess_power(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, 200.0, 1.0
        )
        assert abs(ps_default - ps_explicit) < 1e-10


class TestEnergyHeight:
    """Tests for energy_height (Raymer Eq 17.85)."""

    def test_known_values(self):
        """he = h + V^2/(2g) = 10000 + 250^2/19.6133 = 13186.6 m."""
        he = energy_height(10000.0, 250.0)
        expected = 10000.0 + 250.0 ** 2 / (2.0 * G0)
        assert abs(he - expected) < 0.001

    def test_zero_speed_equals_altitude(self):
        """With V=0, energy height equals geometric altitude."""
        he = energy_height(5000.0, 0.0)
        assert abs(he - 5000.0) < 0.001

    def test_zero_altitude(self):
        """With h=0, energy height is purely kinetic."""
        he = energy_height(0.0, 200.0)
        expected = 200.0 ** 2 / (2.0 * G0)
        assert abs(he - expected) < 0.001

    def test_increases_with_altitude(self):
        he1 = energy_height(5000.0, 200.0)
        he2 = energy_height(10000.0, 200.0)
        assert he2 > he1

    def test_increases_with_speed(self):
        he1 = energy_height(5000.0, 100.0)
        he2 = energy_height(5000.0, 200.0)
        assert he2 > he1


# ============================================================================ #
# Plot Data Generators
# ============================================================================ #

class TestGenerateTurnRateEnvelope:
    """Tests for generate_turn_rate_envelope."""

    def test_returns_correct_keys(self, polar, b737):
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            b737["CLmax"], b737["n_limit"],
        )
        expected_keys = {
            "V", "sustained_turn_rate", "instantaneous_turn_rate",
            "sustained_load_factor", "instantaneous_load_factor",
            "corner_speed", "max_sustained_rate", "max_sustained_speed",
        }
        assert set(result.keys()) == expected_keys

    def test_arrays_same_length(self, polar, b737):
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        n_pts = len(result["V"])
        assert len(result["sustained_turn_rate"]) == n_pts
        assert len(result["instantaneous_turn_rate"]) == n_pts
        assert len(result["sustained_load_factor"]) == n_pts
        assert len(result["instantaneous_load_factor"]) == n_pts

    def test_instantaneous_ge_sustained(self, polar, b737):
        """Instantaneous turn rate must be >= sustained everywhere."""
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            b737["CLmax"], b737["n_limit"],
        )
        diff = result["instantaneous_turn_rate"] - result["sustained_turn_rate"]
        assert np.all(diff >= -0.01)

    def test_corner_speed_positive(self, polar, b737):
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        assert result["corner_speed"] > 0.0

    def test_max_sustained_rate_positive(self, polar, b737):
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        assert result["max_sustained_rate"] > 0.0
        assert result["max_sustained_speed"] > 0.0

    def test_custom_v_range(self, polar, b737):
        v_range = np.linspace(80, 300, 50)
        result = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            v_range=v_range,
        )
        assert len(result["V"]) == 50
        assert abs(result["V"][0] - 80.0) < 0.01

    def test_fighter_has_higher_inst_rates(self, fighter_polar, f16, polar, b737):
        """Fighter should have higher instantaneous turn rates than transport."""
        r_fighter = generate_turn_rate_envelope(
            f16["W_kg"], f16["S_m2"], f16["T_N"], fighter_polar, 0.0,
            f16["CLmax"], f16["n_limit"],
            v_range=np.linspace(100, 300, 50),
        )
        r_transport = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            b737["CLmax"], b737["n_limit"],
            v_range=np.linspace(100, 300, 50),
        )
        assert np.max(r_fighter["instantaneous_turn_rate"]) > \
               np.max(r_transport["instantaneous_turn_rate"])


class TestGeneratePsPlot:
    """Tests for generate_ps_plot."""

    def test_returns_correct_keys(self, polar, b737):
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        assert "mach" in result
        assert "ps" in result
        assert "altitude_m" in result

    def test_default_load_factors(self, polar, b737):
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        assert set(result["ps"].keys()) == {1.0, 3.0, 5.0, 7.0}

    def test_custom_load_factors(self, polar, b737):
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            load_factors=[1, 2],
        )
        assert set(result["ps"].keys()) == {1.0, 2.0}

    def test_ps_decreases_with_n(self, polar, b737):
        """Ps at n=1 should be higher than at n=3 at all Mach numbers."""
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
        )
        assert np.max(result["ps"][1.0]) > np.max(result["ps"][3.0])

    def test_callable_thrust(self, polar, b737):
        """Should accept a callable for variable thrust."""
        def thrust_model(mach, alt_m):
            return 242000.0 * (1.0 - 0.2 * mach)

        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], thrust_model, polar, 0.0,
            load_factors=[1],
        )
        assert len(result["ps"][1.0]) == len(result["mach"])

    def test_scalar_thrust(self, polar, b737):
        """Should accept a scalar float for constant thrust."""
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            load_factors=[1],
        )
        assert isinstance(result["ps"][1.0], np.ndarray)

    def test_custom_mach_range(self, polar, b737):
        mach_range = np.linspace(0.2, 0.8, 30)
        result = generate_ps_plot(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            mach_range=mach_range,
        )
        assert len(result["mach"]) == 30


class TestGeneratePsContours:
    """Tests for generate_ps_contours."""

    def test_returns_correct_keys(self, polar, b737):
        result = generate_ps_contours(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar,
        )
        assert "mach" in result
        assert "altitude" in result
        assert "ps_grid" in result
        assert "load_factor" in result

    def test_grid_shape(self, polar, b737):
        mach_range = np.linspace(0.2, 0.9, 20)
        alt_range = np.linspace(0, 12000, 15)
        result = generate_ps_contours(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar,
            mach_range=mach_range, alt_range=alt_range,
        )
        assert result["ps_grid"].shape == (15, 20)

    def test_ps_higher_at_low_altitude(self, polar, b737):
        """With constant thrust, Ps should be higher at sea level."""
        result = generate_ps_contours(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar,
            mach_range=np.linspace(0.3, 0.7, 10),
            alt_range=np.linspace(0, 10000, 10),
        )
        # Mid-Mach point: Ps at sea level (row 0) vs 10 km (row -1)
        mid_mach = 5
        assert result["ps_grid"][0, mid_mach] > result["ps_grid"][-1, mid_mach]

    def test_callable_thrust(self, polar, b737):
        """Should work with callable thrust model."""
        def thrust_model(mach, alt_m):
            sigma = atmosphere(alt_m)["sigma"]
            return 242000.0 * sigma

        result = generate_ps_contours(
            b737["W_kg"], b737["S_m2"], thrust_model, polar,
            mach_range=np.linspace(0.3, 0.7, 10),
            alt_range=np.linspace(0, 10000, 10),
        )
        assert result["ps_grid"].shape == (10, 10)

    def test_load_factor_stored(self, polar, b737):
        result = generate_ps_contours(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar,
            load_factor=3.0,
        )
        assert result["load_factor"] == 3.0


class TestGeneratePsVsTurnRate:
    """Tests for generate_ps_vs_turn_rate."""

    def test_returns_correct_keys(self, polar, b737):
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        expected_keys = {
            "turn_rate_degs", "ps", "stall_limit_rate", "structural_limit_rate",
        }
        assert set(result.keys()) == expected_keys

    def test_arrays_same_length(self, polar, b737):
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        assert len(result["turn_rate_degs"]) == len(result["ps"])

    def test_zero_turn_rate_at_n_one(self, polar, b737):
        """At n=1 (first point), turn rate should be ~0."""
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        assert abs(result["turn_rate_degs"][0]) < 0.1

    def test_ps_decreases_with_turn_rate(self, polar, b737):
        """Ps should generally decrease as turn rate increases."""
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        # First Ps (at n=1) should be >= last Ps (at max n)
        assert result["ps"][0] >= result["ps"][-1]

    def test_stall_limit_positive(self, polar, b737):
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        assert result["stall_limit_rate"] >= 0.0

    def test_structural_limit_positive(self, polar, b737):
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.5,
        )
        assert result["structural_limit_rate"] >= 0.0

    def test_zero_mach_edge_case(self, polar, b737):
        """Mach=0 should return degenerate but valid result."""
        result = generate_ps_vs_turn_rate(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, mach=0.0,
        )
        assert "turn_rate_degs" in result
        assert "ps" in result


# ============================================================================ #
# Integration Tests
# ============================================================================ #

class TestIntegration:
    """Cross-function consistency checks."""

    def test_ps_zero_at_sustained_n(self, polar, b737):
        """At the sustained n, Ps should be approximately zero."""
        V = 200.0
        n = sustained_load_factor_detailed(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V
        )
        if n > 1.0:
            ps = specific_excess_power(
                b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0, V, n
            )
            assert abs(ps) < 1.0  # Should be very close to zero

    def test_turn_rate_radius_consistency(self):
        """psi_dot * R = V."""
        V = 150.0
        n = 3.0
        tr = turn_rate(V, n)
        r = turn_radius(V, n)
        assert abs(tr * r - V) < 0.01

    def test_bank_angle_load_factor_consistency(self):
        """n = 1/cos(phi)."""
        for n in [1.5, 2.0, 3.0, 5.0]:
            phi = bank_angle(n)
            n_back = 1.0 / math.cos(phi)
            assert abs(n_back - n) < 1e-6

    def test_corner_speed_matches_envelope(self, polar, b737):
        """Corner speed from standalone function should match envelope."""
        V_c_standalone = corner_speed(
            b737["W_kg"], b737["S_m2"], b737["CLmax"],
            b737["n_limit"], 0.0
        )
        env = generate_turn_rate_envelope(
            b737["W_kg"], b737["S_m2"], b737["T_N"], polar, 0.0,
            b737["CLmax"], b737["n_limit"],
        )
        assert abs(V_c_standalone - env["corner_speed"]) < 0.01

    def test_energy_height_additive(self):
        """Energy height at (h, V) should equal h + V^2/(2g)."""
        for h, V in [(0, 0), (1000, 100), (10000, 250), (0, 340)]:
            he = energy_height(h, V)
            expected = h + V ** 2 / (2.0 * G0)
            assert abs(he - expected) < 0.001
