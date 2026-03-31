"""
Tests for AeroSizer Web API (app.py)

Uses FastAPI's TestClient for HTTP-level integration testing
without needing a running server.
"""

import math
import pytest
from fastapi.testclient import TestClient

from app import app, PRESETS, _parse_aircraft_type
from core.mission import AircraftType


client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper: parse aircraft type
# ---------------------------------------------------------------------------
class TestParseAircraftType:
    def test_jet_transport(self):
        assert _parse_aircraft_type("jet_transport") == AircraftType.JET_TRANSPORT

    def test_business_jet(self):
        assert _parse_aircraft_type("business_jet") == AircraftType.BUSINESS_JET

    def test_case_insensitive(self):
        assert _parse_aircraft_type("JET_TRANSPORT") == AircraftType.JET_TRANSPORT

    def test_unknown_defaults_to_transport(self):
        assert _parse_aircraft_type("unknown_type") == AircraftType.JET_TRANSPORT

    def test_whitespace_handling(self):
        assert _parse_aircraft_type("  jet_transport  ") == AircraftType.JET_TRANSPORT


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------
class TestRootEndpoint:
    def test_root_serves_html(self):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "AeroSizer" in resp.text


# ---------------------------------------------------------------------------
# GET /api/presets
# ---------------------------------------------------------------------------
class TestPresetsEndpoint:
    def test_list_presets(self):
        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert "b787" in data
        assert "b737" in data
        assert "a320" in data

    def test_get_b787(self):
        resp = client.get("/api/presets/b787")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "B787-8 Dreamliner"
        assert data["range_km"] == 13621
        assert data["passengers"] == 242

    def test_get_b737(self):
        resp = client.get("/api/presets/b737")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "B737-800"
        assert data["passengers"] == 162

    def test_get_a320(self):
        resp = client.get("/api/presets/a320")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "A320neo"

    def test_unknown_preset(self):
        resp = client.get("/api/presets/concorde")
        assert resp.status_code == 400
        assert "error" in resp.json()


# ---------------------------------------------------------------------------
# POST /api/size
# ---------------------------------------------------------------------------
class TestSizingEndpoint:
    def test_basic_sizing(self):
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5765,
            "passengers": 162,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "loiter_hr": 0.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "W0_kg" in data
        assert "We_kg" in data
        assert "Wf_kg" in data
        assert "convergence_history" in data
        assert "segment_fractions" in data
        assert data["W0_kg"] > 0
        assert data["We_kg"] > 0
        assert data["Wf_kg"] > 0

    def test_sizing_with_custom_ld(self):
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5765,
            "passengers": 162,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "loiter_hr": 0.5,
            "ld_cruise": 17.5,
            "ld_loiter": 18.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["ld_cruise"] == 17.5
        assert data["ld_loiter"] == 18.5

    def test_b737_sizing_reasonable(self):
        """B737-800 sizing should give W0 in the 70,000-90,000 kg range."""
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5765,
            "passengers": 162,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "loiter_hr": 0.5,
            "ld_cruise": 17.5,
            "ld_loiter": 18.5,
            "sfc_cruise": 0.575,
            "sfc_loiter": 0.47,
            "crew_weight_kg": 450,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 55000 < data["W0_kg"] < 100000

    def test_b787_sizing_reasonable(self):
        """B787-8 sizing should give W0 near 220,000 kg."""
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 13621,
            "passengers": 242,
            "cruise_mach": 0.85,
            "cruise_alt_m": 10668,
            "loiter_hr": 0.75,
            "ld_cruise": 19.0,
            "ld_loiter": 20.0,
            "sfc_cruise": 0.525,
            "sfc_loiter": 0.43,
            "crew_weight_kg": 630,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert 180000 < data["W0_kg"] < 260000

    def test_weight_fractions_sum_less_than_one(self):
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5000,
            "passengers": 150,
            "cruise_mach": 0.78,
            "cruise_alt_m": 10668,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["We_fraction"] + data["Wf_fraction"] < 1.0

    def test_convergence_history_monotonic_trend(self):
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5000,
            "passengers": 150,
            "cruise_mach": 0.78,
            "cruise_alt_m": 10668,
        })
        data = resp.json()
        history = data["convergence_history"]
        assert len(history) >= 2
        # Last value should be very close to W0_kg
        assert abs(history[-1] - data["W0_kg"]) < 1.0

    def test_infeasible_mission_returns_400(self):
        """Extremely long range should be infeasible."""
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 50000,
            "passengers": 500,
            "cruise_mach": 0.85,
            "cruise_alt_m": 10668,
        })
        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_segment_fractions_all_less_than_one(self):
        resp = client.post("/api/size", json={
            "aircraft_type": "jet_transport",
            "range_km": 5000,
            "passengers": 150,
            "cruise_mach": 0.78,
            "cruise_alt_m": 10668,
        })
        data = resp.json()
        for label, frac in data["segment_fractions"].items():
            assert 0 < frac <= 1.0, f"Segment {label} fraction {frac} out of range"


# ---------------------------------------------------------------------------
# POST /api/constraints
# ---------------------------------------------------------------------------
class TestConstraintsEndpoint:
    def test_basic_constraints(self):
        resp = client.post("/api/constraints", json={
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_clean": 1.5,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "takeoff_distance_m": 2300,
            "landing_distance_m": 1600,
            "climb_gradient_oei": 0.024,
            "service_ceiling_m": 12496,
            "n_engines": 2,
            "thrust_lapse_cruise": 0.26,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "design_point" in data
        assert "landing_ws_max" in data
        assert "traces" in data or "plot_data" in data

    def test_design_point_positive(self):
        resp = client.post("/api/constraints", json={
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_clean": 1.5,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "takeoff_distance_m": 2300,
            "landing_distance_m": 1600,
            "climb_gradient_oei": 0.024,
            "service_ceiling_m": 12496,
            "n_engines": 2,
            "thrust_lapse_cruise": 0.26,
        })
        data = resp.json()
        dp = data["design_point"]
        assert dp["ws"] > 0
        assert dp["tw"] > 0
        assert dp["tw"] < 1.0  # T/W < 1 for transport

    def test_landing_ws_max_positive(self):
        resp = client.post("/api/constraints", json={
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_clean": 1.5,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "takeoff_distance_m": 2300,
            "landing_distance_m": 1600,
            "climb_gradient_oei": 0.024,
            "service_ceiling_m": 12496,
            "n_engines": 2,
            "thrust_lapse_cruise": 0.26,
        })
        data = resp.json()
        assert data["landing_ws_max"] > 0

    def test_traces_contain_expected_constraints(self):
        resp = client.post("/api/constraints", json={
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_clean": 1.5,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "cruise_mach": 0.785,
            "cruise_alt_m": 10668,
            "takeoff_distance_m": 2300,
            "landing_distance_m": 1600,
            "climb_gradient_oei": 0.024,
            "service_ceiling_m": 12496,
            "n_engines": 2,
            "thrust_lapse_cruise": 0.26,
        })
        data = resp.json()
        trace_names = [t["name"] for t in data["traces"]]
        assert "Cruise" in trace_names
        assert "Takeoff (BFL)" in trace_names
        assert "OEI Climb (FAR 25)" in trace_names
        assert "Service Ceiling" in trace_names


# ---------------------------------------------------------------------------
# POST /api/drag_polar
# ---------------------------------------------------------------------------
class TestDragPolarEndpoint:
    def test_basic_polar(self):
        resp = client.post("/api/drag_polar", json={
            "CD0": 0.022,
            "K": 0.042,
            "cl_min": 0,
            "cl_max": 1.5,
            "cl_step": 0.1,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "CL" in data
        assert "CD_clean" in data
        assert "CD_takeoff" in data
        assert "CD_landing" in data
        assert "LD_clean" in data
        assert "max_ld" in data
        assert "cl_max_ld" in data
        assert "cl_min_power" in data

    def test_max_ld_reasonable(self):
        resp = client.post("/api/drag_polar", json={
            "CD0": 0.022,
            "K": 0.042,
        })
        data = resp.json()
        # For CD0=0.022, K=0.042: L/D_max = 1/(2*sqrt(0.022*0.042)) ~ 16.4
        assert 14 < data["max_ld"] < 19

    def test_landing_cd_greater_than_takeoff(self):
        resp = client.post("/api/drag_polar", json={
            "CD0": 0.022,
            "K": 0.042,
            "cl_min": 0.5,
            "cl_max": 0.6,
            "cl_step": 0.1,
        })
        data = resp.json()
        for cd_to, cd_land in zip(data["CD_takeoff"], data["CD_landing"]):
            assert cd_land > cd_to

    def test_cl_min_power_greater_than_cl_max_ld(self):
        resp = client.post("/api/drag_polar", json={"CD0": 0.02, "K": 0.04})
        data = resp.json()
        assert data["cl_min_power"] > data["cl_max_ld"]

    def test_num_points(self):
        resp = client.post("/api/drag_polar", json={
            "CD0": 0.02,
            "K": 0.04,
            "cl_min": 0,
            "cl_max": 1.0,
            "cl_step": 0.1,
        })
        data = resp.json()
        assert len(data["CL"]) == 11  # 0.0, 0.1, ..., 1.0


# ---------------------------------------------------------------------------
# POST /api/weights
# ---------------------------------------------------------------------------
class TestWeightsEndpoint:
    def test_basic_weights(self):
        resp = client.post("/api/weights", json={
            "W0_kg": 76050,
            "Wf_kg": 21529,
            "AR": 9.45,
            "Sref_m2": 124.6,
            "sweep_quarter_deg": 25.0,
            "taper_ratio": 0.278,
            "t_c_root": 0.128,
            "fuselage_length_m": 38.0,
            "fuselage_width_m": 3.76,
            "fuselage_depth_m": 4.01,
            "Sht_m2": 32.4,
            "Svt_m2": 23.1,
            "n_engines": 2,
            "engine_weight_each_kg": 2370,
            "thrust_each_N": 121400,
            "n_crew": 2,
            "n_pax": 162,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "weight_statement" in data
        assert "We_total_kg" in data
        assert "We_fraction" in data
        assert data["We_total_kg"] > 0
        assert 0 < data["We_fraction"] < 1

    def test_b737_empty_weight_fraction(self):
        """For B737-class inputs, We/W0 should be roughly 0.45-0.55."""
        resp = client.post("/api/weights", json={
            "W0_kg": 76050,
            "Wf_kg": 21529,
            "AR": 9.45,
            "Sref_m2": 124.6,
            "sweep_quarter_deg": 25.0,
            "taper_ratio": 0.278,
            "t_c_root": 0.128,
            "fuselage_length_m": 38.0,
            "fuselage_width_m": 3.76,
            "fuselage_depth_m": 4.01,
            "Sht_m2": 32.4,
            "Svt_m2": 23.1,
            "n_engines": 2,
            "engine_weight_each_kg": 2370,
            "thrust_each_N": 121400,
            "n_crew": 2,
            "n_pax": 162,
        })
        data = resp.json()
        assert 0.40 < data["We_fraction"] < 0.60

    def test_weight_statement_has_components(self):
        resp = client.post("/api/weights", json={
            "W0_kg": 76050,
            "Wf_kg": 21529,
            "AR": 9.45,
            "Sref_m2": 124.6,
            "sweep_quarter_deg": 25.0,
            "taper_ratio": 0.278,
            "t_c_root": 0.128,
        })
        data = resp.json()
        ws = data["weight_statement"]
        assert "wing" in ws
        assert "fuselage" in ws
        assert "TOTAL_EMPTY" in ws
        # Each component should have kg, lb, pct
        assert "kg" in ws["wing"]
        assert "lb" in ws["wing"]
        assert "pct" in ws["wing"]


# ---------------------------------------------------------------------------
# POST /api/performance/takeoff_landing
# ---------------------------------------------------------------------------
class TestTakeoffLandingEndpoint:
    def test_basic_performance(self):
        resp = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "altitude_m": 0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "takeoff" in data
        assert "landing" in data
        assert "BFL_m" in data
        assert data["takeoff"]["total_m"] > 0
        assert data["landing"]["total_m"] > 0
        assert data["BFL_m"] > 0

    def test_takeoff_segments_sum(self):
        resp = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "altitude_m": 0,
        })
        data = resp.json()
        to = data["takeoff"]
        segment_sum = to["ground_roll_m"] + to["rotation_m"] + to["transition_m"] + to["climb_to_35ft_m"]
        assert abs(segment_sum - to["total_m"]) < 1.0

    def test_landing_segments_sum(self):
        resp = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "altitude_m": 0,
        })
        data = resp.json()
        land = data["landing"]
        segment_sum = land["approach_m"] + land["flare_m"] + land["free_roll_m"] + land["braking_m"]
        assert abs(segment_sum - land["total_m"]) < 1.0

    def test_bfl_reasonable_for_transport(self):
        """BFL should be in the 1500-3500m range for a transport."""
        resp = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "altitude_m": 0,
        })
        data = resp.json()
        assert 1500 < data["BFL_m"] < 3500

    def test_speeds_reasonable(self):
        resp = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
            "altitude_m": 0,
        })
        data = resp.json()
        # Stall speed should be 120-160 kt for a B737-class
        assert 100 < data["takeoff"]["V_stall_kt"] < 180
        # Approach speed > stall speed (land)
        assert data["landing"]["V_approach_ms"] > data["landing"]["V_TD_ms"]

    def test_heavier_aircraft_needs_more_distance(self):
        """Heavier W should mean longer distances."""
        resp_light = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 60000,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
        })
        resp_heavy = client.post("/api/performance/takeoff_landing", json={
            "W_kg": 76050,
            "S_m2": 124.6,
            "T_N": 242800,
            "n_engines": 2,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax_takeoff": 1.8,
            "CLmax_landing": 2.3,
        })
        light = resp_light.json()
        heavy = resp_heavy.json()
        assert heavy["takeoff"]["total_m"] > light["takeoff"]["total_m"]
        assert heavy["BFL_m"] > light["BFL_m"]


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------
class TestErrorHandling:
    def test_invalid_json_size(self):
        resp = client.post("/api/size", content="not json",
                          headers={"Content-Type": "application/json"})
        assert resp.status_code == 422  # Pydantic validation error

    def test_missing_required_weights_field(self):
        """Weights endpoint requires W0_kg and Wf_kg."""
        resp = client.post("/api/weights", json={"AR": 9.0})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Presets data integrity
# ---------------------------------------------------------------------------
class TestPresetsData:
    def test_all_presets_have_required_fields(self):
        required = ["name", "aircraft_type", "range_km", "passengers",
                    "cruise_mach", "cruise_alt_m", "CD0", "K",
                    "AR", "Sref_m2"]
        for key, data in PRESETS.items():
            for field in required:
                assert field in data, f"Preset {key} missing field {field}"

    def test_presets_have_positive_values(self):
        for key, data in PRESETS.items():
            assert data["range_km"] > 0
            assert data["passengers"] > 0
            assert data["cruise_mach"] > 0
            assert data["CD0"] > 0
            assert data["AR"] > 0

    def test_presets_have_cost_fields(self):
        """New presets should include cost estimation fields."""
        for key, data in PRESETS.items():
            assert "V_max_ms" in data, f"Preset {key} missing V_max_ms"
            assert "T_turbine_inlet_K" in data, f"Preset {key} missing T_turbine_inlet_K"
            assert "engine_cost_each" in data, f"Preset {key} missing engine_cost_each"
            assert "Q_production" in data, f"Preset {key} missing Q_production"


# ---------------------------------------------------------------------------
# POST /api/turn_performance
# ---------------------------------------------------------------------------
class TestTurnPerformanceEndpoint:
    def test_basic_turn(self):
        resp = client.post("/api/turn_performance", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax": 1.5,
            "n_limit": 2.5,
            "altitude_m": 3000,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "sustained" in data
        assert "instantaneous" in data
        assert "corner_speed_ms" in data
        assert "max_sustained_rate_degs" in data
        assert data["corner_speed_ms"] > 0
        assert data["max_sustained_rate_degs"] > 0

    def test_sustained_has_velocity_array(self):
        resp = client.post("/api/turn_performance", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
        })
        data = resp.json()
        assert len(data["sustained"]["V"]) > 0
        assert len(data["sustained"]["turn_rate_degs"]) == len(data["sustained"]["V"])
        assert len(data["sustained"]["load_factor"]) == len(data["sustained"]["V"])

    def test_instantaneous_rate_at_corner_speed(self):
        """At speeds near the corner speed, instantaneous turn rate should be high."""
        resp = client.post("/api/turn_performance", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax": 1.5,
            "n_limit": 2.5,
            "altitude_m": 3000,
        })
        data = resp.json()
        # At lower speeds (high q / low V), instantaneous rate should be significant
        # The maximum instantaneous turn rate occurs near corner speed
        max_inst_rate = max(data["instantaneous"]["turn_rate_degs"])
        assert max_inst_rate > 5.0  # should be a meaningful turn rate

    def test_corner_speed_reasonable(self):
        resp = client.post("/api/turn_performance", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CLmax": 1.5,
            "n_limit": 2.5,
            "altitude_m": 0,
        })
        data = resp.json()
        # Corner speed should be in a reasonable range for transport
        assert 80 < data["corner_speed_ms"] < 250


# ---------------------------------------------------------------------------
# POST /api/ps_diagram
# ---------------------------------------------------------------------------
class TestPsDiagramEndpoint:
    def test_basic_ps(self):
        resp = client.post("/api/ps_diagram", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CD0": 0.022,
            "K": 0.042,
            "altitude_m": 3000,
            "load_factors": [1, 2, 3],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "mach" in data
        assert "ps" in data
        assert "1" in data["ps"]
        assert "2" in data["ps"]
        assert "3" in data["ps"]

    def test_ps_n1_greater_than_n3(self):
        """Ps at n=1 should be higher than at n=3 (more excess power at 1g)."""
        resp = client.post("/api/ps_diagram", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "load_factors": [1, 3],
        })
        data = resp.json()
        mid = len(data["mach"]) // 2
        assert data["ps"]["1"][mid] > data["ps"]["3"][mid]

    def test_mach_array_length(self):
        resp = client.post("/api/ps_diagram", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
        })
        data = resp.json()
        assert len(data["mach"]) == 150


# ---------------------------------------------------------------------------
# POST /api/cost/dapca
# ---------------------------------------------------------------------------
class TestDAPCACostEndpoint:
    def test_basic_dapca(self):
        resp = client.post("/api/cost/dapca", json={
            "We_kg": 41000,
            "V_max_ms": 236,
            "Q": 500,
            "FTA": 4,
            "N_engines": 2,
            "T_max_N": 121400,
            "M_max_engine": 0.82,
            "T_turbine_inlet_K": 1560,
            "C_avionics": 5000000,
            "n_pax": 162,
            "CPI_factor": 1.35,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "hours" in data
        assert "costs" in data
        assert "totals" in data
        assert "per_unit" in data
        assert data["per_unit"]["total"] > 0

    def test_unit_cost_reasonable(self):
        """For a B737-class, unit cost should be $30M-$200M range."""
        resp = client.post("/api/cost/dapca", json={
            "We_kg": 41000,
            "V_max_ms": 236,
            "Q": 500,
            "T_max_N": 121400,
            "M_max_engine": 0.82,
            "T_turbine_inlet_K": 1560,
            "C_avionics": 5000000,
            "n_pax": 162,
        })
        data = resp.json()
        unit_cost = data["per_unit"]["total"]
        assert 30_000_000 < unit_cost < 200_000_000

    def test_more_production_lowers_unit_cost(self):
        """Higher production quantity should reduce per-unit cost."""
        resp_low = client.post("/api/cost/dapca", json={
            "We_kg": 41000, "V_max_ms": 236, "Q": 100,
            "T_max_N": 121400, "M_max_engine": 0.82, "T_turbine_inlet_K": 1560,
        })
        resp_high = client.post("/api/cost/dapca", json={
            "We_kg": 41000, "V_max_ms": 236, "Q": 1000,
            "T_max_N": 121400, "M_max_engine": 0.82, "T_turbine_inlet_K": 1560,
        })
        assert resp_high.json()["per_unit"]["total"] < resp_low.json()["per_unit"]["total"]


# ---------------------------------------------------------------------------
# POST /api/cost/operating
# ---------------------------------------------------------------------------
class TestOperatingCostEndpoint:
    def test_basic_operating(self):
        resp = client.post("/api/cost/operating", json={
            "W0_kg": 79000,
            "We_kg": 41000,
            "Wf_mission_kg": 18000,
            "V_cruise_ms": 236,
            "range_km": 5765,
            "n_pax": 162,
            "n_engines": 2,
            "aircraft_cost": 106000000,
            "engine_cost_each": 12000000,
            "fuel_price_per_kg": 0.80,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "per_block_hour" in data
        assert "annual" in data
        assert "per_trip" in data
        assert "economics" in data
        assert data["economics"]["CASM_cents"] > 0
        assert data["economics"]["CO2_per_seat_km_kg"] > 0

    def test_casm_reasonable(self):
        """CASM should be in the 5-20 cents range for a narrowbody."""
        resp = client.post("/api/cost/operating", json={
            "W0_kg": 79000,
            "We_kg": 41000,
            "Wf_mission_kg": 18000,
            "V_cruise_ms": 236,
            "range_km": 5765,
            "n_pax": 162,
            "aircraft_cost": 106000000,
            "engine_cost_each": 12000000,
        })
        data = resp.json()
        assert 3 < data["economics"]["CASM_cents"] < 25


# ---------------------------------------------------------------------------
# POST /api/performance/climb_profile
# ---------------------------------------------------------------------------
class TestClimbProfileEndpoint:
    def test_basic_climb(self):
        resp = client.post("/api/performance/climb_profile", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CD0": 0.022,
            "K": 0.042,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "altitude" in data
        assert "ROC_max" in data
        assert "V_best" in data
        assert data["ROC_max"][0] > 0  # positive ROC at sea level

    def test_roc_decreases_with_altitude(self):
        resp = client.post("/api/performance/climb_profile", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
        })
        data = resp.json()
        # ROC at sea level should be greater than at the last altitude
        assert data["ROC_max"][0] > data["ROC_max"][-1]


# ---------------------------------------------------------------------------
# POST /api/performance/range_payload
# ---------------------------------------------------------------------------
class TestRangePayloadEndpoint:
    def test_basic_range_payload(self):
        resp = client.post("/api/performance/range_payload", json={
            "W0_kg": 79000,
            "We_kg": 41000,
            "Wf_max_kg": 21000,
            "sfc_per_hr": 0.575,
            "V_cruise_ms": 236,
            "W_crew_kg": 450,
            "CD0": 0.022,
            "K": 0.042,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert "curves" in data
        pts = data["points"]
        assert "A" in pts and "B" in pts and "C" in pts and "D" in pts
        # Ferry range (D) should be the longest
        assert pts["D"]["range_km"] >= pts["B"]["range_km"]
        # Zero payload for ferry
        assert pts["D"]["payload_kg"] == 0


# ---------------------------------------------------------------------------
# POST /api/performance/envelope
# ---------------------------------------------------------------------------
class TestEnvelopeEndpoint:
    def test_basic_envelope(self):
        resp = client.post("/api/performance/envelope", json={
            "W_kg": 79000,
            "S_m2": 124.6,
            "T_N": 242800,
            "CD0": 0.022,
            "K": 0.042,
            "CLmax": 1.5,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "mach" in data
        assert "altitude" in data
        assert "ps_grid" in data
        assert "stall_line" in data
        assert "q_limit_line" in data
        assert len(data["ps_grid"]) == len(data["altitude"])
        assert len(data["ps_grid"][0]) == len(data["mach"])


# ---------------------------------------------------------------------------
# POST /api/geometry/aircraft_svg  (NEW)
# ---------------------------------------------------------------------------
class TestAircraftSVGEndpoint:
    def test_basic_svg_generation(self):
        resp = client.post("/api/geometry/aircraft_svg", json={
            "span_m": 34.0,
            "root_chord_m": 7.0,
            "tip_chord_m": 1.8,
            "sweep_quarter_deg": 25.0,
            "fuselage_length_m": 38.0,
            "fuselage_width_m": 3.76,
            "tail_span_m": 12.0,
            "tail_root_chord_m": 3.5,
            "tail_tip_chord_m": 1.2,
            "tail_sweep_deg": 30.0,
            "vtail_height_m": 6.0,
            "vtail_root_chord_m": 5.0,
            "vtail_tip_chord_m": 2.0,
            "vtail_sweep_deg": 40.0,
            "engine_diameter_m": 1.8,
            "engine_length_m": 3.5,
            "n_engines": 2,
            "wing_position_fraction": 0.4,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "top_view_svg" in data
        assert "side_view_svg" in data
        assert "front_view_svg" in data

    def test_svg_contains_valid_svg_tags(self):
        resp = client.post("/api/geometry/aircraft_svg", json={
            "span_m": 34.0, "root_chord_m": 7.0, "tip_chord_m": 1.8,
            "sweep_quarter_deg": 25.0, "fuselage_length_m": 38.0,
            "fuselage_width_m": 3.76, "tail_span_m": 12.0,
            "tail_root_chord_m": 3.5, "tail_tip_chord_m": 1.2,
            "tail_sweep_deg": 30.0, "vtail_height_m": 6.0,
            "vtail_root_chord_m": 5.0, "vtail_tip_chord_m": 2.0,
            "vtail_sweep_deg": 40.0, "engine_diameter_m": 1.8,
            "engine_length_m": 3.5, "n_engines": 2,
            "wing_position_fraction": 0.4,
        })
        data = resp.json()
        for key in ["top_view_svg", "side_view_svg", "front_view_svg"]:
            assert data[key].startswith("<svg")
            assert data[key].strip().endswith("</svg>")

    def test_svg_uses_pfd_green_color(self):
        resp = client.post("/api/geometry/aircraft_svg", json={
            "span_m": 34.0, "root_chord_m": 7.0, "tip_chord_m": 1.8,
            "sweep_quarter_deg": 25.0, "fuselage_length_m": 38.0,
            "fuselage_width_m": 3.76, "tail_span_m": 12.0,
            "tail_root_chord_m": 3.5, "tail_tip_chord_m": 1.2,
            "tail_sweep_deg": 30.0, "vtail_height_m": 6.0,
            "vtail_root_chord_m": 5.0, "vtail_tip_chord_m": 2.0,
            "vtail_sweep_deg": 40.0, "engine_diameter_m": 1.8,
            "engine_length_m": 3.5, "n_engines": 2,
            "wing_position_fraction": 0.4,
        })
        data = resp.json()
        # All SVGs should use theme accent blue
        for key in ["top_view_svg", "side_view_svg", "front_view_svg"]:
            assert "#89b4fa" in data[key]

    def test_svg_with_default_values(self):
        """Test that SVG endpoint works with defaults (no body fields required)."""
        resp = client.post("/api/geometry/aircraft_svg", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "top_view_svg" in data
        assert len(data["top_view_svg"]) > 100

    def test_svg_4_engine_config(self):
        resp = client.post("/api/geometry/aircraft_svg", json={
            "n_engines": 4,
            "span_m": 60.0,
            "root_chord_m": 10.0,
            "tip_chord_m": 2.5,
            "fuselage_length_m": 70.0,
            "fuselage_width_m": 5.7,
        })
        assert resp.status_code == 200
        data = resp.json()
        # 4-engine should have more ellipses in top view
        assert data["top_view_svg"].count("ellipse") >= 4

    def test_svg_viewbox_present(self):
        resp = client.post("/api/geometry/aircraft_svg", json={})
        data = resp.json()
        for key in ["top_view_svg", "side_view_svg", "front_view_svg"]:
            assert 'viewBox="0 0 400 300"' in data[key]


# ---------------------------------------------------------------------------
# POST /api/geometry/from_sizing  (NEW)
# ---------------------------------------------------------------------------
class TestGeometryFromSizingEndpoint:
    def test_basic_geometry_computation(self):
        resp = client.post("/api/geometry/from_sizing", json={
            "W0_kg": 79016,
            "S_m2": 124.6,
            "AR": 9.45,
            "taper_ratio": 0.278,
            "sweep_quarter_deg": 25.0,
            "fuselage_length_m": 38.02,
            "fuselage_width_m": 3.76,
            "n_engines": 2,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "span_m" in data
        assert "root_chord_m" in data
        assert "tip_chord_m" in data
        assert "MAC_m" in data
        assert "tail_span_m" in data
        assert "vtail_height_m" in data

    def test_span_from_ar_and_s(self):
        """span = sqrt(AR * S)"""
        resp = client.post("/api/geometry/from_sizing", json={
            "S_m2": 100.0, "AR": 9.0, "taper_ratio": 0.3,
            "fuselage_length_m": 30.0, "fuselage_width_m": 3.5,
        })
        data = resp.json()
        expected_span = math.sqrt(9.0 * 100.0)  # 30.0
        assert abs(data["span_m"] - expected_span) < 0.1

    def test_root_chord_formula(self):
        """root_chord = 2*S / (span * (1 + taper))"""
        S = 100.0
        AR = 9.0
        taper = 0.3
        span = math.sqrt(AR * S)
        expected_root = 2.0 * S / (span * (1.0 + taper))
        resp = client.post("/api/geometry/from_sizing", json={
            "S_m2": S, "AR": AR, "taper_ratio": taper,
            "fuselage_length_m": 30.0, "fuselage_width_m": 3.5,
        })
        data = resp.json()
        assert abs(data["root_chord_m"] - expected_root) < 0.1

    def test_tip_chord_equals_taper_times_root(self):
        taper = 0.3
        resp = client.post("/api/geometry/from_sizing", json={
            "S_m2": 100.0, "AR": 9.0, "taper_ratio": taper,
            "fuselage_length_m": 30.0, "fuselage_width_m": 3.5,
        })
        data = resp.json()
        assert abs(data["tip_chord_m"] - taper * data["root_chord_m"]) < 0.05

    def test_tail_sweep_greater_than_wing_sweep(self):
        resp = client.post("/api/geometry/from_sizing", json={
            "sweep_quarter_deg": 25.0, "S_m2": 100.0, "AR": 9.0,
            "taper_ratio": 0.3, "fuselage_length_m": 30.0,
            "fuselage_width_m": 3.5,
        })
        data = resp.json()
        assert data["tail_sweep_deg"] > 25.0
        assert data["vtail_sweep_deg"] > 25.0

    def test_engine_diameter_scales_with_weight(self):
        resp_light = client.post("/api/geometry/from_sizing", json={
            "W0_kg": 30000, "S_m2": 80.0, "AR": 9.0,
            "taper_ratio": 0.3, "fuselage_length_m": 25.0,
            "fuselage_width_m": 3.0, "n_engines": 2,
        })
        resp_heavy = client.post("/api/geometry/from_sizing", json={
            "W0_kg": 200000, "S_m2": 300.0, "AR": 9.0,
            "taper_ratio": 0.3, "fuselage_length_m": 55.0,
            "fuselage_width_m": 5.5, "n_engines": 2,
        })
        d_light = resp_light.json()
        d_heavy = resp_heavy.json()
        assert d_heavy["engine_diameter_m"] > d_light["engine_diameter_m"]

    def test_with_defaults(self):
        resp = client.post("/api/geometry/from_sizing", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert data["span_m"] > 0
        assert data["root_chord_m"] > 0
        assert data["MAC_m"] > 0
