"""
Performance Analysis Package
Reference: Raymer Ch 17, "Aircraft Design: A Conceptual Approach", 6th Edition

Sub-modules:
    level_flight     - Stall speed, max speed, thrust/power required, drag curves
    climb            - Rate of climb, ceiling, time to climb
    range_endurance  - Breguet range/endurance, cruise-climb, payload-range
    takeoff          - Takeoff/landing distances, balanced field length
"""

from .level_flight import (
    stall_speed,
    max_speed,
    min_drag_speed,
    max_range_speed,
    max_endurance_speed,
    thrust_required,
    power_required,
    generate_tr_curve,
)

from .climb import (
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

from .range_endurance import (
    breguet_range_jet,
    breguet_endurance_jet,
    breguet_range_prop,
    cruise_climb_range,
    specific_air_range,
    fuel_for_range,
    range_payload_diagram,
)

from .takeoff import (
    ground_roll_distance,
    rotation_distance,
    transition_distance,
    climb_to_obstacle,
    total_takeoff_distance,
    balanced_field_length,
    landing_distance,
    generate_takeoff_landing_summary,
)
