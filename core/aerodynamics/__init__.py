"""
Aerodynamics Package
Reference: Raymer Ch 12, "Aircraft Design: A Conceptual Approach", 6th Edition

Sub-modules:
    parasite_drag  - Component buildup method for CD0
    induced_drag   - Oswald efficiency and CDi
    drag_polar     - Complete drag polar with config increments
"""

from .parasite_drag import (
    skin_friction_coeff,
    form_factor_wing,
    form_factor_fuselage,
    form_factor_nacelle,
    reynolds_number,
    component_drag,
    parasite_drag_buildup,
    leakage_protuberance_drag,
)

from .induced_drag import (
    oswald_efficiency_straight,
    oswald_efficiency_swept,
    induced_drag_factor,
    induced_drag_coeff,
)

from .drag_polar import (
    DragPolar,
    create_transport_polar,
)
