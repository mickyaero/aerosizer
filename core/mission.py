"""
Mission Profile Definition
Reference: Raymer Ch 3, Ch 6, Ch 19

Defines the mission as a sequence of segments (warmup, taxi, takeoff,
climb, cruise, loiter, descent, landing). Each segment has a method
for computing its weight fraction Wi/Wi-1.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SegmentType(Enum):
    ENGINE_START = "engine_start"
    TAXI = "taxi"
    TAKEOFF = "takeoff"
    CLIMB = "climb"
    CRUISE = "cruise"
    LOITER = "loiter"
    DESCENT = "descent"
    LANDING = "landing"
    COMBAT = "combat"        # for fighter aircraft
    DASH = "dash"            # high-speed dash segment
    CUSTOM = "custom"


class AircraftType(Enum):
    """Aircraft type for statistical weight estimation. Raymer Table 3.1."""
    JET_TRANSPORT = "jet_transport"
    JET_FIGHTER = "jet_fighter"
    JET_TRAINER = "jet_trainer"
    MILITARY_CARGO = "military_cargo"
    BUSINESS_JET = "business_jet"
    TURBOPROP = "turboprop"
    GA_SINGLE = "ga_single_engine"
    GA_TWIN = "ga_twin_engine"
    FLYING_BOAT = "flying_boat"
    UAV = "uav"


# Raymer Table 3.1: We/W0 = A * W0^C (W0 in lb)
EMPTY_WEIGHT_FRACTION = {
    AircraftType.JET_TRANSPORT: {"A": 1.02, "C": -0.06},
    AircraftType.JET_FIGHTER: {"A": 2.34, "C": -0.13},
    AircraftType.JET_TRAINER: {"A": 1.59, "C": -0.10},
    AircraftType.MILITARY_CARGO: {"A": 0.93, "C": -0.07},
    AircraftType.BUSINESS_JET: {"A": 1.15, "C": -0.05},
    AircraftType.TURBOPROP: {"A": 0.96, "C": -0.05},
    AircraftType.GA_SINGLE: {"A": 2.36, "C": -0.18},
    AircraftType.GA_TWIN: {"A": 1.51, "C": -0.10},
    AircraftType.FLYING_BOAT: {"A": 1.09, "C": -0.05},
    AircraftType.UAV: {"A": 0.86, "C": -0.06},
}


# Default mission segment weight fractions (Raymer Table 6.2)
DEFAULT_WEIGHT_FRACTIONS = {
    SegmentType.ENGINE_START: 0.990,
    SegmentType.TAXI: 0.995,
    SegmentType.TAKEOFF: 0.970,
    SegmentType.CLIMB: 0.985,
    SegmentType.DESCENT: 0.990,
    SegmentType.LANDING: 0.995,
}


@dataclass
class MissionSegment:
    """A single segment of the mission profile.

    Attributes:
        segment_type: type of flight segment
        weight_fraction: Wi/Wi-1 (if known/fixed, e.g. historical)
        range_km: range for cruise segments (km)
        endurance_hr: time for loiter segments (hours)
        altitude_m: segment altitude (m)
        mach: segment Mach number
        speed_ms: segment speed (m/s), alternative to Mach
        ld_ratio: L/D for this segment (can be estimated or specified)
        sfc: specific fuel consumption (1/s for jets, 1/s for props via conversion)
        label: human-readable name
    """
    segment_type: SegmentType
    weight_fraction: Optional[float] = None
    range_km: Optional[float] = None
    endurance_hr: Optional[float] = None
    altitude_m: float = 0.0
    mach: Optional[float] = None
    speed_ms: Optional[float] = None
    ld_ratio: Optional[float] = None
    sfc: Optional[float] = None
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self.segment_type.value.replace("_", " ").title()

        # Use historical default if weight fraction not specified
        # and no range/endurance given for computation
        if (self.weight_fraction is None
                and self.range_km is None
                and self.endurance_hr is None
                and self.segment_type in DEFAULT_WEIGHT_FRACTIONS):
            self.weight_fraction = DEFAULT_WEIGHT_FRACTIONS[self.segment_type]


@dataclass
class MissionProfile:
    """Complete mission profile as a sequence of segments.

    Attributes:
        name: mission name
        aircraft_type: type of aircraft (for statistical estimation)
        segments: ordered list of mission segments
        crew_weight_kg: total crew weight
        payload_weight_kg: payload weight (passengers + cargo)
        passengers: number of passengers (for transport sizing)
        passenger_weight_kg: weight per passenger including luggage (default 100 kg)
        fuel_reserve_fraction: reserve fuel as fraction of mission fuel (default 0.06 = 6%)
        trapped_fuel_fraction: trapped/unusable fuel fraction (included in reserve)
    """
    name: str = "Unnamed Mission"
    aircraft_type: AircraftType = AircraftType.JET_TRANSPORT
    segments: list = field(default_factory=list)
    crew_weight_kg: float = 0.0
    payload_weight_kg: float = 0.0
    passengers: int = 0
    passenger_weight_kg: float = 100.0  # kg per passenger (with luggage)
    fuel_reserve_fraction: float = 0.06  # Raymer: 6% for reserves + trapped
    trapped_fuel_fraction: float = 0.0   # included in reserve if nonzero

    @property
    def total_payload_kg(self) -> float:
        """Total payload = specified payload + passengers * per-pax weight."""
        return self.payload_weight_kg + self.passengers * self.passenger_weight_kg

    def add_segment(self, segment: MissionSegment):
        self.segments.append(segment)

    @classmethod
    def transport_default(cls, range_km: float, passengers: int,
                          cruise_mach: float, cruise_alt_m: float,
                          loiter_hr: float = 0.5) -> "MissionProfile":
        """Create a standard transport mission profile.

        Raymer Figure 6.2: Engine start → Taxi → Takeoff → Climb →
        Cruise → Loiter → Descent → Landing

        Args:
            range_km: design range in km
            passengers: number of passengers
            cruise_mach: cruise Mach number
            cruise_alt_m: cruise altitude in meters
            loiter_hr: loiter endurance in hours (default 30 min)
        """
        mission = cls(
            name=f"Transport {range_km:.0f}km {passengers}pax",
            aircraft_type=AircraftType.JET_TRANSPORT,
            passengers=passengers,
            crew_weight_kg=2 * 90 + 6 * 75,  # 2 pilots + 6 cabin crew
        )

        mission.segments = [
            MissionSegment(SegmentType.ENGINE_START, label="1. Engine Start & Warmup"),
            MissionSegment(SegmentType.TAXI, label="2. Taxi"),
            MissionSegment(SegmentType.TAKEOFF, label="3. Takeoff"),
            MissionSegment(SegmentType.CLIMB, label="4. Climb to Cruise"),
            MissionSegment(
                SegmentType.CRUISE,
                range_km=range_km,
                altitude_m=cruise_alt_m,
                mach=cruise_mach,
                label="5. Cruise",
            ),
            MissionSegment(
                SegmentType.LOITER,
                endurance_hr=loiter_hr,
                altitude_m=cruise_alt_m * 0.5,  # loiter at lower altitude
                label="6. Loiter (Reserve)",
            ),
            MissionSegment(SegmentType.DESCENT, label="7. Descent"),
            MissionSegment(SegmentType.LANDING, label="8. Landing & Taxi"),
        ]

        return mission

    @classmethod
    def b787_8_mission(cls) -> "MissionProfile":
        """B787-8 Dreamliner mission profile from the AIDAA 2017 paper.

        Used as primary validation case.
        Range: 13,621 km (7,355 nm)
        Passengers: 242 (2-class)
        Cruise: Mach 0.85 at 35,000 ft (10,668 m)
        """
        return cls.transport_default(
            range_km=13621.0,
            passengers=242,
            cruise_mach=0.85,
            cruise_alt_m=10668.0,  # 35,000 ft
            loiter_hr=0.75,
        )
