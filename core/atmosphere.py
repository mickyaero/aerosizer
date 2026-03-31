"""
ISA Standard Atmosphere Model
Reference: Raymer Appendix B, ICAO Standard Atmosphere

Provides temperature, pressure, density, and speed of sound
as functions of geometric/geopotential altitude.
"""

import numpy as np

# Constants
R_AIR = 287.058       # J/(kg·K) - specific gas constant for air
GAMMA = 1.4           # ratio of specific heats
G0 = 9.80665          # m/s² - standard gravity
R_EARTH = 6356766.0   # m - Earth's radius (for geopotential conversion)

# Sea-level conditions
T0 = 288.15    # K
P0 = 101325.0  # Pa
RHO0 = 1.225   # kg/m³
A0 = 340.294    # m/s - speed of sound at sea level

# ISA layers: (base altitude m, lapse rate K/m)
# Troposphere: 0-11km, lapse = -6.5 K/km
# Tropopause:  11-20km, lapse = 0
# Stratosphere: 20-32km, lapse = +1.0 K/km
_LAYERS = [
    (0.0,     -0.0065,  288.15,  101325.0),
    (11000.0,  0.0,     216.65,  22632.1),
    (20000.0,  0.001,   216.65,  5474.89),
    (32000.0,  0.0028,  228.65,  868.019),
    (47000.0,  0.0,     270.65,  110.906),
]


def geometric_to_geopotential(h_geometric: float) -> float:
    """Convert geometric altitude to geopotential altitude.

    Args:
        h_geometric: geometric altitude in meters

    Returns:
        geopotential altitude in meters
    """
    return R_EARTH * h_geometric / (R_EARTH + h_geometric)


def geopotential_to_geometric(h_geopot: float) -> float:
    """Convert geopotential altitude to geometric altitude."""
    return R_EARTH * h_geopot / (R_EARTH - h_geopot)


def atmosphere(altitude_m: float, geometric: bool = True) -> dict:
    """Compute ISA atmospheric properties at a given altitude.

    Args:
        altitude_m: altitude in meters
        geometric: if True, input is geometric altitude; if False, geopotential

    Returns:
        dict with keys:
            T: temperature (K)
            P: pressure (Pa)
            rho: density (kg/m³)
            a: speed of sound (m/s)
            mu: dynamic viscosity (Pa·s) via Sutherland's law
            sigma: density ratio rho/rho0
            delta: pressure ratio P/P0
            theta: temperature ratio T/T0
    """
    if geometric:
        h = geometric_to_geopotential(altitude_m)
    else:
        h = altitude_m

    # Find the correct layer
    T_base, P_base, lapse = T0, P0, _LAYERS[0][1]
    for i, (h_base, a_lapse, T_b, P_b) in enumerate(_LAYERS):
        if i + 1 < len(_LAYERS) and h >= _LAYERS[i + 1][0]:
            continue
        T_base = T_b
        P_base = P_b
        lapse = a_lapse
        dh = h - h_base
        break

    # Temperature
    T = T_base + lapse * dh

    # Pressure (gradient vs isothermal layer)
    if abs(lapse) > 1e-10:
        P = P_base * (T / T_base) ** (-G0 / (lapse * R_AIR))
    else:
        P = P_base * np.exp(-G0 * dh / (R_AIR * T_base))

    # Density from ideal gas law
    rho = P / (R_AIR * T)

    # Speed of sound
    a = np.sqrt(GAMMA * R_AIR * T)

    # Dynamic viscosity - Sutherland's law
    # mu = mu_ref * (T/T_ref)^1.5 * (T_ref + S) / (T + S)
    mu_ref = 1.716e-5  # Pa·s at T_ref
    T_ref = 273.15      # K
    S = 110.4            # K - Sutherland constant
    mu = mu_ref * (T / T_ref) ** 1.5 * (T_ref + S) / (T + S)

    return {
        "T": T,
        "P": P,
        "rho": rho,
        "a": a,
        "mu": mu,
        "sigma": rho / RHO0,
        "delta": P / P0,
        "theta": T / T0,
    }


def density_at(altitude_m: float) -> float:
    """Shorthand: get density at altitude (geometric, meters)."""
    return atmosphere(altitude_m)["rho"]


def temperature_at(altitude_m: float) -> float:
    """Shorthand: get temperature at altitude (geometric, meters)."""
    return atmosphere(altitude_m)["T"]


def pressure_at(altitude_m: float) -> float:
    """Shorthand: get pressure at altitude (geometric, meters)."""
    return atmosphere(altitude_m)["P"]


def speed_of_sound_at(altitude_m: float) -> float:
    """Shorthand: get speed of sound at altitude (geometric, meters)."""
    return atmosphere(altitude_m)["a"]


def dynamic_pressure(velocity_ms: float, altitude_m: float) -> float:
    """Compute dynamic pressure q = 0.5 * rho * V².

    Args:
        velocity_ms: true airspeed in m/s
        altitude_m: altitude in meters (geometric)

    Returns:
        dynamic pressure in Pa
    """
    rho = density_at(altitude_m)
    return 0.5 * rho * velocity_ms ** 2


def mach_number(velocity_ms: float, altitude_m: float) -> float:
    """Compute Mach number at given speed and altitude."""
    a = speed_of_sound_at(altitude_m)
    return velocity_ms / a


def true_airspeed(mach: float, altitude_m: float) -> float:
    """Compute true airspeed from Mach number and altitude."""
    a = speed_of_sound_at(altitude_m)
    return mach * a


# Unit conversions
def ft_to_m(ft: float) -> float:
    return ft * 0.3048

def m_to_ft(m: float) -> float:
    return m / 0.3048

def kt_to_ms(kt: float) -> float:
    return kt * 0.514444

def ms_to_kt(ms: float) -> float:
    return ms / 0.514444

def kg_to_lb(kg: float) -> float:
    return kg * 2.20462

def lb_to_kg(lb: float) -> float:
    return lb / 2.20462

def m2_to_ft2(m2: float) -> float:
    return m2 * 10.7639

def ft2_to_m2(ft2: float) -> float:
    return ft2 / 10.7639

def n_to_lbf(n: float) -> float:
    return n / 4.44822

def lbf_to_n(lbf: float) -> float:
    return lbf * 4.44822

def km_to_nm(km: float) -> float:
    return km / 1.852

def nm_to_km(nm: float) -> float:
    return nm * 1.852
