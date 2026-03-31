"""
Statistical Weight Estimation for Transport Aircraft
Reference: Raymer, "Aircraft Design: A Conceptual Approach", 6th Edition, Ch 15
           Equations 15.25 through 15.44 (Transport/Bomber category)

All inputs are SI (kg, m, m^2, radians). Internal calculations convert
to imperial units (lb, ft, ft^2) per the Raymer equations, then convert
the result back to kg.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from ..atmosphere import kg_to_lb, lb_to_kg, m2_to_ft2, ft2_to_m2, m_to_ft, ft_to_m


# ---------------------------------------------------------------------------
# Helper: dynamic pressure in psf from velocity (m/s) and altitude (m)
# ---------------------------------------------------------------------------
def _q_cruise_psf(velocity_ms: float, altitude_m: float) -> float:
    """Return dynamic pressure in lb/ft^2 for Raymer equations.

    Uses ISA sea-level density (1.225 kg/m^3) scaled by sigma.
    For a first-order estimate the caller can pass cruise V and alt.
    This is only used when the user does not inject q directly.
    """
    from ..atmosphere import atmosphere
    atm = atmosphere(altitude_m)
    rho = atm["rho"]  # kg/m^3
    q_pa = 0.5 * rho * velocity_ms ** 2
    # 1 Pa = 0.020885 psf
    return q_pa * 0.020885434


@dataclass
class TransportWeights:
    """Raymer Ch 15 statistical weight equations for Transport aircraft.

    Every method returns weight in **kg**.  Internally each equation
    converts to imperial (lb, ft, ft^2) as published by Raymer, then
    converts the answer back to SI.

    Attributes (all SI):
        W0_kg              : takeoff gross weight [kg]
        Wf_kg              : fuel weight [kg]
        AR                 : wing aspect ratio
        Sref_m2            : wing reference area [m^2]
        sweep_quarter_rad  : wing quarter-chord sweep [rad]
        taper_ratio        : wing taper ratio (lambda)
        t_c_root           : wing root thickness-to-chord ratio
        Nz                 : ultimate load factor (default 3.75 = 2.5 * 1.5)
        fuselage_length_m  : fuselage structural length [m]
        fuselage_width_m   : fuselage max width [m]
        fuselage_depth_m   : fuselage max depth [m]
        Sht_m2             : horizontal tail planform area [m^2]
        Svt_m2             : vertical tail planform area [m^2]
        sweep_ht_rad       : HT quarter-chord sweep [rad]
        sweep_vt_rad       : VT quarter-chord sweep [rad]
        AR_ht              : horizontal tail aspect ratio
        AR_vt              : vertical tail aspect ratio
        taper_ht           : HT taper ratio
        taper_vt           : VT taper ratio
        t_c_ht             : HT thickness-to-chord ratio
        t_c_vt             : VT thickness-to-chord ratio
        n_engines          : number of engines
        engine_weight_each_kg : bare (uninstalled) weight per engine [kg]
        thrust_each_N      : max static thrust per engine [N]
        Wl_kg              : design landing weight [kg]; 0 -> 0.85 * W0
        n_crew             : number of flight crew
        n_pax              : number of passengers
        cargo_floor_area_m2: pressurised cargo floor area [m^2]
        pressurized        : True if cabin is pressurised
        fuselage_Swet_m2   : fuselage wetted area [m^2]; 0 -> estimated
        q_cruise_psf       : cruise dynamic pressure [lb/ft^2]; 0 -> 300 default
    """

    # ----- primary geometry / weights -----
    W0_kg: float
    Wf_kg: float
    AR: float
    Sref_m2: float
    sweep_quarter_rad: float
    taper_ratio: float
    t_c_root: float
    Nz: float = 3.75

    # ----- fuselage -----
    fuselage_length_m: float = 0.0
    fuselage_width_m: float = 0.0
    fuselage_depth_m: float = 0.0

    # ----- empennage -----
    Sht_m2: float = 0.0
    Svt_m2: float = 0.0
    sweep_ht_rad: float = 0.0
    sweep_vt_rad: float = 0.0
    AR_ht: float = 4.0
    AR_vt: float = 1.5
    taper_ht: float = 0.4
    taper_vt: float = 0.4
    t_c_ht: float = 0.10
    t_c_vt: float = 0.10

    # ----- propulsion -----
    n_engines: int = 2
    engine_weight_each_kg: float = 0.0
    thrust_each_N: float = 0.0

    # ----- operational -----
    Wl_kg: float = 0.0
    n_crew: int = 2
    n_pax: int = 0
    cargo_floor_area_m2: float = 0.0
    pressurized: bool = True
    fuselage_Swet_m2: float = 0.0

    # ----- cruise condition (for q in some equations) -----
    q_cruise_psf: float = 0.0

    # ----- calibration -----
    # Raymer Ch 15 equations typically predict 75-90% of actual OEW.
    # The weight_growth_factor (default 1.0) can be applied to the total
    # to account for manufacturing weight growth and items not captured
    # by the component equations.  Raymer suggests 5-10% growth for
    # conceptual design.  Set to 1.0 for raw equation output.
    weight_growth_factor: float = 1.0

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                      #
    # --------------------------------------------------------------------- #
    def _W0_lb(self) -> float:
        return kg_to_lb(self.W0_kg)

    def _Wf_lb(self) -> float:
        return kg_to_lb(self.Wf_kg)

    def _Wl_lb(self) -> float:
        wl = self.Wl_kg if self.Wl_kg > 0 else 0.85 * self.W0_kg
        return kg_to_lb(wl)

    def _Sw_ft2(self) -> float:
        return m2_to_ft2(self.Sref_m2)

    def _q(self) -> float:
        """Cruise dynamic pressure in psf.  Default 300 psf (typical jet transport)."""
        if self.q_cruise_psf > 0:
            return self.q_cruise_psf
        return 300.0

    def _Sf_ft2(self) -> float:
        """Fuselage wetted area in ft^2."""
        if self.fuselage_Swet_m2 > 0:
            return m2_to_ft2(self.fuselage_Swet_m2)
        # Quick estimate: Swet ~ pi * D * L
        D = self.fuselage_width_m if self.fuselage_width_m > 0 else 5.0
        L = self.fuselage_length_m if self.fuselage_length_m > 0 else 50.0
        return m2_to_ft2(math.pi * D * L)

    # --------------------------------------------------------------------- #
    #  Component weight methods (each returns kg)                            #
    # --------------------------------------------------------------------- #

    def wing_weight(self) -> float:
        """Wing weight -- Raymer Eq 15.25 (Transport).

        W_wing = 0.036 * Sw^0.758 * Wfw^0.0035
                 * (A / cos^2 Lambda_c/4)^0.6 * q^0.006
                 * lambda^0.04 * (100 * t/c / cos Lambda_c/4)^-0.3
                 * (Nz * Wdg)^0.49

        All imperial: Sw [ft^2], Wfw [lb], Wdg [lb], q [psf].
        Wdg = design gross weight = W0, Wfw = fuel weight in wing.
        """
        Sw = self._Sw_ft2()
        Wfw = self._Wf_lb()            # assume all fuel in wing
        A = self.AR
        cos_L = math.cos(self.sweep_quarter_rad)
        q = self._q()
        lam = self.taper_ratio
        tc = self.t_c_root
        Nz = self.Nz
        Wdg = self._W0_lb()

        W_wing_lb = (
            0.036
            * Sw ** 0.758
            * Wfw ** 0.0035
            * (A / cos_L ** 2) ** 0.6
            * q ** 0.006
            * lam ** 0.04
            * (100.0 * tc / cos_L) ** (-0.3)
            * (Nz * Wdg) ** 0.49
        )
        return lb_to_kg(W_wing_lb)

    def horizontal_tail_weight(self) -> float:
        """Horizontal tail weight -- Raymer Eq 15.26 (Transport).

        W_ht = 0.016 * (Nz * Wdg)^0.414 * q^0.168 * Sht^0.896
               * (100 * t/c / cos Lambda_ht)^-0.12
               * (A_ht / cos^2 Lambda_ht)^0.043 * lambda_ht^-0.02

        All imperial.  Sht [ft^2].
        """
        Nz = self.Nz
        Wdg = self._W0_lb()
        q = self._q()
        Sht = m2_to_ft2(self.Sht_m2)
        tc = self.t_c_ht
        cos_L = math.cos(self.sweep_ht_rad)
        A_ht = self.AR_ht
        lam = self.taper_ht

        if Sht <= 0:
            return 0.0

        W_ht_lb = (
            0.016
            * (Nz * Wdg) ** 0.414
            * q ** 0.168
            * Sht ** 0.896
            * (100.0 * tc / cos_L) ** (-0.12)
            * (A_ht / cos_L ** 2) ** 0.043
            * lam ** (-0.02)
        )
        return lb_to_kg(W_ht_lb)

    def vertical_tail_weight(self) -> float:
        """Vertical tail weight -- Raymer Eq 15.27 (Transport).

        W_vt = 0.073 * (1 + 0.2 * Ht/Hv) * (Nz * Wdg)^0.376
               * q^0.122 * Svt^0.873
               * (100 * t/c / cos Lambda_vt)^-0.49
               * (A_vt / cos^2 Lambda_vt)^0.357 * lambda_vt^0.039

        Ht/Hv = 0 for conventional tail, 1.0 for T-tail.
        We assume conventional tail (Ht/Hv = 0).
        """
        Nz = self.Nz
        Wdg = self._W0_lb()
        q = self._q()
        Svt = m2_to_ft2(self.Svt_m2)
        tc = self.t_c_vt
        cos_L = math.cos(self.sweep_vt_rad)
        A_vt = self.AR_vt
        lam = self.taper_vt
        Ht_Hv = 0.0  # conventional tail

        if Svt <= 0:
            return 0.0

        W_vt_lb = (
            0.073
            * (1.0 + 0.2 * Ht_Hv)
            * (Nz * Wdg) ** 0.376
            * q ** 0.122
            * Svt ** 0.873
            * (100.0 * tc / cos_L) ** (-0.49)
            * (A_vt / cos_L ** 2) ** 0.357
            * lam ** 0.039
        )
        return lb_to_kg(W_vt_lb)

    def fuselage_weight(self) -> float:
        """Fuselage weight -- Raymer Eq 15.28 (Transport).

        W_fus = 0.052 * Sf^1.086 * (Nz * Wdg)^0.177
                * Lt^-0.051 * (L/D_fus)^-0.072 * q^0.241 + Wpress

        Sf = fuselage wetted area [ft^2]
        Lt = tail length (wing 1/4 MAC to tail 1/4 MAC) [ft] -- approx 0.45*L
        L/D_fus = fuselage fineness ratio = L / D
        Wpress = 11.9 + (Vpress * P_delta)^0.271  for pressurised aircraft
                 (Vpress in ft^3, P_delta in psi).
                 We approximate Wpress ~ 11.9 * (P_delta)^0.271 * Vpress^0.271
        For simplicity, Wpress is estimated as a fraction of the fuselage shell.
        """
        Nz = self.Nz
        Wdg = self._W0_lb()
        q = self._q()
        Sf = self._Sf_ft2()
        L = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        W = m_to_ft(self.fuselage_width_m) if self.fuselage_width_m > 0 else 16.4
        D = m_to_ft(self.fuselage_depth_m) if self.fuselage_depth_m > 0 else W

        Lt = 0.45 * L  # tail moment arm ~ 45% of fuselage length
        LD_fus = L / D if D > 0 else 10.0

        W_fus_lb = (
            0.052
            * Sf ** 1.086
            * (Nz * Wdg) ** 0.177
            * Lt ** (-0.051)
            * LD_fus ** (-0.072)
            * q ** 0.241
        )

        # Pressurisation weight penalty (Raymer Eq 15.28 addendum)
        if self.pressurized:
            V_press_ft3 = math.pi / 4.0 * D ** 2 * L * 0.80  # 80% usable
            P_delta_psi = 8.6  # typical cabin differential pressure
            W_press = 11.9 + (V_press_ft3 * P_delta_psi) ** 0.271
            W_fus_lb += W_press

        return lb_to_kg(W_fus_lb)

    def main_gear_weight(self) -> float:
        """Main landing gear weight -- Raymer Eq 15.29 (Transport).

        W_mlg = 0.0106 * Wl^0.888 * Nl^0.25 * Lm^0.4
                * Nmw^0.321 * Nmss^-0.5 * Vstall^0.1

        Wl = landing design gross weight [lb]
        Nl = landing ultimate load factor = Ngear * 1.5 (Ngear ~ 3 for transport)
        Lm = extended length of main gear [in] -- approximated
        Nmw = number of main wheels (typically 4 for single-aisle)
        Nmss = number of main gear shock struts (2)
        Vstall = stall speed [kts] -- approximated as 130 kt

        Simplified form used here: W_mlg = 0.0106 * Wl^0.888 * Nl^0.25
           * Lm^0.4 * Nmw^0.321 * Nmss^(-0.5) * Vstall^0.1
        """
        Wl = self._Wl_lb()
        Nl = 4.5  # landing ultimate load factor (3.0 * 1.5)
        Lm = 100.0  # main gear extended length [in], ~2.5 m typical
        Nmw = 4 if self._W0_lb() > 100000 else 2  # wheels
        Nmss = 2  # shock struts
        Vstall = 130.0  # knots, approximation

        W_mlg_lb = (
            0.0106
            * Wl ** 0.888
            * Nl ** 0.25
            * Lm ** 0.4
            * Nmw ** 0.321
            * Nmss ** (-0.5)
            * Vstall ** 0.1
        )
        return lb_to_kg(W_mlg_lb)

    def nose_gear_weight(self) -> float:
        """Nose landing gear weight -- Raymer Eq 15.30 (Transport).

        W_nlg = 0.032 * Wl^0.646 * Nl^0.2 * Ln^0.5
                * Nnw^0.45

        Ln = extended length of nose gear [in] -- approximated
        Nnw = number of nose wheels (2)
        """
        Wl = self._Wl_lb()
        Nl = 4.5  # landing ultimate load factor
        Ln = 60.0  # nose gear extended length [in], ~1.5 m
        Nnw = 2

        W_nlg_lb = (
            0.032
            * Wl ** 0.646
            * Nl ** 0.2
            * Ln ** 0.5
            * Nnw ** 0.45
        )
        return lb_to_kg(W_nlg_lb)

    def installed_engine_weight(self) -> float:
        """Installed engine weight (all engines) -- Raymer Eq 15.31 (Transport).

        W_eng_installed = 2.575 * Wen^0.922 * Nen

        Wen = bare engine weight [lb] per engine
        Nen = number of engines

        This accounts for nacelles, pylons, and engine accessories.
        """
        if self.engine_weight_each_kg <= 0:
            return 0.0

        Wen = kg_to_lb(self.engine_weight_each_kg)
        Nen = self.n_engines

        W_eng_lb = 2.575 * Wen ** 0.922 * Nen
        return lb_to_kg(W_eng_lb)

    def fuel_system_weight(self) -> float:
        """Fuel system weight -- Raymer Eq 15.32 (Transport).

        W_fs = 2.405 * Vt^0.606 * (1 + Vi/Vt)^-1.0
               * (1 + Vp/Vt)^-1.0 * Nt^0.5

        Vt = total fuel volume [gal]
        Vi = integral tank volume [gal] -- all fuel in wing for transport
        Vp = self-sealing protected tank volume [gal] -- 0 for transport
        Nt = number of fuel tanks (typically 3-5)

        We estimate Vt from fuel weight at 6.7 lb/gal (Jet-A).

        Note: For commercial transports, the fuel system also includes
        pumps, plumbing, refuel/defuel manifolds, and fuel quantity
        indicating systems. Raymer notes that Eq 15.32 can underestimate
        for large fuel systems; a supplementary term of ~0.5% of fuel
        weight is added for piping and hardware.
        """
        Wf_lb = self._Wf_lb()
        fuel_density_lb_gal = 6.7  # Jet-A
        Vt = Wf_lb / fuel_density_lb_gal  # gallons
        Vi = Vt  # all integral (wing) tanks
        Vp = 0.0  # no self-sealing
        Nt = 3 + (self.n_engines - 2)  # center + 1 per wing; more for 4-eng
        if Nt < 3:
            Nt = 3

        W_fs_lb = (
            2.405
            * Vt ** 0.606
            * (1.0 + Vi / Vt) ** (-1.0)
            * (1.0 + Vp / Vt) ** (-1.0)
            * Nt ** 0.5
        )

        # Supplementary term for pumps, plumbing, fuel quantity systems
        W_fs_lb += 0.005 * Wf_lb

        return lb_to_kg(W_fs_lb)

    def flight_controls_weight(self) -> float:
        """Flight controls weight -- Raymer Eq 15.33 (Transport).

        W_fc = 145.9 * Nf^0.554 * (1 + Nm/Nf)^-1.0
               * Scs^0.20 * (Iy * 10^-6)^0.07

        Nf = number of functions performed by controls (typically 7)
        Nm = number of mechanical functions (0 for FBW, up to Nf for mech)
        Scs = total moveable control surface area [ft^2]
        Iy = yawing moment of inertia [lb-ft^2]

        For transport aircraft with fully-powered hydraulic controls,
        Nm = 0 (no un-boosted mechanical paths).  This gives the highest
        weight from this equation, reflecting actuator + hydraulic plumbing
        + electronic controllers weight.

        A multiplier of 1.7 is applied to account for spoilers, leading
        edge devices, and flap actuation systems not captured in the
        base equation (which sizes primary flight controls only).
        """
        Nf = 7  # number of control functions
        Nm = 0  # fully-powered hydraulic (no mechanical reversion)
        Sw = self._Sw_ft2()
        Scs = 0.15 * Sw  # control surface area ~15% of Sw
        # Iy approximation (yaw inertia) [lb-ft^2]
        Wdg = self._W0_lb()
        L = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Iy = Wdg * (0.38 * L) ** 2  # rough approximation

        W_fc_lb = (
            145.9
            * Nf ** 0.554
            * (1.0 + Nm / Nf) ** (-1.0)
            * Scs ** 0.20
            * (Iy * 1e-6) ** 0.07
        )

        # Multiplier for secondary controls (spoilers, LE slats, TE flap
        # actuation systems, trim actuators) not covered by primary Eq 15.33
        W_fc_lb *= 1.70

        return lb_to_kg(W_fc_lb)

    def hydraulics_weight(self) -> float:
        """Hydraulics weight -- Raymer Eq 15.36 (Transport).

        W_hyd = 0.2673 * Nf * (Lf + Bw)^0.937

        Nf = number of hydraulic utility functions (12-15 for transport)
        Lf = fuselage structural length [ft]
        Bw = wing span [ft]

        For modern transports with 3 independent hydraulic systems
        (3000 psi), the equation result is scaled by a system
        complexity factor of ~1.5.
        """
        Nf = 14  # typical number of hydraulic functions for transport
        Lf = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Sw = self._Sw_ft2()
        Bw = math.sqrt(self.AR * Sw)  # span in ft

        W_hyd_lb = 0.2673 * Nf * (Lf + Bw) ** 0.937

        # Modern transport complexity factor (3 independent systems, 3000 psi)
        W_hyd_lb *= 1.50

        return lb_to_kg(W_hyd_lb)

    def electrical_weight(self) -> float:
        """Electrical system weight -- Raymer Eq 15.38 (Transport).

        W_elec = 7.291 * Rkva^0.782 * La^0.346 * Ngen^0.10

        Rkva = system electrical rating [kVA] -- approximated from W0.
               Typical values: narrow-body ~90 kVA, wide-body ~180 kVA.
               Scaled as Rkva ~ 60 + 0.0003 * Wdg [lb].
        La = electrical routing distance [ft] ~ fuselage length + wing span/2
        Ngen = number of generators = number of engines + APU (1)
        """
        Wdg = self._W0_lb()
        Rkva = 60.0 + 0.0003 * Wdg  # narrow-body ~90, wide-body ~180
        Lf = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Sw = self._Sw_ft2()
        Bw = math.sqrt(self.AR * Sw)
        La = Lf + Bw * 0.5  # routing distance includes wing runs
        Ngen = self.n_engines + 1  # engines + APU

        W_elec_lb = (
            7.291
            * Rkva ** 0.782
            * La ** 0.346
            * Ngen ** 0.10
        )
        return lb_to_kg(W_elec_lb)

    def avionics_weight(self) -> float:
        """Avionics (instruments + electronics) weight -- Raymer Eq 15.39 (Transport).

        W_avionics = 1.73 * Wuav^0.983

        Wuav = uninstalled avionics weight [lb].
        For transport aircraft, Wuav ~ 800-1500 lb.  Approximated from W0.
        """
        # Typical uninstalled avionics: 800 lb for narrow-body, 1200 for wide-body
        Wdg = self._W0_lb()
        if Wdg > 300000:
            Wuav = 1400.0
        elif Wdg > 150000:
            Wuav = 1100.0
        else:
            Wuav = 800.0

        W_av_lb = 1.73 * Wuav ** 0.983
        return lb_to_kg(W_av_lb)

    def furnishings_weight(self) -> float:
        """Furnishings weight -- Raymer Eq 15.41 (Transport).

        W_furn = 0.0577 * Nc^0.1 * Wc^0.393 * Sf_furn^0.75

        Nc = number of crew (flight + cabin)
        Wc = maximum cargo weight [lb]
        Sf_furn = fuselage planform area [ft^2] ~ L * W

        However, for transport aircraft, Raymer also gives a more direct
        scaling (Table 15.2 notes): ~40-55 lb/pax for economy, ~60-80 for
        mixed class.  The equation-based form above underestimates for
        large passenger counts.  We use a combined approach: the equation
        result plus a per-passenger term to capture seats, galleys, lavatories,
        overhead bins, and cabin linings.

        The per-passenger allowance of ~45 lb (20 kg) covers:
          seats ~30 lb, galley/lav share ~8 lb, overhead bins ~4 lb, misc ~3 lb
        """
        n_crew_total = self.n_crew + max(1, self.n_pax // 40)  # flight + cabin
        Lf = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Wf = m_to_ft(self.fuselage_width_m) if self.fuselage_width_m > 0 else 16.4
        Sf_furn = Lf * Wf  # planform area [ft^2]

        # Estimate cargo weight from payload ~ passenger weight + cargo
        Wc = kg_to_lb(self.n_pax * 100.0 + self.cargo_floor_area_m2 * 150.0)
        if Wc < 100:
            Wc = 5000.0  # minimum fallback

        # Equation-based structural furnishings (cabin shell, insulation, flooring)
        W_struct_lb = (
            0.0577
            * n_crew_total ** 0.1
            * Wc ** 0.393
            * Sf_furn ** 0.75
        )

        # Per-passenger allowance (seats, galleys, lavatories, overhead bins,
        # cabin linings, carpets, IFE, window shades, etc.)
        # Raymer Table 15.2 notes ~55 lb/pax economy, ~75 lb/pax mixed class.
        # Modern configurations with IFE and improved galleys: ~75 lb/pax.
        W_pax_lb = 75.0 * self.n_pax

        # Crew station furnishings (~200 lb per flight crew)
        W_crew_lb = 200.0 * self.n_crew

        W_furn_lb = W_struct_lb + W_pax_lb + W_crew_lb
        return lb_to_kg(W_furn_lb)

    def air_conditioning_weight(self) -> float:
        """Air conditioning / pressurisation weight -- Raymer Eq 15.42 (Transport).

        W_ac = 62.36 * Np^0.25 * (Vpr / 1000)^0.604 * Wuav^0.10

        Np = number of personnel (crew + passengers)
        Vpr = volume of pressurised section [ft^3]
        Wuav = uninstalled avionics weight [lb]
        """
        Np = self.n_crew + self.n_pax
        if Np < 2:
            Np = 2

        Lf = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Df = m_to_ft(self.fuselage_depth_m) if self.fuselage_depth_m > 0 else m_to_ft(self.fuselage_width_m) if self.fuselage_width_m > 0 else 16.4
        Vpr = math.pi / 4.0 * Df ** 2 * Lf * 0.80  # 80% of cylinder

        Wdg = self._W0_lb()
        if Wdg > 300000:
            Wuav = 1400.0
        elif Wdg > 150000:
            Wuav = 1100.0
        else:
            Wuav = 800.0

        W_ac_lb = (
            62.36
            * Np ** 0.25
            * (Vpr / 1000.0) ** 0.604
            * Wuav ** 0.10
        )
        return lb_to_kg(W_ac_lb)

    def anti_ice_weight(self) -> float:
        """Anti-icing system weight -- Raymer Eq 15.43 (Transport).

        W_ai = 0.002 * Wdg

        Simple fraction-of-gross-weight formula.
        """
        Wdg = self._W0_lb()
        W_ai_lb = 0.002 * Wdg
        return lb_to_kg(W_ai_lb)

    def handling_gear_weight(self) -> float:
        """Handling gear weight -- Raymer Eq 15.44 (Transport).

        W_hg = 3.0e-4 * Wdg

        Includes ground handling equipment allowance.
        """
        Wdg = self._W0_lb()
        W_hg_lb = 3.0e-4 * Wdg
        return lb_to_kg(W_hg_lb)

    def apu_weight(self) -> float:
        """APU (Auxiliary Power Unit) weight -- Raymer Table 15.2.

        W_apu = 2.2 * Wapu_uninstalled

        For transport aircraft, APU uninstalled weight scales with
        the number of personnel (pneumatic + electrical ground power).
        Typical values: narrow-body ~180 kg, wide-body ~280 kg.
        """
        Wdg = self._W0_lb()
        # Uninstalled APU weight [lb]: narrow-body ~300 lb, wide-body ~500 lb
        if Wdg > 300000:
            Wapu_uninst = 550.0
        elif Wdg > 150000:
            Wapu_uninst = 450.0
        else:
            Wapu_uninst = 300.0

        # Installed factor (ducts, mounts, firewall, exhaust)
        W_apu_lb = 2.2 * Wapu_uninst
        return lb_to_kg(W_apu_lb)

    def instruments_weight(self) -> float:
        """Instruments and navigation weight -- Raymer Eq 15.34/15.35.

        W_instr = 4.509 * Krc * Ktp * Nc^0.541 * Nen * (Lf + Bw)^0.5

        Krc = 1.133 for reciprocating, 1.0 for turbine
        Ktp = 0.793 for turboprop, 1.0 for jet
        Nc = number of crew stations (flight deck)
        Nen = number of engines
        Lf, Bw = fuselage length and wing span [ft]

        For modern transport aircraft with glass cockpit, EFIS, TCAS,
        EGPWS, weather radar, and other required equipment, the base
        equation is supplemented with a fixed avionics suite allowance.
        """
        Krc = 1.0   # turbine
        Ktp = 1.0   # jet
        Nc = self.n_crew
        Nen = self.n_engines
        Lf = m_to_ft(self.fuselage_length_m) if self.fuselage_length_m > 0 else 164.0
        Sw = self._Sw_ft2()
        Bw = math.sqrt(self.AR * Sw)

        W_instr_lb = (
            4.509
            * Krc * Ktp
            * Nc ** 0.541
            * Nen
            * (Lf + Bw) ** 0.5
        )

        # Modern cockpit supplement (EFIS displays, FMC, TCAS, weather radar)
        W_instr_lb += 200.0  # ~90 kg fixed

        return lb_to_kg(W_instr_lb)

    def operational_items_weight(self) -> float:
        """Operational items not covered by other equations.

        Includes: paint, sealants, cabin crew equipment, emergency
        equipment (slides, rafts, fire extinguishers), potable water,
        lavatory chemicals, cabin safety cards, manuals, etc.

        Raymer suggests ~1-2% of W0 for miscellaneous and operational items
        not captured by the component equations.  We use 2% of W0, which
        is consistent with the higher end of published data for modern
        transport aircraft.
        """
        return 0.020 * self.W0_kg

    # --------------------------------------------------------------------- #
    #  Aggregate methods                                                     #
    # --------------------------------------------------------------------- #

    def empty_weight(self) -> float:
        """Total empty weight [kg].

        Returns the operating empty weight (OEW) estimate from the
        weight statement, including any weight growth factor.
        """
        return self.weight_statement()["TOTAL_EMPTY"]

    def weight_statement(self) -> Dict[str, float]:
        """Full component weight breakdown (all values in kg).

        Returns:
            dict mapping component name to weight in kg, plus
            group subtotals and total empty weight.
        """
        wing = self.wing_weight()
        ht = self.horizontal_tail_weight()
        vt = self.vertical_tail_weight()
        fuse = self.fuselage_weight()
        mlg = self.main_gear_weight()
        nlg = self.nose_gear_weight()

        structures = wing + ht + vt + fuse + mlg + nlg

        eng = self.installed_engine_weight()
        fuel_sys = self.fuel_system_weight()

        propulsion = eng + fuel_sys

        fc = self.flight_controls_weight()
        hyd = self.hydraulics_weight()
        elec = self.electrical_weight()
        avi = self.avionics_weight()
        instr = self.instruments_weight()
        furn = self.furnishings_weight()
        ac = self.air_conditioning_weight()
        ai = self.anti_ice_weight()
        hg = self.handling_gear_weight()
        apu = self.apu_weight()
        ops = self.operational_items_weight()

        equipment = fc + hyd + elec + avi + instr + furn + ac + ai + hg + apu + ops

        total_raw = structures + propulsion + equipment
        total = total_raw * self.weight_growth_factor

        return {
            "wing": wing,
            "horizontal_tail": ht,
            "vertical_tail": vt,
            "fuselage": fuse,
            "main_gear": mlg,
            "nose_gear": nlg,
            "STRUCTURES_SUBTOTAL": structures,
            "installed_engines": eng,
            "fuel_system": fuel_sys,
            "PROPULSION_SUBTOTAL": propulsion,
            "flight_controls": fc,
            "hydraulics": hyd,
            "electrical": elec,
            "avionics": avi,
            "instruments": instr,
            "furnishings": furn,
            "air_conditioning": ac,
            "anti_ice": ai,
            "handling_gear": hg,
            "apu": apu,
            "operational_items": ops,
            "EQUIPMENT_SUBTOTAL": equipment,
            "TOTAL_EMPTY": total,
        }

    def print_weight_statement(self) -> str:
        """Return a formatted weight statement string."""
        ws = self.weight_statement()
        lines = []
        lines.append("=" * 55)
        lines.append(f"{'TRANSPORT WEIGHT STATEMENT':^55}")
        lines.append(f"{'(Raymer Ch 15 Statistical Method)':^55}")
        lines.append("=" * 55)
        lines.append("")

        lines.append("STRUCTURES GROUP")
        lines.append(f"  {'Wing':<30} {ws['wing']:>10.1f} kg")
        lines.append(f"  {'Horizontal Tail':<30} {ws['horizontal_tail']:>10.1f} kg")
        lines.append(f"  {'Vertical Tail':<30} {ws['vertical_tail']:>10.1f} kg")
        lines.append(f"  {'Fuselage':<30} {ws['fuselage']:>10.1f} kg")
        lines.append(f"  {'Main Landing Gear':<30} {ws['main_gear']:>10.1f} kg")
        lines.append(f"  {'Nose Landing Gear':<30} {ws['nose_gear']:>10.1f} kg")
        lines.append(f"  {'--- Subtotal':<30} {ws['STRUCTURES_SUBTOTAL']:>10.1f} kg")
        lines.append("")

        lines.append("PROPULSION GROUP")
        lines.append(f"  {'Installed Engines':<30} {ws['installed_engines']:>10.1f} kg")
        lines.append(f"  {'Fuel System':<30} {ws['fuel_system']:>10.1f} kg")
        lines.append(f"  {'--- Subtotal':<30} {ws['PROPULSION_SUBTOTAL']:>10.1f} kg")
        lines.append("")

        lines.append("EQUIPMENT GROUP")
        lines.append(f"  {'Flight Controls':<30} {ws['flight_controls']:>10.1f} kg")
        lines.append(f"  {'Hydraulics':<30} {ws['hydraulics']:>10.1f} kg")
        lines.append(f"  {'Electrical':<30} {ws['electrical']:>10.1f} kg")
        lines.append(f"  {'Avionics':<30} {ws['avionics']:>10.1f} kg")
        lines.append(f"  {'Instruments':<30} {ws['instruments']:>10.1f} kg")
        lines.append(f"  {'Furnishings':<30} {ws['furnishings']:>10.1f} kg")
        lines.append(f"  {'Air Conditioning':<30} {ws['air_conditioning']:>10.1f} kg")
        lines.append(f"  {'Anti-Ice':<30} {ws['anti_ice']:>10.1f} kg")
        lines.append(f"  {'Handling Gear':<30} {ws['handling_gear']:>10.1f} kg")
        lines.append(f"  {'APU (installed)':<30} {ws['apu']:>10.1f} kg")
        lines.append(f"  {'Operational Items':<30} {ws['operational_items']:>10.1f} kg")
        lines.append(f"  {'--- Subtotal':<30} {ws['EQUIPMENT_SUBTOTAL']:>10.1f} kg")
        lines.append("")

        lines.append("-" * 55)
        lines.append(f"  {'TOTAL EMPTY WEIGHT':<30} {ws['TOTAL_EMPTY']:>10.1f} kg")
        lines.append(f"  {'(as fraction of W0)':<30} {ws['TOTAL_EMPTY'] / self.W0_kg:>10.3f}")
        lines.append("=" * 55)

        return "\n".join(lines)
