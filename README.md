# AeroSizer

**Aircraft Conceptual Design & Sizing Tool**

A web-based tool for aerospace engineers and students to design, size, and analyze aircraft using Daniel Raymer's *"Aircraft Design: A Conceptual Approach"* (6th Edition, 2018) as the computational backbone.

Every equation is traceable to its Raymer chapter and equation number. Built as both a professional sizing tool and a learning platform.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-571%20passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Quick Start

```bash
pip install fastapi uvicorn numpy scipy
cd Aircraft_Design
python -m uvicorn app:app --port 8080
# Open http://localhost:8080
```

Select a preset (B787-8, B737-800, A320neo) and hit **SIZE AIRCRAFT**.

---

## What It Does

AeroSizer walks you through the full Raymer conceptual design pipeline:

| Module | Raymer Chapter | Description |
|--------|---------------|-------------|
| **Initial Sizing** | Ch 3, 6 | Iterative W0 estimation via Breguet fuel fractions |
| **Constraint Diagram** | Ch 5 | T/W vs W/S with cruise, takeoff, landing, climb, ceiling constraints |
| **Aerodynamics** | Ch 12 | Component buildup drag, Oswald efficiency, drag polar |
| **Weights** | Ch 15 | 20-component statistical weight equations for transports |
| **Performance** | Ch 17 | Level flight, climb, range, takeoff/landing, turn, Ps plots |
| **Cost** | Ch 18 | DAPCA IV acquisition cost + airline operating economics |
| **Optimization** | Ch 19 | Carpet plots, parametric trade studies, growth sensitivity |

### Key Features

- **9-tab interactive UI** with dark coding theme and Plotly charts
- **Dynamic 3-view aircraft SVG** that updates as you change geometry
- **Aircraft presets**: B787-8 Dreamliner, B737-800, A320neo
- **Equation tooltips**: Every result references its Raymer equation number
- **Carpet plot optimizer**: Parametric T/W x W/S sweep to find minimum-weight design
- **Trade studies**: AR, sweep, range, payload, SFC sensitivity analysis
- **Cost estimation**: DAPCA IV model + DOC/seat-mile + CO2 per seat-km
- **Report generation**: Printable summary with all results

---

## Validation

Validated against the B787-8 Dreamliner (data from Mukhopadhyay, Micky & Pant, AIDAA 2017):

| Parameter | AeroSizer | Actual B787-8 | Accuracy |
|-----------|-----------|---------------|----------|
| MTOW | 221,219 kg | 227,930 kg | 97% |
| Wingspan (from sizing) | 60.16 m | 60.17 m | 99.99% |
| Cruise L/D | 18.2 | ~18-19 | Within range |

---

## Architecture

```
Aircraft_Design/
├── app.py                          # FastAPI web server (17 endpoints)
├── core/
│   ├── atmosphere.py               # ISA standard atmosphere + unit conversions
│   ├── mission.py                  # Mission profile definition & segment types
│   ├── sizing/
│   │   ├── initial_sizing.py       # W0 iteration (Breguet method)
│   │   └── tw_ws_selection.py      # Constraint diagram analysis
│   ├── aerodynamics/
│   │   ├── parasite_drag.py        # Component buildup (Cf, FF, Q, Swet)
│   │   ├── induced_drag.py         # Oswald efficiency, K factor
│   │   └── drag_polar.py           # CD = CD0 + K*CL^2
│   ├── weights/
│   │   └── statistical_weights.py  # Transport weight equations (Eqs 15.25-15.63)
│   ├── performance/
│   │   ├── level_flight.py         # Stall, max speed, Tr curves
│   │   ├── climb.py                # ROC, ceiling, time-to-climb
│   │   ├── range_endurance.py      # Breguet, cruise-climb, range-payload
│   │   ├── takeoff.py              # Ground roll, BFL, landing
│   │   └── turn.py                 # Turn rate, Ps, energy-maneuverability
│   ├── cost/
│   │   ├── dapca_iv.py             # Modified DAPCA IV (Eqs 18.1-18.9)
│   │   └── operating_cost.py       # DOC, crew, fuel, maintenance
│   └── optimization/
│       ├── carpet_plot.py          # Sizing matrix & carpet plots
│       └── trade_studies.py        # Parametric sweeps & growth sensitivity
├── static/
│   └── index.html                  # Single-page app (Plotly + vanilla JS)
└── tests/                          # 571 tests
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/size` | POST | Run initial sizing |
| `/api/constraints` | POST | Constraint diagram analysis |
| `/api/drag_polar` | POST | Compute drag polar |
| `/api/weights` | POST | Detailed weight breakdown |
| `/api/performance/takeoff_landing` | POST | Takeoff & landing distances |
| `/api/performance/climb_profile` | POST | ROC vs altitude |
| `/api/performance/range_payload` | POST | Range-payload diagram |
| `/api/turn_performance` | POST | Turn rate envelope |
| `/api/ps_diagram` | POST | Specific excess power |
| `/api/cost/dapca` | POST | DAPCA IV acquisition cost |
| `/api/cost/operating` | POST | Operating cost & economics |
| `/api/geometry/aircraft_svg` | POST | 3-view SVG generation |
| `/api/geometry/from_sizing` | POST | Compute geometry from sizing |
| `/api/presets/{name}` | GET | Aircraft preset data |

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## References

1. Raymer, D.P., *Aircraft Design: A Conceptual Approach*, 6th Edition, AIAA, 2018.
2. Mukhopadhyay, Micky & Pant, *"Conceptual Sizing of Boeing 787-8 Dreamliner"*, AIDAA 2017.

---

## Tech Stack

- **Backend**: Python 3.11+ / FastAPI
- **Computation**: NumPy, SciPy
- **Charts**: Plotly.js
- **Frontend**: Vanilla JS, single HTML file
- **Font**: Share Tech Mono (aviation monospace)

---

## License

MIT
