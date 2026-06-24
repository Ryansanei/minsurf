"""Named presets for the CLI and demo.

Each entry maps a preset name to a dict of default parameters that are
passed to the solve pipeline.  Users can override any key via CLI flags.
"""

from __future__ import annotations

PRESETS: dict[str, dict] = {
    "catenoid": {
        "boundary": "two-rings",
        "separation": 1.0,
        "radius": 1.0,
        "n_theta": 48,
        "n_z": 32,
        "method": "implicit",
        "tau": 0.05,
        "max_iter": 2000,
        "tol": 1e-6,
        "export": "obj",
        "validate_exact": "catenoid",
        "description": "Catenoid between two coaxial rings. Validates against the exact solution.",
    },
    "two-rings": {
        "boundary": "two-rings",
        "separation": 1.0,
        "radius": 1.0,
        "n_theta": 48,
        "n_z": 32,
        "method": "implicit",
        "tau": 0.05,
        "max_iter": 2000,
        "tol": 1e-6,
        "export": "obj",
        "description": "Two-ring stability study; try --separation 1.6 to trigger collapse.",
    },
    "saddle-ring": {
        "boundary": "disk",
        "amplitude": 0.55,
        "k": 2,
        "n_theta": 48,
        "n_radial": 20,
        "radius": 1.0,
        "method": "implicit",
        "tau": 0.05,
        "max_iter": 2000,
        "tol": 1e-6,
        "export": "obj",
        "description": "Saddle-shaped minimal surface from z = A sin(2θ) boundary.",
    },
    "wavy-ring": {
        "boundary": "disk",
        "amplitude": 0.55,
        "k": 3,
        "n_theta": 48,
        "n_radial": 20,
        "radius": 1.0,
        "method": "implicit",
        "tau": 0.05,
        "max_iter": 2000,
        "tol": 1e-6,
        "export": "obj",
        "description": "Monkey-saddle from z = A sin(3θ) boundary.",
    },
    "enneper": {
        "boundary": "analytic",
        "n_u": 32,
        "n_v": 32,
        "r_max": 0.8,
        "export": "obj",
        "description": "Enneper surface (analytic; no flow needed).",
    },
    "helicoid": {
        "boundary": "analytic",
        "c": 1.0,
        "n_u": 24,
        "n_v": 48,
        "export": "obj",
        "description": "Helicoid (analytic; no flow needed).",
    },
}


def get_preset(name: str) -> dict:
    """Return a copy of the preset parameters for 'name'."""
    if name not in PRESETS:
        raise ValueError(
            f"Unknown preset {name!r}. Available: {', '.join(PRESETS)}"
        )
    return dict(PRESETS[name])
