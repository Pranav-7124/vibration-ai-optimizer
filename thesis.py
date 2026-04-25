"""
thesis.py — Physics-based vibration dataset generator
Models a cantilever beam with an impact damper using forced vibration equations.

Key thesis insights encoded:
- Optimal mass ratio ≈ 0.04 (4% of primary mass)
- Optimal clearance ≈ 0.5 mm (normalized)
- Optimal location ≈ 0.6 L (60% along beam length)
- Near-resonance operation (β ≈ 1) maximizes damper effectiveness
"""

import pandas as pd
import numpy as np

np.random.seed(42)

N_SAMPLES = 5000

# ─── System Parameters ────────────────────────────────────────────────────────
FN = 50.0        # Natural frequency of primary beam (Hz) — thesis baseline
ZETA_0 = 0.01   # Structural (inherent) damping ratio of beam
F0_K = 1.0      # Static deflection = F0/k (normalized)

# ─── Physics Model ────────────────────────────────────────────────────────────

def impact_damper_zeta(mass_ratio, clearance, location):
    """
    Compute effective additional damping ratio contributed by the impact damper.

    Based on energy dissipation principles from impact damper literature:
    - Mass ratio effectiveness peaks at ~0.04 (Gaussian bell)
    - Clearance has an optimal around 0.5 (below → no impact, above → slip-through)
    - Location peaks at antinode near 0.6L (mode shape weighting)
    """
    # Mass ratio contribution — peaks at μ = 0.04
    mu_eff = np.exp(-((mass_ratio - 0.04) ** 2) / (2 * 0.015 ** 2))

    # Clearance contribution — peaks at δ = 0.5, vanishes at extremes
    cl_eff = np.exp(-((clearance - 0.5) ** 2) / (2 * 0.22 ** 2))

    # Location contribution — peaks near 0.6L (first-mode antinode region)
    loc_eff = np.sin(np.pi * location) * np.exp(-((location - 0.60) ** 2) / (2 * 0.12 ** 2))
    loc_eff = np.clip(loc_eff, 0, None)

    # Combined effective damping from impact damper
    zeta_impact = 0.20 * mu_eff * cl_eff * loc_eff
    return zeta_impact


def forced_vibration_amplitude(frequency, mass_ratio, clearance, location):
    """
    Compute steady-state vibration amplitude using forced vibration theory.

    X = (F0/k) / sqrt((1 - β²)² + (2 ζ_total β)²)

    where β = f / fn  (frequency ratio)
          ζ_total = ζ_structural + ζ_impact_damper
    """
    beta = frequency / FN                    # Frequency ratio
    zeta_impact = impact_damper_zeta(mass_ratio, clearance, location)
    zeta_total = ZETA_0 + zeta_impact        # Total effective damping

    denom = np.sqrt((1 - beta ** 2) ** 2 + (2 * zeta_total * beta) ** 2)
    amplitude = F0_K / denom
    return amplitude, zeta_total, zeta_impact


# ─── Dataset Generation ───────────────────────────────────────────────────────

data = []

for _ in range(N_SAMPLES):
    # Sample parameters
    frequency   = np.random.uniform(35, 70)       # Hz — sweep around resonance
    mass_ratio  = np.random.uniform(0.005, 0.10)  # 0.5% – 10% of primary mass
    clearance   = np.random.uniform(0.10, 1.20)   # normalized clearance
    location    = np.random.uniform(0.25, 0.80)   # fraction of beam length

    # Compute physics-based amplitude
    amplitude, zeta_total, zeta_impact = forced_vibration_amplitude(
        frequency, mass_ratio, clearance, location
    )

    # Add small realistic measurement noise (1% std dev)
    noise = np.random.normal(0, 0.01 * amplitude)
    amplitude = abs(amplitude + noise)

    # Derived features (useful for model)
    beta = frequency / FN
    damping_ratio = zeta_total

    data.append([
        frequency, mass_ratio, clearance, location,
        beta, damping_ratio, amplitude
    ])

# ─── Save Dataset ─────────────────────────────────────────────────────────────

df = pd.DataFrame(data, columns=[
    "frequency", "mass_ratio", "clearance", "location",
    "freq_ratio", "damping_ratio", "amplitude"
])

df.to_csv("vibration_data.csv", index=False)

print(f"[OK] Dataset created: {len(df)} samples")
print(f"   Amplitude range: {df['amplitude'].min():.4f} - {df['amplitude'].max():.4f}")
print(f"   Min amplitude row:\n{df.loc[df['amplitude'].idxmin()]}")