"""Small, validated scientific calculations used by the virtual lab."""
from __future__ import annotations
import math

R_GAS = 8.314462618  # J mol-1 K-1
FARADAY = 96485.33212  # C mol-1


def water_pkw(temperature_c: float = 25.0) -> float:
    """Approximate pKw of liquid water from 0 to 100 °C.

    Correlation: pKw ≈ 4471/T + 0.01706T − 6.0875, T in kelvin.
    It is a useful educational approximation, not a substitute for a validated
    thermodynamic database outside ordinary liquid-water conditions.
    """
    if not math.isfinite(temperature_c) or not 0.0 <= temperature_c <= 100.0:
        raise ValueError("This water-ionization approximation is valid from 0 to 100 °C.")
    temperature_k = temperature_c + 273.15
    return 4471.0 / temperature_k + 0.01706 * temperature_k - 6.0875


def strong_acid_strong_base_ph(acid_molarity: float, acid_ml: float,
                               base_molarity: float, base_ml: float,
                               temperature_c: float = 25.0) -> float:
    """pH for an ideal strong monoprotic acid/strong base titration.

    Uses the full water autoionization balance rather than forcing pH=7 at
    equivalence or clipping tiny excess concentrations. Concentrations are
    treated as activities (ideal solution); activity coefficients, dilution
    non-ideality, and polyprotic/weak-acid equilibria are not included.
    """
    values = (acid_molarity, acid_ml, base_molarity, base_ml, temperature_c)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("All inputs must be finite numbers.")
    if acid_molarity <= 0 or base_molarity <= 0 or acid_ml <= 0 or base_ml < 0:
        raise ValueError("Acid/base concentrations and acid volume must be positive; base volume cannot be negative.")
    if not 0.0 <= temperature_c <= 100.0:
        raise ValueError("Temperature must be between 0 and 100 °C for this model.")

    total_l = (acid_ml + base_ml) / 1000.0
    if total_l <= 0 or not math.isfinite(total_l):
        raise ValueError("Total solution volume must be positive and finite.")

    # Moles are calculated in mol; concentration difference is mol/L.
    acid_mol = acid_molarity * acid_ml / 1000.0
    base_mol = base_molarity * base_ml / 1000.0
    excess_h = (acid_mol - base_mol) / total_l
    kw = 10.0 ** (-water_pkw(temperature_c))

    # Solve [H+] - [OH-] = excess_h and [H+][OH-] = Kw.
    # This quadratic remains stable at equivalence and at very small excess.
    h_concentration = 0.5 * (excess_h + math.hypot(excess_h, 2.0 * math.sqrt(kw)))
    # For strongly basic solutions the expression above can lose precision by
    # cancellation, so compute OH- directly and derive [H+] = Kw/[OH-].
    if excess_h < 0:
        oh_concentration = 0.5 * (-excess_h + math.hypot(excess_h, 2.0 * math.sqrt(kw)))
        h_concentration = kw / oh_concentration

    # Do not clip to the textbook 0–14 range: ideal concentration-based pH can
    # fall outside it for concentrated solutions, although activity effects then
    # become important and this model should be interpreted cautiously.
    return -math.log10(h_concentration)


def first_order_concentration(initial_concentration: float, rate_constant: float, time_s: float) -> float:
    values = (initial_concentration, rate_constant, time_s)
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError("Concentration, rate constant, and time must be finite numbers.")
    if initial_concentration < 0 or rate_constant < 0 or time_s < 0:
        raise ValueError("Concentration, rate constant, and time must be non-negative.")
    return initial_concentration * math.exp(-rate_constant * time_s)


def first_order_half_life(rate_constant: float) -> float:
    if not math.isfinite(float(rate_constant)) or rate_constant <= 0:
        raise ValueError("Rate constant must be a finite number greater than zero.")
    return math.log(2.0) / rate_constant


def nernst_potential(e_standard_v: float, electrons: int, quotient: float, temperature_c: float = 25.0) -> float:
    if not all(math.isfinite(float(v)) for v in (e_standard_v, quotient, temperature_c)):
        raise ValueError("Potential, reaction quotient, and temperature must be finite numbers.")
    if electrons < 1 or quotient <= 0 or temperature_c <= -273.15:
        raise ValueError("Electrons must be positive, Q must be positive, and temperature must be above absolute zero.")
    return e_standard_v - (R_GAS * (temperature_c + 273.15) / (electrons * FARADAY)) * math.log(quotient)


def beer_lambert_absorbance(molar_absorptivity: float, path_length_cm: float, concentration_mol_l: float) -> float:
    if not all(math.isfinite(float(v)) for v in (molar_absorptivity, path_length_cm, concentration_mol_l)):
        raise ValueError("Absorptivity, path length, and concentration must be finite numbers.")
    if molar_absorptivity < 0 or path_length_cm <= 0 or concentration_mol_l < 0:
        raise ValueError("Absorptivity and concentration must be non-negative; path length must be positive.")
    return molar_absorptivity * path_length_cm * concentration_mol_l


def beer_lambert_concentration(absorbance: float, molar_absorptivity: float, path_length_cm: float) -> float:
    if not all(math.isfinite(float(v)) for v in (absorbance, molar_absorptivity, path_length_cm)):
        raise ValueError("Absorbance, absorptivity, and path length must be finite numbers.")
    if absorbance < 0 or molar_absorptivity <= 0 or path_length_cm <= 0:
        raise ValueError("Absorbance must be non-negative; absorptivity and path length must be positive.")
    return absorbance / (molar_absorptivity * path_length_cm)


def arrhenius_rate_constant(pre_exponential_s_inv: float, activation_energy_kj_mol: float, temperature_k: float) -> float:
    if not all(math.isfinite(float(v)) for v in (pre_exponential_s_inv, activation_energy_kj_mol, temperature_k)):
        raise ValueError("Arrhenius inputs must be finite numbers.")
    if pre_exponential_s_inv <= 0 or activation_energy_kj_mol < 0 or temperature_k <= 0:
        raise ValueError("Pre-exponential factor and temperature must be positive; activation energy cannot be negative.")
    exponent = -(activation_energy_kj_mol * 1000.0) / (R_GAS * temperature_k)
    return pre_exponential_s_inv * math.exp(max(-745.0, min(700.0, exponent)))
