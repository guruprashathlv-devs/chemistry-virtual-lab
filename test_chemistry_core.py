import math
import unittest

from chemistry_core import (
    arrhenius_rate_constant,
    beer_lambert_absorbance,
    beer_lambert_concentration,
    first_order_concentration,
    first_order_half_life,
    nernst_potential,
    strong_acid_strong_base_ph,
)


class ChemistryCoreTests(unittest.TestCase):
    def test_titration_equivalence_uses_temperature_dependent_water_ionization(self):
        # The water pKw correlation gives a neutral pH very close to 7 at 25 °C.
        self.assertAlmostEqual(strong_acid_strong_base_ph(0.1, 25, 0.1, 25), 6.9974, places=3)

    def test_titration_at_zero_base_includes_water_autoionization(self):
        ph = strong_acid_strong_base_ph(0.1, 25, 0.1, 0)
        self.assertAlmostEqual(ph, 1.0, places=3)

    def test_titration_is_monotonic_around_equivalence(self):
        before = strong_acid_strong_base_ph(0.1, 25, 0.1, 24.9)
        equivalence = strong_acid_strong_base_ph(0.1, 25, 0.1, 25.0)
        after = strong_acid_strong_base_ph(0.1, 25, 0.1, 25.1)
        self.assertLess(before, equivalence)
        self.assertLess(equivalence, after)

    def test_titration_before_equivalence_is_acidic(self):
        self.assertLess(strong_acid_strong_base_ph(0.1, 25, 0.1, 10), 7.0)

    def test_first_order_half_life(self):
        k = 0.2
        half = first_order_half_life(k)
        self.assertAlmostEqual(first_order_concentration(2.0, k, half), 1.0, places=10)

    def test_nernst_q_one_returns_standard_potential(self):
        self.assertAlmostEqual(nernst_potential(1.1, 2, 1.0), 1.1, places=12)

    def test_beer_lambert_round_trip(self):
        absorbance = beer_lambert_absorbance(150.0, 1.0, 0.002)
        self.assertAlmostEqual(absorbance, 0.3)
        self.assertAlmostEqual(beer_lambert_concentration(absorbance, 150.0, 1.0), 0.002)

    def test_arrhenius_rate_increases_with_temperature(self):
        low = arrhenius_rate_constant(1e10, 50, 280)
        high = arrhenius_rate_constant(1e10, 50, 320)
        self.assertGreater(high, low)

    def test_invalid_nernst_quotient_rejected(self):
        with self.assertRaises(ValueError):
            nernst_potential(1.1, 2, 0)


if __name__ == "__main__":
    unittest.main()
