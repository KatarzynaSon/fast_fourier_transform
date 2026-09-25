"""Tests for the FFT core module."""

import math
import unittest

from fast_fourier_transform.core import convolution, fft, ifft, polynomial_multiply


class TestFFT(unittest.TestCase):
    def assertComplexAlmostEqual(self, first, second, places=9):
        """Assert that two complex numbers are almost equal."""
        self.assertAlmostEqual(first.real, second.real, places=places)
        self.assertAlmostEqual(first.imag, second.imag, places=places)

    def assertComplexListAlmostEqual(self, first, second, places=9):
        """Assert that two lists of complex numbers are element-wise almost equal."""
        self.assertEqual(len(first), len(second))
        for a, b in zip(first, second):
            self.assertComplexAlmostEqual(a, b, places=places)

    def test_fft_single_element(self):
        """FFT of a single element is that element."""
        self.assertComplexListAlmostEqual(fft([5 + 2j]), [5 + 2j])

    def test_fft_constant_signal(self):
        """FFT of a constant signal has energy only at DC."""
        result = fft([1, 1, 1, 1])
        expected = [4 + 0j, 0j, 0j, 0j]
        self.assertComplexListAlmostEqual(result, expected)

    def test_fft_delta(self):
        """FFT of a delta at position 0 is all ones."""
        result = fft([1, 0, 0, 0])
        expected = [1 + 0j, 1 + 0j, 1 + 0j, 1 + 0j]
        self.assertComplexListAlmostEqual(result, expected)

    def test_fft_known_frequency(self):
        """FFT of a pure cosine wave."""
        # cos(2*pi*1*n/4) = [1, 0, -1, 0]
        result = fft([1, 0, -1, 0])
        # DFT has spikes at k=1 and k=3, each of magnitude 2.
        expected = [0j, 2 + 0j, 0j, 2 + 0j]
        self.assertComplexListAlmostEqual(result, expected)

    def test_fft_length_not_power_of_two(self):
        """Non-power-of-two length raises ValueError."""
        with self.assertRaises(ValueError):
            fft([1, 2, 3])

    def test_ifft_inverts_fft(self):
        """ifft(fft(x)) == x for a random power-of-two sequence."""
        x = [1 + 2j, -3 + 0.5j, 4 - 1j, 0.2 + 0.3j]
        result = ifft(fft(x))
        self.assertComplexListAlmostEqual(result, x)

    def test_ifft_length_not_power_of_two(self):
        """Non-power-of-two length raises ValueError."""
        with self.assertRaises(ValueError):
            ifft([1, 2, 3])

    def test_convolution_simple(self):
        """Convolution of two short sequences."""
        a = [1, 2, 3]
        b = [4, 5]
        # Expected: [4, 13, 22, 15]
        result = convolution(a, b)
        expected = [4 + 0j, 13 + 0j, 22 + 0j, 15 + 0j]
        self.assertComplexListAlmostEqual(result, expected)

    def test_convolution_empty_raises(self):
        """Empty input raises ValueError."""
        with self.assertRaises(ValueError):
            convolution([], [1, 2])
        with self.assertRaises(ValueError):
            convolution([1, 2], [])

    def test_convolution_identity(self):
        """Convolution with a delta returns the original sequence."""
        a = [2, -1, 3]
        delta = [1]
        result = convolution(a, delta)
        self.assertComplexListAlmostEqual(result, [complex(x) for x in a])

    def test_polynomial_multiply_basic(self):
        """Multiply two simple polynomials."""
        # (1 + x) * (1 - x) = 1 - x^2
        p = [1, 1]
        q = [1, -1]
        result = polynomial_multiply(p, q)
        expected = [1 + 0j, 0j, -1 + 0j]
        self.assertComplexListAlmostEqual(result, expected)

    def test_polynomial_multiply_degree(self):
        """Degree of product is sum of degrees."""
        p = [1, 2, 3]   # degree 2
        q = [4, 5, 6]   # degree 2
        result = polynomial_multiply(p, q)
        self.assertEqual(len(result), 5)  # 2+2+1 = 5 coefficients
        # Check a known coefficient: x^2 term: 1*6 + 2*5 + 3*4 = 28
        self.assertComplexAlmostEqual(result[2], 28 + 0j)


if __name__ == "__main__":
    unittest.main()
