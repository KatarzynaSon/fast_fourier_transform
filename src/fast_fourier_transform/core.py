"""Core FFT routines.

This module implements a radix-2 Cooley-Tukey FFT for sequences whose length
is a power of two. The implementation is iterative, which avoids recursion
overhead and Python recursion-depth limits for large transforms.

The input may be any iterable of complex numbers (including real numbers,
which are promoted to complex). The output is a list of complex numbers.
"""

from __future__ import annotations

from cmath import exp, pi
from typing import Iterable, List, Sequence, Union

Complex = Union[int, float, complex]


def _is_power_of_two(n: int) -> bool:
    """Return True if n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def _bit_reverse_index(index: int, bits: int) -> int:
    """Return the bit-reversed value of index using the low `bits` bits.

    For example, with bits=3, index 1 (001b) becomes 4 (100b).
    """
    result = 0
    for _ in range(bits):
        result = (result << 1) | (index & 1)
        index >>= 1
    return result


def fft(values: Sequence[Complex]) -> List[complex]:
    """Compute the forward discrete Fourier transform.

    Parameters
    ----------
    values : sequence of complex-compatible numbers
        The input sequence. Its length must be a power of two.

    Returns
    -------
    list of complex
        The DFT of the input, ordered from frequency 0 to N-1.

    Raises
    ------
    ValueError
        If the length is not a power of two.
    """
    n = len(values)
    if not _is_power_of_two(n):
        raise ValueError("fft length must be a power of two")

    # Work with a list of complex numbers.
    a = [complex(x) for x in values]

    # Bit-reversal permutation. This reorders the input so that the iterative
    # Cooley-Tukey butterfly stages can operate in-place.
    bits = n.bit_length() - 1
    for i in range(n):
        j = _bit_reverse_index(i, bits)
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Iterative butterfly stages.
    length = 2
    while length <= n:
        # Principal primitive root of unity for this stage length.
        angle = -2.0 * pi / length
        w_len = complex(exp(1j * angle))
        half = length // 2
        for start in range(0, n, length):
            w = 1 + 0j
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half] * w
                a[start + j] = u + v
                a[start + j + half] = u - v
                w *= w_len
        length <<= 1

    return a


def ifft(values: Sequence[Complex]) -> List[complex]:
    """Compute the inverse discrete Fourier transform.

    The inverse is scaled by 1/N so that ifft(fft(x)) == x within
    floating-point accuracy.

    Parameters
    ----------
    values : sequence of complex-compatible numbers
        The input sequence in frequency order. Length must be a power of two.

    Returns
    -------
    list of complex
        The inverse DFT of the input.
    """
    n = len(values)
    if not _is_power_of_two(n):
        raise ValueError("ifft length must be a power of two")

    a = [complex(x) for x in values]

    # Bit-reversal permutation, same as forward transform.
    bits = n.bit_length() - 1
    for i in range(n):
        j = _bit_reverse_index(i, bits)
        if i < j:
            a[i], a[j] = a[j], a[i]

    # Butterfly stages with positive exponent, then scale by 1/N.
    length = 2
    while length <= n:
        angle = 2.0 * pi / length
        w_len = complex(exp(1j * angle))
        half = length // 2
        for start in range(0, n, length):
            w = 1 + 0j
            for j in range(half):
                u = a[start + j]
                v = a[start + j + half] * w
                a[start + j] = u + v
                a[start + j + half] = u - v
                w *= w_len
        length <<= 1

    scale = 1.0 / n
    return [x * scale for x in a]


def convolution(a: Sequence[Complex], b: Sequence[Complex]) -> List[complex]:
    """Compute the linear convolution of two sequences.

    The result length is len(a) + len(b) - 1. The FFT size is the smallest
    power of two greater than or equal to that length. The returned values are
    complex, even if the inputs were real.

    Parameters
    ----------
    a, b : sequences of complex-compatible numbers
        The sequences to convolve. Both must be non-empty.

    Returns
    -------
    list of complex
        The convolution.

    Raises
    ------
    ValueError
        If either sequence is empty.
    """
    if len(a) == 0 or len(b) == 0:
        raise ValueError("convolution requires non-empty sequences")

    result_len = len(a) + len(b) - 1
    # Find the next power of two >= result_len.
    n = 1
    while n < result_len:
        n <<= 1

    # Zero-pad to length n.
    a_padded = [complex(x) for x in a] + [0j] * (n - len(a))
    b_padded = [complex(x) for x in b] + [0j] * (n - len(b))

    fa = fft(a_padded)
    fb = fft(b_padded)

    # Pointwise multiply in the frequency domain.
    fc = [x * y for x, y in zip(fa, fb)]

    result = ifft(fc)

    # Trim to the true convolution length.
    return result[:result_len]


def polynomial_multiply(p: Sequence[Complex], q: Sequence[Complex]) -> List[complex]:
    """Multiply two polynomials given as coefficient lists.

    Coefficients are ordered from lowest degree to highest degree. For example,
    [1, 2, 3] represents 1 + 2x + 3x^2.

    This is a thin wrapper around `convolution`; polynomial multiplication is
    exactly the convolution of the coefficient lists.

    Parameters
    ----------
    p, q : sequences of complex-compatible numbers
        The coefficient lists of the two polynomials. Both must be non-empty.

    Returns
    -------
    list of complex
        The coefficients of the product polynomial, ordered from lowest to
        highest degree.
    """
    return convolution(p, q)
