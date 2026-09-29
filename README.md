# fast_fourier_transform

A small, dependency-free radix-2 Cooley-Tukey FFT for complex numbers, with helpers for convolution and polynomial multiplication.

```python
from fast_fourier_transform import fft, ifft, convolution, polynomial_multiply

# Forward transform
signal = [1, 0, -1, 0]
freq = fft(signal)
print(freq)  # [0j, (2+0j), 0j, (2+0j)]

# Inverse transform
reconstructed = ifft(freq)
print(reconstructed)  # [1+0j, 0j, -1+0j, 0j]

# Convolution
result = convolution([1, 2, 3], [4, 5])
print(result)  # [4+0j, 13+0j, 22+0j, 15+0j]

# Polynomial multiplication (coefficients low to high)
product = polynomial_multiply([1, 1], [1, -1])
print(product)  # [1+0j, 0j, -1+0j]  (1 - x^2)
```

## Why this library exists

Many environments cannot install NumPy or SciPy, but still need an FFT for tasks such as multiplying polynomials or performing linear convolution. This library provides a compact, readable implementation of the classic iterative radix-2 Cooley-Tukey algorithm using only Python's standard `cmath` module.

The main trade-off is that input lengths must be powers of two. The convolution and polynomial multiplication helpers automatically zero-pad to the next power of two, so callers only need to remember the restriction when using `fft` or `ifft` directly. A recursive implementation would be shorter but risks hitting Python's recursion limit on large inputs; the iterative version avoids that.

## Edge cases

- `fft` and `ifft` raise `ValueError` if the input length is not a power of two.
- `convolution` and `polynomial_multiply` raise `ValueError` on empty inputs.
- All functions return `complex` values, even when the inputs are real, because the frequency-domain representation is generally complex.

## Design notes

The window stores values eagerly rather than keeping running aggregates. Running
sums drift with floating point over long streams, and recomputing from a small
buffer is cheap enough that the drift is not worth the speed.

