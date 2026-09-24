"""Fast Fourier Transform library.

Provides radix-2 Cooley-Tukey FFT for complex sequences, suitable for
convolution and polynomial multiplication.
"""

from .core import fft, ifft, convolution, polynomial_multiply

__all__ = ["fft", "ifft", "convolution", "polynomial_multiply"]
