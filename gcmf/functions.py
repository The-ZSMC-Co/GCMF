"""Predefined GCMF function families (Table 1, draft-dutta-gcmf-01),
excluding CUSTOM (0000) and LITERAL (0001), which are handled directly
by the encoder/decoder modules.

Each predefined family provides a parameter encoding (`write_params` /
`read_params`, Section 4.9.1) and an exact-integer evaluator
(`evaluate`, Table 1's formula column). MODULAR (0b1010) is listed here
for completeness of the address table but is not implemented: its
parameter body is itself a nested GCMF function, which this v0.1
reference implementation does not yet support (see README "Roadmap").
"""

from __future__ import annotations

import math

from .bitstream import BitReader, BitWriter

CUSTOM = 0b0000
LITERAL = 0b0001
CONSTANT = 0b0010
LINEAR = 0b0011
ARITHMETIC = 0b0100
GEOMETRIC = 0b0101
POLYNOMIAL = 0b0110
EXPONENTIAL = 0b0111
LOGARITHMIC = 0b1000
POWER = 0b1001
MODULAR = 0b1010
FIBONACCI = 0b1011
FACTORIAL = 0b1100
TRIANGULAR = 0b1101
SQUARE = 0b1110
CUBE = 0b1111

NAMES = {
    CUSTOM: "CUSTOM",
    LITERAL: "LITERAL",
    CONSTANT: "CONSTANT",
    LINEAR: "LINEAR",
    ARITHMETIC: "ARITHMETIC",
    GEOMETRIC: "GEOMETRIC",
    POLYNOMIAL: "POLYNOMIAL",
    EXPONENTIAL: "EXPONENTIAL",
    LOGARITHMIC: "LOGARITHMIC",
    POWER: "POWER",
    MODULAR: "MODULAR",
    FIBONACCI: "FIBONACCI",
    FACTORIAL: "FACTORIAL",
    TRIANGULAR: "TRIANGULAR",
    SQUARE: "SQUARE",
    CUBE: "CUBE",
}

# Predefined functions that take no parameters (Section 4.9.1).
ZERO_PARAM = {FIBONACCI, FACTORIAL, TRIANGULAR, SQUARE, CUBE}


def write_params(bw: BitWriter, code: int, params: tuple) -> None:
    if code == CONSTANT:
        (c,) = params
        bw.write_zigzag_vli(c)
    elif code in (LINEAR, ARITHMETIC, GEOMETRIC, EXPONENTIAL):
        a, b = params
        bw.write_zigzag_vli(a)
        bw.write_zigzag_vli(b)
    elif code in (LOGARITHMIC, POWER):
        a, b, c = params
        bw.write_zigzag_vli(a)
        bw.write_zigzag_vli(b)
        bw.write_zigzag_vli(c)
    elif code == POLYNOMIAL:
        coeffs = params
        bw.write_vli(len(coeffs) - 1)  # n = degree
        for a_i in coeffs:
            bw.write_zigzag_vli(a_i)
    elif code in ZERO_PARAM:
        pass
    else:
        raise ValueError(f"cannot encode parameters for function code {code:#06b}")


def read_params(br: BitReader, code: int) -> tuple:
    if code == CONSTANT:
        return (br.read_zigzag_vli(),)
    if code in (LINEAR, ARITHMETIC, GEOMETRIC, EXPONENTIAL):
        return (br.read_zigzag_vli(), br.read_zigzag_vli())
    if code in (LOGARITHMIC, POWER):
        return (br.read_zigzag_vli(), br.read_zigzag_vli(), br.read_zigzag_vli())
    if code == POLYNOMIAL:
        n = br.read_vli()
        return tuple(br.read_zigzag_vli() for _ in range(n + 1))
    if code in ZERO_PARAM:
        return ()
    raise ValueError(f"cannot decode parameters for function code {code:#06b}")


def evaluate(code: int, x: int, params: tuple) -> int:
    """Evaluate F(x) exactly. Raises ValueError for undefined evaluations
    (Section 4.11.1: implementations MUST reject rather than approximate)."""
    if code == CONSTANT:
        (c,) = params
        return c
    if code == LINEAR:
        a, b = params
        return a * x + b
    if code == ARITHMETIC:
        a, d = params
        return a + x * d
    if code == GEOMETRIC:
        a, r = params
        return a * (r ** x)
    if code == EXPONENTIAL:
        a, b = params
        return a * (b ** x)
    if code == POLYNOMIAL:
        return sum(a_i * (x ** i) for i, a_i in enumerate(params))
    if code == LOGARITHMIC:
        a, b, c = params
        if x <= 0 or b <= 1:
            raise ValueError("LOGARITHMIC is undefined for this x or base")
        exact = a * math.log(x, b) + c
        rounded = round(exact)
        if abs(exact - rounded) > 1e-9:
            raise ValueError("LOGARITHMIC evaluation is not exactly integral")
        return rounded
    if code == POWER:
        a, b, c = params
        return a * (x ** b) + c
    if code == FIBONACCI:
        if x < 0:
            raise ValueError("FIBONACCI is undefined for negative x")
        f0, f1 = 0, 1
        for _ in range(x):
            f0, f1 = f1, f0 + f1
        return f0
    if code == FACTORIAL:
        if x < 0:
            raise ValueError("FACTORIAL is undefined for negative x")
        return math.factorial(x)
    if code == TRIANGULAR:
        return x * (x + 1) // 2
    if code == SQUARE:
        return x * x
    if code == CUBE:
        return x * x * x
    raise ValueError(
        f"function code {code:#06b} ({NAMES.get(code, '?')}) is not "
        f"implemented in this reference implementation"
    )
