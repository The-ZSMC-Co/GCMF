"""GCMF encoder: candidate search and argmin-length selection
(Sections 3, 4.21-4.23 of draft-dutta-gcmf-01).

The specification encodes an abstract integer sequence; it does not
say how a caller's source data maps onto that sequence. This reference
implementation exposes that mapping as an explicit `word_size`
parameter (1, 2, 4, or 8 bytes per element, little-endian) -- see the
module docstring in `decoder.py` for the full rationale.

This encoder searches, for every input, all of:

  * Predefined families: CONSTANT, LINEAR, ARITHMETIC, GEOMETRIC,
    EXPONENTIAL, POLYNOMIAL (degree 2-3, solved exactly via Gaussian
    elimination over the rationals), FIBONACCI, FACTORIAL, TRIANGULAR,
    SQUARE, CUBE.
  * MODULAR (Section 4.9.1): F(x) = G(x) mod m, currently searched for
    an affine G(x) = ax + b against a fixed candidate set of moduli
    (this is what represents periodic/wrapping data, e.g. a counter
    that resets every 256 values -- not representable by any
    non-modular predefined family).
  * CUSTOM bytecode (Sections 4.8-4.12): a library of parameterized
    program *shapes* (XOR mask, AND-mask power-of-two modulus,
    absolute value of an affine function, clamped affine via MIN,
    integer floor-division) is fit against the data the same way a
    predefined family is, by solving for that shape's small parameter
    set and verifying an exact match -- this is automated parameter
    search, not a user-supplied formula. In addition, for inputs of at
    most `BRUTE_FORCE_MAX_ELEMENTS` elements, a bounded exhaustive
    search (`_brute_force_custom_search`) enumerates every valid
    straight-line stack program up to `BRUTE_FORCE_MAX_LENGTH`
    instructions over a restricted opcode/constant alphabet, and keeps
    the shortest one that reproduces the data exactly. This directly
    implements Section 3's C_I (the implementation-searchable subset
    of the theoretical candidate set C): the specification explicitly
    does not require finding the global optimum, only the best
    candidate this implementation is able to construct.

Per Sections 4.14.3 and 4.15.5, BASE does not affect the binary
encoding of any field and MUST NOT be used as a selection criterion
when it has no effect -- which, given this encoder's field encodings,
is always the case. This encoder therefore fixes BASE at a constant
default rather than searching it, since a "search across bases" that
never changes any candidate's length would contradict Section
4.15.5's explicit prohibition.

For every successful candidate, this encoder computes the *complete*
encoded length (magic, version, function address, body, domain, and
base -- Section 3.1) and only keeps candidates strictly smaller than
the current best, initialised to the LITERAL fallback (Section
4.22/4.23): this encoder never emits a mathematical representation
that is not actually smaller than storing the bytes directly.

Not yet implemented (see README "Roadmap"): residual (near-fit)
representations, and searching LOGARITHMIC/POWER as predefined
functions (their codecs/evaluators exist in `functions.py` for
forward compatibility with streams from other GCMF encoders, but
exact integer fits for those families are rare over a
consecutive-integer domain, so this encoder does not spend search
time on them).
"""

from __future__ import annotations

from fractions import Fraction

from . import bytecode as bc
from . import functions as fn
from .bitstream import BitWriter

MAGIC = b"GCMF"
VERSION = 0
DEFAULT_BASE = 10  # Section 4.14.3/4.15.5: BASE never affects encoded
                    # length in this implementation, so it is fixed
                    # rather than searched (see module docstring).

VALID_WORD_SIZES = (1, 2, 4, 8)

BRUTE_FORCE_MAX_ELEMENTS = 64
BRUTE_FORCE_MAX_LENGTH = 4
BRUTE_FORCE_CONST_RANGE = range(-4, 5)
BRUTE_FORCE_MAX_NODES = 300_000

_LIMITS = bc.DEFAULT_LIMITS


def _bytes_to_elements(data: bytes, word_size: int) -> list:
    if word_size not in VALID_WORD_SIZES:
        raise ValueError(f"word_size must be one of {VALID_WORD_SIZES}, got {word_size}")
    if len(data) % word_size != 0:
        raise ValueError(
            f"len(data)={len(data)} is not a multiple of word_size={word_size}; "
            f"pad the input first, or use a word_size that divides its length"
        )
    return [int.from_bytes(data[i:i + word_size], "little")
            for i in range(0, len(data), word_size)]


# ---------------------------------------------------------------------------
# Candidate byte-stream construction
# ---------------------------------------------------------------------------

def _write_function_body(bw: BitWriter, code: int, body) -> None:
    """Write a function's type-specific body. `code`'s discriminant bits
    (5-bit outer Function Address, or MODULAR's bare 4-bit nested type
    -- Section 4.9.1) must already have been written by the caller."""
    if code == fn.CUSTOM:
        instructions = body
        program = bc.assemble(instructions)
        bw.write_vli(program.bit_length())
        bw.write_bitwriter(program)
    else:
        fn.write_params(bw, code, body)


def _candidate_bytes_for_spec(code: int, body, x_start: int, x_end: int,
                               base: int = DEFAULT_BASE) -> bytes:
    """Build a complete candidate stream for any of: a predefined
    function (`body` = params tuple), CUSTOM (`body` = instruction
    list), or MODULAR (`body` = (inner_code, inner_body, m))."""
    bw = BitWriter()
    bw.write_bits(0, 1)      # residual flag R = 0 (residuals: future work)
    bw.write_bits(code, 4)   # function type address
    if code == fn.MODULAR:
        inner_code, inner_body, m = body
        bw.write_bits(inner_code, 4)  # bare 4-bit nested type, no R bit
        _write_function_body(bw, inner_code, inner_body)
        bw.write_vli(m)
    else:
        _write_function_body(bw, code, body)
    bw.write_vli(x_start)
    bw.write_vli(x_end)
    bw.write_vli(base)
    return MAGIC + bytes([VERSION]) + bw.to_bytes()


def _literal_bytes(data: bytes) -> bytes:
    bw = BitWriter()
    bw.write_bits(0, 1)       # residual flag MUST be 0 for LITERAL
    bw.write_bits(fn.LITERAL, 4)
    bw.align_to_byte()        # 3 zero padding bits (Section 4.18)
    return MAGIC + bytes([VERSION]) + bw.to_bytes() + data


# ---------------------------------------------------------------------------
# Predefined-family search (unchanged behaviour from v0.1)
# ---------------------------------------------------------------------------

def _fits(code: int, params: tuple, elements, x_start: int) -> bool:
    for i, d in enumerate(elements):
        try:
            v = fn.evaluate(code, x_start + i, params)
        except (ValueError, OverflowError):
            return False
        if v != d:
            return False
    return True


def _try_constant(elements):
    return (elements[0],) if len(set(elements)) == 1 else None


def _try_linear(elements):
    b = elements[0]
    a = (elements[1] - elements[0]) if len(elements) >= 2 else 0
    return (a, b) if _fits(fn.LINEAR, (a, b), elements, 0) else None


def _try_arithmetic(elements):
    a = elements[0]
    d = (elements[1] - elements[0]) if len(elements) >= 2 else 0
    return (a, d) if _fits(fn.ARITHMETIC, (a, d), elements, 0) else None


def _try_geometric_family(elements, code: int):
    if len(elements) < 2 or elements[0] == 0 or elements[1] % elements[0] != 0:
        return None
    params = (elements[0], elements[1] // elements[0])
    return params if _fits(code, params, elements, 0) else None


def _solve_vandermonde_int(xs, ys):
    """Solve sum(a_i * x^i) = y for i in 0..len(xs)-1 exactly, returning
    an integer coefficient tuple, or None if no exact-integer solution
    exists (used to find genuinely integer-coefficient POLYNOMIAL fits,
    since integer outputs on a consecutive-integer domain do not imply
    integer standard-form coefficients in general)."""
    n = len(xs)
    matrix = [[Fraction(x) ** i for i in range(n)] + [Fraction(y)]
              for x, y in zip(xs, ys)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if matrix[r][col] != 0), None)
        if pivot is None:
            return None
        matrix[col], matrix[pivot] = matrix[pivot], matrix[col]
        pv = matrix[col][col]
        matrix[col] = [v / pv for v in matrix[col]]
        for r in range(n):
            if r != col and matrix[r][col] != 0:
                factor = matrix[r][col]
                matrix[r] = [matrix[r][k] - factor * matrix[col][k] for k in range(n + 1)]
    coeffs = []
    for row in matrix:
        val = row[-1]
        if val.denominator != 1:
            return None
        coeffs.append(int(val))
    return tuple(coeffs)


def _try_polynomial(elements, degree: int):
    if len(elements) < degree + 1:
        return None
    xs = list(range(degree + 1))
    ys = elements[: degree + 1]
    coeffs = _solve_vandermonde_int(xs, ys)
    if coeffs is None:
        return None
    return coeffs if _fits(fn.POLYNOMIAL, coeffs, elements, 0) else None


def _try_zero_param(elements, code: int):
    return () if _fits(code, (), elements, 0) else None


# ---------------------------------------------------------------------------
# MODULAR search (Section 4.9.1): F(x) = G(x) mod m
# ---------------------------------------------------------------------------

# A fixed, cheap candidate modulus set: every power of two up to 2**32
# (covers common wraparound widths: nibble, byte, word, dword, ...),
# plus the data's own range. Checking one modulus is O(n), and this
# set has fixed size, so this search stays linear in input size
# regardless of the modulus values themselves.
_MODULAR_CANDIDATE_MODULI = tuple(1 << k for k in range(1, 33))


def _try_modular_affine(elements):
    """Search for F(x) = (a*x + b) mod m: the case that matters most in
    practice is a wrapping counter/index/timestamp column, which no
    non-modular predefined family can represent once it wraps."""
    n = len(elements)
    if n < 2:
        return None
    a = elements[1] - elements[0]
    b = elements[0]
    if a == 0:
        return None  # constant case is already covered by CONSTANT
    candidates = set(_MODULAR_CANDIDATE_MODULI)
    candidates.add(max(elements) + 1)
    for m in sorted(mod for mod in candidates if mod >= 1):
        if all((a * i + b) % m == d for i, d in enumerate(elements)):
            return (fn.LINEAR, (a, b), m)
    return None


# ---------------------------------------------------------------------------
# CUSTOM search: parameterized program-shape templates
# ---------------------------------------------------------------------------

def _custom_fits(instructions, elements) -> bool:
    for i, d in enumerate(elements):
        try:
            v = bc.evaluate(instructions, i, _LIMITS)
        except bc.BytecodeError:
            return False
        if isinstance(v, float):
            if not v.is_integer():
                return False
            v = int(v)
        if v != d:
            return False
    return True


def _try_custom_xor_constant(elements):
    """F(x) = x XOR k. Since 0 XOR k = k, k is read directly off the
    first element."""
    if not elements:
        return None
    k = elements[0]
    instrs = [bc.X, (bc.CONST, k), bc.XOR, bc.END]
    return instrs if _custom_fits(instrs, elements) else None


def _try_custom_and_mask(elements):
    """F(x) = x AND (2^bits - 1), i.e. x mod 2^bits computed via a
    bitmask instead of MOD -- included alongside `_try_modular_affine`
    so the encoder genuinely competes CUSTOM against MODULAR for
    power-of-two wraparound rather than assuming one is always better."""
    if not elements:
        return None
    for bits in range(1, 33):
        mask = (1 << bits) - 1
        instrs = [bc.X, (bc.CONST, mask), bc.AND, bc.END]
        if _custom_fits(instrs, elements):
            return instrs
    return None


def _try_custom_abs_affine(elements):
    """F(x) = |a*x + b| -- a V-shaped sequence, not representable by
    any predefined family (all of which are monotonic or periodic)."""
    if len(elements) < 2:
        return None
    diff = elements[1] - elements[0]
    for a in (diff, -diff):
        for b in (elements[0], -elements[0]):
            instrs = [(bc.CONST, a), bc.X, bc.MUL, (bc.CONST, b), bc.ADD, bc.ABS, bc.END]
            if _custom_fits(instrs, elements):
                return instrs
    return None


def _try_custom_min_clamp(elements):
    """F(x) = min(a*x + b, cap) -- an affine ramp that saturates, e.g. a
    counter that stops increasing once it hits a ceiling."""
    if len(elements) < 2:
        return None
    a = elements[1] - elements[0]
    b = elements[0]
    cap = max(elements)
    instrs = [(bc.CONST, a), bc.X, bc.MUL, (bc.CONST, b), bc.ADD, (bc.CONST, cap), bc.MIN, bc.END]
    return instrs if _custom_fits(instrs, elements) else None


def _try_custom_floor_div(elements):
    """F(x) = floor(x / k) -- a staircase, e.g. downsampled indices."""
    if not elements:
        return None
    for k in range(2, 17):
        instrs = [bc.X, (bc.CONST, k), bc.DIV, bc.FLOOR, bc.END]
        if _custom_fits(instrs, elements):
            return instrs
    return None


def _brute_force_custom_search(elements):
    """Bounded exhaustive search over straight-line stack programs of at
    most `BRUTE_FORCE_MAX_LENGTH` instructions (excluding END), built
    from a restricted opcode/constant alphabet, returning the shortest
    program (in instruction count) that reproduces `elements` exactly,
    or None. This is a real, deterministic search -- not a heuristic
    guess and not a user-supplied formula -- over the bounded candidate
    space C_I this implementation is able to construct (Section 3);
    the space is intentionally small enough to finish quickly, capped
    additionally by `BRUTE_FORCE_MAX_NODES` visited search-tree nodes.
    """
    binary_ops = (bc.ADD, bc.SUB, bc.MUL, bc.MOD, bc.AND, bc.XOR, bc.MIN, bc.MAX)
    unary_ops = (bc.NEG, bc.ABS)

    best = None
    nodes_visited = 0

    def leaves():
        yield [bc.X], 1
        for v in BRUTE_FORCE_CONST_RANGE:
            yield [(bc.CONST, v)], 1

    def extend(instrs, stack_size):
        nonlocal best, nodes_visited
        nodes_visited += 1
        if nodes_visited > BRUTE_FORCE_MAX_NODES:
            return
        if stack_size == 1:
            candidate = instrs + [bc.END]
            if best is None or len(candidate) < len(best):
                if _custom_fits(candidate, elements):
                    best = candidate
        if len(instrs) >= BRUTE_FORCE_MAX_LENGTH:
            return
        for extra, delta in leaves():
            extend(instrs + extra, stack_size + delta)
        if stack_size >= 1:
            for op in unary_ops:
                extend(instrs + [op], stack_size)
            extend(instrs + [bc.DUP], stack_size + 1)
        if stack_size >= 2:
            for op in binary_ops:
                extend(instrs + [op], stack_size - 1)

    extend([], 0)
    return best


_CUSTOM_TEMPLATES = (
    _try_custom_xor_constant,
    _try_custom_and_mask,
    _try_custom_abs_affine,
    _try_custom_min_clamp,
    _try_custom_floor_div,
)


# ---------------------------------------------------------------------------
# Top-level search
# ---------------------------------------------------------------------------

_PREDEFINED_SEARCHERS = (
    (fn.CONSTANT, _try_constant),
    (fn.LINEAR, _try_linear),
    (fn.ARITHMETIC, _try_arithmetic),
    (fn.GEOMETRIC, lambda e: _try_geometric_family(e, fn.GEOMETRIC)),
    (fn.EXPONENTIAL, lambda e: _try_geometric_family(e, fn.EXPONENTIAL)),
    (fn.POLYNOMIAL, lambda e: _try_polynomial(e, 2)),
    (fn.POLYNOMIAL, lambda e: _try_polynomial(e, 3)),
    (fn.FIBONACCI, lambda e: _try_zero_param(e, fn.FIBONACCI)),
    (fn.FACTORIAL, lambda e: _try_zero_param(e, fn.FACTORIAL)),
    (fn.TRIANGULAR, lambda e: _try_zero_param(e, fn.TRIANGULAR)),
    (fn.SQUARE, lambda e: _try_zero_param(e, fn.SQUARE)),
    (fn.CUBE, lambda e: _try_zero_param(e, fn.CUBE)),
)


def encode(data: bytes, word_size: int = 1) -> bytes:
    """Encode `data` as a GCMF byte string, selecting the shortest valid
    representation this encoder can find (Section 4.22) across every
    predefined family, MODULAR, and CUSTOM (templates plus bounded
    brute-force search), falling back to LITERAL when nothing found is
    both exact and compressive.

    `word_size` (1, 2, 4, or 8) controls how many bytes of `data` make up
    one sequence element (little-endian) when searching for a functional
    fit; it must be passed identically to `decode()` -- see the
    "IMPORTANT" note on `decoder.decode()`. It has no effect on the
    LITERAL fallback, which always stores `data` byte-for-byte.
    """
    best = _literal_bytes(data)
    if not data:
        return best

    elements = _bytes_to_elements(data, word_size)
    x_end = len(elements) - 1

    def consider(code, body):
        nonlocal best
        candidate = _candidate_bytes_for_spec(code, body, x_start=0, x_end=x_end)
        if len(candidate) < len(best):
            best = candidate

    for code, search in _PREDEFINED_SEARCHERS:
        params = search(elements)
        if params is not None:
            consider(code, params)

    modular_body = _try_modular_affine(elements)
    if modular_body is not None:
        consider(fn.MODULAR, modular_body)

    for template in _CUSTOM_TEMPLATES:
        instructions = template(elements)
        if instructions is not None:
            consider(fn.CUSTOM, instructions)

    if len(elements) <= BRUTE_FORCE_MAX_ELEMENTS:
        instructions = _brute_force_custom_search(elements)
        if instructions is not None:
            consider(fn.CUSTOM, instructions)

    return best
