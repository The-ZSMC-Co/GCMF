"""GCMF decoder: strict, spec-conformant parsing and validation
(Sections 4.24-4.25 of draft-dutta-gcmf-01).

This reference decoder implements:

  * LITERAL                        (Section 4.18)
  * All predefined functions except MODULAR's own arithmetic
    (Section 4.9), via `functions.py`
  * CUSTOM bytecode functions      (Sections 4.8, 4.10-4.12), via
    `bytecode.py`, including the security limits of Section 6
    (bounded stack depth, numeric magnitude, POW exponent, domain
    size, and wall-clock time)
  * MODULAR functions              (Section 4.9.1: F(x) = G(x) mod m),
    where the nested G(x) may itself be PREDEFINED, CUSTOM, or
    (recursively) MODULAR, up to `MAX_MODULAR_NESTING`

together with the subset of Section 4.25's mandatory rejection checks
that apply to the above: invalid magic, an unsupported version, the
reserved LITERAL+residual combination, truncated/malformed VLIs
(raised by BitReader), X_end < X_start, BASE < 2, stack underflow /
invalid stack termination / undefined bytecode operations (raised by
`bytecode.py`), and evaluated values outside the element range implied
by `word_size`.

Section 4.9.1 does not give an explicit bit-layout table for MODULAR's
nested G(x) the way Section 4.17 does for a residual function. This
implementation resolves that gap the same way Section 4.17 resolves it
for residuals: G(x) is read as a bare 4-bit function type (no Residual
Flag bit -- Section 4.9.1 says the nesting is "excluding residual
encoding"), followed by that function type's body (CUSTOM's
length+bytecode, or a PREDEFINED function's parameters), with no
separate X_start/X_end/Base for G(x) -- it is evaluated over the same
domain as the enclosing MODULAR representation. LITERAL is rejected as
a nested function type, mirroring Section 4.17's explicit prohibition
on a LITERAL residual, since G(x) must be "a mathematical function"
(Section 4.9.1).

This implementation still raises NotImplementedError -- rather than
silently misinterpreting the stream -- for residual (R = 1)
representations, which remain unimplemented (see README "Roadmap").

IMPORTANT -- `word_size` is a caller-supplied convention, not a wire
field: GCMF's abstract integer sequence says nothing about how many
bytes each element occupies, so this reference implementation cannot
recover `word_size` from the stream itself. `decode()` MUST be called
with the same `word_size` that was passed to `encode()`, or a
predefined/CUSTOM/MODULAR stream will be silently unpacked at the
wrong width and produce incorrect output of the wrong length rather
than an error (there is no checksum to detect this). LITERAL streams
are unaffected, since they store the original bytes verbatim
regardless of `word_size`.
"""

from __future__ import annotations

from . import bytecode as bc
from . import functions as fn
from .bitstream import BitReader
from .encoder import MAGIC, VERSION, VALID_WORD_SIZES

MAX_MODULAR_NESTING = 8


class GCMFError(ValueError):
    """A GCMF stream failed decoder validation (Section 4.25)."""


def decode(blob: bytes, word_size: int = 1, limits: bc.VMLimits = bc.DEFAULT_LIMITS) -> bytes:
    if word_size not in VALID_WORD_SIZES:
        raise ValueError(f"word_size must be one of {VALID_WORD_SIZES}, got {word_size}")

    if len(blob) < 5:
        raise GCMFError("truncated GCMF stream: shorter than the fixed header")
    if blob[:4] != MAGIC:
        raise GCMFError("invalid magic number: expected b'GCMF' (0x47434D46)")

    version = blob[4]
    if version != VERSION:
        raise GCMFError(f"unsupported GCMF version: {version}")

    br = BitReader(blob[5:])
    residual_flag = br.read_bits(1)
    code = br.read_bits(4)

    if code == fn.LITERAL:
        if residual_flag != 0:
            raise GCMFError("reserved LITERAL+residual combination (address 10001)")
        br.align_to_byte_expect_zero(3)
        return br.remaining_bytes()

    if residual_flag == 1:
        raise NotImplementedError(
            "residual (near-fit) representations are not yet implemented "
            "in this reference decoder"
        )

    try:
        spec = _read_function_body(br, code, limits, depth=0)
    except bc.BytecodeError as exc:
        raise GCMFError(str(exc)) from exc

    x_start = br.read_vli()
    x_end = br.read_vli()
    base = br.read_vli()

    if x_end < x_start:
        raise GCMFError("X_end must be >= X_start")
    if base < 2:
        raise GCMFError("BASE must be >= 2")
    if x_end - x_start + 1 > limits.max_domain_size:
        raise GCMFError("domain size exceeds the maximum permitted by this implementation")

    max_value = (1 << (8 * word_size)) - 1
    out = bytearray()
    try:
        for x in range(x_start, x_end + 1):
            value = _evaluate_spec(spec, x, limits)
            value = _finalize_value(value, x, max_value)
            out.extend(value.to_bytes(word_size, "little"))
    except bc.BytecodeError as exc:
        raise GCMFError(str(exc)) from exc
    except ValueError as exc:
        raise GCMFError(f"undefined evaluation: {exc}") from exc
    return bytes(out)


def _read_function_body(br: BitReader, code: int, limits: bc.VMLimits, depth: int):
    """Read a function body given its (already-consumed) type code,
    returning an internal spec tuple consumed by `_evaluate_spec`.
    Used for both the outer representation and, recursively, for
    MODULAR's nested G(x) (Section 4.9.1)."""
    if code == fn.CUSTOM:
        length_bits = br.read_vli()
        instructions = bc.read_program(br, length_bits, limits)
        return ("custom", instructions)

    if code == fn.MODULAR:
        if depth >= MAX_MODULAR_NESTING:
            raise GCMFError("MODULAR nesting exceeds the maximum permitted by this implementation")
        inner_code = br.read_bits(4)
        if inner_code == fn.LITERAL:
            raise GCMFError("MODULAR's nested G(x) MUST be a mathematical function, not LITERAL")
        if inner_code not in fn.NAMES and inner_code != fn.CUSTOM:
            raise GCMFError(f"invalid nested function type address in MODULAR: {inner_code:#06b}")
        inner_spec = _read_function_body(br, inner_code, limits, depth + 1)
        m = br.read_vli()
        if m < 1:
            raise GCMFError("MODULAR modulus m MUST satisfy m >= 1")
        return ("modular", inner_spec, m)

    if code not in fn.NAMES:
        raise GCMFError(f"invalid function type address: {code:#06b}")

    params = fn.read_params(br, code)
    return ("predefined", code, params)


def _evaluate_spec(spec, x: int, limits: bc.VMLimits):
    kind = spec[0]
    if kind == "predefined":
        _, code, params = spec
        return fn.evaluate(code, x, params)
    if kind == "custom":
        _, instructions = spec
        return bc.evaluate(instructions, x, limits)
    if kind == "modular":
        _, inner_spec, m = spec
        g = _evaluate_spec(inner_spec, x, limits)
        g_int = g if isinstance(g, int) else (int(g) if isinstance(g, float) and g.is_integer() else None)
        if g_int is None:
            raise GCMFError("MODULAR's nested G(x) did not evaluate to an exact integer")
        return g_int % m
    raise AssertionError(f"unreachable spec kind {kind!r}")


def _finalize_value(value, x: int, max_value: int) -> int:
    if isinstance(value, float):
        if not value.is_integer():
            raise GCMFError(f"evaluated value at x={x} is not an exact integer ({value})")
        value = int(value)
    if not (0 <= value <= max_value):
        raise GCMFError(
            f"evaluated value {value} at x={x} is outside the element range [0, {max_value}]"
        )
    return value
