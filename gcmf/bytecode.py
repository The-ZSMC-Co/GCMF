"""GCMF CUSTOM function bytecode (Sections 4.8-4.12, Appendix Table 2).

CUSTOM functions embed an executable, stack-based bytecode program
directly in the stream. Every opcode is exactly 5 bits; CONST is the
only opcode with an operand (a signed GCMF-VLI value). This module
provides:

  * OPCODES / MNEMONICS  -- the full 32-entry opcode table (Table 2);
                             the 5-bit address space is fully allocated,
                             so every possible 5-bit value is a valid
                             opcode (Section 4.28).
  * Program               -- a decoded/assembled instruction sequence
  * assemble()             -- build bytecode bits from a Python-level
                               instruction list (used by the encoder's
                               CUSTOM search)
  * read_program()         -- parse a bytecode blob of a declared bit
                               length into a Program (Section 4.10)
  * evaluate()              -- execute a Program for one domain value x
                               under configurable security limits
                               (Section 6, Section 4.25)

Security note (Section 6): a decoder executing CUSTOM bytecode is
executing untrusted, attacker-supplied instructions. `VMLimits` bounds
stack depth, numerical magnitude, exponent size, and per-domain
evaluation count/wall-clock time, since the bytecode itself has no
loop or branch instructions (Table 2 defines none), so the only ways
an adversarial stream can force excessive work are: an oversized
domain (X_end - X_start), oversized numeric intermediates (repeated
MUL/POW), or an oversized declared Function Length.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Union

from .bitstream import BitReader, BitWriter, zigzag_decode, zigzag_encode

CONST = 0b00000
X = 0b00001
ADD = 0b00010
SUB = 0b00011
MUL = 0b00100
DIV = 0b00101
MOD = 0b00110
POW = 0b00111
NEG = 0b01000
ABS = 0b01001
SQRT = 0b01010
LOG = 0b01011
LOG10 = 0b01100
EXP = 0b01101
SIN = 0b01110
COS = 0b01111
TAN = 0b10000
ASIN = 0b10001
ACOS = 0b10010
ATAN = 0b10011
FLOOR = 0b10100
CEIL = 0b10101
ROUND = 0b10110
MIN = 0b10111
MAX = 0b11000
AND = 0b11001
OR = 0b11010
XOR = 0b11011
SHL = 0b11100
SHR = 0b11101
DUP = 0b11110
END = 0b11111

MNEMONICS = {
    CONST: "CONST", X: "X", ADD: "ADD", SUB: "SUB", MUL: "MUL", DIV: "DIV",
    MOD: "MOD", POW: "POW", NEG: "NEG", ABS: "ABS", SQRT: "SQRT", LOG: "LOG",
    LOG10: "LOG10", EXP: "EXP", SIN: "SIN", COS: "COS", TAN: "TAN",
    ASIN: "ASIN", ACOS: "ACOS", ATAN: "ATAN", FLOOR: "FLOOR", CEIL: "CEIL",
    ROUND: "ROUND", MIN: "MIN", MAX: "MAX", AND: "AND", OR: "OR", XOR: "XOR",
    SHL: "SHL", SHR: "SHR", DUP: "DUP", END: "END",
}

# Ops that take one operand at assembly time (a zigzag-VLI constant).
_HAS_OPERAND = {CONST}

# Binary ops: pop 2 push 1. Unary ops: pop 1 push 1.
_BINARY = {ADD, SUB, MUL, DIV, MOD, POW, MIN, MAX, AND, OR, XOR, SHL, SHR}
_UNARY = {NEG, ABS, SQRT, LOG, LOG10, EXP, SIN, COS, TAN, ASIN, ACOS, ATAN,
          FLOOR, CEIL, ROUND}


class BytecodeError(ValueError):
    """A CUSTOM bytecode program is malformed or violates a security limit
    (Sections 4.10, 4.11, 6)."""


Instruction = Union[int, tuple]  # opcode, or (CONST, value)
Program = list  # list[Instruction]


@dataclass
class VMLimits:
    """Implementation-defined resource limits (Section 4.25, Section 6).

    These bound CUSTOM/MODULAR evaluation cost since the bytecode has
    no loop or branch instructions, so runaway cost can only come from
    an oversized domain, oversized numeric magnitude, or an oversized
    exponent -- not from the program looping.
    """

    max_stack_depth: int = 64
    max_program_instructions: int = 512
    max_numeric_magnitude: int = 1 << 512
    max_pow_exponent: int = 100_000
    max_domain_size: int = 10_000_000
    max_wall_seconds: float = 5.0


DEFAULT_LIMITS = VMLimits()


def assemble(instructions: Program) -> BitWriter:
    """Encode a list of instructions (ints, or (CONST, value) tuples) into
    a bit-packed bytecode body per Section 4.10. Does not include the
    leading Function Length field; callers prepend that separately."""
    bw = BitWriter()
    for instr in instructions:
        if isinstance(instr, tuple):
            opcode, operand = instr
            if opcode not in _HAS_OPERAND:
                raise BytecodeError(f"opcode {MNEMONICS.get(opcode)} does not take an operand")
            bw.write_bits(opcode, 5)
            bw.write_zigzag_vli(operand)
        else:
            if instr in _HAS_OPERAND:
                raise BytecodeError(f"opcode {MNEMONICS.get(instr)} requires an operand")
            bw.write_bits(instr, 5)
    return bw


def read_program(br: BitReader, length_bits: int, limits: VMLimits = DEFAULT_LIMITS) -> Program:
    """Read exactly `length_bits` bits as a bytecode program (Section
    4.10). The decoder MUST NOT read beyond the declared function
    length; this is enforced by bounding a sub-reader to that many bits."""
    if length_bits <= 0:
        raise BytecodeError("Function Length must be positive")
    if length_bits > limits.max_program_instructions * 40:
        # Generous per-instruction upper bound (5-bit opcode + up to a
        # multi-byte operand); rejects absurd declared lengths early
        # rather than allocating/reading them, per Section 6's
        # decompression-bomb guidance.
        raise BytecodeError("declared Function Length exceeds this implementation's limit")

    # BitReader has no sub-reader helper; read the raw bits directly and
    # re-wrap them so we can enforce a hard boundary independent of br.
    raw_bits = [br.read_bits(1) for _ in range(length_bits)]
    sub = _BitsReader(raw_bits)

    instructions: Program = []
    saw_end = False
    while sub.remaining() > 0:
        if len(instructions) >= limits.max_program_instructions:
            raise BytecodeError("program exceeds the maximum instruction count")
        opcode = sub.read(5)
        if opcode == CONST:
            if sub.remaining() < 8:  # a GCMF-VLI unit is 1 continuation bit + 7 payload bits
                raise BytecodeError("truncated CONST operand")
            operand = _read_vli_from_bits(sub)
            instructions.append((CONST, zigzag_decode(operand)))
        else:
            instructions.append(opcode)
        if opcode == END:
            saw_end = True
            break

    if not saw_end:
        raise BytecodeError("bytecode does not contain a terminating END within the declared length")

    trailing = sub.remaining()
    if trailing:
        # Section 4.8: "Any unused bits between the final END instruction
        # and the declared function boundary MUST be zero."
        for _ in range(trailing):
            if sub.read(1) != 0:
                raise BytecodeError("non-zero padding after END within declared Function Length")

    return instructions


class _BitsReader:
    """A tiny MSB-first bit reader over an in-memory list of bits, used
    to enforce CUSTOM's declared Function Length as a hard boundary."""

    def __init__(self, bits: list) -> None:
        self._bits = bits
        self._pos = 0

    def remaining(self) -> int:
        return len(self._bits) - self._pos

    def read(self, nbits: int) -> int:
        if self._pos + nbits > len(self._bits):
            raise BytecodeError("read past the declared Function Length")
        value = 0
        for _ in range(nbits):
            value = (value << 1) | self._bits[self._pos]
            self._pos += 1
        return value


def _read_vli_from_bits(sub: "_BitsReader") -> int:
    value = 0
    while True:
        cont = sub.read(1)
        group = sub.read(7)
        value = (value << 7) | group
        if cont == 0:
            return value


def evaluate(instructions: Program, x: int, limits: VMLimits = DEFAULT_LIMITS):
    """Execute `instructions` for a single domain value `x` (Section
    4.11). Returns an int (or a float that is exactly integral -- the
    caller is responsible for the final int-or-reject check, matching
    Section 4.11.1's requirement that only the final output need be
    representable in the target numerical domain)."""
    stack: list = []

    def push(v):
        if len(stack) >= limits.max_stack_depth:
            raise BytecodeError("stack depth exceeds the maximum permitted by this implementation")
        _check_magnitude(v, limits)
        stack.append(v)

    def pop():
        if not stack:
            raise BytecodeError("stack underflow")
        return stack.pop()

    for instr in instructions:
        opcode, operand = (instr, None) if not isinstance(instr, tuple) else instr

        if opcode == CONST:
            push(operand)
        elif opcode == X:
            push(x)
        elif opcode == END:
            break
        elif opcode in _BINARY:
            b = pop()
            a = pop()
            push(_apply_binary(opcode, a, b, limits))
        elif opcode in _UNARY:
            a = pop()
            push(_apply_unary(opcode, a))
        elif opcode == DUP:
            a = pop()
            push(a)
            push(a)
        else:
            raise BytecodeError(f"unknown opcode {opcode!r}")  # unreachable: address space is fully allocated

    if len(stack) != 1:
        raise BytecodeError(
            f"program terminated with {len(stack)} stack value(s); END requires exactly one"
        )
    return stack[0]


def _check_magnitude(v, limits: VMLimits) -> None:
    if isinstance(v, int) and abs(v) > limits.max_numeric_magnitude:
        raise BytecodeError("intermediate value exceeds the maximum numerical magnitude")
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        raise BytecodeError("intermediate value is not finite")


def _as_int(v):
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    raise BytecodeError("operation requires an exact integer operand")


def _apply_binary(opcode: int, a, b, limits: VMLimits):
    if opcode == ADD:
        return a + b
    if opcode == SUB:
        return a - b
    if opcode == MUL:
        return a * b
    if opcode == DIV:
        if b == 0:
            raise BytecodeError("division by zero")
        if isinstance(a, int) and isinstance(b, int) and a % b == 0:
            return a // b
        # Section 4.11.1: exact integer arithmetic is required unless an
        # operation is "explicitly defined to use another numerical
        # domain". DIV of non-multiples has no exact-integer result;
        # this implementation treats that as the real (float) domain,
        # to be reconciled back to an integer via FLOOR/CEIL/ROUND if
        # the program needs one -- see README for this interpretation.
        return a / b
    if opcode == MOD:
        b_i = _as_int(b)
        if b_i <= 0:
            raise BytecodeError("MOD requires a positive modulus")
        return _as_int(a) % b_i
    if opcode == POW:
        if abs(b if isinstance(b, int) else int(b)) > limits.max_pow_exponent:
            raise BytecodeError("POW exponent exceeds the maximum permitted by this implementation")
        try:
            return a ** b
        except (OverflowError, ValueError) as exc:
            raise BytecodeError(f"POW is undefined for these operands: {exc}") from exc
    if opcode == MIN:
        return min(a, b)
    if opcode == MAX:
        return max(a, b)
    if opcode == AND:
        return _as_int(a) & _as_int(b)
    if opcode == OR:
        return _as_int(a) | _as_int(b)
    if opcode == XOR:
        return _as_int(a) ^ _as_int(b)
    if opcode == SHL:
        shift = _as_int(b)
        if shift < 0:
            raise BytecodeError("SHL requires a non-negative shift amount")
        return _as_int(a) << shift
    if opcode == SHR:
        shift = _as_int(b)
        if shift < 0:
            raise BytecodeError("SHR requires a non-negative shift amount")
        return _as_int(a) >> shift
    raise BytecodeError(f"unhandled binary opcode {opcode!r}")


def _apply_unary(opcode: int, a):
    if opcode == NEG:
        return -a
    if opcode == ABS:
        return abs(a)
    if opcode == SQRT:
        if a < 0:
            raise BytecodeError("SQRT is undefined for a negative operand")
        if isinstance(a, int):
            r = math.isqrt(a)
            return r if r * r == a else math.sqrt(a)
        return math.sqrt(a)
    if opcode == LOG:
        if a <= 0:
            raise BytecodeError("LOG is undefined for a non-positive operand")
        return math.log(a)
    if opcode == LOG10:
        if a <= 0:
            raise BytecodeError("LOG10 is undefined for a non-positive operand")
        return math.log10(a)
    if opcode == EXP:
        try:
            return math.exp(a)
        except OverflowError as exc:
            raise BytecodeError(f"EXP overflowed: {exc}") from exc
    if opcode == SIN:
        return math.sin(a)
    if opcode == COS:
        return math.cos(a)
    if opcode == TAN:
        return math.tan(a)
    if opcode == ASIN:
        if not (-1 <= a <= 1):
            raise BytecodeError("ASIN is undefined outside [-1, 1]")
        return math.asin(a)
    if opcode == ACOS:
        if not (-1 <= a <= 1):
            raise BytecodeError("ACOS is undefined outside [-1, 1]")
        return math.acos(a)
    if opcode == ATAN:
        return math.atan(a)
    if opcode == FLOOR:
        return math.floor(a)
    if opcode == CEIL:
        return math.ceil(a)
    if opcode == ROUND:
        return round(a)
    raise BytecodeError(f"unhandled unary opcode {opcode!r}")


def evaluate_domain(instructions: Program, x_start: int, x_end: int,
                     limits: VMLimits = DEFAULT_LIMITS):
    """Evaluate `instructions` for every x in [x_start, x_end], enforcing
    the domain-size and wall-clock limits (Section 6)."""
    n = x_end - x_start + 1
    if n > limits.max_domain_size:
        raise BytecodeError("domain size exceeds the maximum permitted by this implementation")
    deadline = time.monotonic() + limits.max_wall_seconds
    results = []
    for i, x in enumerate(range(x_start, x_end + 1)):
        if i % 1024 == 0 and time.monotonic() > deadline:
            raise BytecodeError("CUSTOM evaluation exceeded the maximum wall-clock time")
        results.append(evaluate(instructions, x, limits))
    return results
