"""Bit-level I/O primitives for the GCMF binary format.

GCMF (draft-dutta-gcmf-01) packs fields at the bit level, most-significant
bit first, and several fields (the 5-bit Function Address, GCMF-VLI
integers) are not byte-aligned (Section 4.5 of the specification). This
module implements the three primitives the rest of the codec is built on:

  * BitWriter / BitReader   -- MSB-first bit packing/unpacking
  * write_vli / read_vli    -- GCMF-VLI variable-length integers (Section 4.6)
  * zigzag_encode / decode  -- ZigZag mapping used for signed parameters
                                before GCMF-VLI encoding (Section 4.7)
"""

from __future__ import annotations


class BitWriter:
    """Accumulates individual bits (MSB-first) and packs them into bytes."""

    def __init__(self) -> None:
        self._bits: list[int] = []

    def write_bits(self, value: int, nbits: int) -> None:
        if nbits <= 0:
            return
        if value < 0 or value >= (1 << nbits):
            raise ValueError(f"value {value} does not fit in {nbits} bits")
        for i in range(nbits - 1, -1, -1):
            self._bits.append((value >> i) & 1)

    def write_vli(self, value: int) -> None:
        """Write an unsigned integer using GCMF-VLI (Section 4.6)."""
        groups = _vli_groups(value)
        last = len(groups) - 1
        for i, group in enumerate(groups):
            self.write_bits(1 if i < last else 0, 1)  # continuation bit C
            self.write_bits(group, 7)                  # 7-bit payload

    def write_zigzag_vli(self, value: int) -> None:
        """Write a signed integer using ZigZag + GCMF-VLI (Section 4.7)."""
        self.write_vli(zigzag_encode(value))

    def align_to_byte(self) -> int:
        """Pad with zero bits until the buffer length is a multiple of 8.

        Returns the number of padding bits written. Used for LITERAL's
        mandatory 3-bit alignment padding (Section 4.18); most other GCMF
        fields are *not* required to be byte-aligned (Section 4.5).
        """
        pad = (-len(self._bits)) % 8
        self._bits.extend([0] * pad)
        return pad

    def write_bitwriter(self, other: "BitWriter") -> None:
        """Append another BitWriter's bits directly (no re-alignment).
        Used to splice an assembled CUSTOM bytecode body into the
        enclosing candidate's bitstream."""
        self._bits.extend(other._bits)

    def bit_length(self) -> int:
        return len(self._bits)

    def to_bytes(self) -> bytes:
        pad = (-len(self._bits)) % 8
        bits = self._bits + [0] * pad
        out = bytearray(len(bits) // 8)
        for i, bit in enumerate(bits):
            if bit:
                out[i // 8] |= 1 << (7 - (i % 8))
        return bytes(out)


class BitReader:
    """Reads individual bits (MSB-first) out of a byte string."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._pos = 0
        self._nbits = len(data) * 8

    def remaining_bits(self) -> int:
        return self._nbits - self._pos

    def read_bits(self, nbits: int) -> int:
        if nbits <= 0:
            return 0
        if self._pos + nbits > self._nbits:
            raise ValueError("truncated GCMF stream: not enough bits remaining")
        value = 0
        for _ in range(nbits):
            byte = self._data[self._pos // 8]
            bit = (byte >> (7 - (self._pos % 8))) & 1
            value = (value << 1) | bit
            self._pos += 1
        return value

    def read_vli(self) -> int:
        """Read an unsigned GCMF-VLI integer (Section 4.6). Rejects a
        stream that runs out of bits before a terminating unit (C=0)."""
        value = 0
        while True:
            cont = self.read_bits(1)
            group = self.read_bits(7)
            value = (value << 7) | group
            if cont == 0:
                return value

    def read_zigzag_vli(self) -> int:
        return zigzag_decode(self.read_vli())

    def align_to_byte_expect_zero(self, nbits: int) -> None:
        """Consume `nbits` padding bits, raising if any are non-zero
        (Section 4.5: "Padding bits MUST be zero")."""
        for _ in range(nbits):
            if self.read_bits(1) != 0:
                raise ValueError("non-zero padding bit where zero padding is required")

    def byte_position(self) -> int:
        if self._pos % 8 != 0:
            raise ValueError("reader is not currently at a byte boundary")
        return self._pos // 8

    def remaining_bytes(self) -> bytes:
        return self._data[self.byte_position():]


def _vli_groups(value: int) -> list[int]:
    if value < 0:
        raise ValueError("GCMF-VLI encodes unsigned integers only; use zigzag for signed values")
    groups = [value & 0x7F]
    value >>= 7
    while value:
        groups.append(value & 0x7F)
        value >>= 7
    groups.reverse()  # GCMF-VLI is big-endian: most-significant group first
    return groups


def zigzag_encode(n: int) -> int:
    """Z(n) = 2n for n >= 0, -2n-1 for n < 0 (Section 4.7)."""
    return (n << 1) if n >= 0 else (-(n << 1) - 1)


def zigzag_decode(z: int) -> int:
    return (z >> 1) if (z & 1) == 0 else -((z + 1) >> 1)
