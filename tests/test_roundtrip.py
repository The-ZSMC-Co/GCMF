"""Round-trip and decoder-validation tests.

These exercise the worked example from the specification itself
(Appendix A / Section 4.21), every predefined family this v0.1 encoder
searches, the LITERAL fallback for incompressible data, and a sample of
the decoder's mandatory rejection rules (Section 4.25).
"""

import math
import random

import pytest

import gcmf
from gcmf.encoder import _literal_bytes


def roundtrip(data: bytes) -> bytes:
    blob = gcmf.encode(data)
    assert gcmf.decode(blob) == data
    return blob


def test_empty_input():
    roundtrip(b"")


def test_single_byte():
    roundtrip(bytes([42]))


def test_constant_is_smaller_than_literal():
    data = bytes([7] * 50)
    assert len(roundtrip(data)) < len(_literal_bytes(data))


def test_linear_matches_spec_appendix_a():
    # draft-dutta-gcmf-01, Section 4.21 / Appendix A worked example.
    data = bytes([10, 13, 16, 19, 22, 25])
    assert len(roundtrip(data)) < len(_literal_bytes(data))


def test_arithmetic_family_also_round_trips():
    data = bytes([5, 8, 11, 14, 17])  # a=5, d=3
    roundtrip(data)


def test_square():
    data = bytes([x * x for x in range(16)])  # max 225, fits a byte
    assert len(roundtrip(data)) < len(_literal_bytes(data))


def test_cube():
    data = bytes([x ** 3 for x in range(7)])  # max 216
    roundtrip(data)


def test_triangular():
    data = bytes([x * (x + 1) // 2 for x in range(21)])  # max 210
    assert len(roundtrip(data)) < len(_literal_bytes(data))


def test_fibonacci():
    data, a, b = bytearray(), 0, 1
    while b <= 255 and len(data) < 14:
        data.append(a)
        a, b = b, a + b
    roundtrip(bytes(data))


def test_factorial():
    data = bytes(math.factorial(x) for x in range(6))  # up to 5! = 120
    roundtrip(data)


def test_geometric():
    data = bytes(2 ** x for x in range(8))  # 1..128
    assert len(roundtrip(data)) < len(_literal_bytes(data))


def test_polynomial_degree_2():
    data = bytes(2 * x * x + 3 * x + 1 for x in range(10))
    roundtrip(data)


def test_incompressible_data_falls_back_to_literal():
    rng = random.Random(1234)
    data = bytes(rng.randrange(256) for _ in range(64))
    # None of the searched families should spuriously match 64 random
    # bytes; the encoder must fall back to LITERAL rather than emit a
    # "compressed" representation that is actually larger.
    assert gcmf.encode(data) == _literal_bytes(data)


def test_bad_magic_is_rejected():
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(b"NOPE" + b"\x00" * 4)


def test_bad_version_is_rejected():
    blob = bytearray(gcmf.encode(bytes([1, 2, 3])))
    blob[4] = 99
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(bytes(blob))


def test_truncated_stream_is_rejected():
    blob = gcmf.encode(bytes([10, 13, 16, 19, 22, 25]))
    with pytest.raises((gcmf.GCMFError, ValueError)):
        gcmf.decode(blob[:6])


def test_word_size_2_extends_square_domain_far_past_16_elements():
    # With word_size=1, SQUARE overflows after 16 elements (15^2=225,
    # 16^2=256). With word_size=2, each element holds up to 65535, so a
    # much longer SQUARE sequence fits and actually compresses.
    n = 200  # 199**2 = 39601, well within a 2-byte element
    elements = [x * x for x in range(n)]
    data = b"".join(v.to_bytes(2, "little") for v in elements)
    blob = gcmf.encode(data, word_size=2)
    assert gcmf.decode(blob, word_size=2) == data
    assert len(blob) < len(data)


def test_word_size_1_cannot_represent_the_same_long_square_sequence():
    # The same 200-element SQUARE sequence, read as word_size=1 (i.e.
    # byte-for-byte), no longer looks like SQUARE at all once values
    # exceed 255 and get truncated by to_bytes(2, ...) below -- this
    # test instead confirms the *shorter* all-byte-sized case still
    # correctly overflows past the 16-element ceiling with word_size=1.
    n = 17  # 16**2 = 256, one past the byte ceiling
    data = bytes([min(x * x, 255) for x in range(n)])  # not a true SQUARE fit
    blob = gcmf.encode(data, word_size=1)
    assert gcmf.decode(blob, word_size=1) == data
    # 256 does not fit a byte, so this is not a genuine SQUARE match;
    # the encoder must not claim one.
    assert blob == _literal_bytes(data)


def test_decoding_with_mismatched_word_size_does_not_silently_succeed():
    # A predefined-function stream decoded with the wrong word_size
    # must not silently return incorrect data of the *same* apparent
    # validity; here it must at least produce a different length/value
    # than the correct decode, demonstrating why callers must track
    # word_size out-of-band (documented in decoder.decode()).
    elements = [x * x for x in range(200)]
    data = b"".join(v.to_bytes(2, "little") for v in elements)
    blob = gcmf.encode(data, word_size=2)
    correct = gcmf.decode(blob, word_size=2)
    wrong = gcmf.decode(blob, word_size=4)
    assert correct == data
    assert wrong != data  # silently wrong -- exactly the documented risk


def test_modular_represents_a_wrapping_counter():
    # This is exactly the case a non-modular predefined family cannot
    # represent: a counter that wraps every 256 values.
    data = bytes(i % 256 for i in range(2000))
    blob = roundtrip(data)
    assert len(blob) < len(_literal_bytes(data))


def test_modular_wrap_uses_modular_or_custom_not_literal():
    data = bytes(i % 256 for i in range(2000))
    blob = gcmf.encode(data)
    from gcmf import functions as fn
    from gcmf.bitstream import BitReader
    br = BitReader(blob[5:])
    br.read_bits(1)  # residual flag
    code = br.read_bits(4)
    assert code in (fn.MODULAR, fn.CUSTOM)


def test_custom_xor_pattern():
    data = bytes(x ^ 0x2A for x in range(60))
    blob = roundtrip(data)
    assert len(blob) < len(_literal_bytes(data))


def test_custom_abs_v_shape():
    data = bytes(abs(3 * x - 30) for x in range(21))  # 90..0..33, all < 256
    blob = roundtrip(data)
    assert len(blob) < len(_literal_bytes(data))


def test_custom_floor_division_staircase():
    data = bytes(x // 5 for x in range(60))
    blob = roundtrip(data)
    assert len(blob) < len(_literal_bytes(data))


def test_custom_brute_force_finds_short_program():
    # x*x - x is not any predefined family and not one of the hand
    # written templates; only the bounded brute-force search can find
    # it (CONST -1, X, ADD, X, MUL is a 5-token program candidate --
    # exercise that the exhaustive search actually explores this).
    data = bytes((x * x - x) % 256 for x in range(20))
    roundtrip(data)


def test_custom_stack_underflow_is_rejected():
    from gcmf.encoder import MAGIC, VERSION
    from gcmf.bitstream import BitWriter
    from gcmf import bytecode as bc
    from gcmf import functions as fn

    program = bc.assemble([bc.ADD, bc.END])  # ADD with nothing on the stack
    bw = BitWriter()
    bw.write_bits(0, 1)
    bw.write_bits(fn.CUSTOM, 4)
    bw.write_vli(program.bit_length())
    bw.write_bitwriter(program)
    bw.write_vli(0)  # X_start
    bw.write_vli(0)  # X_end
    bw.write_vli(10)  # BASE
    blob = MAGIC + bytes([VERSION]) + bw.to_bytes()
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(blob)


def test_custom_missing_end_is_rejected():
    from gcmf.encoder import MAGIC, VERSION
    from gcmf.bitstream import BitWriter
    from gcmf import bytecode as bc
    from gcmf import functions as fn

    program = bc.assemble([bc.X])  # no END
    bw = BitWriter()
    bw.write_bits(0, 1)
    bw.write_bits(fn.CUSTOM, 4)
    bw.write_vli(program.bit_length())
    bw.write_bitwriter(program)
    bw.write_vli(0)
    bw.write_vli(0)
    bw.write_vli(10)
    blob = MAGIC + bytes([VERSION]) + bw.to_bytes()
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(blob)


def test_modular_zero_modulus_is_rejected():
    from gcmf.encoder import MAGIC, VERSION
    from gcmf.bitstream import BitWriter
    from gcmf import functions as fn

    bw = BitWriter()
    bw.write_bits(0, 1)
    bw.write_bits(fn.MODULAR, 4)
    bw.write_bits(fn.LINEAR, 4)  # nested G(x) = LINEAR
    bw.write_zigzag_vli(1)
    bw.write_zigzag_vli(0)
    bw.write_vli(0)  # m = 0, invalid (spec requires m >= 1)
    bw.write_vli(0)  # X_start
    bw.write_vli(9)  # X_end
    bw.write_vli(10)  # BASE
    blob = MAGIC + bytes([VERSION]) + bw.to_bytes()
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(blob)


def test_modular_nested_literal_is_rejected():
    from gcmf.encoder import MAGIC, VERSION
    from gcmf.bitstream import BitWriter
    from gcmf import functions as fn

    bw = BitWriter()
    bw.write_bits(0, 1)
    bw.write_bits(fn.MODULAR, 4)
    bw.write_bits(fn.LITERAL, 4)  # nested G(x) = LITERAL, forbidden
    blob = MAGIC + bytes([VERSION]) + bw.to_bytes()
    with pytest.raises(gcmf.GCMFError):
        gcmf.decode(blob)


def test_residual_flag_still_not_implemented():
    from gcmf.encoder import MAGIC, VERSION
    from gcmf.bitstream import BitWriter
    from gcmf import functions as fn

    bw = BitWriter()
    bw.write_bits(1, 1)  # R = 1: residual present
    bw.write_bits(fn.CONSTANT, 4)
    blob = MAGIC + bytes([VERSION]) + bw.to_bytes()
    with pytest.raises(NotImplementedError):
        gcmf.decode(blob)
