from gcmf.bitstream import BitReader, BitWriter, zigzag_decode, zigzag_encode


def test_zigzag_round_trip():
    for n in [0, 1, -1, 2, -2, 12345, -12345, 0x7FFFFFFF, -0x7FFFFFFF]:
        assert zigzag_decode(zigzag_encode(n)) == n


def test_vli_round_trip_various_magnitudes():
    for value in [0, 1, 127, 128, 16383, 16384, 2**32, 2**64]:
        bw = BitWriter()
        bw.write_vli(value)
        br = BitReader(bw.to_bytes())
        assert br.read_vli() == value


def test_bit_packing_is_msb_first_and_non_byte_aligned():
    bw = BitWriter()
    bw.write_bits(0b1, 1)
    bw.write_bits(0b0110, 4)
    data = bw.to_bytes()
    assert data == bytes([0b10110000])  # 5 bits set, 3 zero-padding bits


def test_align_to_byte_pads_with_zero_bits():
    bw = BitWriter()
    bw.write_bits(0b10110, 5)
    pad = bw.align_to_byte()
    assert pad == 3
    assert bw.bit_length() == 8
