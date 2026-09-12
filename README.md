# gcmf

A reference encoder/decoder for **General-Purpose Compression via
Mathematical Functions (GCMF)**, as specified in the Internet-Draft
[`draft-dutta-gcmf-01`](https://datatracker.ietf.org/doc/html/draft-dutta-gcmf-01).

Instead of storing a sequence of bytes directly, GCMF searches for a
compact mathematical function (linear, polynomial, geometric,
Fibonacci, and other predefined forms) that reproduces the sequence
exactly, storing only the function's parameters and a small domain
descriptor. When no function reproduces the data exactly and
compactly, the format falls back to storing the original bytes
verbatim (`LITERAL`), so encoded output is never more than a few bytes
larger than the input.

## Install

```bash
pip install -e .
# or, to also install test dependencies:
pip install -e ".[test]"
```

## Usage

```python
import gcmf

blob = gcmf.encode(bytes([10, 13, 16, 19, 22, 25]))
assert gcmf.decode(blob) == bytes([10, 13, 16, 19, 22, 25])
print(len(blob))  # smaller than the 6-byte input
```

Command line:

```bash
python -m gcmf encode input.bin output.gcmf
python -m gcmf decode output.gcmf roundtrip.bin
```

### Element width (`word_size`)

By default each sequence element is one byte (0–255). Families whose
output grows quickly — `SQUARE`, `CUBE`, `FACTORIAL`, `FIBONACCI`,
`TRIANGULAR`, `GEOMETRIC` — therefore only fit a handful of elements
before a single byte can no longer hold the result (e.g. `SQUARE` tops
out at 16 elements, since 16² = 256). This is a property of how this
library chunks a file into elements, not a limitation of the GCMF
format itself — the specification defines an abstract integer
sequence and does not mandate 1-byte elements.

Pass `word_size=2/4/8` to interpret the input as wider little-endian
elements instead:

```python
elements = [x * x for x in range(65536)]
data = b"".join(v.to_bytes(4, "little") for v in elements)
blob = gcmf.encode(data, word_size=4)     # 262144 bytes -> 11 bytes
assert gcmf.decode(blob, word_size=4) == data
```

**`word_size` must be supplied identically to `encode()` and
`decode()`.** It is a caller-side convention, not a field in the GCMF
stream itself (the specification doesn't define one), so a
mismatched `word_size` at decode time will not raise an error — it
will silently unpack a predefined-function stream at the wrong width
and return incorrect data of the wrong length. `LITERAL` streams are
unaffected, since they always store the original bytes verbatim.
Applications embedding `gcmf` are responsible for tracking the chosen
`word_size` out-of-band (a file extension, a wrapping container
header, or a fixed convention).

## Implementation status

This implements the full binary format for all three representation
kinds the specification defines -- LITERAL, PREDEFINED, and CUSTOM --
plus MODULAR:

- The GCMF-VLI variable-length integer format and ZigZag signed mapping
- Bit-level (non-byte-aligned) reading and writing
- The `LITERAL` fallback, including its 3-bit alignment padding
- All predefined functions except LOGARITHMIC/POWER search (decode is
  supported for forward compatibility; see below): `CONSTANT`,
  `LINEAR`, `ARITHMETIC`, `GEOMETRIC`, `EXPONENTIAL`, `POLYNOMIAL`
  (degree 2–3, solved via exact rational Gaussian elimination),
  `FIBONACCI`, `FACTORIAL`, `TRIANGULAR`, `SQUARE`, `CUBE`
- **`CUSTOM`**: a full 32-opcode stack-machine bytecode assembler and
  evaluator (`bytecode.py`), with the security limits Section 6 calls
  for (bounded stack depth, numerical magnitude, `POW` exponent,
  domain size, and wall-clock time). The encoder does not require a
  user to type a formula: it searches for a fit automatically, via (a)
  a library of parameterized program *shapes* (XOR mask, power-of-two
  AND mask, absolute value of an affine function, clamped affine, and
  integer floor-division), each solved for its own small parameter set
  the same way a predefined family is fit, and (b) for inputs of at
  most 64 elements, a bounded *exhaustive* search over every valid
  straight-line program up to 4 instructions from a restricted
  opcode/constant alphabet -- genuine automated program synthesis
  within a documented, bounded search space (Section 3's C_I), not a
  heuristic guess and not an operator-supplied formula.
- **`MODULAR`** (`F(x) = G(x) mod m`): searched for an affine
  `G(x) = ax + b` against a fixed set of candidate moduli (every power
  of two up to 2³², plus the data's own range). This is what
  represents periodic/wrapping data -- e.g. a counter or timestamp
  column that resets every 256 values -- which no non-modular
  predefined family can express. The nested `G(x)` may itself be
  PREDEFINED, CUSTOM, or (recursively) MODULAR, up to a bounded
  nesting depth; Section 4.9.1 doesn't give an explicit bit layout for
  this nesting the way Section 4.17 does for residuals, so this
  implementation mirrors that section's design (a bare 4-bit function
  type, no residual) -- documented as an interpretation choice in
  `decoder.py`.
- A subset of the decoder's mandatory rejection rules (bad magic,
  unsupported version, reserved LITERAL+residual combination, `X_end <
  X_start`, `BASE < 2`, truncated streams, stack underflow, missing
  `END`, invalid stack termination, undefined bytecode operations,
  out-of-range evaluations, invalid `MODULAR` modulus/nesting)

### Roadmap (not yet implemented)

- Residual (near-fit) encoding, i.e. `R = 1` streams
- Encoder search over `LOGARITHMIC` and `POWER` as predefined
  functions (exact integer fits for these are rare over a
  consecutive-integer domain, so the encoder does not currently spend
  search time on them; streams from other encoders using them still
  decode correctly)
- `CUSTOM`/brute-force search parameters (max length, constant range,
  element-count ceiling) are fixed constants in `encoder.py`; making
  them configurable, and widening the brute-force search, are natural
  follow-ups

BASE (Section 4.14) is intentionally *not* searched: Sections 4.14.3
and 4.15.5 state that BASE does not affect the binary encoding of any
field, and explicitly prohibit using it as a selection criterion when
it has no effect on encoded length -- which, given this
implementation's field encodings, is always the case.

Contributions implementing any of the above are welcome.

## Testing

```bash
pytest
```

The test suite includes the specification's own worked example
(`[10, 13, 16, 19, 22, 25]` → `LINEAR`, Section 4.21/Appendix A), a
round-trip test for every implemented predefined family, a
`MODULAR`-represented wrapping counter (2,000 bytes → 16 bytes), CUSTOM
template matches (XOR mask, absolute-value V-shape, floor-division
staircase), a case only the bounded brute-force search finds, a
fallback test confirming random/incompressible data is correctly
stored as `LITERAL`, and tests for the decoder's mandatory rejection
rules -- including CUSTOM stack underflow, a missing `END`, and an
invalid `MODULAR` modulus/nested-LITERAL.

## License

MIT — see [LICENSE](LICENSE).
