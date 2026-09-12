# Changelog

## Unreleased

- Implemented `CUSTOM` bytecode functions in full: all 32 opcodes
  (Appendix Table 2), a stack-machine evaluator with Section 6's
  security limits (stack depth, numerical magnitude, `POW` exponent,
  domain size, wall-clock time), and an automated encoder-side search
  (parameterized program-shape templates plus a bounded exhaustive
  search over short programs) -- not an operator-supplied formula.
- Implemented `MODULAR` functions (`F(x) = G(x) mod m`), including
  nested `PREDEFINED`/`CUSTOM`/(recursive) `MODULAR` bodies, with an
  encoder search for affine `G(x) mod m` fits (the representation for
  periodic/wrapping data, e.g. a counter that resets every 256 values).
- Documented and left unimplemented: searching BASE, since Sections
  4.14.3/4.15.5 state it never affects encoded length in this
  implementation and explicitly prohibit using it as a selection
  criterion in that case.
- Added a `word_size` parameter (1/2/4/8 bytes, little-endian) to
  `encode()`/`decode()` and the CLI, so growth-based families
  (`SQUARE`, `CUBE`, `FACTORIAL`, `FIBONACCI`, `TRIANGULAR`,
  `GEOMETRIC`) are no longer artificially capped at the handful of
  elements that fit in a single byte. This is a library-side chunking
  choice, not a wire-format change; `word_size` is not recorded in the
  stream and must be tracked out-of-band by the caller (documented in
  `decoder.decode()` and the README).

## 0.1.0 — Initial release

- GCMF-VLI and ZigZag primitives; MSB-first bit-level reader/writer
- `LITERAL` fallback encode/decode
- Encoder search + decoder support for `CONSTANT`, `LINEAR`,
  `ARITHMETIC`, `GEOMETRIC`, `EXPONENTIAL`, `POLYNOMIAL` (degree 2–3),
  `FIBONACCI`, `FACTORIAL`, `TRIANGULAR`, `SQUARE`, `CUBE`
- Decode-only support for `LOGARITHMIC` and `POWER` parameter codecs
- CLI (`python -m gcmf encode|decode`)
- Test suite covering the specification's worked example, round-trips
  for every implemented family, LITERAL fallback correctness, and
  several mandatory decoder rejection rules
- Not yet implemented: `CUSTOM` bytecode, `MODULAR`, residual encoding
