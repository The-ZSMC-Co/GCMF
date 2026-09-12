"""Command-line interface: `python -m gcmf encode|decode <input> <output>`."""

from __future__ import annotations

import argparse
import sys

from . import GCMFError, decode, encode


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gcmf", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    enc = sub.add_parser("encode", help="encode a file to GCMF")
    enc.add_argument("input")
    enc.add_argument("output")
    enc.add_argument("--word-size", type=int, choices=(1, 2, 4, 8), default=1,
                      help="bytes per sequence element (little-endian); "
                           "must match the --word-size used to decode (default: 1)")

    dec = sub.add_parser("decode", help="decode a GCMF file")
    dec.add_argument("input")
    dec.add_argument("output")
    dec.add_argument("--word-size", type=int, choices=(1, 2, 4, 8), default=1,
                      help="must match the --word-size used to encode (default: 1)")

    args = parser.parse_args(argv)

    with open(args.input, "rb") as handle:
        data = handle.read()

    try:
        if args.command == "encode":
            result, action = encode(data, word_size=args.word_size), "Encoded"
        else:
            result, action = decode(data, word_size=args.word_size), "Decoded"
    except (GCMFError, NotImplementedError) as exc:
        print(f"gcmf: error: {exc}", file=sys.stderr)
        return 1

    with open(args.output, "wb") as handle:
        handle.write(result)

    print(f"{action} {len(data)} bytes -> {len(result)} bytes ({args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
