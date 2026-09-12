"""gcmf: a reference implementation of General-Purpose Compression via
Mathematical Functions (GCMF), as specified in the Internet-Draft
draft-dutta-gcmf-01 <https://datatracker.ietf.org/doc/html/draft-dutta-gcmf-01>.

    >>> import gcmf
    >>> blob = gcmf.encode(bytes([10, 13, 16, 19, 22, 25]))
    >>> gcmf.decode(blob) == bytes([10, 13, 16, 19, 22, 25])
    True

See README.md for the current implementation status and roadmap.
"""

from .decoder import GCMFError, decode
from .encoder import encode

__all__ = ["encode", "decode", "GCMFError"]
__version__ = "0.1.0"
