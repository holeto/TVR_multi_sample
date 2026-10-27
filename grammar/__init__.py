"""Grammar of TRANCE transformations and the decoder's masking rules.

- cfg: the context-free grammar, parse trees and leftmost derivations.
- geometry: grid, visibility and overlaps, as in the generator.
- executor: per-member state and the valid steps V.
- decoder: the masked leftmost derivation (GVAE stack decoder's mask).
- data: programs and init_desc of generated data.
"""
from .cfg import K, NOOP, RULES, derivation, parse, program_tokens
from .decoder import MaskedDecoder, MaskError, accepts
from .geometry import Geometry

__all__ = [
    "K",
    "NOOP",
    "RULES",
    "derivation",
    "parse",
    "program_tokens",
    "MaskedDecoder",
    "MaskError",
    "accepts",
    "Geometry",
]
