#!/usr/bin/env python
"""Compatibility entry point for the 82M -> KaiNomos-110M migration.

The implementation lives in :mod:`migrate_vocab` because the tokenizer change
is an inseparable part of this migration. Only tensors whose name, shape, and
role match are copied; added layers and 110M-only mechanisms remain fresh.
"""

from migrate_vocab import (
    build_vocab_embedding,
    main,
    migrate,
    source_layer_map,
)

# Kept for existing imports. Unlike the old implementation, this mapping omits
# added depths instead of cloning earlier layers into them.
build_source_layer_map = source_layer_map

__all__ = [
    "build_source_layer_map",
    "build_vocab_embedding",
    "migrate",
    "source_layer_map",
]


if __name__ == "__main__":
    raise SystemExit(main())
