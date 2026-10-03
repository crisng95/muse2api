"""Dependency-free token estimate, used for both the ``usage`` block and billing.

Each CJK ideograph, kana or hangul character counts as one token; the rest of the
text counts as one token per four characters (rounded up).
"""

from __future__ import annotations

import math
import re

_WIDE = re.compile(
    "[ᄀ-ᇿ぀-ヿ㄰-㆏ㇰ-ㇿ㐀-䶿一-鿿"
    "가-힯豈-﫿ｦ-ﾟ\U00020000-\U0003ffff]"
)


def count_tokens(text: str) -> int:
    if not text:
        return 0
    wide = len(_WIDE.findall(text))
    return max(1, wide + math.ceil((len(text) - wide) / 4))
