from __future__ import annotations

import hashlib
import math
import re

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{1,}|[0-9]{2,}|[\u4e00-\u9fff]+")


def _tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer((text or "").lower()):
        token = match.group(0).strip()
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            if len(token) <= 2:
                tokens.append(token)
            else:
                tokens.append(token)
                tokens.extend(token[idx : idx + 2] for idx in range(len(token) - 1))
            continue
        tokens.append(token)
    return tokens


class HashingEmbeddingProvider:
    def __init__(self, *, dims: int = 256) -> None:
        self.dims = dims

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dims
        tokens = _tokenize(text)
        if not tokens:
            return vector

        for token in tokens:
            slot = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16) % self.dims
            vector[slot] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]

    def tokenize(self, text: str) -> list[str]:
        return _tokenize(text)
