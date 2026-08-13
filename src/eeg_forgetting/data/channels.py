from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .contracts import DatasetProtocolError, load_yaml


def _token(value: str) -> str:
    return "".join(character for character in value.upper() if character.isalnum())


@dataclass(frozen=True)
class ChannelRegistry:
    aliases: dict[str, tuple[str, ...]]
    montages: dict[str, tuple[str, ...]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ChannelRegistry":
        document = load_yaml(path)
        aliases = {
            str(canonical): tuple(str(alias) for alias in values)
            for canonical, values in dict(document["aliases"]).items()
        }
        montages = {
            str(name): tuple(str(channel) for channel in values)
            for name, values in dict(document["montages"]).items()
        }
        return cls(aliases=aliases, montages=montages)

    def canonicalize(self, name: str) -> str:
        query = _token(name)
        for canonical, aliases in self.aliases.items():
            if query in {_token(canonical), *(_token(alias) for alias in aliases)}:
                return canonical
        return name.upper()

    def indices(self, source_names: Sequence[str], montage: str) -> tuple[int, ...]:
        if montage not in self.montages:
            raise DatasetProtocolError(f"unknown montage: {montage}")
        canonical_source = [self.canonicalize(name) for name in source_names]
        duplicates = {name for name in canonical_source if canonical_source.count(name) > 1}
        if duplicates:
            raise DatasetProtocolError(f"duplicate canonical channels: {sorted(duplicates)}")
        missing = [name for name in self.montages[montage] if name not in canonical_source]
        if missing:
            raise DatasetProtocolError(f"{montage}: missing channels {missing}")
        return tuple(canonical_source.index(name) for name in self.montages[montage])
