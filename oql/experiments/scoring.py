"""Shared scoring helpers for experiment scripts."""

from __future__ import annotations

import itertools

from quotient_learning.types import ObservationHistory


def partitions_agree(
    got: dict[ObservationHistory, int],
    truth: dict[ObservationHistory, int],
) -> bool:
    """Exact recovery on the common context set: identical induced equivalence."""
    common = [h for h in got if h in truth]
    for a, b in itertools.combinations(common, 2):
        if (got[a] == got[b]) != (truth[a] == truth[b]):
            return False
    return True


def pair_disagreement_rate(
    got: dict[ObservationHistory, int],
    truth: dict[ObservationHistory, int],
) -> float:
    """Fraction of common context pairs whose same/different label disagrees."""
    common = [h for h in got if h in truth]
    pairs = list(itertools.combinations(common, 2))
    if not pairs:
        return 0.0
    bad = sum(
        1 for a, b in pairs if (got[a] == got[b]) != (truth[a] == truth[b])
    )
    return bad / len(pairs)


def merge_result_rows(csv_path, new_rows: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    """Merge new result rows into an existing CSV by key, replacing matches.

    Prevents a subset invocation (--envs X --seeds 0) from clobbering the
    full committed table; returns the merged row list in stable order
    (existing rows first, new keys appended).
    """
    import csv as _csv
    from pathlib import Path as _Path

    csv_path = _Path(csv_path)
    new_by_key = {tuple(str(r[k]) for k in key_fields): r for r in new_rows}
    merged: list[dict] = []
    if csv_path.exists():
        with csv_path.open() as f:
            for r in _csv.DictReader(f):
                key = tuple(str(r[k]) for k in key_fields)
                merged.append(new_by_key.pop(key, r))
    merged.extend(new_by_key.values())
    return merged
