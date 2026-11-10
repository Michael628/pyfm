"""Pooled split-phase planning primitives for ``hadrons_lma_new``.

Pure planning — no config or emission imports. The caller (lma_new /
strategy) pools each split entry's run refs in entry order into
:class:`PooledRef` items, each carrying its source count (its entry's
``noise`` — phase sizes count sources, as the cost model does) and its
placement weight (noise × rungs: the CHAIN ladder's length, 1 otherwise).
The planners here fix phase boundaries and subgrid placement:

* :func:`plan_phase_sizes` back-packs phases from the END of the pool:
  the last phase takes whole refs from the back while they fit
  ``subgrid_last_phase_size`` sources, then resident phases of
  ``subgrid_phase_size`` while more than one phase's worth remains, and
  phase 0 takes what is left — so the short phase is FIRST (20 sources
  at 8/16 is [4, 16]; 28 is [4, 8, 16]; 48 is [8, 8, 8, 8, 16]; 7 is
  [7], a pool that fits one last phase being that phase alone).
* :func:`plan_phase_counts` is the count path's back-packing: from the
  back, each phase's target is ``roundup_ranks(ceil(rem/phases_left))``
  sources, whole refs, leaving at least one ref for every remaining
  phase (10 refs in 3 phases on 2 subgrids is [2, 4, 4]).
* :func:`place_greedy` places the heaviest ref first onto the
  least-loaded subgrid, ties to the lowest subgrid index, equal weights
  in pool order — round-robin by pool position under equal weights.
  Loads reset each phase.

A phase never exceeds its size while it holds more than one ref (a ref
larger than the phase size runs alone); boundaries fall on whole refs; a
short phase is allowed.
"""

import typing as t

from pyfm.tasks.hadrons.types import SourceRef


class RefPlacement(t.NamedTuple):
    """Where a ref's tagged modules run: phase index, subgrid index."""

    phase: int
    subgrid: int


class PooledRef(t.NamedTuple):
    """One entry's ref pooled into the program, in entry order.

    ``entry`` is the ref's entry position in ``high_modes_config`` order;
    ``sources`` is the ref's source count (its entry's ``noise``);
    ``weight`` is its greedy-placement weight (noise × rungs).
    """

    entry: int
    ref: SourceRef
    sources: int
    weight: int


class ProgramPlan(t.NamedTuple):
    """A phased pool with per-ref placement.

    ``phases`` is front-to-back (phase 0 first), each the pool items that
    phase runs; ``placement`` maps ``(entry, ref)`` to where that ref's
    tagged modules run.
    """

    phases: t.List[t.List[PooledRef]]
    placement: t.Dict[t.Tuple[int, SourceRef], RefPlacement]


def _ceil_div(value: int, step: int) -> int:
    return -(-value // step)


def _round_up(value: int, step: int) -> int:
    return _ceil_div(value, step) * step


def validate_phase_sizes(phase_size: int, last_size: int, subgrid_ranks: int) -> None:
    """Both sizes must be positive multiples of ``subgrid_ranks``."""
    for name, size in (
        ("subgrid_phase_size", phase_size),
        ("subgrid_last_phase_size", last_size),
    ):
        if size < 1 or size % subgrid_ranks != 0:
            raise ValueError(
                f"{name} ({size}) must be a positive multiple of "
                f"subgrid_ranks ({subgrid_ranks})."
            )


def _take_from_back(
    pool: t.List[int], capacity: int, max_refs: int | None = None
) -> int:
    """Pop whole refs from the back while they fit ``capacity`` sources.

    Always takes at least one ref (a ref larger than ``capacity`` runs
    alone); ``max_refs`` caps how many (the count path's one-ref reserve).
    Returns the number of refs taken.
    """
    taken = taken_refs = 0
    while pool and (max_refs is None or taken_refs < max_refs):
        if taken_refs and taken + pool[-1] > capacity:
            break
        taken += pool.pop()
        taken_refs += 1
    return taken_refs


def plan_phase_sizes(
    sources: t.Sequence[int], phase_size: int, last_size: int
) -> t.List[int]:
    """Refs per phase, front to back, back-packed from the end.

    The last phase takes whole refs from the back up to ``last_size``
    sources; then resident phases of ``phase_size`` while more than one
    phase's worth of sources remains; phase 0 takes what is left (the
    short phase, first). A pool that fits one last phase is that phase
    alone (no empty phase 0). Empty pool -> no phases.
    """
    pool = list(sources)
    back: t.List[int] = []
    capacity = last_size
    while pool:
        back.append(_take_from_back(pool, capacity))
        if not pool:
            return back[::-1]
        if sum(pool) <= phase_size:
            return [len(pool), *back[::-1]]
        capacity = phase_size
    return []


def plan_phase_counts(
    sources: t.Sequence[int], n_phases: int, subgrid_ranks: int
) -> t.List[int]:
    """Refs per phase for exactly ``n_phases`` phases, back-packed.

    From the back, each phase's target is
    ``roundup_ranks(ceil(remaining_sources / phases_left))`` capped at
    the remaining sources; whole refs are taken while they fit and while
    at least one ref stays available for every remaining phase (phase 0
    included). The short phase is first.
    """
    if not 1 <= n_phases <= len(sources):
        raise ValueError(
            f"subgrid_phases ({n_phases}) exceeds the pooled source refs "
            f"({len(sources)}); every phase needs at least one ref."
        )
    pool = list(sources)
    back: t.List[int] = []
    for phases_left in range(n_phases, 1, -1):
        remaining = sum(pool)
        target = min(
            _round_up(_ceil_div(remaining, phases_left), subgrid_ranks), remaining
        )
        back.append(
            _take_from_back(pool, target, max_refs=len(pool) - phases_left + 1)
        )
    back.append(len(pool))
    return back[::-1]


def place_greedy(weights: t.Sequence[int], subgrid_ranks: int) -> t.List[int]:
    """Subgrid per item (parallel to ``weights``): heaviest first onto
    the least-loaded subgrid, ties to the lowest subgrid index, equal
    weights in pool order (round-robin under equal weights).
    """
    loads = [0] * subgrid_ranks
    out = [0] * len(weights)
    for i in sorted(range(len(weights)), key=lambda i: (-weights[i], i)):
        subgrid = min(range(subgrid_ranks), key=lambda s: (loads[s], s))
        out[i] = subgrid
        loads[subgrid] += weights[i]
    return out


def plan_program(
    pool: t.Sequence[PooledRef],
    *,
    subgrid_ranks: int,
    n_phases: int | None = None,
    phase_size: int | None = None,
    last_size: int | None = None,
) -> ProgramPlan:
    """Phase the pooled refs and place them: the program's layout.

    Exactly one of ``n_phases`` (count path) and ``phase_size`` (size
    path) must be set; ``last_size`` defaults to ``phase_size``. Sizes
    are validated against ``subgrid_ranks`` here as well (defense in
    depth — the config layer validates earlier with the same helper).
    """
    if (n_phases is None) == (phase_size is None):
        raise ValueError(
            "plan_program needs exactly one of n_phases and phase_size."
        )
    if phase_size is not None:
        effective_last = last_size if last_size is not None else phase_size
        validate_phase_sizes(phase_size, effective_last, subgrid_ranks)
        counts = plan_phase_sizes(
            [p.sources for p in pool], phase_size, effective_last
        )
    else:
        counts = plan_phase_counts([p.sources for p in pool], n_phases, subgrid_ranks)
    phases: t.List[t.List[PooledRef]] = []
    placement: t.Dict[t.Tuple[int, SourceRef], RefPlacement] = {}
    start = 0
    for phase, count in enumerate(counts):
        chunk = list(pool[start : start + count])
        start += count
        subgrids = place_greedy([p.weight for p in chunk], subgrid_ranks)
        phases.append(chunk)
        for item, subgrid in zip(chunk, subgrids):
            placement[(item.entry, item.ref)] = RefPlacement(phase, subgrid)
    return ProgramPlan(phases=phases, placement=placement)
