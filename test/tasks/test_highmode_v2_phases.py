"""Tests for highmode_v2/phases.py — pooled back-packing (sizes and
count), greedy placement, and the composed program plan."""

import pytest

from pyfm.tasks.hadrons.types import SourceRef
from pyfm.tasks.hadrons.highmode_v2.phases import (
    PooledRef,
    place_greedy,
    plan_phase_counts,
    plan_phase_sizes,
    plan_program,
    validate_phase_sizes,
)


def ref_pool(sources, weights=None, entry=0):
    """A pool of grid refs at t0 = 0..n-1 with per-ref source counts."""
    weights = sources if weights is None else weights
    return [
        PooledRef(
            entry=entry,
            ref=SourceRef(label=f"t{t}", axis=str(t), t0=t),
            sources=s,
            weight=w,
        )
        for t, (s, w) in enumerate(zip(sources, weights))
    ]


class TestPlanPhaseSizes:
    @pytest.mark.parametrize(
        "total, phase_size, last_size, want",
        [
            (20, 8, 16, [4, 16]),
            (28, 8, 16, [4, 8, 16]),
            (48, 8, 16, [8, 8, 8, 8, 16]),
            (7, 8, 16, [7]),  # fits the last phase alone
            (16, 8, 8, [8, 8]),  # ph-b8i: two resident-sized phases
            (8, 8, 16, [8]),  # one phase after the pack is freed
        ],
    )
    def test_handoff_examples(self, total, phase_size, last_size, want):
        assert plan_phase_sizes([1] * total, phase_size, last_size) == want

    def test_boundaries_fall_on_whole_refs(self):
        # a 2-source ref never splits: the last phase takes three refs
        # (6 sources), phase 0 the fourth
        assert plan_phase_sizes([2, 2, 2, 2], 6, 6) == [1, 3]

    def test_mixed_noise_last_phase_boundary(self):
        # fused-48: 20 hh refs of 2 sources, then 8 bias refs of 1 — the
        # last phase takes the 8 bias + 4 hh refs (16 sources exactly)
        assert plan_phase_sizes([2] * 20 + [1] * 8, 8, 16) == [4, 4, 4, 4, 12]

    def test_oversized_ref_runs_alone(self):
        # a ref larger than the phase size occupies a phase alone; the
        # loop never stalls and never splits a ref
        assert plan_phase_sizes([4, 1], 2, 2) == [1, 1]

    def test_empty_pool_is_no_phases(self):
        assert plan_phase_sizes([], 8, 8) == []


class TestPlanPhaseCounts:
    def test_short_phase_first(self):
        # 10 refs, 3 phases, 2 subgrids: [2, 4, 4] (back-packed; the old
        # front-to-back math gave [4, 4, 2])
        assert plan_phase_counts([1] * 10, 3, 2) == [2, 4, 4]

    def test_n96_shapes(self):
        # 96 sources as 48 two-noise refs: 6 phases of 16, 12 of 8
        assert plan_phase_counts([2] * 48, 6, 4) == [8] * 6
        assert plan_phase_counts([2] * 48, 12, 4) == [4] * 12

    def test_every_phase_keeps_one_ref(self):
        assert plan_phase_counts([1, 1], 2, 4) == [1, 1]

    def test_too_many_phases_rejected(self):
        with pytest.raises(ValueError, match="exceeds the pooled source refs"):
            plan_phase_counts([1, 1], 3, 2)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError):
            plan_phase_counts([1], 0, 2)


class TestPlaceGreedy:
    def test_equal_weights_round_robin(self):
        assert place_greedy([1] * 8, 4) == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_heaviest_first_least_loaded(self):
        # w3 -> sg0; w2 -> sg1; w1 -> sg1 (loads 3, 2)
        assert place_greedy([3, 2, 1], 2) == [0, 1, 1]

    def test_ties_lowest_subgrid(self):
        assert place_greedy([5, 5], 3) == [0, 1]

    def test_o_tier_bias_alone(self):
        # 6 sloppy refs (w1) then 1 chained bias ref (w2): the bias lands
        # alone on subgrid 0, the sloppy two each on 1-3
        out = place_greedy([1] * 6 + [2], 4)
        assert out[6] == 0
        assert out[:6] == [1, 2, 3, 1, 2, 3]


class TestValidatePhaseSizes:
    def test_multiples_accepted(self):
        validate_phase_sizes(8, 16, 4)

    def test_non_multiple_rejected(self):
        with pytest.raises(ValueError, match="subgrid_phase_size"):
            validate_phase_sizes(9, 16, 4)

    def test_last_non_multiple_rejected(self):
        with pytest.raises(ValueError, match="subgrid_last_phase_size"):
            validate_phase_sizes(8, 10, 4)

    def test_non_positive_rejected(self):
        with pytest.raises(ValueError):
            validate_phase_sizes(0, 8, 4)


class TestPlanProgram:
    def _fused_pool(self):
        hh = [
            PooledRef(
                0, SourceRef(f"t{2 * t}", str(2 * t), 2 * t), sources=2, weight=2
            )
            for t in range(20)
        ]
        hbias = [
            PooledRef(
                1, SourceRef(f"n{i}", f"n{i}", (7 * i) % 48), sources=1, weight=2
            )
            for i in range(8)
        ]
        return hh + hbias

    def test_fused_48_layout_and_placement(self):
        plan = plan_program(
            self._fused_pool(), subgrid_ranks=4, phase_size=8, last_size=16
        )
        assert [len(p) for p in plan.phases] == [4, 4, 4, 4, 12]
        assert [sum(i.sources for i in p) for p in plan.phases] == [8, 8, 8, 8, 16]
        last = plan.phases[-1]
        per_subgrid = [0, 0, 0, 0]
        for item in last:
            per_subgrid[plan.placement[(item.entry, item.ref)].subgrid] += 1
        assert per_subgrid == [3, 3, 3, 3]
        # the bias refs all land in the last phase
        assert all(
            plan.placement[(1, r)].phase == 4
            for r in [i.ref for i in self._fused_pool()[20:]]
        )

    def test_ph_b8i_tags(self):
        # 8 slices x noise 2, size 8 (last defaults): [4, 4] refs; tags
        # t0 % 4, phase t0 // 4
        pool = [PooledRef(0, SourceRef(f"t{t}", str(t), t), 2, 2) for t in range(8)]
        plan = plan_program(pool, subgrid_ranks=4, phase_size=8)
        assert [len(p) for p in plan.phases] == [4, 4]
        for item in [i for p in plan.phases for i in p]:
            pl = plan.placement[(item.entry, item.ref)]
            assert pl.subgrid == item.ref.t0 % 4
            assert pl.phase == item.ref.t0 // 4

    def test_count_path_phases_the_pool(self):
        plan = plan_program(ref_pool([1] * 10), subgrid_ranks=2, n_phases=3)
        assert [len(p) for p in plan.phases] == [2, 4, 4]

    def test_o_tier_one_phase(self):
        # 6 sloppy (1 source, w1) + 1 chained bias (1 source, w2): ONE
        # phase; bias alone on its subgrid, sloppy two each on the others
        pool = [
            PooledRef(0, SourceRef(f"t{8 * t}", str(8 * t), 8 * t), 1, 1)
            for t in range(6)
        ]
        pool += [PooledRef(1, SourceRef("n0", "n0", 13), 1, 2)]
        plan = plan_program(pool, subgrid_ranks=4, phase_size=8)
        assert len(plan.phases) == 1
        bias = plan.placement[(1, pool[-1].ref)]
        assert bias.phase == 0
        sloppy = [plan.placement[(0, r.ref)] for r in pool[:6]]
        assert {p.subgrid for p in sloppy} == {1, 2, 3}
        assert bias.subgrid not in {p.subgrid for p in sloppy}

    def test_exclusivity_enforced(self):
        pool = ref_pool([1])
        with pytest.raises(ValueError, match="exactly one"):
            plan_program(pool, subgrid_ranks=4, n_phases=2, phase_size=4)
        with pytest.raises(ValueError, match="exactly one"):
            plan_program(pool, subgrid_ranks=4)

    def test_sizes_validated(self):
        with pytest.raises(ValueError, match="subgrid_phase_size"):
            plan_program(
                ref_pool([1] * 12), subgrid_ranks=4, phase_size=6, last_size=12
            )
