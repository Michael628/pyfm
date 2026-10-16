"""QED block-access and derivation layer for the connected four-point
contraction (``contract_qed``).

The producer ``hadrons_qed_mf`` stores its noise structure INVERTED
relative to ``hadrons_sib_mf``: the PHOTON axis splits files (one module
per sample with ``nEmFields=1``, the ``{em_index}`` stem token plus the
per-sample seed baked into the ``{file_key}`` ext), while ALL
``N = qed.batch.noise`` fermion-noise worlds live INSIDE every n-legged
file's matrix axes —

- the ``noise_fv_vec`` left axis is ``3N`` (source-major: entry ``3*n+c``
  is world ``n``, color ``c``; upstream ``DilutedNoise.hpp`` getFerm);
- the h-solve ket axis is ``3N*nt`` (world-MAJOR blocks of ``3*nt``;
  within a world, slice-major ``3*t+c`` — the upstream RandomWall column
  order ``3*(i*nSlices+j)+k``);

so fermion-world selection is axis slicing here, not file selection
(D13). The tab is a single ``[nt, 2*n_eig, 3N]`` file (ket noise-major,
3 columns per world — exactly what the producer's precon consumes).

Derivations port the validated sib pair-basis identity
(``pyfm/a2a/sib_derive.py``; <= 5.5e-8 against the HadronsMILC harness):
``lp``/``np`` are exact offline arithmetic from ``ll``/``nl``/``tab``/
evals per fermion world; ``pure_lh = lh - lp`` and ``pure_nh = nh - np``
strip the deflation subspace the h solves deliberately leave in. Ket-side
eval folding follows the writer convention (``raw = stored *
eval(defl)``; ``block(m) = raw / eval(m)``); all mass dependence is
``w_rows(m)`` or the per-column refold.

j1l vertices sum their four J_kA_k component files inline (D2) — every
read of a j1l block loads all four channels and sums (the vertex operator
IS Σ_k J_kA_k). Everything is cached per content-affecting axis tuple
(the d38ef6c lesson): file-granular keys for the stored reads,
result-granular keys for derived/refolded blocks.
"""

import typing as t

import h5py
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm.a2a.sib_derive import (
    SCALAR_GAMMA,
    eval_pairs,
    ket_refold,
    load_evals,
    w_rows,
)
from pyfm.domain import MassDict, Outfile
from pyfm.domain.blockkeys import (
    EM_FAMILY_JA,
    EM_FAMILY_J1L,
    em_components,
    em_file_key,
    em_index_suffix,
    mass_token,
    vec_file_key,
)

EM_FAMILIES = frozenset({EM_FAMILY_JA, EM_FAMILY_J1L})


def load_block(path: str, dataset: str) -> xp.ndarray:
    """Load one block file as complex128 ``[nt, N_left, N_right]``.

    ``dataset`` is the full HDF5 group name: the filename stem (the
    ``file_key``) for qed block files, ``G1_G1_0_0_0`` for the tab — one
    uniform reader keyed on the producer's grammar (the
    ``_compare_h5_file`` convention).
    """
    with h5py.File(path, "r") as f:
        if dataset not in f or "a2aMatrix" not in f[dataset]:
            raise ValueError(
                f"dataset {dataset!r}/a2aMatrix not found in {path!r}"
            )
        d = f[dataset]["a2aMatrix"][()]
    return xp.asarray((d["re"] + 1j * d["im"]).astype(np.complex128))


def _lp_np_product(raw: xp.ndarray, wt: xp.ndarray) -> xp.ndarray:
    """Concatenate ``raw[t] @ wt[t']`` over lattice times (col order 3t+c).

    Port of ``SIBBlockAccessor._lp_np_product`` (sib_derive) — identical
    math; kept local so each accessor layer imports independently. Both
    are pinned by synthetic-surface tests.
    """
    return xp.stack(
        [
            xp.concatenate(
                [xp.matmul(raw[t], wt[s]) for s in range(wt.shape[0])],
                axis=-1,
            )
            for t in range(raw.shape[0])
        ],
        axis=0,
    )


class QEDBlockAccessor:
    """Load-or-derive accessor over the hadrons_qed_mf block surface.

    Loads the emitted blocks (``ll``/``nl`` at defl_mass, ``lh``/``nh``
    per operations mass, per photon world for EM families), slices
    fermion worlds from the in-file matrix axes, sums j1l component
    channels inline, derives the removed ``lp``/``np`` pairs via the sib
    pair-basis identity, refolds eig-ket sides between masses, and
    exposes ``pure_lh = lh - lp`` / ``pure_nh = nh - np`` per world.

    ``blocks`` is the producer's block Outfile with
    ``{family}/{leg_pair}/{mass}/{em_index}`` stem tokens and the
    ``{file_key}`` ext intact (everything else — {series}/{cfg}/{eigs}/
    {noise} — must already be resolved); ``tab`` the producer's tab label
    (``{gamma}`` ext); ``evalfile`` fully resolved. ``em_seeds`` are the
    pre-expanded per-world photon seed strings (D10) — the EM file keys
    bake them in. Path formatting raises loudly on any remaining brace.
    """

    def __init__(
        self,
        *,
        blocks: Outfile,
        tab: Outfile,
        evalfile: Outfile,
        mass: MassDict,
        defl_mass: str,
        noise: int,
        em_seeds: t.Tuple[str, ...],
    ) -> None:
        self.blocks = blocks
        self.tab_outfile = tab
        self.evalfile = evalfile
        self.mass = mass
        self.defl_mass = defl_mass
        self.noise = noise
        self.em_seeds = tuple(em_seeds)
        if self.noise < 1:
            raise ValueError(f"noise must be >= 1; got {self.noise}")
        self._cache: t.Dict[t.Tuple, xp.ndarray] = {}
        self._lam: xp.ndarray | None = None
        self._nt: int | None = None

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def nt(self) -> int:
        """Lattice time extent, from the tab's output-time axis."""
        if self._nt is None:
            self._nt = int(self.tab().shape[0])
        return self._nt

    def cols(self, block: xp.ndarray, t_pin: int) -> xp.ndarray:
        """Time-pinned column selection of a per-world h/p block.

        The last axis is one world's ``3*nt`` columns (slice-major
        ``3*t + c``); view as ``(nt, 3)`` and take time ``t_pin``,
        giving ``[..., 3]``. Width and bounds verified loudly.
        """
        nt = self.nt()
        if not 0 <= t_pin < nt:
            raise ValueError(f"pinned time {t_pin} outside [0, {nt})")
        if block.shape[-1] != nt * 3:
            raise ValueError(
                f"block last axis {block.shape[-1]} does not match the "
                f"per-world width {nt}*3"
            )
        return block.reshape(block.shape[:-1] + (nt, 3))[..., t_pin, :]

    # ------------------------------------------------------------------
    # weights (sib_derive ports, verbatim semantics)
    # ------------------------------------------------------------------
    def lam(self) -> xp.ndarray:
        """``sqrt(lambda^2)`` of the base pack (cached)."""
        if self._lam is None:
            self._lam = load_evals(self.evalfile.filename)
        return self._lam

    def n_eig(self) -> int:
        """Number of eigenpairs (rows are 2*n_eig: parity partners)."""
        return int(self.lam().shape[0])

    def _mass_value(self, label: str) -> float:
        if label not in self.mass:
            raise ValueError(
                f"mass label {label!r} not in mass parameters "
                f"({sorted(self.mass.keys())})"
            )
        return self.mass[label]

    def w_rows(self, mass_label: str) -> xp.ndarray:
        """Pair-basis row weights at ``mass_label`` (the p-side weight)."""
        return w_rows(self.lam(), self._mass_value(mass_label))

    def ket_refold(self, mass_from: str, mass_to: str) -> xp.ndarray:
        """Per-column refold ratio between two mass labels."""
        return ket_refold(
            self.lam(),
            self._mass_value(mass_from),
            self._mass_value(mass_to),
        )

    # ------------------------------------------------------------------
    # paths
    # ------------------------------------------------------------------
    def _check_resolved(self, path: str) -> str:
        if "{" in path:
            raise ValueError(f"unresolved path tokens in {path!r}")
        return path

    def _format(self, outfile: Outfile, **tokens: str) -> str:
        """Format an Outfile path; a missing token becomes a loud
        ValueError (str.format would raise an opaque KeyError first)."""
        try:
            return self._check_resolved(outfile.filename.format(**tokens))
        except KeyError as e:
            raise ValueError(
                f"path token {e} is unresolved in {outfile.filename!r}; "
                "resolve every token except family/leg_pair/mass/em_index/"
                "file_key/gamma before constructing the accessor"
            ) from e

    def block_path(
        self,
        *,
        family: str,
        leg_pair: str,
        mass_stem: str,
        file_key: str,
        em_index: int | None = None,
    ) -> str:
        """Formatted path of one block file (the producer's grammar)."""
        return self._format(
            self.blocks,
            family=family,
            leg_pair=leg_pair,
            mass=mass_stem,
            em_index=em_index_suffix(em_index),
            file_key=file_key,
        )

    def tab_path(self) -> str:
        """Formatted path of the scalar tab file."""
        return self._format(self.tab_outfile, gamma=SCALAR_GAMMA)

    # ------------------------------------------------------------------
    # fermion-world axis slicing (D13 — worlds live inside the files)
    # ------------------------------------------------------------------
    def _eta_world(self, arr: xp.ndarray, world: int) -> xp.ndarray:
        """``[nt, 3N, ...] -> [nt, 3, ...]``: one world's eta rows.

        The left axis is source-major (entry ``3*n + c``), so world
        ``n`` occupies rows ``3n..3n+3``.
        """
        n = self.noise
        if arr.shape[1] != 3 * n:
            raise ValueError(
                f"eta axis {arr.shape[1]} does not match 3*noise = {3 * n} "
                "(the full-volume noise left axis is 3N, source-major)"
            )
        if not 0 <= world < n:
            raise ValueError(f"fermion world {world} outside [0, {n})")
        return arr[:, 3 * world : 3 * world + 3, ...]

    def _h_world(self, arr: xp.ndarray, world: int) -> xp.ndarray:
        """``[..., 3N*nt] -> [..., 3*nt]``: one world's h ket block.

        The ket axis is world-MAJOR blocks of ``3*nt`` (upstream
        RandomWall order ``3*(i*nSlices+j)+k``); within a world the
        columns are slice-major ``3*t + c``.
        """
        n = self.noise
        w3 = 3 * self.nt()
        if arr.shape[-1] != w3 * n:
            raise ValueError(
                f"h axis {arr.shape[-1]} does not match 3*nt*noise = "
                f"{w3 * n} (world-major blocks of {w3} slice-major columns)"
            )
        if not 0 <= world < n:
            raise ValueError(f"fermion world {world} outside [0, {n})")
        return arr[..., w3 * world : w3 * (world + 1)]

    # ------------------------------------------------------------------
    # channel reads (j1l sums its four component files inline — D2)
    # ------------------------------------------------------------------
    def _file_keys(
        self,
        family: str,
        gamma: str | None,
        mom: str | None,
        em_world: int | None,
    ) -> t.List[str]:
        """File keys of one block's channel files (j1l yields four)."""
        if family in EM_FAMILIES:
            if em_world is None:
                raise ValueError(
                    f"EM family {family!r} blocks require an em_world "
                    "(the photon axis splits the files)"
                )
            if mom is not None:
                raise ValueError(
                    "EM family blocks are pinned to zero momentum; "
                    f"got mom={mom!r}"
                )
            if not 0 <= em_world < len(self.em_seeds):
                raise ValueError(
                    f"em world {em_world} outside [0, {len(self.em_seeds)})"
                )
            seed = self.em_seeds[em_world]
            return [em_file_key(family, seed, c) for c in em_components(family)]
        if mom is None:
            raise ValueError(
                f"vec family {family!r} blocks require a momentum elem"
            )
        if em_world is not None:
            raise ValueError("em_world is only valid for EM families")
        return [vec_file_key(gamma, mom)]

    def _read_file(
        self,
        *,
        family: str,
        leg_pair: str,
        mass_label: str | None,
        gamma: str | None,
        mom: str | None,
        em_world: int | None,
    ) -> xp.ndarray:
        """Read (and channel-sum) one block file, cached at file grain."""
        key = ("file", family, leg_pair, mass_label, gamma, mom, em_world)
        if key not in self._cache:
            keys = self._file_keys(family, gamma, mom, em_world)
            out: xp.ndarray | None = None
            for k in keys:
                arr = load_block(
                    self.block_path(
                        family=family,
                        leg_pair=leg_pair,
                        mass_stem=mass_token(mass_label),
                        file_key=k,
                        em_index=em_world,
                    ),
                    k,
                )
                if out is None:
                    out = arr
                else:
                    if arr.shape != out.shape:
                        raise ValueError(
                            f"{family}/{leg_pair} channel files disagree on "
                            f"shape: {out.shape} vs {arr.shape}"
                        )
                    out = out + arr
            self._cache[key] = out
        return self._cache[key]

    # ------------------------------------------------------------------
    # stored / raw pieces
    # ------------------------------------------------------------------
    def tab(self) -> xp.ndarray:
        """``tab = <l|eta>`` unweighted ``[nt, 2*n_eig, 3N]`` (cached)."""
        key = ("tab",)
        if key not in self._cache:
            tab = load_block(self.tab_path(), f"{SCALAR_GAMMA}_0_0_0")
            if tab.shape[-1] != 3 * self.noise:
                raise ValueError(
                    f"tab has {tab.shape[-1]} columns; the single-file tab "
                    f"carries 3*noise = {3 * self.noise} (noise-major)"
                )
            if tab.shape[1] != 2 * self.n_eig():
                raise ValueError(
                    f"tab rows {tab.shape[1]} do not match the evals file "
                    f"({2 * self.n_eig()})"
                )
            self._cache[key] = tab
        return self._cache[key]

    def wt(self, world: int, mass_label: str) -> xp.ndarray:
        """``w_rows(mass) * tab[:, :, 3w:3w+3]`` — ``[nt, 2*n_eig, 3]``."""
        key = ("wt", world, mass_label)
        if key not in self._cache:
            w = self.w_rows(mass_label)[:, None]
            stored = self.tab()[:, :, 3 * world : 3 * world + 3]
            self._cache[key] = xp.stack(
                [stored[tt] * w for tt in range(stored.shape[0])], axis=0
            )
        return self._cache[key]

    def _ll_stored(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None,
    ) -> xp.ndarray:
        """Stored ``ll`` (ket folded at defl) ``[nt, 2*n_eig, 2*n_eig]``."""
        key = ("ll_stored", family, gamma, mom, em_world)
        if key not in self._cache:
            arr = self._read_file(
                family=family, leg_pair="ll", mass_label=None,
                gamma=gamma, mom=mom, em_world=em_world,
            )
            if arr.shape[1] != 2 * self.n_eig() or arr.shape[2] != 2 * self.n_eig():
                raise ValueError(
                    f"ll {family}/{gamma}/{mom} axes {arr.shape[1:]} do not "
                    f"match the evals file ({2 * self.n_eig()} per side)"
                )
            self._cache[key] = arr
        return self._cache[key]

    def _ll_raw(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None,
    ) -> xp.ndarray:
        """``ll`` with the ket-side eval folding removed (mass-free)."""
        key = ("ll_raw", family, gamma, mom, em_world)
        if key not in self._cache:
            ev = eval_pairs(self.lam(), self._mass_value(self.defl_mass))
            self._cache[key] = (
                self._ll_stored(family, gamma, mom, em_world)
                * ev[None, None, :]
            )
        return self._cache[key]

    def _nl_file(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None,
    ) -> xp.ndarray:
        """Stored ``nl`` file ``[nt, 3N, 2*n_eig]`` (all worlds; folded)."""
        key = ("nl_file", family, gamma, mom, em_world)
        if key not in self._cache:
            arr = self._read_file(
                family=family, leg_pair="nl", mass_label=None,
                gamma=gamma, mom=mom, em_world=em_world,
            )
            if arr.shape[1] != 3 * self.noise:
                raise ValueError(
                    f"nl {family}/{gamma}/{mom} left axis {arr.shape[1]} "
                    f"does not match 3*noise = {3 * self.noise}"
                )
            if arr.shape[2] != 2 * self.n_eig():
                raise ValueError(
                    f"nl {family}/{gamma}/{mom} ket axis {arr.shape[2]} "
                    f"does not match the evals file ({2 * self.n_eig()})"
                )
            self._cache[key] = arr
        return self._cache[key]

    def _nl_raw(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None, world: int,
    ) -> xp.ndarray:
        """``<eta_world|Gamma|l>`` unfolded ``[nt, 3, 2*n_eig]``.

        qed_mf emits ``nl`` for EVERY family (momentum projection
        destroys the scalar tab identity) — always a stored read, never
        a tab derivation (unlike sib).
        """
        key = ("nl_raw", family, gamma, mom, em_world, world)
        if key not in self._cache:
            ev = eval_pairs(self.lam(), self._mass_value(self.defl_mass))
            self._cache[key] = (
                self._eta_world(
                    self._nl_file(family, gamma, mom, em_world), world
                )
                * ev[None, None, :]
            )
        return self._cache[key]

    # ------------------------------------------------------------------
    # public mass-resolved / derived blocks
    # ------------------------------------------------------------------
    def ll(
        self, family: str, gamma: str | None, mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """``ll`` refolded to ``mass_label`` ``[nt, 2*n_eig, 2*n_eig]``."""
        key = ("ll", family, gamma, mom, em_world, mass_label)
        if key not in self._cache:
            ratio = self.ket_refold(self.defl_mass, mass_label)
            self._cache[key] = (
                self._ll_stored(family, gamma, mom, em_world)
                * ratio[None, None, :]
            )
        return self._cache[key]

    def nl(
        self, family: str, gamma: str | None, world: int, mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """``nl`` at ``mass_label`` for one world ``[nt, 3, 2*n_eig]``
        (carries the writer's folding at ``mass_label``)."""
        key = ("nl", family, gamma, mom, em_world, world, mass_label)
        if key not in self._cache:
            ev = eval_pairs(self.lam(), self._mass_value(mass_label))
            self._cache[key] = (
                self._nl_raw(family, gamma, mom, em_world, world)
                / ev[None, None, :]
            )
        return self._cache[key]

    def _lh_file(
        self, family: str, gamma: str | None, mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """Stored ``lh`` file ``[nt, 2*n_eig, 3N*nt]`` (all worlds)."""
        key = ("lh_file", family, gamma, mom, em_world, mass_label)
        if key not in self._cache:
            arr = self._read_file(
                family=family, leg_pair="lh", mass_label=mass_label,
                gamma=gamma, mom=mom, em_world=em_world,
            )
            if arr.shape[1] != 2 * self.n_eig():
                raise ValueError(
                    f"lh {family}/{gamma}/{mom} rows {arr.shape[1]} do not "
                    f"match the evals file ({2 * self.n_eig()})"
                )
            if arr.shape[2] != 3 * self.noise * self.nt():
                raise ValueError(
                    f"lh {family}/{gamma}/{mom} ket axis {arr.shape[2]} "
                    f"does not match 3*nt*noise = "
                    f"{3 * self.noise * self.nt()}"
                )
            self._cache[key] = arr
        return self._cache[key]

    def _nh_file(
        self, family: str, gamma: str | None, mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """Stored ``nh`` file ``[nt, 3N, 3N*nt]`` (all world pairs)."""
        key = ("nh_file", family, gamma, mom, em_world, mass_label)
        if key not in self._cache:
            arr = self._read_file(
                family=family, leg_pair="nh", mass_label=mass_label,
                gamma=gamma, mom=mom, em_world=em_world,
            )
            if arr.shape[1] != 3 * self.noise:
                raise ValueError(
                    f"nh {family}/{gamma}/{mom} left axis {arr.shape[1]} "
                    f"does not match 3*noise = {3 * self.noise}"
                )
            if arr.shape[2] != 3 * self.noise * self.nt():
                raise ValueError(
                    f"nh {family}/{gamma}/{mom} ket axis {arr.shape[2]} "
                    f"does not match 3*nt*noise = "
                    f"{3 * self.noise * self.nt()}"
                )
            self._cache[key] = arr
        return self._cache[key]

    def _derived_lp(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None, world: int, mass_label: str,
    ) -> xp.ndarray:
        """Derived ``lp`` ``[nt, 2*n_eig, 3*nt]`` for one world."""
        key = ("lp", family, gamma, mom, em_world, world, mass_label)
        if key not in self._cache:
            self._cache[key] = _lp_np_product(
                self._ll_raw(family, gamma, mom, em_world),
                self.wt(world, mass_label),
            )
        return self._cache[key]

    def _derived_np(
        self, family: str, gamma: str | None, mom: str | None,
        em_world: int | None, world_n: int, world_hp: int,
        mass_label: str,
    ) -> xp.ndarray:
        """Derived ``np`` ``[nt, 3, 3*nt]``: ``nl_raw_n @ w*tab_hp``."""
        key = ("np", family, gamma, mom, em_world, world_n, world_hp, mass_label)
        if key not in self._cache:
            self._cache[key] = _lp_np_product(
                self._nl_raw(family, gamma, mom, em_world, world_n),
                self.wt(world_hp, mass_label),
            )
        return self._cache[key]

    def pure_lh(
        self, family: str, gamma: str | None, world: int, mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """``lh - lp`` for one world ``[nt, 2*n_eig, 3*nt]``."""
        key = ("pure_lh", family, gamma, mom, em_world, world, mass_label)
        if key not in self._cache:
            self._cache[key] = (
                self._h_world(
                    self._lh_file(
                        family, gamma, mass_label, mom=mom, em_world=em_world
                    ),
                    world,
                )
                - self._derived_lp(
                    family, gamma, mom, em_world, world, mass_label
                )
            )
        return self._cache[key]

    def pure_nh(
        self, family: str, gamma: str | None, world_n: int, world_hp: int,
        mass_label: str, *,
        mom: str | None = None, em_world: int | None = None,
    ) -> xp.ndarray:
        """``nh - np`` for one world pair ``[nt, 3, 3*nt]``."""
        key = (
            "pure_nh", family, gamma, mom, em_world,
            world_n, world_hp, mass_label,
        )
        if key not in self._cache:
            self._cache[key] = (
                self._eta_world(
                    self._h_world(
                        self._nh_file(
                            family, gamma, mass_label,
                            mom=mom, em_world=em_world,
                        ),
                        world_hp,
                    ),
                    world_n,
                )
                - self._derived_np(
                    family, gamma, mom, em_world,
                    world_n, world_hp, mass_label,
                )
            )
        return self._cache[key]
