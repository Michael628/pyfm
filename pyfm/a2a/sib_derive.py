"""Pair-basis derivation layer for the connected-SIB three-point contraction.

The producer task ``hadrons_sib_mf`` (split-noise mode) stores only the
structurally-underivable blocks: ``ll`` (every Gamma family), ``nl`` (the
vector families only), ``lh``/``nh`` (per operations mass), the per-world
scalar ``tab`` overlap tables, and the epack evals file. The removed
derivable pairs are exact offline arithmetic from those files (research
``2026-10-04_17-36-40``; validated by
``../HadronsMILC/test/compare_sib_pair_identity.py`` at <= 5.5e-8):

- ``lp_Γ^m[t][:, (s,c)] = (ll_Γ[t] * eval(defl)) @ (w_rows(m) * tab_w[s])[:, c]``
- ``np_Γ^m[t][:, (s,c)] = nl_raw_Γ,w[t] @ (w_rows(m) * tab_w'[s])[:, c]``
  where ``nl_raw`` is the eval-unfolded stored ``nl`` (vector families) or
  ``conj(tab_w[t]).T`` (the scalar family);
- ``nl_s^m[t][j, 2a]   = conj(tab_w[t][2a, j]) / eval_a(m)``
  ``nl_s^m[t][j, 2a+1] = conj(tab_w[t][2a+1, j]) / conj(eval_a(m))``.

Weights (validated diagonal row-scaling form, plan revision
2026-10-04T22:55:14): with ``m = 2*mass`` (MILC convention),
``lam_k = sqrt(evals[k])`` of the BASE pack, and
``invmag_k = 1/(m^2 + lam_k^2)``: row ``2k`` carries
``invmag*(m - i*lam_k)`` and row ``2k+1`` carries ``invmag*(m + i*lam_k)``.
The stored rows are the (e+-o)/sqrt(2) rotation of the research doublet in
which the pair operator is diagonal.

Ket-side eval folding: every stored block whose ket leg is the eigenvector
set (``ll``, vector ``nl``) carries the writer's folding
``stored_col(2a) *= eval_a(defl_mass)``, ``stored_col(2a+1) *= conj(eval_a)``
with ``eval_a(mass) = 2*mass + i*lam_a`` (the inverse documented at
``LoadMesonField.hpp:388-397``). Refolding to another mass is the per-column
ratio ``eval(defl)/eval(m)`` — the same convention as
``pyfm/a2a/mesonloader.py::meson_mass_alter``. ``h``/``p`` columns carry no
folding (they are not eigenvector sets), and the tab is stored unweighted.

All mass dependence is either ``w_rows(m)`` (p-side derivations) or the
per-column ket refold (stored eig-ket sides): raw blocks are mass-independent
because the eigenvectors are mass-shift invariant.

Full-volume column addressing: each world file's h/p columns are
``3*nt`` in slice-major order ``3*t + c`` (nsrc=1 per world) — upstream
``StagRandomWall`` derives ``nSlices = nt/min(tStep, nt)`` with
``t0 < tStep``, so production always covers every timeslice and the
slice index equals the lattice time. A noise junction pins the column
block to the eta-side field's time row ``t``; ``cols`` verifies the
width loudly against ``3*nt``.
"""

import typing as t

import h5py
import numpy as np

try:
    import cupy as xp
except ImportError:
    import numpy as xp

from pyfm.domain import MassDict, Outfile

SCALAR_GAMMA = "G1_G1"


def w_rows(lam: xp.ndarray, mass: float) -> xp.ndarray:
    """Per-row pair weight, diagonal in the stored row basis.

    Row ``2k`` gets ``invmag*(m - i*lam_k)``, row ``2k+1`` gets
    ``invmag*(m + i*lam_k)`` with ``m = 2*mass`` (MILC convention).
    """
    m = 2.0 * mass
    lam = xp.asarray(lam)
    invmag = 1.0 / (m * m + lam * lam)
    w = xp.empty(2 * lam.shape[0], dtype=xp.complex128)
    w[0::2] = invmag * (m - 1j * lam)
    w[1::2] = invmag * (m + 1j * lam)
    return w


def eval_pairs(lam: xp.ndarray, mass: float) -> xp.ndarray:
    """Ket-side folding vector at ``mass``: even ``2m + i*lam``, odd conj."""
    m = 2.0 * mass
    lam = xp.asarray(lam)
    ev = xp.empty(2 * lam.shape[0], dtype=xp.complex128)
    ev[0::2] = m + 1j * lam
    ev[1::2] = m - 1j * lam
    return ev


def ket_refold(lam: xp.ndarray, mass_from: float, mass_to: float) -> xp.ndarray:
    """Per-column refold ratio ``eval(mass_from)/eval(mass_to)``.

    Multiplying a stored (defl-folded) block's ket columns by this ratio
    re-expresses it at ``mass_to``; ``mass_from == mass_to`` gives 1.
    """
    return eval_pairs(lam, mass_from) / eval_pairs(lam, mass_to)


def load_evals(path: str) -> xp.ndarray:
    """Load the base-pack eigenvalue file and return ``lam = sqrt(lambda^2)``.

    The file is the ``EigenPackExtractEvals`` saveResult form: a flat
    ``evals.<traj>.h5`` with a single 1-d RealD ``evals`` dataset.
    """
    with h5py.File(path, "r") as f:
        if "evals" not in f:
            raise ValueError(f"dataset 'evals' not found in {path!r}")
        ev = np.array(f["evals"][()], dtype=np.float64)
    if ev.ndim != 1 or ev.size < 1:
        raise ValueError(
            f"evals file {path!r}: expected a 1-d RealD dataset, got {ev.shape}"
        )
    return xp.asarray(np.sqrt(ev))


def load_block(path: str, gamma: str) -> xp.ndarray:
    """Load one meson-field block file as complex128 ``[nt, N_i, N_j]``."""
    with h5py.File(path, "r") as f:
        ds = f"{gamma}_0_0_0"
        if ds not in f or "a2aMatrix" not in f[ds]:
            raise ValueError(f"dataset {ds!r}/a2aMatrix not found in {path!r}")
        d = f[ds]["a2aMatrix"][()]
    return xp.asarray((d["re"] + 1j * d["im"]).astype(np.complex128))


def _index_suffix(index: int | None) -> str:
    """``'_n<k>'`` stem-suffix fragment; ``''`` for an absent axis."""
    return "" if index is None else f"_n{index}"


class SIBBlockAccessor:
    """Load-or-derive accessor over the split-noise SIB block surface.

    Loads the emitted blocks (``ll``, vector ``nl``, ``lh``, ``nh``), derives
    the removed ones (``lp``, ``np``, scalar ``nl``) via the pair-basis
    identity, refolds eig-ket sides between masses, and exposes pure-high
    differences ``pure_lh = lh - lp`` and ``pure_nh = nh - np``. Everything is
    cached per ``(kind, gamma, worlds, mass)`` — the d38ef6c lesson: the cache
    key carries every axis that changes content.

    ``blocks``/``tab`` are the producer's Outfile labels with
    ``{leg_pair}``/``{mass}``/``{n_index}``/``{hp_index}``/``{gamma}`` tokens
    intact (anything else, e.g. ``{series}``, must already be resolved);
    ``evalfile`` is fully resolved. Path formatting raises loudly on any
    remaining brace so a stale grammar cannot silently point elsewhere.
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
    ) -> None:
        self.blocks = blocks
        self.tab_outfile = tab
        self.evalfile = evalfile
        self.mass = mass
        self.defl_mass = defl_mass
        self.noise = noise
        self._cache: t.Dict[t.Tuple, xp.ndarray] = {}
        self._lam: xp.ndarray | None = None
        self._nt: int | None = None

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def nt(self) -> int:
        """Lattice time extent (== the wall-slice count; full coverage).

        Derived from the data: the per-world tab is stored at every
        lattice time ``[nt, 2*n_eig, 3]``, and full-volume noise means
        every timeslice is a wall slice (upstream ``StagRandomWall``:
        ``nSlices = nt/min(tStep, nt)``, ``t0 < tStep``).
        """
        if self._nt is None:
            self._nt = int(self.tab(0).shape[0])
        return self._nt

    def cols(self, block: xp.ndarray, t_pin: int) -> xp.ndarray:
        """Time-pinned column selection of an h/p block.

        The last axis (``3*nt``, slice-major ``3*t + c``) is viewed as
        ``(nt, 3)`` and the block for time ``t_pin`` extracted, giving
        ``[..., 3]``. Width and bounds are verified loudly — a block
        whose columns don't cover every lattice time breaks the
        full-volume data contract.
        """
        nt = self.nt()
        if not 0 <= t_pin < nt:
            raise ValueError(f"pinned time {t_pin} outside [0, {nt})")
        if block.shape[-1] != nt * 3:
            raise ValueError(
                f"block last axis {block.shape[-1]} does not match the "
                f"full-volume width {nt}*3"
            )
        return block.reshape(block.shape[:-1] + (nt, 3))[..., t_pin, :]

    # ------------------------------------------------------------------
    # weights
    # ------------------------------------------------------------------
    def lam(self) -> xp.ndarray:
        """``sqrt(lambda^2)`` of the base pack (cached)."""
        if self._lam is None:
            self._lam = load_evals(self.evalfile.filename)
        return self._lam

    def n_eig(self) -> int:
        """Number of eigenpairs (rows are 2*n_eig: one per parity partner)."""
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
        """Format an Outfile path, converting a missing token into a loud
        ValueError (str.format would raise an opaque KeyError first)."""
        try:
            return self._check_resolved(outfile.filename.format(**tokens))
        except KeyError as e:
            raise ValueError(
                f"path token {e} is unresolved in {outfile.filename!r}; "
                "resolve every token except leg_pair/mass/n_index/hp_index/"
                "gamma before constructing the accessor"
            ) from e

    def block_path(
        self,
        *,
        leg_pair: str,
        mass_token: str,
        gamma: str,
        n_index: int | None = None,
        hp_index: int | None = None,
    ) -> str:
        """Formatted path of one block file (the producer's grammar)."""
        return self._format(
            self.blocks,
            leg_pair=leg_pair,
            mass=mass_token,
            gamma=gamma,
            n_index=_index_suffix(n_index),
            hp_index=_index_suffix(hp_index),
        )

    def tab_path(self, world: int) -> str:
        """Formatted path of one world's tab file."""
        return self._format(
            self.tab_outfile, gamma=SCALAR_GAMMA, n_index=_index_suffix(world)
        )

    # ------------------------------------------------------------------
    # stored / raw pieces
    # ------------------------------------------------------------------
    def tab(self, world: int) -> xp.ndarray:
        """``tab_w = <l|eta_w>`` unweighted ``[nt, 2*n_eig, 3]`` (cached)."""
        key = ("tab", world)
        if key not in self._cache:
            tab = load_block(self.tab_path(world), SCALAR_GAMMA)
            if tab.shape[-1] != 3:
                raise ValueError(
                    f"tab world {world} has {tab.shape[-1]} columns; the "
                    "split-noise per-world tab carries exactly 3"
                )
            self._cache[key] = tab
        return self._cache[key]

    def wt(self, world: int, mass_label: str) -> xp.ndarray:
        """``w_rows(mass) * tab_w[t]`` per lattice time ``[nt, 2*n_eig, 3]``."""
        key = ("wt", world, mass_label)
        if key not in self._cache:
            w = self.w_rows(mass_label)[:, None]
            stored = self.tab(world)
            self._cache[key] = xp.stack(
                [stored[tt] * w for tt in range(stored.shape[0])], axis=0
            )
        return self._cache[key]

    def _ll_stored(self, gamma: str) -> xp.ndarray:
        key = ("ll_stored", gamma)
        if key not in self._cache:
            arr = load_block(
                self.block_path(
                    leg_pair="ll", mass_token="", gamma=gamma
                ),
                gamma,
            )
            if arr.shape[-1] != 2 * self.n_eig():
                raise ValueError(
                    f"ll gamma {gamma!r} ket axis {arr.shape[-1]} does not "
                    f"match the evals file ({2 * self.n_eig()})"
                )
            self._cache[key] = arr
        return self._cache[key]

    def _ll_raw(self, gamma: str) -> xp.ndarray:
        """``ll`` with the ket-side eval folding removed (mass-independent)."""
        key = ("ll_raw", gamma)
        if key not in self._cache:
            ev = eval_pairs(self.lam(), self._mass_value(self.defl_mass))
            self._cache[key] = self._ll_stored(gamma) * ev[None, None, :]
        return self._cache[key]

    def _nl_raw(self, gamma: str, world: int) -> xp.ndarray:
        """``<eta_w|Gamma|l>`` unfolded ``[nt, 3, 2*n_eig]``.

        Vector families load the stored block and remove the ket folding;
        the scalar family derives it as ``conj(tab_w).T``.
        """
        key = ("nl_raw", gamma, world)
        if key not in self._cache:
            if gamma == SCALAR_GAMMA:
                raw = xp.conjugate(self.tab(world).transpose(0, 2, 1))
            else:
                stored = load_block(
                    self.block_path(
                        leg_pair="nl",
                        mass_token="",
                        gamma=gamma,
                        n_index=world,
                    ),
                    gamma,
                )
                if stored.shape[-1] != 2 * self.n_eig():
                    raise ValueError(
                        f"nl gamma {gamma!r} ket axis {stored.shape[-1]} "
                        f"does not match the evals file "
                        f"({2 * self.n_eig()})"
                    )
                ev = eval_pairs(self.lam(), self._mass_value(self.defl_mass))
                raw = stored * ev[None, None, :]
            self._cache[key] = raw
        return self._cache[key]

    # ------------------------------------------------------------------
    # public mass-resolved blocks
    # ------------------------------------------------------------------
    def ll(self, gamma: str, mass_label: str) -> xp.ndarray:
        """``ll_Gamma`` refolded to ``mass_label`` ``[nt, 2*n_eig, 2*n_eig]``."""
        key = ("ll", gamma, mass_label)
        if key not in self._cache:
            ratio = self.ket_refold(self.defl_mass, mass_label)
            self._cache[key] = self._ll_stored(gamma) * ratio[None, None, :]
        return self._cache[key]

    def nl(self, gamma: str, world: int, mass_label: str) -> xp.ndarray:
        """``nl`` at ``mass_label`` ``[nt, 3, 2*n_eig]``.

        Vector families load the stored block (ket-folded at the deflation
        mass); the scalar family derives it from the tab. Either way the
        returned block carries the writer's folding at ``mass_label``:
        raw ``<eta_w|Gamma|l>`` divided by ``eval(mass_label)`` per ket
        column parity (mass-shift invariance of the eigenvectors means the
        raw block is mass-independent; only the folding changes).
        """
        key = ("nl", gamma, world, mass_label)
        if key not in self._cache:
            ev = eval_pairs(self.lam(), self._mass_value(mass_label))
            self._cache[key] = self._nl_raw(gamma, world) / ev[None, None, :]
        return self._cache[key]

    def lh(self, gamma: str, world: int, mass_label: str) -> xp.ndarray:
        """Stored ``lh`` ``[nt, 2*n_eig, 3*nt]`` (no folding on h)."""
        key = ("lh", gamma, world, mass_label)
        if key not in self._cache:
            arr = load_block(
                self.block_path(
                    leg_pair="lh",
                    mass_token=f"_m{mass_label}",
                    gamma=gamma,
                    hp_index=world,
                ),
                gamma,
            )
            self._shape_check_h(gamma, "lh", arr, world, None)
            self._cache[key] = arr
        return self._cache[key]

    def nh(
        self, gamma: str, world_n: int, world_hp: int, mass_label: str
    ) -> xp.ndarray:
        """Stored ``nh`` ``[nt, 3, 3*nt]`` (no folding on h)."""
        key = ("nh", gamma, world_n, world_hp, mass_label)
        if key not in self._cache:
            arr = load_block(
                self.block_path(
                    leg_pair="nh",
                    mass_token=f"_m{mass_label}",
                    gamma=gamma,
                    n_index=world_n,
                    hp_index=world_hp,
                ),
                gamma,
            )
            self._shape_check_h(gamma, "nh", arr, world_n, world_hp)
            self._cache[key] = arr
        return self._cache[key]

    def _shape_check_h(
        self,
        gamma: str,
        leg_pair: str,
        arr: xp.ndarray,
        world_n: int | None,
        world_hp: int | None,
    ) -> None:
        n_eig = 2 * self.n_eig()
        expected_rows = 3 if leg_pair == "nh" else n_eig
        if arr.shape[1] != expected_rows or arr.shape[2] != self.nt() * 3:
            raise ValueError(
                f"{leg_pair} gamma {gamma!r} worlds "
                f"(n={world_n}, hp={world_hp}) has shape {arr.shape}; "
                f"expected [nt, {expected_rows}, {self.nt() * 3}]"
            )

    # ------------------------------------------------------------------
    # derived blocks / pure-high differences
    # ------------------------------------------------------------------
    def _derived_lp(self, gamma: str, world: int, mass_label: str) -> xp.ndarray:
        """Derived ``lp`` ``[nt, 2*n_eig, 3*nt]`` (slice-major cols)."""
        key = ("lp", gamma, world, mass_label)
        if key not in self._cache:
            wt = self.wt(world, mass_label)
            raw = self._ll_raw(gamma)
            self._cache[key] = self._lp_np_product(raw, wt)
        return self._cache[key]

    def _derived_np(
        self, gamma: str, world_n: int, world_hp: int, mass_label: str
    ) -> xp.ndarray:
        """Derived ``np`` ``[nt, 3, 3*nt]``: ``nl_raw @ w*tab_{hp}``."""
        key = ("np", gamma, world_n, world_hp, mass_label)
        if key not in self._cache:
            wt = self.wt(world_hp, mass_label)
            raw = self._nl_raw(gamma, world_n)
            self._cache[key] = self._lp_np_product(raw, wt)
        return self._cache[key]

    @staticmethod
    def _lp_np_product(raw: xp.ndarray, wt: xp.ndarray) -> xp.ndarray:
        """Concatenate ``raw[t] @ wt[t']`` over lattice times (col order 3t+c).

        ``raw`` is ``[nt, R, 2*n_eig]``; ``wt`` is ``[nt, 2*n_eig, 3]``;
        the result is ``[nt, R, 3*nt]`` — for each ``t`` and time ``t'``,
        column ``(t', c)`` = ``raw[t] @ (w * tab[t'])[:, c]``.
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

    def pure_lh(self, gamma: str, world: int, mass_label: str) -> xp.ndarray:
        """``lh - lp`` (the pure-high ℓ-leg block) ``[nt, 2*n_eig, 3*nt]``."""
        key = ("pure_lh", gamma, world, mass_label)
        if key not in self._cache:
            self._cache[key] = (
                self.lh(gamma, world, mass_label)
                - self._derived_lp(gamma, world, mass_label)
            )
        return self._cache[key]

    def pure_nh(
        self, gamma: str, world_n: int, world_hp: int, mass_label: str
    ) -> xp.ndarray:
        """``nh - np`` (the pure-high n-leg block) ``[nt, 3, 3*nt]``."""
        key = ("pure_nh", gamma, world_n, world_hp, mass_label)
        if key not in self._cache:
            self._cache[key] = (
                self.nh(gamma, world_n, world_hp, mass_label)
                - self._derived_np(gamma, world_n, world_hp, mass_label)
            )
        return self._cache[key]
