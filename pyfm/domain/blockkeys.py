"""Shared block-addressing helpers for mf producers and contraction consumers.

The block-file grammar — stem tokens plus ``{file_key}`` file keys — is the
declared interface between the ``hadrons_*_mf`` producers and the
``contract_*`` consumers: ``Outfile`` labels are shared between the two
tasks' params stanzas and the filenames are the API. These pure helpers
compose and tokenize that grammar ONCE, here in ``pyfm.domain`` (the layer
both ``pyfm.tasks`` and ``pyfm.a2a`` already import), so neither side
imports the other and the grammar cannot drift between emission and
consumption.

Grammar elements:

- momentum: ``expand_momenta`` (the conjugate ±p expansion),
  ``mom_token`` (``"1 0 0"`` → ``"1_0_0"``), ``ZERO_MOM`` (the EM-insertion
  momentum pin).
- file keys (the ``{file_key}`` ext token of the ``cfg_qed_h5`` grammar):
  ``vec_file_key`` ``{GammaName}_{p1_p2_p3}``, ``aslash_file_key``
  ``{seed}_{a}_{p}``, ``onelink_file_key`` ``{seed}_{j}_{k}_{p}``, and the
  family-dispatching ``em_file_key``/``em_components``.
- stem suffix tokens: ``index_suffix`` (``_n<k>`` fermion-noise world),
  ``em_index_suffix`` (``_em<i>`` photon world), ``mass_token``
  (``_m<label>``; ``''`` for the reference pairs).
- EM family identifiers: ``EM_FAMILY_JA`` / ``EM_FAMILY_J1L`` plus the
  vec-family nicknames ``VEC_FAMILY_TOKEN`` (s/vl/vo) — module-family
  concepts, NOT ``Gamma`` members themselves.

Every function is pure (no config objects, no I/O) so the module is safe to
import anywhere — including the MPI-lazy ``pyfm.a2a`` import chain and test
fixtures.
"""

import typing as t

from pyfm.domain.ops import Gamma

# The zero-momentum elem the EM insertion families are pinned to (momentum
# lives on the non-ASlash legs only; qed_mf D3).
ZERO_MOM = "0 0 0"

# EM module-family identifiers: ja = local A_slash J·A, j1l = per-component
# J_kA_k one-link. The token rides output stems ({family}) and module names
# (mf_ja_* / mf_j1l_*).
EM_FAMILY_JA = "ja"
EM_FAMILY_J1L = "j1l"

# Output-stem family tokens for the vec Γ families (the sib s/vl/vo
# nicknames); EM families use EM_FAMILY_JA/EM_FAMILY_J1L directly.
# Module-family concepts, NOT ``Gamma`` members themselves.
VEC_FAMILY_TOKEN = {
    Gamma.SCALAR_LOCAL: "s",
    Gamma.VEC_LOCAL: "vl",
    Gamma.VEC_ONELINK: "vo",
}


def expand_momenta(momentum) -> t.List[str]:
    """±p conjugate expansion of the configured momentum list.

    Each configured p yields ``+p`` then ``-p`` (in order), rendered as
    space-separated component strings and deduplicated
    first-occurrence (a ``[0, 0, 0]`` entry collapses to the single
    ``"0 0 0"``; ``int(-0) == 0`` in Python so ``"-0"`` never renders).
    The list feeds the vec family's ``mom`` option and the catalog
    momentum axis.
    """
    elems: t.List[str] = []
    for p in momentum:
        for sign in (1, -1):
            tok = " ".join(str(sign * int(c)) for c in p)
            if tok not in elems:
                elems.append(tok)
    return elems


def mom_token(mom: str) -> str:
    """Filename token of one momentum elem (``"1 0 0"`` -> ``"1_0_0"``)."""
    return mom.replace(" ", "_")


def index_suffix(index: int | None) -> str:
    """``'_n<k>'`` fermion-noise-world stem-suffix fragment; ``''`` if absent."""
    return "" if index is None else f"_n{index}"


def em_index_suffix(index: int | None) -> str:
    """``'_em<k>'`` photon-world stem-suffix fragment; ``''`` if absent."""
    return "" if index is None else f"_em{index}"


def mass_token(mass_label: str | None) -> str:
    """``'_m<label>'`` stem token; ``''`` for the reference (mass-free) pairs."""
    return f"_m{mass_label}" if mass_label else ""


def vec_file_key(gamma_name: str, mom: str) -> str:
    """vec file_key: ``{GammaName}_{p1_p2_p3}`` — the upstream per-
    gamma×momentum file grammar (one HDF5 per pair, g-major mu axis)."""
    return f"{gamma_name}_{mom_token(mom)}"


def aslash_file_key(seed: str, sample: int, mom: str) -> str:
    """ja file_key: ``{seed}_{a}_{p}`` — a is the sample index within the
    module (always 0 for the split_em nEmFields=1 worlds; the world is
    identified by the per-sample seed baked into the key)."""
    return f"{seed}_{sample}_{mom_token(mom)}"


def onelink_file_key(seed: str, sample: int, component: int, mom: str) -> str:
    """j1l file_key: ``{seed}_{j}_{k}_{p}`` — j sample (0), k = J_kA_k
    component 0..3 (one file channel per component)."""
    return f"{seed}_{sample}_{component}_{mom_token(mom)}"


def em_components(family: str) -> t.List[t.Optional[int]]:
    """Per-file channel axes of an EM family: ja writes one channel (the
    summed a index); j1l writes four J_kA_k components."""
    return list(range(4)) if family == EM_FAMILY_J1L else [None]


def em_file_key(family: str, seed: str, component: int | None = None) -> str:
    """World+channel-composed file_key of one EM block file (zero momentum).

    The seed is the per-sample ``EmSeedString`` (``{seed}_n{i}`` — shared
    across ja/j1l so world i of both families consumes identical photons).
    """
    if family == EM_FAMILY_JA:
        return aslash_file_key(seed, 0, ZERO_MOM)
    if family == EM_FAMILY_J1L:
        if component is None:
            raise ValueError(
                "j1l file keys require a J_kA_k component (0..3); got None"
            )
        return onelink_file_key(seed, 0, component, ZERO_MOM)
    raise ValueError(
        f"unknown EM family {family!r}; expected {EM_FAMILY_JA!r} or "
        f"{EM_FAMILY_J1L!r}"
    )
