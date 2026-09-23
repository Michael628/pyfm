import typing as t

"""
This module contains python wrappers for Hadrons modules.
"""


def xml_wrapper(runid: str, sched: str, cfg: str, mpi_split: str | None = None) -> dict:
    params = {
        "grid": {
            "parameters": {
                "runId": runid,
                "trajCounter": {
                    "start": cfg,
                    "end": "10000",
                    "step": "10000",
                },
                "genetic": {
                    "popSize": "20",
                    "maxGen": "1000",
                    "maxCstGen": "100",
                    "mutationRate": "0.1",
                },
                "graphFile": "",
                "scheduleFile": sched,
                "saveSchedule": "false",
                "parallelWriteMaxRetry": "-1",
            },
            "modules": {},
        },
    }

    if mpi_split is not None:
        params["grid"]["parameters"]["split"] = {"mpiSplit": mpi_split}

    return params


def load_milcv5(name: str, file: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MIO::LoadMilc"},
        "options": {"file": file, "exitOnChecksumMismatch": "false"},
    }


def load_ildg(name: str, file: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MIO::LoadIldg"},
        "options": {"file": file},
    }


def save_ildg(name: str, gauge: str, filestem: str, ensemble_label: str = "") -> t.Dict:
    """Wrap ``MIO::SaveIldg``.

    Writes the named gauge field to ``<filestem>.<traj>`` as an ILDG file. The
    trajectory number is appended by the C++ module, matching ``LoadIldg``'s
    read convention, so ``filestem`` should be the bare path (no extension).
    """
    return {
        "id": {"name": name, "type": "MIO::SaveIldg"},
        "options": {
            "gauge": gauge,
            "fileStem": filestem,
            "ensembleLabel": ensemble_label,
        },
    }


def unit_gauge(name: str) -> t.Dict:
    return {"id": {"name": name, "type": "MGauge::Unit"}}


def cast_gauge(name: str, field: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MUtilities::GaugeSinglePrecisionCast"},
        "options": {"field": field},
    }


def apbc_gauge(name: str, gauge: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MGauge::APBCGauge"},
        "options": {"gauge": gauge, "boundary": "1 1 1 -1"},
    }


def hisq_smear(name: str, gauge: str, boundary: str = "1 1 1 -1") -> t.Dict:
    """Wrap ``MGauge::HISQSmear``.

    Smears the thin ``gauge`` field into fat7 + Naik (long) links, producing
    outputs named ``<name>_fat`` and ``<name>_long``. The KS phases and
    ``boundary`` are baked into the links via rephase, so the downstream
    ``ImprovedStaggered`` action consumes them with a plain ``1 1 1 1``
    boundary (no double rephase), matching the loaded-link convention.
    """
    return {
        "id": {"name": name, "type": "MGauge::HISQSmear"},
        "options": {"gauge": gauge, "boundary": boundary},
    }


def action(name: str, mass: str, gauge_fat: str, gauge_long: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MAction::ImprovedStaggeredMILC"},
        "options": {
            "mass": mass,
            "gaugefat": gauge_fat,
            "gaugelong": gauge_long,
        },
    }


def action_float(*args, **kwargs) -> t.Dict:
    res = action(*args, **kwargs)
    res["id"]["type"] = "MAction::ImprovedStaggeredMILCF"
    return res


def op(name: str, action: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MFermion::StagOperators"},
        "options": {"action": action},
    }


def irl(
    name: str,
    op: str,
    alpha: str,
    beta: str,
    npoly: str,
    nstop: str,
    nk: str,
    nm: str,
    multifile: str,
    residual: str,
    output: str,
) -> t.Dict:
    return {
        "id": {"name": name, "type": "MSolver::StagFermionIRL"},
        "options": {
            "op": op,
            "lanczosParams": {
                "Cheby": {"alpha": alpha, "beta": beta, "Npoly": npoly},
                "Nstop": nstop,
                "Nk": nk,
                "Nm": nm,
                "resid": residual,
                "MaxIt": "5000",
                "betastp": "0",
                "MinRes": "0",
            },
            "evenEigen": "false",
            "redBlack": "true",
            "output": output,
            "multiFile": multifile,
        },
    }


def epack_load(name: str, filestem: str, size: str, multifile: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MIO::StagLoadFermionEigenPack"},
        "options": {
            "redBlack": "true",
            "filestem": filestem,
            "multiFile": multifile,
            "size": size,
            "Ls": "1",
        },
    }


def eval_save(name: str, eigen_pack: str, output: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MUtilities::EigenPackExtractEvals"},
        "options": {"eigenPack": eigen_pack, "output": output},
    }


def epack_modify(name: str, eigen_pack: str, mass: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MUtilities::ModifyEigenPackMILC"},
        "options": {
            "eigenPack": eigen_pack,
            "mass": mass,
            "evenEigen": "false",
            "normalizeCheckerboard": "false",
        },
    }


def spin_taste(name: str) -> t.Dict:
    return {"id": {"name": name, "type": "MFermion::SpinTaste"}}


def sink(name: str, mom: str) -> t.Dict:
    return {"id": {"name": name, "type": "MSink::ScalarPoint"}, "options": {"mom": mom}}


def noise_rw(name: str, nsrc: str, t0: str, tstep: str, noise: str = "") -> t.Dict:
    return {
        "id": {"name": name, "type": "MSource::StagRandomWall"},
        "options": {
            "nSrc": nsrc,
            "tStep": tstep,
            "t0": t0,
            "colorDiag": "true",
            "noise": noise,
        },
    }


def time_diluted_noise(name: str, nsrc: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MNoise::StagTimeDilutedSpinColorDiagonal"},
        "options": {"nsrc": nsrc, "tStep": "1"},
    }


def full_volume_noise(name: str, nsrc: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MNoise::StagFullVolumeSpinColorDiagonal"},
        "options": {"nsrc": nsrc},
    }


def split_vec(name: str, source: str, indices: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MUtilities::StagSourcePickIndices"},
        "options": {
            "source": source,
            "indices": indices,
        },
    }


def rb_cg(name: str, action: str, residual: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MSolver::RBPrecCGMILC"},
        "options": {
            "action": action,
            "maxIteration": "10000",
            "residual": residual,
            "isEven": "false",
        },
    }


def cg(name: str, action: str, residual: str, guesser: str = "") -> t.Dict:
    return {
        "id": {"name": name, "type": "MSolver::StagCGMILC"},
        "options": {
            "action": action,
            "maxIteration": "10000",
            "residual": residual,
            "guesser": guesser,
        },
    }


def mixed_precision_cg(
    name: str, outer_action: str, inner_action: str, residual: str
) -> t.Dict:
    return {
        "id": {"name": name, "type": "MSolver::StagMixedPrecisionCG"},
        "options": {
            "outerAction": outer_action,
            "innerAction": inner_action,
            "maxOuterIteration": "10000",
            "maxInnerIteration": "10000",
            "residual": residual,
            "isEven": "false",
        },
    }


def lma_solver(name: str, action: str, low_modes: str) -> t.Dict:
    return {
        "id": {
            "name": name,
            "type": "MSolver::StagLMA",
        },
        "options": {
            "action": action,
            "lowModes": low_modes,
            "projector": "false",
            "eigStart": "0",
            "nEigs": "-1",
        },
    }


def quark_prop(
    name: str,
    source: str,
    solver: str,
    guess: str,
    gammas: str,
    gauge: str,
    apply_g5: str,
    subgrid: int | None = None,
) -> t.Dict:
    module = {
        "id": {
            "name": name,
            "type": "MFermion::StagGaugeProp",
        },
        "options": {
            "source": source,
            "solver": solver,
            "guess": guess,
            "spinTaste": {"gammas": gammas, "gauge": gauge, "applyG5": apply_g5},
        },
    }
    if subgrid is not None:
        module["subgrid"] = subgrid
    return module


def prop_contract(
    name: str,
    source: str,
    sink: str,
    sink_fn: str,
    source_shift: str,
    source_gammas: str,
    sink_gammas: str,
    gauge: str,
    output: str,
    apply_g5: str,
    subgrid: int | None = None,
) -> t.Dict:
    module = {
        "id": {
            "name": name,
            "type": "MContraction::StagMeson",
        },
        "options": {
            "source": source,
            "sink": sink,
            "sinkFunc": sink_fn,
            "sourceShift": source_shift,
            "sourceGammas": source_gammas,
            "sinkSpinTaste": {
                "gammas": sink_gammas,
                "gauge": gauge,
                "applyG5": apply_g5,
            },
            "output": output,
        },
    }
    if subgrid is not None:
        module["subgrid"] = subgrid
    return module


def meson_field(
    name: str,
    action: str,
    block: str,
    gammas: str,
    gauge: str,
    low_modes: str,
    left: str,
    right: str,
    output: str,
    apply_g5: str,
    cb_pairs_left: str = "",
    cb_pairs_right: str = "",
) -> t.Dict:
    """Wrap ``MContraction::StagA2AMesonField``.

    With ``cb_pairs_left``/``cb_pairs_right`` naming two distinct
    ``MUtilities::EigenPackCBPairs`` instances, the eigenvector rows come from
    on-demand checkerboard pairs (HadronsMILC 4d96243) — the options are
    injected only when set, keeping the legacy emission byte-identical. The
    C++ setup requires both set together (MesonField.hpp).
    """
    if bool(cb_pairs_left) != bool(cb_pairs_right):
        raise ValueError(
            "cbPairsLeft and cbPairsRight must be set together (two distinct "
            "MUtilities::EigenPackCBPairs instances); got "
            f"{cb_pairs_left!r} / {cb_pairs_right!r}."
        )
    options = {
        "action": action,
        "block": block,
        "mom": {
            "elem": "0 0 0",
        },
        "spinTaste": {"gammas": gammas, "gauge": gauge, "applyG5": apply_g5},
        "lowModes": low_modes,
        "left": left,
        "right": right,
        "output": output,
    }
    if cb_pairs_left:
        options["cbPairsLeft"] = cb_pairs_left
        options["cbPairsRight"] = cb_pairs_right
    return {
        "id": {
            "name": name,
            "type": "MContraction::StagA2AMesonField",
        },
        "options": options,
    }


def eigen_pack_cb_pairs(name: str, eigen_pack: str, action: str) -> t.Dict:
    """Wrap ``MUtilities::EigenPackCBPairs`` — on-demand CB pair source over a
    checkerboarded eigenpack (odd partner filled from Meooe on demand)."""
    return {
        "id": {"name": name, "type": "MUtilities::EigenPackCBPairs"},
        "options": {"action": action, "eigenPack": eigen_pack},
    }


def load_meson_field(
    name: str, file: str, dataset: str, side: str = "", low_modes: str = ""
) -> t.Dict:
    """Wrap ``MIO::LoadMesonField`` — read back a StagA2AMesonField file.

    ``file`` may embed the ``@traj@`` token (replaced with the trajectory
    counter at execute time; passes through pyfm's ``{...}`` format strings
    untouched). ``side``/``low_modes`` stay empty for the solver-input use.
    """
    return {
        "id": {"name": name, "type": "MIO::LoadMesonField"},
        "options": {
            "file": file,
            "dataset": dataset,
            "side": side,
            "lowModes": low_modes,
        },
    }


# The dead file-driven LMA solver wrapper formerly defined at this
# position was removed with the load-mode chain swap: HadronsMILC deleted
# that module type in dacc3fa, and lma_meson_field_prop below is its
# eager-producer replacement.


def lma_meson_field_prop(
    name: str,
    action: str,
    low_modes: str,
    meson_field: str,
    ta: str,
    tb: str,
    tstep: str,
    gammas: str,
    apply_g5: str,
    noise_index: str = "0",
    noise: str = "",
    n_noise: str = "1",
) -> t.Dict:
    """Wrap ``MFermion::StagLMAMesonFieldProp`` — eager low-mode producer.

    Reconstructs one ``PropagatorField`` per output name from precomputed
    meson-field files (the GaugeProp-style replacement of the former
    ``MSolver::StagLMAMesonField`` family, HadronsMILC dacc3fa): outputs
    are ``<name>_t<t>`` for a single gamma (or empty ``gammas``) and
    ``<name>_t<t>_<spin>_<taste>`` for multiple gammas, one per ``t`` in
    ``[tA, tB]`` stride ``tStep`` — the raw gamma labels (independent of
    ``applyG5``) name the per-gamma outputs, exactly GaugeProp's
    per-gamma guess grammar. ``meson_field`` is a whitespace-separated
    parallel list of ``MIO::LoadMesonField`` module names, one per gamma
    in raw order (each loader must hold the file produced under the
    applyG5-conjugated gamma; miswiring fails at the execute-time
    metadata cross-check). ``gauge`` stays empty — the module applies no
    spin-taste operator (fatal otherwise). ``noise`` names a
    ``<fvnoise>_vec`` object to enable the pairing/normalization
    self-check (run once per noise window). ``noise_index``/``n_noise``
    select the noise windows (HadronsMILC 8ea54f1): the module
    reconstructs ``noise_index..noise_index+n_noise-1``, window n
    reading three adjacent table columns (``noise_index`` counts noises,
    not columns); the execute-time coverage check needs
    ``(noise_index + n_noise) * 3`` columns. ``n_noise == 1`` publishes
    a scalar ``PropagatorField`` per output name (legacy grammar);
    ``n_noise > 1`` publishes a noise-major
    ``std::vector<PropagatorField>`` — the element-wise vector contract
    ``MContraction::Meson`` averages over, and the per-noise guess
    alignment GaugeProp's vector overload provides.
    """
    return {
        "id": {
            "name": name,
            "type": "MFermion::StagLMAMesonFieldProp",
        },
        "options": {
            "action": action,
            "lowModes": low_modes,
            "mesonField": meson_field,
            "spinTaste": {"gammas": gammas, "gauge": "", "applyG5": apply_g5},
            "noiseIndex": noise_index,
            "nNoise": n_noise,
            "tA": ta,
            "tB": tb,
            "tStep": tstep,
            "eigStart": "0",
            "nEigs": "-1",
            "negFirst": "",
            "pairScale": "",
            "noise": noise,
        },
    }


def qed_meson_field(
    name: str,
    action: str,
    block: str,
    em_fn: str,
    n_em_fields: str,
    em_seed_string: str,
    low_modes: str,
    left: str,
    right: str,
    output: str,
) -> t.Dict:
    return {
        "id": {
            "name": name,
            "type": "MContraction::StagA2AASlashMesonField",
        },
        "options": {
            "action": action,
            "block": block,
            "mom": {
                "elem": "0 0 0",
            },
            "EmFunc": em_fn,
            "nEmFields": n_em_fields,
            "EmSeedString": em_seed_string,
            "lowModes": low_modes,
            "left": left,
            "right": right,
            "output": output,
        },
    }


def em_fn(name: str, gauge: str, zm_scheme: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MGauge::StochEmFunc"},
        "options": {"gauge": gauge, "zmScheme": zm_scheme},
    }


def em_field(name: str, gauge: str, zm_scheme: str, improvement: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MGauge::StochEm"},
        "options": {"gauge": gauge, "zmScheme": zm_scheme, "improvement": improvement},
    }


def seq_aslash(name: str, q: str, ta: str, tb: str, em_field: str, mom: str) -> t.Dict:
    return {
        "id": {"name": name, "type": "MSource::StagSeqAslash"},
        "options": {
            "q": q,
            "tA": ta,
            "tB": tb,
            "emField": em_field,
            "mom": mom,
        },
    }


def seq_gamma(
    name: str,
    q: str,
    ta: str,
    tb: str,
    gammas: str,
    gauge: str,
    apply_g5: str,
    mom: str,
) -> t.Dict:
    return {
        "id": {"name": name, "type": "MSource::StagSeqGamma"},
        "options": {
            "q": q,
            "tA": ta,
            "tB": tb,
            "mom": mom,
            "spinTaste": {"gammas": gammas, "gauge": gauge, "applyG5": apply_g5},
        },
    }


def save_vector(name: str, field: str, output: str, multifile: str = "false") -> t.Dict:
    return {
        "id": {
            "name": name,
            "type": "MIO::SaveStagVector",
        },
        "options": {"field": field, "multiFile": multifile, "output": output},
    }


def a2a_vector(
    name: str, noise: str, action: str, low_modes: str, solver: str, high_output: str
) -> t.Dict:
    return {
        "id": {
            "name": name,
            "type": "MSolver::StagA2AVectors",
        },
        "options": {
            "noise": noise,
            "action": action,
            "lowModes": low_modes,
            "solver": solver,
            "highOutput": high_output,
            "norm2": "1.0",
            "highMultiFile": "false",
        },
    }


def load_vectors(
    name: str, filestem: str, size: str, multifile: str = "false"
) -> t.Dict:
    return {
        "id": {
            "name": name,
            "type": "MIO::StagLoadA2AVectors",
        },
        "options": {"filestem": filestem, "multiFile": multifile, "size": size},
    }
