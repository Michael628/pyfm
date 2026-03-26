import typing as t

"""
This module contains python wrappers for Hadrons modules.
"""


def xml_wrapper(run_seed: str, series: str, cfg: str) -> dict:
    params = dict(
        grid=dict(parameters=dict(runSeed=run_seed, trajectory=cfg, series=series))
    )

    return params


def gauge_files(link: str, fatlink: str, longlink: str) -> t.Dict:
    return dict(type="file", link=link, fatlink=fatlink, longlink=longlink)


def action(label: str, mass: str) -> t.Dict:
    return dict(
        label=label,
        boundary="1 1 1 1",
        mass=mass,
        c1="1.0",
        c2="1.0",
        tad="1.0",
        twist="0 0 0",
    )


def irl(
    alpha: str,
    beta: str,
    npoly: str,
    nstop: str,
    nk: str,
    nm: str,
    residual: str,
) -> t.Dict:
    return dict(
        lanczosParams=dict(
            Cheby=dict(alpha=alpha, beta=beta, Npoly=npoly),
            Nstop=nstop,
            Nk=nk,
            Nm=nm,
            resid=residual,
            MaxIt="5000",
            betastp="0",
            MinRes="0",
        ),
        evenEigen="false",
    )


def epack(
    op_type: str,
    file: str,
    size: str,
    eval_save: str,
    /,
    seed: str = "epack",
    multifile: str = "false",
    mass: str | None = None,
    **irl_kwargs,
) -> t.Dict:
    epack = dict(
        type=op_type,
        checker="odd",
        seed=seed,
        size=size,
        file=file,
        multiFile=multifile,
        evalSave=eval_save,
    )
    if op_type == "solve":
        if mass is None:
            raise ValueError('mass must be specified for epack type "solve"')
        epack["action"] = action("irl", mass)
        epack["irl"] = irl(**irl_kwargs)

    return epack


def lma(
    projector: str = "false",
    eig_start: str = "0",
    n_eigs: str = "-1",
) -> t.Dict:
    return dict(projector=projector, eigStart=eig_start, nEigs=n_eigs)


def mpcg(
    residual: str = "1e-08",
    max_inner_iteration: str = "10000",
    max_outer_iteration: str = "10000",
) -> t.Dict:
    return dict(
        maxInnerIteration=max_inner_iteration,
        maxOuterIteration=max_outer_iteration,
        residual=residual,
    )


def random_wall_source(
    t_step: str,
    t0: str,
    n_src: str,
    seed: str = "noise",
) -> t.Dict:
    return dict(tStep=t_step, t0=t0, nSrc=n_src, seed=seed)


def spin_taste(gammas: str, apply_g5: str = "true") -> t.Dict:
    return dict(applyG5=apply_g5, gammas=gammas)


def contraction(
    antiquark_solver: str,
    antiquark_action: str,
    quark_solver: str,
    quark_action: str,
    quark: t.Dict,
    antiquark: t.Dict,
    sink: t.Dict,
    output: str,
) -> t.Dict:
    return dict(
        antiquarkAction=antiquark_action,
        antiquarkSolver=antiquark_solver,
        quarkAction=quark_action,
        quarkSolver=quark_solver,
        quark=quark,
        antiquark=antiquark,
        sink=sink,
        output=output,
    )


def meson_field(
    block: str,
    mass: str,
    output: str,
    mom: t.List[str] = ["0", "0", "0"],
    **spin_taste_kwargs,
) -> t.Dict:
    return dict(
        block=block,
        action=action("a2a", mass),
        output=output,
        spinTaste=spin_taste(**spin_taste_kwargs),
        mom=dict(elem=" ".join(mom)),
    )
