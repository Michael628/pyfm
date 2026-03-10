# Tasks API

Task handlers generate input files for external programs (MILC, Hadrons), validate task completion, and aggregate output. See `pyfm/tasks/hadrons/lmi.py` for the exemplary implementation.

## Registration

::: pyfm.tasks.register

## Hadrons Tasks

::: pyfm.tasks.hadrons.lmi
::: pyfm.tasks.hadrons.gauge
::: pyfm.tasks.hadrons.meson
::: pyfm.tasks.hadrons.epack
::: pyfm.tasks.hadrons.raw
::: pyfm.tasks.hadrons.types
::: pyfm.tasks.hadrons.highmode.strategy
::: pyfm.tasks.hadrons.highmode.sib
::: pyfm.tasks.hadrons.highmode.twopoint

## MILC Tasks

::: pyfm.tasks.milc.smear

## Contraction Tasks

::: pyfm.tasks.contract.diagram
::: pyfm.tasks.contract.contraction
::: pyfm.tasks.contract.mesonloader
