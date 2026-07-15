#!/bin/bash
# GPU memory monitor for Frontier (AMD MI250X).
# Analogue of the Aurora `gpumem` script, which streams telemetry via xpu-smi.
# rocm-smi has no continuous dump mode, so the sampling loop lives here.
# Sample every 10s (matches the Aurora `xpu-smi -i10`).

if ! which rocm-smi >&/dev/null; then
  module load rocm
fi

while true; do
  echo "=== $(hostname) $(date) ==="
  rocm-smi --showmeminfo vram
  sleep 10
done
