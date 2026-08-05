#!/bin/bash
set -e

# Official evaluation notes released on 2026-06-02:
# - Timed infer range includes moving the batch to device, model forward,
#   sigmoid, masking, CPU copy, and list conversion.
# - Input sampling/truncation is not allowed.
# - Unsafe structural pruning is not allowed: no layer-count reduction,
#   hidden-size reduction, attention-head deletion, or FFN-channel deletion.
# - Loading a checkpoint and then manually changing learned weights is high risk.
# - Safe inference optimizations are allowed when the network structure is kept.
# - Online infer data is different from the baseline dataset.
# - Online hardware/environment: A800, GCC 9.4.0 on Ubuntu 20.04, CUDA 12.6.3.

echo "build env success"
