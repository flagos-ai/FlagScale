#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

# Environment shared by local development and the generic image workflow.
: "${FLAGSCALE_HOME:=/opt/flagscale}"
: "${FLAGSCALE_DEPS:=$FLAGSCALE_HOME/deps}"
: "${DTK_HOME:=/opt/dtk}"

export FLAGSCALE_HOME FLAGSCALE_DEPS DTK_HOME
export PATH="$DTK_HOME/bin:$DTK_HOME/cuda/cuda-12/bin:$PATH"
export LD_LIBRARY_PATH="$DTK_HOME/lib:$DTK_HOME/lib64:$DTK_HOME/cuda/cuda-12/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
