# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native KERV speculative-decoding runtime and FlagScale launch helpers."""

from .launcher import build_kerv_command, launch_kerv_stage, load_kerv_config
from .runtime import (
    CandidateTree,
    DraftCallable,
    KERVConfig,
    KERVRuntime,
    KERVStepResult,
    VerificationResult,
    VerifyCallable,
    build_candidate_tree,
    compute_dynamic_threshold,
    generate_candidate_tree,
    kalman_predict,
    verify_candidates,
)

__all__ = [
    "CandidateTree",
    "DraftCallable",
    "KERVConfig",
    "KERVRuntime",
    "KERVStepResult",
    "VerificationResult",
    "VerifyCallable",
    "build_candidate_tree",
    "build_kerv_command",
    "compute_dynamic_threshold",
    "generate_candidate_tree",
    "kalman_predict",
    "launch_kerv_stage",
    "load_kerv_config",
    "verify_candidates",
]
