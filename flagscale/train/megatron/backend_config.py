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

"""Configure FlagOS backend selection before backend-dependent initialization."""

import os
import sys


_BACKEND_ENVIRONMENT_ARGUMENTS = (
    ("mg_fl_prefer", "MG_FL_PREFER"),
    ("te_fl_prefer", "TE_FL_PREFER"),
    ("te_fl_per_op", "TE_FL_PER_OP"),
    ("te_fl_allow_vendors", "TE_FL_ALLOW_VENDORS"),
    ("te_fl_deny_vendors", "TE_FL_DENY_VENDORS"),
)


def configure_backend_environment(args) -> None:
    """Apply backend-selection arguments before the first FL-dispatched call.

    TransformerEngine-FL reads its environment-backed selection policy lazily and
    caches it. Avoid importing TransformerEngine here, but invalidate the policy if
    it was imported earlier in the process so that the next dispatch sees the new
    environment.
    """
    te_policy_configured = False

    for argument_name, environment_name in _BACKEND_ENVIRONMENT_ARGUMENTS:
        value = getattr(args, argument_name, None)
        if value is None or value == "":
            continue

        os.environ[environment_name] = str(value)
        if environment_name.startswith("TE_FL_"):
            te_policy_configured = True

    if not te_policy_configured:
        return

    policy_module = sys.modules.get("transformer_engine.plugin.core.policy")
    reset_global_policy = getattr(policy_module, "reset_global_policy", None)
    if callable(reset_global_policy):
        reset_global_policy()
