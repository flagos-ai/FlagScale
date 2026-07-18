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

# Namespace package configuration
# This allows megatron.core and megatron.plugin to be imported from the installed
# megatron-core package, while other modules (training, rl, post_training, legacy)
# are imported from this source directory.

# Make this a namespace package to allow imports from multiple locations
try:
    import pkgutil

    __path__ = pkgutil.extend_path(__path__, __name__)
except (AttributeError, NameError):
    # If __path__ doesn't exist yet, create it
    import os

    __path__ = [os.path.dirname(__file__)]

# The installed megatron-core package will have megatron.core and megatron.plugin
# They will be automatically available through the namespace package mechanism
# No need to explicitly import them here
