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

"""FlagScale entrypoint for KERV speculative VLA inference."""

import argparse

from flagscale.models.kerv import launch_kerv_stage, load_kerv_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch KERV inference")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_kerv_config(args.config_path)
    if config.stage != "inference":
        raise ValueError(f"expected KERV inference stage, got: {config.stage}")
    launch_kerv_stage(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
