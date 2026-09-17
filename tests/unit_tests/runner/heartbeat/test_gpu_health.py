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

import json

from flagscale.runner.heartbeat.gpu_health import parse_hygon_smi, parse_nvidia_smi_csv


def _row(
    *,
    corrected="0",
    uncorrected="0",
    retired_pending="[N/A]",
    remap_pending="No",
    remap_failure="No",
    thermal="Not Active",
    recovery="None",
):
    return ", ".join(
        [
            "0",
            "GPU-test",
            "00000000:10:00.0",
            "570.00",
            "42",
            "75",
            "1024",
            "81920",
            "250.0",
            "400.0",
            thermal,
            corrected,
            uncorrected,
            retired_pending,
            remap_pending,
            remap_failure,
            recovery,
        ]
    )


def test_healthy_sample_is_typed_without_creating_a_warning():
    baseline = {}
    gpu = parse_nvidia_smi_csv(_row(corrected="3"), baseline)[0]

    assert gpu["status"] == "healthy"
    assert gpu["temperature_c"] == 42
    assert gpu["volatile_corrected_ecc"] == 3
    assert gpu["volatile_corrected_ecc_delta"] == 0
    assert gpu["retired_pages_pending"] is None


def test_new_corrected_ecc_and_thermal_slowdown_are_warnings():
    baseline = {"GPU-test": 3}
    gpu = parse_nvidia_smi_csv(
        _row(corrected="4", thermal="Active", remap_pending="Yes"), baseline
    )[0]

    assert gpu["status"] == "warning"
    assert gpu["volatile_corrected_ecc_delta"] == 1
    assert {issue["reason"] for issue in gpu["issues"]} == {
        "new_volatile_corrected_ecc",
        "hardware_thermal_slowdown",
        "row_remap_pending",
    }


def test_uncorrected_ecc_remap_failure_and_recovery_action_are_unhealthy():
    gpu = parse_nvidia_smi_csv(_row(uncorrected="1", remap_failure="Yes", recovery="Reset"))[0]

    assert gpu["status"] == "unhealthy"
    assert {issue["reason"] for issue in gpu["issues"]} == {
        "volatile_uncorrected_ecc_present",
        "row_remap_failure",
        "gpu_recovery_action_required",
    }


def _hygon_inventory():
    return json.dumps(
        {
            "card0": {
                "Unique ID": "DCU-test",
                "PCI Bus": "0000:09:00.0 --> SN: test",
                "Temperature (Sensor edge) (C)": "53.0",
                "Temperature (Sensor junction) (C)": "59.0",
                "Temperature (Sensor mem) (C)": "52.0",
                "HCU memory use (%)": "25",
                "Vram memory ECC status": "ON",
            }
        }
    )


def _hygon_ras(*, corrected=0, uncorrected=0):
    return json.dumps(
        {
            "card0": {
                "Block umc is": f"ENABLED (ue: {uncorrected}, ce: {corrected})",
                "Block gfx is": "ENABLED (ue: 0, ce: 0)",
            }
        }
    )


def test_hygon_healthy_sample_has_identity_temperature_memory_and_ras():
    gpu = parse_hygon_smi(
        _hygon_inventory(),
        _hygon_ras(corrected=3),
        "HCU[0] : Bus Id : 0000:09:00.0 : Healthy",
        {},
    )[0]

    assert gpu["status"] == "healthy"
    assert gpu["uuid"] == "DCU-test"
    assert gpu["pci_bus_id"] == "0000:09:00.0"
    assert gpu["temperature_c"] == 53.0
    assert gpu["memory_utilization_percent"] == 25.0
    assert gpu["volatile_corrected_ecc_delta"] == 0


def test_hygon_new_corrected_error_warns_and_uncorrected_error_fails():
    warning_gpu = parse_hygon_smi(
        _hygon_inventory(),
        _hygon_ras(corrected=4),
        "HCU[0] : Bus Id : 0000:09:00.0 : Healthy",
        {"DCU-test": 3},
    )[0]
    unhealthy_gpu = parse_hygon_smi(
        _hygon_inventory(),
        _hygon_ras(uncorrected=1),
        "HCU[0] : Bus Id : 0000:09:00.0 : Healthy",
    )[0]

    assert warning_gpu["status"] == "warning"
    assert warning_gpu["issues"] == [{"severity": "warning", "reason": "new_ras_corrected_ecc"}]
    assert unhealthy_gpu["status"] == "unhealthy"
    assert unhealthy_gpu["issues"] == [
        {"severity": "unhealthy", "reason": "ras_uncorrected_ecc_present"}
    ]
