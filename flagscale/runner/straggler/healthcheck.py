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

"""Network health helpers for straggler workflows."""

import socket
import subprocess
import time


class NetworkHealthChecker:
    """Check basic network reachability and quality."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.node_health: dict[int, bool] = {}
        self.latency_matrix: dict[tuple[int, int], float] = {}

    def check_node_connectivity(
        self,
        node_ips: list[str],
        port: int = 29500,
        timeout: float = 5.0,
    ) -> dict[str, bool]:
        results = {}
        for ip in node_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                sock.close()
                results[ip] = result == 0
            except Exception:
                results[ip] = False
        return results

    def measure_latency(
        self,
        node_ips: list[str],
        port: int = 29500,
        num_pings: int = 3,
    ) -> dict[str, float]:
        latencies = {}
        for ip in node_ips:
            try:
                result = subprocess.run(
                    ["ping", "-c", str(num_pings), "-W", "1", ip],
                    capture_output=True,
                    text=True,
                )
                if result.returncode != 0:
                    latencies[ip] = float("inf")
                    continue

                avg_latency = float("inf")
                for line in result.stdout.splitlines():
                    if "avg" not in line and "Average" not in line:
                        continue
                    parts = line.split("/")
                    if len(parts) >= 5:
                        avg_latency = float(parts[4])
                        break
                latencies[ip] = avg_latency
            except Exception:
                latencies[ip] = float("inf")
        return latencies

    def check_bandwidth(
        self,
        node_ips: list[str],
        test_size: int = 1024 * 1024,
    ) -> dict[str, float]:
        bandwidths = {}
        for ip in node_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)
                start_time = time.time()
                sock.connect((ip, 29500))
                end_time = time.time()
                sock.close()

                elapsed = end_time - start_time
                if elapsed > 0:
                    bandwidths[ip] = (test_size / elapsed) / (1024 * 1024)
                else:
                    bandwidths[ip] = 0.0
            except Exception:
                bandwidths[ip] = 0.0
        return bandwidths

    def comprehensive_health_check(
        self,
        node_ips: list[str],
        port: int = 29500,
    ) -> dict[str, dict[str, object]]:
        results = {}
        connectivity = self.check_node_connectivity(node_ips, port)
        latencies = self.measure_latency(node_ips, port)
        bandwidths = self.check_bandwidth(node_ips)

        for ip in node_ips:
            results[ip] = {
                "connectivity": connectivity.get(ip, False),
                "latency_ms": latencies.get(ip, float("inf")),
                "bandwidth_mbps": bandwidths.get(ip, 0.0),
                "healthy": connectivity.get(ip, False),
            }
        return results

    def identify_unhealthy_nodes(
        self,
        health_results: dict[str, dict[str, object]],
        max_latency_ms: float = 100.0,
        min_bandwidth_mbps: float = 10.0,
    ) -> list[str]:
        unhealthy = []
        for ip, metrics in health_results.items():
            if not metrics["connectivity"]:
                unhealthy.append(ip)
            elif metrics["latency_ms"] > max_latency_ms:
                unhealthy.append(ip)
            elif metrics["bandwidth_mbps"] < min_bandwidth_mbps:
                unhealthy.append(ip)
        return unhealthy

    def get_network_summary(
        self,
        node_ips: list[str],
        port: int = 29500,
    ) -> dict[str, object]:
        health_results = self.comprehensive_health_check(node_ips, port)
        unhealthy = self.identify_unhealthy_nodes(health_results)

        total_nodes = len(node_ips)
        healthy_nodes = total_nodes - len(unhealthy)
        reachable_nodes = [ip for ip, metrics in health_results.items() if metrics["connectivity"]]

        avg_latency = 0.0
        if reachable_nodes:
            avg_latency = sum(health_results[ip]["latency_ms"] for ip in reachable_nodes) / len(
                reachable_nodes
            )

        avg_bandwidth = 0.0
        if reachable_nodes:
            avg_bandwidth = sum(
                health_results[ip]["bandwidth_mbps"] for ip in reachable_nodes
            ) / len(reachable_nodes)

        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": unhealthy,
            "health_percentage": (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "average_latency_ms": avg_latency,
            "average_bandwidth_mbps": avg_bandwidth,
            "network_healthy": len(unhealthy) == 0,
        }

    def save_health_report(
        self,
        health_results: dict[str, dict[str, object]],
        filepath: str,
    ):
        try:
            with open(filepath, "w") as file_obj:
                file_obj.write("Network Health Check Report\n")
                file_obj.write("=" * 50 + "\n\n")
                for ip, metrics in health_results.items():
                    file_obj.write(f"Node: {ip}\n")
                    file_obj.write(f"  Connectivity: {metrics['connectivity']}\n")
                    file_obj.write(f"  Latency: {metrics['latency_ms']:.2f} ms\n")
                    file_obj.write(f"  Bandwidth: {metrics['bandwidth_mbps']:.2f} Mbps\n")
                    file_obj.write(f"  Healthy: {metrics['healthy']}\n")
                    file_obj.write("\n")
        except Exception as exc:
            print(f"Warning: Could not save health report: {exc}")


class ElasticTrainingHealthChecker(NetworkHealthChecker):
    """Health checker variant for elastic training."""

    def __init__(self, rank: int = 0, world_size: int = 1):
        super().__init__(rank, world_size)
        self.health_history: list[dict[str, object]] = []

    def monitor_elastic_health(
        self,
        node_ips: list[str],
        port: int = 29500,
        check_interval: float = 30.0,
        num_checks: int = 10,
    ) -> list[dict[str, object]]:
        results = []
        for check_id in range(num_checks):
            check_result = {
                "check_id": check_id,
                "timestamp": time.time(),
                "health_results": self.comprehensive_health_check(node_ips, port),
            }
            results.append(check_result)
            self.health_history.append(check_result)
            if check_id < num_checks - 1:
                time.sleep(check_interval)
        return results

    def detect_unstable_nodes(
        self,
        health_history: list[dict[str, object]],
        instability_threshold: float = 0.3,
    ) -> list[str]:
        node_failures = {}
        for check in health_history:
            for ip, metrics in check["health_results"].items():
                if not metrics["connectivity"]:
                    node_failures[ip] = node_failures.get(ip, 0) + 1

        unstable_nodes = []
        total_checks = len(health_history)
        for ip, failures in node_failures.items():
            failure_rate = failures / total_checks
            if failure_rate >= instability_threshold:
                unstable_nodes.append(ip)
        return unstable_nodes
