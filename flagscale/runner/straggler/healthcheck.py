"""
Network health check for distributed training.
"""

import socket
import subprocess
import time

from typing import Dict, List, Optional, Tuple


class NetworkHealthChecker:
    """
    Checks network connectivity and health before and during training.
    This is particularly useful for:
    - Elastic training scenarios (nodes can join/leave)
    - Multi-node training setup validation
    - Ongoing health monitoring during training
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.node_health: Dict[int, bool] = {}
        self.latency_matrix: Dict[Tuple[int, int], float] = {}

    def check_node_connectivity(
        self, node_ips: List[str], port: int = 29500, timeout: float = 5.0
    ) -> Dict[str, bool]:
        """
        Check connectivity to a list of node IPs.
        Args:
            node_ips: List of IP addresses to check
            port: Port to test connectivity on
            timeout: Timeout for each connection attempt
        Returns:
            Dictionary mapping IP to connectivity status
        """
        results = {}

        for ip in node_ips:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                result = sock.connect_ex((ip, port))
                sock.close()

                results[ip] = result == 0
            except Exception as e:
                results[ip] = False

        return results

    def measure_latency(
        self, node_ips: List[str], port: int = 29500, num_pings: int = 3
    ) -> Dict[str, float]:
        """
        Measure network latency to nodes using ping.
        Args:
            node_ips: List of IP addresses
            port: Port to test (if using TCP)
            num_pings: Number of ping attempts
        Returns:
            Dictionary mapping IP to average latency in ms
        """
        latencies = {}

        for ip in node_ips:
            try:
                # Use ping to measure latency
                result = subprocess.run(
                    ["ping", "-c", str(num_pings), "-W", "1", ip],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                if result.returncode == 0:
                    # Parse ping output to extract average latency
                    lines = result.stdout.split('\n')
                    for line in lines:
                        if 'avg' in line or 'Average' in line:
                            # Extract latency value (format varies by system)
                            parts = line.split('/')
                            if len(parts) >= 5:
                                avg_latency = float(parts[4])
                                latencies[ip] = avg_latency
                                break
                    else:
                        # Try alternative parsing
                        latencies[ip] = 0.0
                else:
                    latencies[ip] = float('inf')
            except Exception as e:
                latencies[ip] = float('inf')

        return latencies

    def check_bandwidth(
        self, node_ips: List[str], test_size: int = 1024 * 1024  # 1 MB
    ) -> Dict[str, float]:
        """
        Estimate bandwidth to nodes using a simple test.
        Note: This is a simplified bandwidth estimation.
        In practice, you'd use tools like iperf for accurate measurements.
        Args:
            node_ips: List of IP addresses
            test_size: Size of test data in bytes
        Returns:
            Dictionary mapping IP to bandwidth in Mbps
        """
        bandwidths = {}

        for ip in node_ips:
            try:
                # Create a socket and measure transfer time
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10.0)

                start_time = time.time()
                # Attempt connection (not actual data transfer)
                sock.connect((ip, 29500))
                end_time = time.time()
                sock.close()

                # Simple bandwidth estimate based on connection time
                elapsed = end_time - start_time
                if elapsed > 0:
                    # Rough estimate: connection overhead / test_size
                    # This is not accurate but gives a relative measure
                    bandwidth_mbps = (test_size / elapsed) / (1024 * 1024)
                    bandwidths[ip] = bandwidth_mbps
                else:
                    bandwidths[ip] = 0.0
            except Exception:
                bandwidths[ip] = 0.0

        return bandwidths

    def comprehensive_health_check(
        self, node_ips: List[str], port: int = 29500
    ) -> Dict[str, Dict[str, any]]:
        """
        Perform a comprehensive health check on all nodes.
        Args:
            node_ips: List of IP addresses
            port: Port to check
        Returns:
            Dictionary mapping IP to health metrics
        """
        results = {}

        # Check connectivity
        connectivity = self.check_node_connectivity(node_ips, port)

        # Measure latency
        latencies = self.measure_latency(node_ips, port)

        # Estimate bandwidth
        bandwidths = self.check_bandwidth(node_ips)

        # Compile results
        for ip in node_ips:
            results[ip] = {
                "connectivity": connectivity.get(ip, False),
                "latency_ms": latencies.get(ip, float('inf')),
                "bandwidth_mbps": bandwidths.get(ip, 0.0),
                "healthy": connectivity.get(ip, False),
            }

        return results

    def identify_unhealthy_nodes(
        self,
        health_results: Dict[str, Dict[str, any]],
        max_latency_ms: float = 100.0,
        min_bandwidth_mbps: float = 10.0,
    ) -> List[str]:
        """
        Identify nodes that are unhealthy based on thresholds.
        Args:
            health_results: Results from comprehensive_health_check
            max_latency_ms: Maximum acceptable latency in ms
            min_bandwidth_mbps: Minimum acceptable bandwidth in Mbps
        Returns:
            List of unhealthy node IPs
        """
        unhealthy = []

        for ip, metrics in health_results.items():
            if not metrics["connectivity"]:
                unhealthy.append(ip)
            elif metrics["latency_ms"] > max_latency_ms:
                unhealthy.append(ip)
            elif metrics["bandwidth_mbps"] < min_bandwidth_mbps:
                unhealthy.append(ip)

        return unhealthy

    def get_network_summary(self, node_ips: List[str], port: int = 29500) -> Dict[str, any]:
        """
        Get a summary of network health across all nodes.
        Args:
            node_ips: List of IP addresses
            port: Port to check
        Returns:
            Summary dictionary with statistics
        """
        health_results = self.comprehensive_health_check(node_ips, port)
        unhealthy = self.identify_unhealthy_nodes(health_results)

        # Calculate statistics
        total_nodes = len(node_ips)
        healthy_nodes = total_nodes - len(unhealthy)

        # Average latency (only for reachable nodes)
        reachable_nodes = [ip for ip, metrics in health_results.items() if metrics["connectivity"]]
        avg_latency = 0.0
        if reachable_nodes:
            latencies = [health_results[ip]["latency_ms"] for ip in reachable_nodes]
            avg_latency = sum(latencies) / len(latencies)

        # Average bandwidth
        avg_bandwidth = 0.0
        if reachable_nodes:
            bandwidths = [health_results[ip]["bandwidth_mbps"] for ip in reachable_nodes]
            avg_bandwidth = sum(bandwidths) / len(bandwidths)

        return {
            "total_nodes": total_nodes,
            "healthy_nodes": healthy_nodes,
            "unhealthy_nodes": unhealthy,
            "health_percentage": (healthy_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "average_latency_ms": avg_latency,
            "average_bandwidth_mbps": avg_bandwidth,
            "network_healthy": len(unhealthy) == 0,
        }

    def save_health_report(self, health_results: Dict[str, Dict[str, any]], filepath: str):
        """
        Save health check results to a file.
        Args:
            health_results: Results from comprehensive_health_check
            filepath: Path to save the report
        """
        try:
            with open(filepath, 'w') as f:
                f.write("Network Health Check Report\n")
                f.write("=" * 50 + "\n\n")

                for ip, metrics in health_results.items():
                    f.write(f"Node: {ip}\n")
                    f.write(f"  Connectivity: {'✓' if metrics['connectivity'] else '✗'}\n")
                    f.write(f"  Latency: {metrics['latency_ms']:.2f} ms\n")
                    f.write(f"  Bandwidth: {metrics['bandwidth_mbps']:.2f} Mbps\n")
                    f.write(f"  Healthy: {'✓' if metrics['healthy'] else '✗'}\n")
                    f.write("\n")
        except Exception as e:
            print(f"Warning: Could not save health report: {e}")


class ElasticTrainingHealthChecker(NetworkHealthChecker):
    """
    Specialized health checker for elastic training scenarios.
    Elastic training requires more frequent checks and the ability
    to handle changing node sets.
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        super().__init__(rank, world_size)
        self.health_history: List[Dict] = []

    def monitor_elastic_health(
        self,
        node_ips: List[str],
        port: int = 29500,
        check_interval: float = 30.0,
        num_checks: int = 10,
    ) -> List[Dict[str, any]]:
        """
        Monitor health over time for elastic training.
        Args:
            node_ips: List of IP addresses to monitor
            port: Port to check
            check_interval: Time between checks in seconds
            num_checks: Number of checks to perform
        Returns:
            List of health check results over time
        """
        results = []

        for i in range(num_checks):
            check_result = {
                "check_id": i,
                "timestamp": time.time(),
                "health_results": self.comprehensive_health_check(node_ips, port),
            }

            results.append(check_result)
            self.health_history.append(check_result)

            if i < num_checks - 1:  # Don't sleep after the last check
                time.sleep(check_interval)

        return results

    def detect_unstable_nodes(
        self, health_history: List[Dict[str, any]], instability_threshold: float = 0.3
    ) -> List[str]:
        """
        Detect nodes that are unstable (connectivity issues).
        Args:
            health_history: Results from monitor_elastic_health
            instability_threshold: Fraction of checks that must fail for a node to be considered unstable
        Returns:
            List of unstable node IPs
        """
        node_failures = {}

        for check in health_history:
            for ip, metrics in check["health_results"].items():
                if not metrics["connectivity"]:
                    if ip not in node_failures:
                        node_failures[ip] = 0
                    node_failures[ip] += 1

        unstable_nodes = []
        total_checks = len(health_history)

        for ip, failures in node_failures.items():
            failure_rate = failures / total_checks
            if failure_rate >= instability_threshold:
                unstable_nodes.append(ip)

        return unstable_nodes
