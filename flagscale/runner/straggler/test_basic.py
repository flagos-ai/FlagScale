#!/usr/bin/env python3
"""
Basic test script for straggler module.
This script tests the basic functionality of the straggler detection module.
"""

import sys
import time

# Test imports
try:
    from flagscale.runner.straggler.config import StragglerConfig

    print("✓ Successfully imported StragglerConfig")
except ImportError as e:
    print(f"✗ Failed to import StragglerConfig: {e}")
    sys.exit(1)

try:
    from flagscale.runner.straggler.detector import StragglerDetector

    print("✓ Successfully imported StragglerDetector")
except ImportError as e:
    print(f"✗ Failed to import StragglerDetector: {e}")
    sys.exit(1)

try:
    from flagscale.runner.straggler.report import StragglerReport

    print("✓ Successfully imported StragglerReport")
except ImportError as e:
    print(f"✗ Failed to import StragglerReport: {e}")
    sys.exit(1)

try:
    from flagscale.runner.straggler.section import SectionContext

    print("✓ Successfully imported SectionContext")
except ImportError as e:
    print(f"✗ Failed to import SectionContext: {e}")
    sys.exit(1)

try:
    from flagscale.runner.straggler.comm import CommStatsCollector

    print("✓ Successfully imported CommStatsCollector")
except ImportError as e:
    print(f"✗ Failed to import CommStatsCollector: {e}")
    sys.exit(1)

try:
    from flagscale.runner.straggler.healthcheck import NetworkHealthChecker

    print("✓ Successfully imported NetworkHealthChecker")
except ImportError as e:
    print(f"✗ Failed to import NetworkHealthChecker: {e}")
    sys.exit(1)

# Test basic functionality
print("\n--- Testing Basic Functionality ---")

# Test StragglerConfig
config = StragglerConfig(
    enabled=True,
    scores_to_compute="all",
    profiling_interval=10,
    report_interval_steps=100,
    monitor_sections=["dataloader", "forward", "backward", "optimizer"],
)
print(f"✓ Created StragglerConfig: enabled={config.enabled}")

# Test StragglerDetector
detector = StragglerDetector(config=config, rank=0, world_size=4, node_name="node-0")
print(f"✓ Created StragglerDetector: rank={detector.rank}")

# Test SectionContext
print("\n--- Testing SectionContext ---")
with SectionContext(detector, "test_section", profile_cuda=False) as ctx:
    time.sleep(0.1)  # Simulate work
print("✓ SectionContext works correctly")

# Test StragglerReport
report = StragglerReport(
    step=100,
    section_scores={"forward": {0: 1.0, 1: 0.8, 2: 0.6}},
    gpu_scores={0: 1.0, 1: 0.9, 2: 0.7},
    straggler_ranks=[2],
    node_names={0: "node-0", 1: "node-1", 2: "node-2"},
)
print(f"✓ Created StragglerReport: step={report.step}")

# Test report methods
report_dict = report.to_dict()
print(f"✓ to_dict() works: {len(report_dict)} fields")

report_text = report.to_text()
print(f"✓ to_text() works: {len(report_text)} characters")

# Test identify_stragglers
stragglers = report.identify_stragglers(threshold=1.5)
print(f"✓ identify_stragglers() works: {stragglers}")

# Test CommStatsCollector
print("\n--- Testing CommStatsCollector ---")
collector = CommStatsCollector(enabled=True)
collector.set_backend_info("nccl", 4, 0)
print(f"✓ Created CommStatsCollector: backend={collector.backend}")

# Record a dummy operation
start_time = time.perf_counter()
time.sleep(0.01)
end_time = time.perf_counter()
collector.record_operation(
    op_type="all_reduce",
    op_name="gradient_sync",
    start_time=start_time,
    end_time=end_time,
    data_size=1024,
)
print("✓ Recorded communication operation")

# Test NetworkHealthChecker
print("\n--- Testing NetworkHealthChecker ---")
checker = NetworkHealthChecker(rank=0, world_size=4)
print(f"✓ Created NetworkHealthChecker: world_size={checker.world_size}")

print("\n=== All tests passed! ===")
