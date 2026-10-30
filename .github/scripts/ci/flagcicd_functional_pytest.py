# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Bind numerical-comparison JUnit to its actual runner arguments."""

import pytest


@pytest.fixture(autouse=True)
def functional_identity(request, record_testsuite_property):
    for field in ("platform", "device", "task", "model", "case"):
        record_testsuite_property("flagcicd." + field, str(request.config.getoption("--" + field)))
