#!/bin/bash

# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

before_file=${1:?pip-check before file is required}
after_file=${2:?pip-check after file is required}

[ -f "$before_file" ] || {
    echo "pip-check before file not found: $before_file" >&2
    exit 1
}
[ -f "$after_file" ] || {
    echo "pip-check after file not found: $after_file" >&2
    exit 1
}

normalize() {
    sed '/^[[:space:]]*$/d; /^No broken requirements found\.$/d' "$1" | \
        LC_ALL=C sort -u
}

new_conflicts=$(comm -13 <(normalize "$before_file") <(normalize "$after_file"))

if [ -n "$new_conflicts" ]; then
    echo "New Python dependency conflicts were introduced:" >&2
    printf '%s\n' "$new_conflicts" >&2
    exit 1
fi

remaining_conflicts=$(normalize "$after_file")
if [ -n "$remaining_conflicts" ]; then
    echo "Existing Python dependency conflicts (unchanged or reduced):"
    printf '%s\n' "$remaining_conflicts"
fi
