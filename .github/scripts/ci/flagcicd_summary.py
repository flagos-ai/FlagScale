# Copyright 2026 FlagOS Contributors
# Licensed under the Apache License, Version 2.0.
"""Human-readable presentation of a plan; never changes selection or identity."""

import html
import json
from collections import Counter


def cell(value):
    return html.escape(str(value), quote=True).replace("|", "&#124;").replace("\n", " ")


def render_summary(plan):
    lines = [
        "## Test plan",
        "",
        f"- Stage: **{cell(plan['profile'])}** · Selection: **{cell(plan['selection'])}**",
        f"- Platforms: {', '.join(cell(p) for p in plan['platforms']) or 'None (common checks only)'}",
        "- Common checks: lint and CI controller tests.",
        "- Counts below are test files and configured functional cases, not collected pytest test counts.",
        "",
        "| Platform | Device | Unit scope | Unit files | Processes | Train | Hetero | Benchmark |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for platform in plan["platforms"]:
        part = plan["slices"][platform]
        for device in part["devices"]:
            counts = Counter(c["task"] for c in part["cases"] if c["device"] == device)
            values = [
                platform,
                device,
                part["unit_mode"],
                len(part["unit_files"][device]),
                part["unit_ranks"][device],
                counts["train"],
                counts["hetero_train"],
                counts["benchmark"],
            ]
            lines.append("| " + " | ".join(cell(v) for v in values) + " |")
    full = [p for p in plan["platforms"] if plan["slices"][p]["unit_mode"] == "full"]
    if full:
        lines += [
            "",
            "Configured full unit scope: "
            + ", ".join(cell(p) for p in full)
            + ". An affected request can conservatively expand to full scope; see selection reasons below.",
        ]
    if plan["gaps"]:
        lines += [
            "",
            f"**Validation gaps: {len(plan['gaps'])}.** See details; this plan does not claim coverage for missing cases.",
        ]
    lines += [
        "",
        "<details>",
        "<summary>Selection reasons, groups and validation gaps</summary>",
        "",
    ]
    for key in ("reasons", "direct_groups", "selected_groups", "gaps"):
        lines += [f"**{cell(key)}**", ""] + [f"- {cell(v)}" for v in plan[key]] + [""]
    lines += [
        "</details>",
        "",
        "<details>",
        "<summary>Plan identity and complete plan</summary>",
        "",
        f"Plan hash: <code>{cell(plan['plan_hash'])}</code>",
        "",
        "<pre>" + html.escape(json.dumps(plan, indent=2)) + "</pre>",
        "",
        "</details>",
        "",
        "The complete machine-readable plan is also available in the `ci-plan` artifact.",
        "",
    ]
    return "\n".join(lines)
