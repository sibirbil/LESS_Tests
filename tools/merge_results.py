"""Merge a results directory produced on another machine into this one.

Two machines may run different methods on the same problems, so each ends up
with its own copy of every <PROBLEM>.json. This merges the source copy into the
target one, key by key. Both directories must carry the same _protocol.json
(same CV settings, grid size and library versions). A value present on both
sides must be identical; anything else stops the merge before a file is written.

    python tools/merge_results.py <source_dir> <target_dir> [--dry-run]
"""

import argparse
import json
import os
import sys


def merge(dst, src, path=""):
    """Merge src into dst in place; return a list of conflicting key paths."""
    conflicts = []
    for key, value in src.items():
        where = f"{path}/{key}"
        if key not in dst:
            dst[key] = value
        elif isinstance(dst[key], dict) and isinstance(value, dict):
            conflicts += merge(dst[key], value, where)
        elif dst[key] != value and not where.endswith("/host"):
            conflicts.append(where)
    return conflicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with open(os.path.join(args.source, "_protocol.json")) as f:
        src_protocol = json.load(f)
    with open(os.path.join(args.target, "_protocol.json")) as f:
        dst_protocol = json.load(f)
    if src_protocol != dst_protocol:
        diff = {k for k in src_protocol.keys() | dst_protocol.keys()
                if src_protocol.get(k) != dst_protocol.get(k)}
        sys.exit(f"protocols differ in {sorted(diff)}; not merging")

    merged, problems = {}, []
    for name in sorted(os.listdir(args.source)):
        if not name.endswith(".json") or name.startswith("_"):
            continue
        with open(os.path.join(args.source, name)) as f:
            src = json.load(f)
        target_path = os.path.join(args.target, name)
        dst = {}
        if os.path.exists(target_path):
            with open(target_path) as f:
                dst = json.load(f)
        # The per-method host lives under _manifest/methods/<name>/host and may
        # legitimately differ; every other differing value is a conflict.
        conflicts = merge(dst, src)
        if conflicts:
            problems.append((name, conflicts))
        merged[target_path] = dst

    if problems:
        for name, conflicts in problems:
            print(f"{name}: {len(conflicts)} conflicting values, e.g. {conflicts[:3]}")
        sys.exit("conflicts found; nothing written")

    for path, payload in merged.items():
        print(f"{'would write' if args.dry_run else 'writing'} {path}")
        if not args.dry_run:
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)


if __name__ == "__main__":
    main()
