#!/usr/bin/env python3
"""Print every out.X.p under DIR that is either older than its sibling in.X.p, or is missing"""

import sys
from pathlib import Path
from argparse import ArgumentParser


def main(args):
    root = Path(args.DIR)
    if not root.is_dir():
        sys.exit(f"{root}: not a directory")

    verbose = args.verbose
    # Find in files, and check if the corresponding out file is older, or is missing
    if verbose:
        print(f"CHECKING {root}")
    for inp in root.glob("in.*.p"):          # rglob = recursive; use root.glob for one level
        if not inp.is_file():
            continue
        outp = inp.with_name(inp.name.replace("in.", "out.")) 
        # Check if the otu file exists
        msg = ""
        if outp.exists():
            if outp.stat().st_mtime_ns < inp.stat().st_mtime_ns:
                msg = ["STALE OUT", inp]
        else:
            msg = ["MISSING OUT", inp]

        if len(msg):
            msg = "\t".join(str(x) for x in msg) if verbose else msg[-1]
            print(msg)


if __name__ == "__main__":
    parser = ArgumentParser(description="Print every out.X.p under DIR that is older than its sibling in.X.p.")
    parser.add_argument("DIR", help="Directory to check")
    # Verposity flag
    parser.add_argument("-v", "--verbose", action="store_true", help="Print verbose output")
    args = parser.parse_args()
    main(args)
