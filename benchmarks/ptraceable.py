"""Run a Python module after allowing any process to attach to this one with ptrace.

    python -m benchmarks.ptraceable stallfree.server.api --model ...

Used only for the profiling run: in the benchmark containers a sampling profiler (py-spy) may
attach only to its own descendants, and the server is started by the driver, not by the
profiler. ``prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY)`` lifts that restriction for this process.
Nothing else changes: the module runs as ``__main__`` with the remaining arguments.
"""

from __future__ import annotations

import ctypes
import runpy
import sys

PR_SET_PTRACER = 0x59616D61
PR_SET_PTRACER_ANY = ctypes.c_ulong(-1 & 0xFFFFFFFFFFFFFFFF)


def main() -> None:
    module, args = sys.argv[1], sys.argv[2:]
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0) != 0:
        print(f"prctl(PR_SET_PTRACER) failed: errno {ctypes.get_errno()}", file=sys.stderr)
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
