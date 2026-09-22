"""
Process pools that fail loudly, and do not deadlock.

Two failure modes have cost real runs here, and a bare
``ProcessPoolExecutor`` reports neither usefully:

* **Silent deadlock.** ``fork`` copies only the calling thread but every
  lock in whatever state it was in, so a child can inherit a held lock no
  surviving thread will release. Importing numpy or torch is enough to
  make the parent multi-threaded (64 threads each, measured 2026-09-14),
  so any pool forked after those imports can hang forever with no error.
  ``forkserver`` forks from a separate, idle server process instead.

* **A worker dying.** agama is built without ``-DNDEBUG``, so a failed
  assertion calls ``abort()`` (SIGABRT, exit -6) which ``except
  Exception`` cannot catch; the OOM killer sends SIGKILL (-9). Either way
  the executor raises a bare ``BrokenProcessPool`` naming no cause.
  ``faulthandler`` in each worker turns that into a C-level traceback on
  stderr naming the exact Python frame that died.

Example:
    >>> with process_pool(8) as pool:          # doctest: +SKIP
    ...     results = list(pool.map(work, items))
"""

import faulthandler
import multiprocessing as mp
import signal
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from contextlib import contextmanager

# Reason: threadpool_limits restores the old limit when the object is
# collected, so the worker has to keep a reference alive.
_LIMITS = None


def _init_worker(threads_per_worker: int) -> None:
    """Run in each worker at start-up: crash dumps and thread limits."""
    global _LIMITS
    faulthandler.enable()
    if threads_per_worker is None:
        return
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    # Reason: the OMP_NUM_THREADS environment variable is read when the
    # BLAS loads, which has already happened by the time an initializer
    # runs, so it has to be done through threadpoolctl at runtime.
    _LIMITS = threadpool_limits(limits=threads_per_worker)


def describe_exit(code: int) -> str:
    """
    Render a worker exit code, naming the signal if it was killed.

    Args:
        code: A ``multiprocessing.Process.exitcode``.

    Returns:
        Human-readable description.

    Example:
        >>> describe_exit(-6)
        '-6 (SIGABRT: a C-level abort(), e.g. an agama assertion)'
    """
    if code is None:
        return "still running"
    if code >= 0:
        return str(code)
    try:
        name = signal.Signals(-code).name
    except ValueError:
        return str(code)
    hint = {
        "SIGABRT": "a C-level abort(), e.g. an agama assertion",
        "SIGKILL": "killed from outside, usually the OOM killer",
        "SIGSEGV": "a segfault in a C extension",
    }.get(name)
    return f"{code} ({name}{': ' + hint if hint else ''})"


@contextmanager
def process_pool(max_workers: int, threads_per_worker: int = 1,
                 method: str = "forkserver", **kwargs):
    """
    A ProcessPoolExecutor that will not deadlock and explains its deaths.

    Args:
        max_workers: Number of worker processes.
        threads_per_worker: BLAS/OpenMP threads each worker may use, or
            None to leave them alone. One is right for a pool of
            single-threaded tasks; the default would otherwise give every
            worker a full thread pool and oversubscribe the machine.
        method: Start method. ``forkserver`` is safe after numpy/torch
            are imported; ``fork`` is faster to start but can deadlock.
        **kwargs: Passed through to ProcessPoolExecutor.

    Yields:
        The executor.

    Raises:
        BrokenProcessPool: re-raised with any worker exit codes that
            could still be read, and a pointer to the stderr traceback.
    """
    ctx = mp.get_context(method)
    pool = ProcessPoolExecutor(
        max_workers=max_workers, mp_context=ctx, initializer=_init_worker,
        initargs=(threads_per_worker,), **kwargs)
    # TRADEOFF: ProcessPoolExecutor clears _processes before raising, so
    # the exit codes are read from a snapshot taken while the workers are
    # alive. It is a private attribute and the snapshot races with worker
    # replacement, so treat a hit as a bonus -- the faulthandler dump on
    # stderr is the reliable diagnostic.
    try:
        yield pool
    except BrokenProcessPool as exc:
        codes = sorted({p.exitcode for p in getattr(pool, "_processes",
                                                    {}).values()
                        if p.exitcode is not None})
        detail = ", ".join(describe_exit(c) for c in codes) or "not readable"
        raise BrokenProcessPool(
            f"{exc} | worker exit codes: {detail} | a 'Fatal Python error' "
            f"traceback naming the frame should be on stderr above"
        ) from exc
    finally:
        pool.shutdown(wait=False)


def _identity(i):
    """Top-level so it is picklable."""
    return i


def _self_check():
    """A worker aborts; the error must name SIGABRT or point to stderr."""
    with process_pool(2, threads_per_worker=1) as pool:
        assert list(pool.map(_identity, range(4))) == [0, 1, 2, 3]
    print("process_pool: normal map OK")

    assert describe_exit(-6).startswith("-6 (SIGABRT")
    assert describe_exit(-9).startswith("-9 (SIGKILL")
    assert describe_exit(0) == "0"
    print("describe_exit: signal naming OK")

    try:
        with process_pool(2, threads_per_worker=1) as pool:
            list(pool.map(_abort_on, range(6)))
    except BrokenProcessPool as exc:
        assert "worker exit codes" in str(exc), exc
        print(f"process_pool: abort reported as -> {str(exc)[-96:]}")
    else:
        raise AssertionError("aborting worker did not raise")


def _abort_on(i):
    """Top-level so it is picklable; aborts like a failed C assertion."""
    import os
    import time

    time.sleep(0.05)
    if i == 4:
        os.abort()
    return i


if __name__ == "__main__":
    _self_check()
