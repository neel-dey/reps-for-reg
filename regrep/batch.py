"""Register the pairs of a CSV file on a pool of GPUs.

The GPUs are cut into groups of ``gpus_per_pair``. Each group gets one
worker process, which takes the next pair from a shared queue, makes a run
plan for its GPUs and registers the pair. A worker keeps the anatomix
network loaded from one pair to the next. A pair that fails is reported and
the worker goes on with the next pair.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing
import queue
import time
from dataclasses import dataclass

log = logging.getLogger(__name__)

INPUT_COLUMNS = ("fixed_ct", "fixed_pet", "moving_ct", "moving_pet")
PAIR_COLUMNS = (*INPUT_COLUMNS, "output")


@dataclass(frozen=True)
class BatchSettings:
    """The options that every worker uses."""

    mode: str
    chunk_stage1: int | str
    chunk_stage2: int | str
    config_path: str | None
    keep_workspace: bool
    verbose: bool


def read_pairs(path: str) -> list[dict[str, str]]:
    """Read the pair table and check it.

    The ``output`` column names the output folder of each pair
    (``regrep.io.PairOutputs``). The paths are relative to the current
    directory, and spaces around them are removed. A ``ValueError`` is
    raised if a column is missing, an entry is empty, two rows have the same
    output folder, or an output file is an input file of any row
    (``regrep.io.file_key`` decides when two paths are one file).
    """
    from regrep.io import PairOutputs, file_key

    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{path}: no pairs")
    missing = [c for c in PAIR_COLUMNS if c not in rows[0]]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    for number, row in enumerate(rows, start=2):
        empty = [c for c in PAIR_COLUMNS if not (row.get(c) or "").strip()]
        if empty:
            raise ValueError(f"{path}, line {number}: empty {empty}")
    pairs = [{c: row[c].strip() for c in PAIR_COLUMNS} for row in rows]

    folders: dict[tuple, int] = {}
    for number, pair in enumerate(pairs, start=2):
        key = file_key(pair["output"])
        if key in folders:
            raise ValueError(f"{path}, lines {folders[key]} and {number}: two pairs "
                             "have the same output folder")
        folders[key] = number
    inputs = {file_key(pair[c]): (number, pair[c])
              for number, pair in enumerate(pairs, start=2) for c in INPUT_COLUMNS}
    for number, pair in enumerate(pairs, start=2):
        for output in PairOutputs(pair["output"]).files():
            if file_key(output) in inputs:
                line, name = inputs[file_key(output)]
                raise ValueError(f"{path}, line {number}: the output {output} is "
                                 f"the input file {name} of line {line}")
    return pairs


def _worker(indices: list[int], tasks, results, settings: BatchSettings) -> None:
    """Register pairs from ``tasks`` on the GPUs ``indices``.

    The worker stops when the queue gives ``None``.
    """
    from regrep.cli import configure_logging, stop_on_signals

    stop_on_signals()
    configure_logging(settings.verbose,
                      prefix=f"[GPUs {','.join(str(i) for i in indices)}] ")

    import torch

    from regrep import runtime
    from regrep.config import load_config
    from regrep.io import PairInputs, working_grid_size
    from regrep.pipeline import Models, register_pair

    devices = runtime.select_gpus(",".join(str(i) for i in indices))
    config = load_config(settings.config_path)
    models = Models()
    while True:
        task = tasks.get()
        if task is None:
            return
        index, pair = task
        output = pair["output"]
        try:
            torch.cuda.empty_cache()
            plan = runtime.plan(devices, settings.mode,
                                working_grid_size(pair["fixed_ct"], config.grid),
                                working_grid_size(pair["moving_ct"], config.grid),
                                settings.chunk_stage1, settings.chunk_stage2)
            inputs = PairInputs(pair["fixed_ct"], pair["fixed_pet"],
                                pair["moving_ct"], pair["moving_pet"])
            seconds = register_pair(inputs, output, plan, config,
                                    keep_workspace=settings.keep_workspace,
                                    models=models)
            results.put((index, None, seconds))
        except Exception as error:
            log.exception("pair %s failed", output)
            results.put((index, f"{type(error).__name__}: {error}", None))


def run_batch(pairs: list[dict[str, str]], gpus: str | None, gpus_per_pair: int,
              mode: str, chunk_stage1: int | str, chunk_stage2: int | str,
              config_path: str | None, overwrite: bool, keep_workspace: bool,
              verbose: bool) -> dict[str, str]:
    """Register many pairs in parallel.

    Parameters
    ----------
    pairs
        Rows of :func:`read_pairs`.
    gpus
        The GPUs to use, e.g. ``"0,1,2,3"``; ``None`` uses every visible GPU.
    gpus_per_pair
        Number of GPUs in each group. GPUs left over after the last full
        group are not used.
    mode, chunk_stage1, chunk_stage2
        Passed to ``regrep.runtime.plan`` for each pair.
    config_path
        YAML or JSON file of setting overrides.
    overwrite
        Register pairs whose output files all exist already. Without it they
        are skipped.
    keep_workspace
        Keep the workspace of each pair.
    verbose
        Print debug messages.

    Returns
    -------
    dict
        The failed pairs: output folder to error message. Empty when every
        pair succeeded.
    """
    from regrep import runtime
    from regrep.io import PairOutputs

    indices = runtime.gpu_indices(gpus)
    if gpus_per_pair < 1 or gpus_per_pair > len(indices):
        raise ValueError(f"--gpus-per-pair must be between 1 and {len(indices)}")
    groups = [indices[i:i + gpus_per_pair]
              for i in range(0, len(indices) - gpus_per_pair + 1, gpus_per_pair)]

    todo = []
    for pair in pairs:
        if not overwrite and PairOutputs(pair["output"]).exist():
            log.info("skip %s: the outputs exist", pair["output"])
        else:
            todo.append(pair)
    if not todo:
        log.info("nothing to do")
        return {}
    log.info("%d pair(s) on %d group(s) of GPUs: %s", len(todo), len(groups),
             "; ".join(",".join(str(i) for i in g) for g in groups))

    settings = BatchSettings(mode, chunk_stage1, chunk_stage2, config_path,
                             keep_workspace, verbose)
    context = multiprocessing.get_context("spawn")
    tasks, results = context.Queue(), context.Queue()
    for task in enumerate(todo):
        tasks.put(task)
    workers = [context.Process(target=_worker, args=(group, tasks, results, settings))
               for group in groups[:len(todo)]]
    for _ in workers:
        tasks.put(None)

    started = time.perf_counter()
    done: dict[int, str | None] = {}
    try:
        for worker in workers:
            worker.start()
        while len(done) < len(todo):
            try:
                index, error, seconds = results.get(timeout=5.0)
            except queue.Empty:
                if not any(worker.is_alive() for worker in workers):
                    break
                continue
            done[index] = error
            state = "failed" if error else f"done in {seconds:.1f} s"
            log.info("[%d/%d] %s %s", len(done), len(todo), todo[index]["output"],
                     state)
    finally:
        # Pairs that no worker took stay in the queue. Without this call the
        # thread that feeds the queue would keep this process alive at exit,
        # waiting to write them.
        tasks.cancel_join_thread()
        tasks.close()
        results.close()
        for worker in workers:
            if worker.is_alive() and len(done) < len(todo):
                worker.terminate()
        for worker in workers:
            # A worker that did not start has no process id.
            if worker.pid is not None:
                worker.join()

    failures = {}
    for index, pair in enumerate(todo):
        if index not in done:
            failures[pair["output"]] = "the worker process ended without a result"
        elif done[index]:
            failures[pair["output"]] = done[index]
    log.info("%d of %d pair(s) registered in %.1f s", len(todo) - len(failures),
             len(todo), time.perf_counter() - started)
    for output, error in failures.items():
        log.error("failed: %s: %s", output, error)
    return failures
