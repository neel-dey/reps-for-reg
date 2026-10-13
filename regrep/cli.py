"""The ``regrep`` command line.

.. code-block:: text

    regrep register FIXED_CT FIXED_PET MOVING_CT MOVING_PET OUTPUT [options]
    regrep register-batch PAIRS_CSV [options]
    regrep download-weights
    regrep evaluate --field FIELD --fixed-ct ... | --csv FIELDS_CSV [options]

``register`` follows the five-argument interface of the PSMAReg challenge:
the baseline CT and PET are the fixed images, the follow-up CT and PET the
moving images, and OUTPUT receives the displacement field. A failed
registration writes no file and exits with status 1.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import warnings

from regrep.batch import PAIR_COLUMNS

log = logging.getLogger("regrep")

EVALUATION_COLUMNS = ("field", "fixed_ct", "fixed_labels", "moving_labels",
                      "fixed_lesions", "moving_lesions", "moving_pet")


def _chunk(value: str) -> int | str:
    """Parse a chunk setting: ``auto``, ``full`` or a positive channel count."""
    if value in ("auto", "full"):
        return value
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected auto, full or a channel count, got {value!r}") from error
    if number < 1:
        raise argparse.ArgumentTypeError("a channel count must be positive")
    return number


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    # regrep.runtime imports torch; main() sets the CUDA allocator option
    # before it builds the parser.
    from regrep.runtime import MODES

    parser.add_argument("--mode", choices=MODES, default="auto",
                        help="serial runs every step in this process, one after "
                             "another; concurrent runs the segmentation in worker "
                             "processes while the features and stage 1 are "
                             "computed; auto picks one from the free GPU memory "
                             "(default: auto)")
    parser.add_argument("--gpus", default=None,
                        help="comma-separated GPU indices, e.g. 0 or 0,1; they "
                             "are the indices that nvidia-smi shows, or positions "
                             "in CUDA_VISIBLE_DEVICES when it is set (default: all "
                             "visible GPUs)")
    for stage in (1, 2):
        parser.add_argument(f"--chunk-stage{stage}", type=_chunk, default="auto",
                            metavar="{auto,full,N}",
                            help=f"channels per chunk in stage {stage}: auto, full "
                                 "(one chunk per GPU) or a number (default: auto)")
    parser.add_argument("--config", help="YAML or JSON file that overrides settings")
    parser.add_argument("--keep-workspace", action="store_true",
                        help="concurrent mode: keep the temporary workspace with "
                             "the worker logs and label maps")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="also print debug messages")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="regrep",
        description="Longitudinal whole-body PSMA PET/CT registration.")
    commands = parser.add_subparsers(dest="command", required=True)

    register = commands.add_parser(
        "register", help="register one pair",
        description="Register the follow-up (moving) CT and PET to the baseline "
                    "(fixed) CT and PET and write the displacement field.")
    for name, text in (("fixed_ct", "baseline CT (the fixed image)"),
                       ("fixed_pet", "baseline PET (the fixed image)"),
                       ("moving_ct", "follow-up CT (the moving image)"),
                       ("moving_pet", "follow-up PET (the moving image)")):
        register.add_argument(name, metavar=name.upper(), help=text)
    register.add_argument("output", metavar="OUTPUT",
                          help="the displacement field to write (.nii.gz)")
    _add_run_options(register)

    batch = commands.add_parser(
        "register-batch", help="register the pairs of a CSV file",
        description="Register many pairs. The CSV file has the columns "
                    + ", ".join(PAIR_COLUMNS) + ". Pairs run in parallel, one "
                    "worker process per group of GPUs.")
    batch.add_argument("pairs", metavar="PAIRS_CSV", help="the table of pairs")
    batch.add_argument("--gpus-per-pair", type=int, default=1,
                       help="GPUs per pair (default: 1)")
    batch.add_argument("--overwrite", action="store_true",
                       help="register pairs whose output exists already")
    _add_run_options(batch)

    download = commands.add_parser(
        "download-weights", help="download the network weights",
        description="Download the anatomix, TotalSegmentator and LION weights "
                    "that are missing. Run this before going offline.")
    download.add_argument("--config", help="YAML or JSON file that overrides settings")
    download.add_argument("-v", "--verbose", action="store_true",
                          help="also print debug messages")

    evaluate = commands.add_parser(
        "evaluate", help="score displacement fields with the challenge metrics",
        description="Score one field, or each row of a CSV file with the columns "
                    + ", ".join(EVALUATION_COLUMNS) + " and an optional column "
                    "name, which labels the row in the output.")
    evaluate.add_argument("--csv", help="CSV file with one field per row")
    evaluate.add_argument("--field", help="displacement field (.nii.gz, 3 x X x Y x Z)")
    evaluate.add_argument("--fixed-ct", help="baseline CT (the fixed image)")
    evaluate.add_argument("--fixed-labels", help="CT organ labels of the fixed image")
    evaluate.add_argument("--moving-labels", help="CT organ labels of the moving image")
    evaluate.add_argument("--fixed-lesions",
                          help="PET lesion labels of the fixed image")
    evaluate.add_argument("--moving-lesions",
                          help="PET lesion labels of the moving image")
    evaluate.add_argument("--moving-pet", help="follow-up PET (the moving image)")
    evaluate.add_argument("--output", help="write the scores to a .csv or .json file")
    evaluate.add_argument("--device", default=None,
                          help="device of the warps (default: cuda if available)")
    evaluate.add_argument("-v", "--verbose", action="store_true",
                          help="also print debug messages")
    return parser


def configure_logging(verbose: bool, prefix: str = "") -> None:
    """Print regrep's messages; print other libraries' only with ``verbose``.

    Without ``verbose``, messages of other libraries are shown from level
    ERROR on, and nnU-Net's warning about the plans format of the
    TotalSegmentator models is hidden.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        f"%(asctime)s {prefix}%(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
    if not verbose:
        handler.addFilter(lambda record: record.name.startswith("regrep")
                          or record.levelno >= logging.ERROR)
        warnings.filterwarnings("ignore", message="Detected old nnU-Net plans format")
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        handlers=[handler], force=True)


def stop_on_signals() -> None:
    """Turn SIGTERM and SIGHUP into a normal exit, so that the clean-up code runs."""
    def handler(signum, frame):
        raise SystemExit(128 + signum)

    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, handler)


def _missing_weights(config) -> bool:
    """Log the network weights that are missing; return True if there are any."""
    from regrep import weights

    problems = weights.missing(config)
    if problems:
        log.error("weights are missing (run `regrep download-weights`):\n  %s",
                  "\n  ".join(problems))
    return bool(problems)


def _register(args) -> int:
    from regrep import runtime
    from regrep.config import load_config
    from regrep.io import PairInputs, working_grid_size
    from regrep.pipeline import register_pair

    config = load_config(args.config)
    if _missing_weights(config):
        return 1
    inputs = PairInputs(args.fixed_ct, args.fixed_pet, args.moving_ct,
                        args.moving_pet)
    devices = runtime.select_gpus(args.gpus)
    if args.gpus is not None and len(devices) == 1:
        log.info("using GPU %s; the messages below call it cuda:0", args.gpus)
    elif args.gpus is not None:
        log.info("using GPUs %s; the messages below call them cuda:0 to cuda:%d",
                 args.gpus, len(devices) - 1)
    plan = runtime.plan(devices, args.mode,
                        working_grid_size(args.fixed_ct, config.grid),
                        working_grid_size(args.moving_ct, config.grid),
                        args.chunk_stage1, args.chunk_stage2)
    register_pair(inputs, args.output, plan, config,
                  keep_workspace=args.keep_workspace)
    return 0


def _register_batch(args) -> int:
    from regrep.batch import read_pairs, run_batch
    from regrep.config import load_config

    if _missing_weights(load_config(args.config)):
        return 1
    pairs = read_pairs(args.pairs)
    failures = run_batch(pairs, gpus=args.gpus, gpus_per_pair=args.gpus_per_pair,
                         mode=args.mode, chunk_stage1=args.chunk_stage1,
                         chunk_stage2=args.chunk_stage2, config_path=args.config,
                         overwrite=args.overwrite,
                         keep_workspace=args.keep_workspace, verbose=args.verbose)
    return 1 if failures else 0


def _download_weights(args) -> int:
    from regrep import weights
    from regrep.config import load_config

    config = load_config(args.config)
    weights.download_all(config)
    problems = weights.missing(config)
    if problems:
        log.error("still missing:\n  %s", "\n  ".join(problems))
        return 1
    log.info("all weights are present")
    return 0


def _evaluation_rows(args) -> list[dict[str, str]]:
    if args.csv:
        with open(args.csv, newline="") as handle:
            rows = list(csv.DictReader(handle))
        missing = [c for c in EVALUATION_COLUMNS if rows and c not in rows[0]]
        if missing:
            raise ValueError(f"{args.csv}: missing columns {missing}")
        return rows
    row = {c: getattr(args, c) for c in EVALUATION_COLUMNS}
    absent = [c for c, v in row.items() if not v]
    if absent:
        raise ValueError("give --csv or all of: " + ", ".join(
            "--" + c.replace("_", "-") for c in absent))
    return [row]


# The evaluation inputs on the grid of the fixed CT and on the grid of the
# moving PET.
_FIXED_GRID = ("fixed_labels", "fixed_lesions")
_MOVING_GRID = ("moving_labels", "moving_lesions")


def _evaluation_volumes(row: dict[str, str]):
    """Read the volumes of one evaluation row and check their grids.

    The label maps must be on the grid of their reference image, the fixed CT
    or the moving PET (``regrep.io.same_grid``).

    Returns
    -------
    tuple
        The arrays by column name, the affine of the fixed CT and the affine
        of the moving PET.
    """
    import nibabel as nib

    from regrep.io import read_array, same_grid

    images = {c: nib.load(row[c]) for c in EVALUATION_COLUMNS if c != "field"}
    arrays = {c: read_array(image, row[c]) for c, image in images.items()}
    for reference, columns in (("fixed_ct", _FIXED_GRID), ("moving_pet", _MOVING_GRID)):
        for column in columns:
            if not same_grid(arrays[column].shape, images[column].affine,
                             arrays[reference].shape, images[reference].affine):
                raise ValueError(
                    f"{row[column]} is not on the grid of {row[reference]}: shape "
                    f"{arrays[column].shape} and affine\n{images[column].affine}\n"
                    f"against shape {arrays[reference].shape} and affine\n"
                    f"{images[reference].affine}")
    return arrays, images["fixed_ct"].affine, images["moving_pet"].affine


def _evaluate(args) -> int:
    import torch

    from regrep.evaluation.metrics import METRICS, evaluate_pair, read_field
    from regrep.io import check_output, voxel_size
    from regrep.registration.transforms import VoxelMap

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    rows = _evaluation_rows(args)
    if args.output:
        check_output(args.output, [row[c] for row in rows for c in EVALUATION_COLUMNS])
    results = []
    for row in rows:
        volumes, fixed_affine, moving_affine = _evaluation_volumes(row)
        scores = evaluate_pair(
            read_field(row["field"]), volumes["fixed_ct"], volumes["fixed_labels"],
            volumes["moving_labels"], volumes["fixed_lesions"],
            volumes["moving_lesions"], volumes["moving_pet"], voxel_size(fixed_affine),
            VoxelMap.between(fixed_affine, moving_affine).jacobian, device=device)
        name = row.get("name") or row["field"]
        results.append({"name": name, **scores})
        log.info("%s: %s", name, ", ".join(
            f"{m} {scores[m]:.6g}" for m in METRICS))

    if args.output and args.output.endswith(".json"):
        # JSON has no NaN; an undefined metric is written as null.
        cleaned = [{k: None if isinstance(v, float) and math.isnan(v) else v
                    for k, v in result.items()} for result in results]
        with open(args.output, "w") as handle:
            json.dump(cleaned, handle, indent=1)
    else:
        handle = open(args.output, "w", newline="") if args.output else sys.stdout
        try:
            writer = csv.writer(handle)
            writer.writerow(["name", *METRICS])
            for result in results:
                writer.writerow([result["name"], *(repr(result[m]) for m in METRICS)])
        finally:
            if args.output:
                handle.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the command line; return the exit status."""
    # The memory figures of the run plan assume this allocator setting. It
    # must be set before torch starts its CUDA allocator.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    # GPU indices follow the order of nvidia-smi, unless the user set another
    # order. The worker processes inherit this setting.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    args = _parser().parse_args(argv)
    configure_logging(args.verbose)
    stop_on_signals()
    commands = {"register": _register, "register-batch": _register_batch,
                "download-weights": _download_weights, "evaluate": _evaluate}
    try:
        return commands[args.command](args)
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130
    except Exception as error:
        log.debug("traceback", exc_info=True)
        log.error("%s: %s", type(error).__name__, error)
        if args.command == "register":
            log.error("no output was written to %s", args.output)
        return 1
