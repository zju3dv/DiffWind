"""Sequential zero-force training, continuation, and frozen replay."""
import os
import time
from pathlib import Path

import numpy as np

from .config import input_manifest, scene_config, sha256
from .io import (
    code_manifest,
    load_state,
    prepare_renderer,
    read_json,
    render_views,
    run_lock,
    save_array,
    write_json,
)


def status(output, phase, **values):
    record = dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), **values)
    write_json(output / "status.json", record)
    print(record, flush=True)


def committed_frames(output):
    records = [read_json(p) for p in sorted((output / "frames").glob("*.json"))]
    if [r["frame"] for r in records] != list(range(1, len(records) + 1)):
        raise ValueError("Committed frames are not a contiguous prefix")
    for record in records:
        for key in ("force", "state"):
            if sha256(output / record[key]) != record[key + "_sha256"]:
                raise ValueError("Committed checkpoint changed: " + record[key])
    return records


def replay(engine, output, records, render=False):
    state, diagnostics = engine.initial, []
    for record in records:
        frame = record["frame"]
        state = engine.forward(
            state, np.load(str(output / record["force"]), allow_pickle=False)
        )
        expected = load_state(output / record["state"])
        errors = {k: float(np.max(np.abs(state[k] - expected[k]))) for k in state}
        diagnostics.append(dict(frame=frame, max_abs=errors))
        write_json(output / "replay.json", diagnostics)
        if render:
            for split in ("train", "test"):
                render_views(engine, state, output / "render", frame, split)
        print(
            "Replayed frame %d/%d, position error %.3g"
            % (frame, len(records), errors["x"]),
            flush=True,
        )
    return state


def verify_run(record, spec, binary):
    if record["inputs"] != input_manifest(spec, record["frames"]):
        raise ValueError("Run inputs changed; use a fresh output directory")
    if record["code"] != code_manifest() or record["rasterizer_sha256"] != binary:
        raise ValueError("Run implementation changed; use a fresh output directory")


def evaluate_run(engine, output, run, binary):
    from .evaluation import score

    verify_run(run, engine.spec, binary)
    frozen = read_json(output / "frozen.json")
    records = committed_frames(output)
    if (
        run["frames"] != 48
        or len(records) != 48
        or records != frozen["frames"]
        or sha256(output / "run.json") != frozen["run_sha256"]
    ):
        raise ValueError("Evaluation requires a frozen 48-frame reconstruction")
    status(output, "replaying", completed_frames=48)
    replay(engine, output, records, render=True)
    # Score the exported test-view images.
    result = score(
        engine.spec["gt"], output / "render", output / "evaluation", run["scene"]
    )
    result["frozen_sha256"] = sha256(output / "frozen.json")
    write_json(output / "evaluation" / "metrics.json", result)
    status(output, "complete", completed_frames=48, test_psnr=result["psnr"])
    return result


def execute(args):
    output = Path(args.output).resolve()
    with run_lock(output):
        try:
            return _execute(args, output)
        except Exception as error:
            status(output, "failed", error=repr(error))
            raise


def _execute(args, output):
    binary = prepare_renderer(args.gpu)
    from .engine import ReconstructionEngine, zero_force
    from .audit import audit
    from .optimizer import optimize_window

    if args.command == "evaluate":
        run = read_json(output / "run.json")
        spec = scene_config(run["scene"], args.data_root)
        verify_run(run, spec, binary)
        engine = ReconstructionEngine(run["scene"], args.data_root, output / ".cache")
        return evaluate_run(engine, output, run, binary)

    spec = scene_config(args.scene, args.data_root)
    defaults = spec["settings"]["optimization"]
    settings = {
        k: getattr(args, k, None) if getattr(args, k, None) is not None else defaults[k]
        for k in ("frames", "min_updates", "max_updates", "initial_lr")
    }
    if (
        not 1 <= settings["frames"] <= 48
        or not 1 <= settings["min_updates"] <= settings["max_updates"]
        or not np.isfinite(settings["initial_lr"])
        or settings["initial_lr"] <= 0
    ):
        raise ValueError("Invalid frame range or optimization settings")
    manifest = input_manifest(spec, settings["frames"])
    if (output / "run.json").exists():
        if not args.resume or args.command != "train":
            raise ValueError(
                "Output already has a run; use --resume or a fresh directory"
            )
        run = read_json(output / "run.json")
        if run["scene"] != args.scene or any(
            run[k] != value for k, value in settings.items()
        ):
            raise ValueError("Resume settings differ from the saved run")
        verify_run(run, spec, binary)
    else:
        if args.resume:
            raise ValueError("No run.json to resume")
        if any(p.name not in (".lock", "status.json") for p in output.iterdir()):
            raise ValueError("Use an empty output directory for a new run")
        run = dict(
            scene=args.scene,
            **settings,
            inputs=manifest,
            code=code_manifest(),
            rasterizer_sha256=binary,
            created_unix=time.time(),
            initialization="rest state; zero force for every new frame",
            selection="four training views only",
            covariance="F Sigma_0 F^T",
        )
        write_json(output / "run.json", run)
    engine = ReconstructionEngine(args.scene, args.data_root, output / ".cache")
    write_json(output / "engine.json", engine.metadata)
    if (
        not (output / "audit" / "audit.json").exists()
        or not read_json(output / "audit" / "audit.json")["passed"]
    ):
        status(output, "gradient_audit")
        audit(engine, output / "audit")
    if args.command == "audit":
        status(output, "audit_passed")
        return
    records = committed_frames(output)
    state = engine.initial
    if records:
        # Restore the latest committed simulation state.
        status(output, "restoring_checkpoint", completed_frames=len(records))
        state = load_state(output / records[-1]["state"])
    for frame in range(len(records) + 1, run["frames"] + 1):
        status(output, "training", frame=frame, completed_frames=frame - 1)
        window = output / "windows" / ("%03d" % frame)
        window.mkdir(parents=True, exist_ok=True)
        attempt = len(list(window.glob("attempt_*"))) + 1
        # Optimize the next frame from zero force.
        force, state, result = optimize_window(
            engine,
            state,
            zero_force(),
            frame,
            window / ("attempt_%03d" % attempt),
            initial_lr=run["initial_lr"],
            min_updates=run["min_updates"],
            max_updates=run["max_updates"],
        )
        force_path = Path("forces") / ("%05d.npy" % frame)
        state_path = Path("states") / ("%05d.npz" % frame)
        save_array(output / force_path, force)
        save_array(output / state_path, state)
        render_views(engine, state, output / "training_render", frame, "train")
        record = dict(
            frame=frame,
            force=str(force_path),
            state=str(state_path),
            force_sha256=sha256(output / force_path),
            state_sha256=sha256(output / state_path),
            attempt=attempt,
            result=result,
        )
        write_json(output / "frames" / ("%05d.json" % frame), record)
    verify_run(run, spec, binary)
    records = committed_frames(output)
    if run["frames"] < 48:
        status(output, "training_prefix_complete", completed_frames=len(records))
        return
    frozen = output / "frozen.json"
    if not frozen.exists():
        write_json(
            frozen,
            dict(
                run_sha256=sha256(output / "run.json"),
                frames=records,
                frozen_unix=time.time(),
            ),
        )
    return evaluate_run(engine, output, run, binary)
