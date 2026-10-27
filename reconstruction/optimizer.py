"""Adaptive force optimization with Adam and backtracking."""
import json
import time
import numpy as np


class TransactionalAdam:
    def __init__(self, shape, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = np.zeros(shape, dtype=np.float32)
        self.v = np.zeros(shape, dtype=np.float32)
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def propose(self, gradient, lr, reset=False):
        t = 1 if reset else self.t + 1
        m = (
            (1 - self.beta1) * gradient
            if reset
            else self.beta1 * self.m + (1 - self.beta1) * gradient
        )
        v = (
            (1 - self.beta2) * gradient**2
            if reset
            else self.beta2 * self.v + (1 - self.beta2) * gradient**2
        )
        step = (
            -lr
            * (m / (1 - self.beta1**t))
            / (np.sqrt(v / (1 - self.beta2**t)) + self.epsilon)
        )
        return step.astype(np.float32), (m, v, t)

    def commit(self, proposed):
        self.m, self.v, self.t = proposed


def optimize_window(
    adapter,
    state,
    initial_force,
    frame,
    output,
    initial_lr=0.001,
    min_updates=12,
    max_updates=64,
    loss_mode="mse",
):
    output.mkdir(parents=True, exist_ok=False)
    force = initial_force.copy()
    optimizer = TransactionalAdam(force.shape)
    initial, _, initial_state = adapter.evaluate(
        state, force, frame, loss_mode=loss_mode
    )
    best = dict(initial)
    best_force = force.copy()
    best_state = initial_state
    best_iteration = -1
    lr = initial_lr
    rows = []
    accepted_count = 0
    history = []
    reductions = 0
    restart = False
    stalls = 0
    stop_reason = "update_budget_exhausted"
    anchor_force = force.copy()
    recoveries = []

    def record_recovery(iteration, error, action):
        event = dict(
            frame=frame,
            iteration=iteration,
            error=str(error),
            action=action,
            next_lr=lr,
            time_unix=time.time(),
        )
        recoveries.append(event)
        with (output / "recovery.jsonl").open("a") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    for iteration in range(max_updates):
        start = time.time()
        gradient = None
        for retry in range(3):
            try:
                before, candidate_gradient, _ = adapter.evaluate(
                    state, force, frame, gradient=True, loss_mode=loss_mode
                )
                if (
                    not np.isfinite(before["loss"])
                    or not np.isfinite(candidate_gradient).all()
                ):
                    raise FloatingPointError("Nonfinite objective or force gradient")
                gradient = candidate_gradient
                anchor_force = force.copy()
                break
            except FloatingPointError as error:
                # Rebuild the gradient trajectory from the input state.
                lr *= 0.25
                restart = True
                force = (anchor_force if retry == 0 else initial_force).copy()
                record_recovery(
                    iteration,
                    error,
                    "restore_last_valid_adjoint"
                    if retry == 0
                    else "restore_initial_force",
                )
        if gradient is None:
            if accepted_count == 0:
                raise FloatingPointError(
                    "No valid adjoint after three recovery attempts"
                )
            stop_reason = "adjoint_recovery_exhausted_return_valid_best"
            break
        norm = float(np.linalg.norm(gradient))
        if norm == 0:
            stop_reason = "zero_gradient"
            break
        step, proposal = optimizer.propose(gradient, lr, reset=restart)
        slope = float(np.sum(gradient * step, dtype=np.float64))
        reset = restart or slope >= 0
        if reset:
            step, proposal = optimizer.propose(gradient, lr, reset=True)
        slope = float(np.sum(gradient * step, dtype=np.float64))
        restart = False
        attempts = []
        accepted = False
        after = before
        for backtrack in range(9):
            factor = 0.5**backtrack
            try:
                candidate = force + factor * step
                metric, _, next_state = adapter.evaluate(
                    state, candidate, frame, loss_mode=loss_mode
                )
                decrease = before["loss"] - metric["loss"]
                accepted = bool(
                    np.isfinite(metric["loss"])
                    and decrease > max(0.0, -1e-4 * factor * slope)
                )
                ratio = decrease / max(-factor * slope, 1e-30)
                trial = dict(
                    backtrack=backtrack,
                    lr=lr * factor,
                    loss=metric["loss"],
                    accepted=accepted,
                    ratio=ratio,
                )
            except FloatingPointError as error:
                trial = dict(
                    backtrack=backtrack,
                    lr=lr * factor,
                    accepted=False,
                    error=str(error),
                )
            attempts.append(trial)
            if accepted:
                force = candidate
                optimizer.commit(proposal)
                after = metric
                accepted_count += 1
                lr = (
                    lr
                    * factor
                    * (
                        1.25
                        if ratio > 0.75 and backtrack == 0
                        else 0.5
                        if ratio < 0.1
                        else 1.0
                    )
                )
                if metric["train_psnr"] > best["train_psnr"]:
                    best = dict(metric)
                    best_force = force.copy()
                    best_state = next_state
                    best_iteration = iteration
                stalls = 0
                break
        if not accepted:
            lr *= 0.25
            restart = True
            stalls += 1
        history.append(after["loss"])
        plateau = False
        if iteration + 1 >= min_updates and len(history) >= 7:
            improvement = (history[-7] - min(history[-6:])) / max(
                abs(history[-7]), 1e-30
            )
            plateau = improvement < 1e-3
            if plateau:
                lr *= 0.5
                reductions += 1
                history = []
        row = dict(
            frame=frame,
            iteration=iteration,
            loss_before=before["loss"],
            loss_after=after["loss"],
            train_psnr=after["train_psnr"],
            best_train_psnr=best["train_psnr"],
            accepted=accepted,
            moments_reset=reset,
            gradient_norm=norm,
            gradient_abs_max=float(np.abs(gradient).max()),
            image_F_gradient_norm=before.get("image_F_gradient_norm"),
            next_lr=lr,
            plateau_reduction=plateau,
            attempts=attempts,
            state_diagnostics=after["state_diagnostics"],
            seconds=time.time() - start,
        )
        rows.append(row)
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        (output / "progress.json").write_text(json.dumps(row, indent=2))
        np.save(str(output / "best_force.npy"), best_force)
        print(
            json.dumps(
                {
                    k: row[k]
                    for k in [
                        "frame",
                        "iteration",
                        "loss_after",
                        "train_psnr",
                        "accepted",
                        "next_lr",
                        "seconds",
                    ]
                }
            ),
            flush=True,
        )
        if plateau and reductions >= 2:
            stop_reason = "training_plateau_after_two_lr_reductions"
            break
        if stalls >= 3 or lr < initial_lr * 1e-7:
            stop_reason = "numerical_stall"
            break
    # Evaluate the force and state passed to the next frame.
    for label, value in (
        ("best", best_force),
        ("last_valid_adjoint", anchor_force),
        ("initial", initial_force),
    ):
        try:
            metric, _, next_state = adapter.evaluate(
                state, value, frame, loss_mode=loss_mode
            )
            if not np.isfinite(metric["loss"]):
                raise FloatingPointError("Nonfinite final objective")
            best, best_force, best_state = dict(metric), value.copy(), next_state
            if label != "best":
                stop_reason = "final_replay_recovered_" + label
            break
        except FloatingPointError as error:
            record_recovery(len(rows), error, "final_replay_" + label)
    else:
        raise FloatingPointError("No replayable state after rollback")
    np.save(str(output / "best_force.npy"), best_force)
    result = dict(
        frame=frame,
        initial=initial,
        best=best,
        best_iteration=best_iteration,
        updates=len(rows),
        accepted_updates=accepted_count,
        next_lr=lr,
        stop_reason=stop_reason,
        selection="four training views only",
        image_domain=adapter.image_domain,
        gradient_mode="full_x_and_F",
        loss_mode=loss_mode,
        numerical_recoveries=len(recoveries),
    )
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return best_force, best_state, result
