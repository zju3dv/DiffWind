"""Finite differences of the force-to-MPM-to-RGB chain at zero force."""
import numpy as np

from .engine import zero_force
from .io import export_image, write_json


def audit(engine, output):
    output.mkdir(parents=True, exist_ok=True)
    state, force = engine.initial, zero_force()
    if np.any(state["v"]) or np.any(state["C"]):
        raise ValueError("Reconstruction presets must start at rest")
    if not np.array_equal(
        state["F"], np.broadcast_to(np.eye(3, dtype=np.float32), state["F"].shape)
    ):
        raise ValueError("Initial deformation must be identity")
    import torch

    for view in range(4):
        with torch.no_grad():
            image, _, _ = engine.render(state, view)
        export_image(image, output / ("rest_train_view%d.png" % view))
    metrics, gradient, central = engine.evaluate(state, force, 1, gradient=True)
    _, repeat, _ = engine.evaluate(state, force, 1, gradient=True)
    norm = float(np.linalg.norm(gradient))
    direction = gradient / max(norm, 1e-30)
    adjoint = float(np.sum(gradient.astype(np.float64) * direction))
    _, gx, gf = engine.objective(central, 1, gradient=True)
    report = dict(
        scene=engine.config["scene"],
        passed=False,
        metrics=metrics,
        zero_force_initialization=True,
        gradient_norm=norm,
        repeat_relative_l2=float(np.linalg.norm(repeat - gradient) / max(norm, 1e-30)),
        samples=[],
    )
    for epsilon in engine.config["audit_epsilons"]:
        plus, _, ps = engine.evaluate(state, force + epsilon * direction, 1)
        minus, _, ms = engine.evaluate(state, force - epsilon * direction, 1)
        contraction = float(
            (
                np.sum(gx.astype(np.float64) * (ps["x"] - ms["x"]))
                + np.sum(gf.astype(np.float64) * (ps["F"] - ms["F"]))
            )
            / (2 * epsilon)
        )
        fd = (plus["loss"] - minus["loss"]) / (2 * epsilon)
        report["samples"].append(
            dict(
                epsilon=epsilon,
                adjoint=adjoint,
                mpm_fd=contraction,
                image_fd=fd,
                mpm_relative_error=abs(adjoint - contraction)
                / max(abs(adjoint), abs(contraction), 1e-30),
                full_relative_error=abs(adjoint - fd)
                / max(abs(adjoint), abs(fd), 1e-30),
            )
        )
        write_json(output / "audit.json", report)
        print(
            "Gradient audit epsilon=%g: MPM %.3f%%, full chain %.3f%%"
            % (
                epsilon,
                100 * report["samples"][-1]["mpm_relative_error"],
                100 * report["samples"][-1]["full_relative_error"],
            ),
            flush=True,
        )
    report["passed"] = bool(
        norm > 0
        and report["repeat_relative_l2"] < 0.01
        and max(s["mpm_relative_error"] for s in report["samples"]) < 0.01
        and max(s["full_relative_error"] for s in report["samples"]) < 0.20
    )
    write_json(output / "audit.json", report)
    if not report["passed"]:
        raise RuntimeError("Gradient audit failed; see " + str(output / "audit.json"))
    return report
