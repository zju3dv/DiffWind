"""python -m reconstruction {train,audit,evaluate,status} --help"""
import argparse
import json
from pathlib import Path

from .config import SCENES


def main():
    parser = argparse.ArgumentParser(description="Automatic WD-Objects reconstruction")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "audit", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument(
            "--data-root", type=Path, required=True, help="WD-Objects/paper_evaluation"
        )
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--gpu", default="0")
        if name != "evaluate":
            command.add_argument("--scene", choices=SCENES, required=True)
            command.add_argument(
                "--frames", type=int, help="1-48; a shorter run trains only"
            )
            command.add_argument("--initial-lr", type=float)
            command.add_argument("--min-updates", type=int)
            command.add_argument("--max-updates", type=int)
            command.add_argument("--resume", action="store_true")
    command = commands.add_parser("status")
    command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "status":
        print((args.output / "status.json").read_text())
        progress = sorted((args.output / "windows").glob("*/attempt_*/progress.json"))
        if progress:
            row = json.loads(progress[-1].read_text())
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "frame",
                            "iteration",
                            "loss_after",
                            "best_train_psnr",
                            "next_lr",
                        )
                    },
                    indent=2,
                )
            )
        return
    from .runner import execute

    execute(args)


if __name__ == "__main__":
    main()
