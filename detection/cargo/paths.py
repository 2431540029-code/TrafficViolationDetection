from pathlib import Path


DEFAULT_CARGO_RUN_NAME = "cargo_protrusion_v3_fixed"


def latest_cargo_checkpoint(base_dir):
    run_root = Path(base_dir) / "runs" / "detect"
    candidates = list(
        run_root.glob(f"{DEFAULT_CARGO_RUN_NAME}*/weights/best.pt")
    )
    if candidates:
        def checkpoint_order(path):
            run_name = path.parent.parent.name
            suffix = run_name[len(DEFAULT_CARGO_RUN_NAME):]
            run_number = int(suffix) if suffix.isdigit() else -1
            return path.stat().st_mtime_ns, run_number, run_name

        return max(candidates, key=checkpoint_order)
    return (
        run_root
        / DEFAULT_CARGO_RUN_NAME
        / "weights"
        / "best.pt"
    )
