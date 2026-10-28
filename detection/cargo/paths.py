from pathlib import Path


DEFAULT_CARGO_RUN_NAME = "cargo_protrusion_v4_only"
RELABELED_CARGO_RUN_NAME = "cargo_protrusion_relabel_v1"
LEGACY_CARGO_RUN_NAME = "cargo_protrusion_v3_fixed"


def latest_cargo_checkpoint(base_dir):
    run_root = Path(base_dir) / "runs" / "detect"

    def newest_checkpoint(run_name):
        candidates = list(run_root.glob(f"{run_name}*/weights/best.pt"))
        if candidates:
            def checkpoint_order(path):
                name = path.parent.parent.name
                suffix = name[len(run_name):]
                run_number = int(suffix) if suffix.isdigit() else -1
                return path.stat().st_mtime_ns, run_number, name

            return max(candidates, key=checkpoint_order)
        return None

    checkpoint = newest_checkpoint(RELABELED_CARGO_RUN_NAME)
    if checkpoint is None:
        checkpoint = newest_checkpoint(DEFAULT_CARGO_RUN_NAME)
    if checkpoint is None:
        checkpoint = newest_checkpoint(LEGACY_CARGO_RUN_NAME)
    if checkpoint is not None:
        return checkpoint
    return (
        run_root
        / DEFAULT_CARGO_RUN_NAME
        / "weights"
        / "best.pt"
    )
