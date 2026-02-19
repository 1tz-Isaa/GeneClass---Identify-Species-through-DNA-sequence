"""Archive current training artifacts and reset experiment state.

Usage:
  python3 reset_training_state.py
"""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path


def move_contents(src: Path, dst: Path) -> int:
    if not src.exists():
        return 0

    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for item in src.iterdir():
        target = dst / item.name
        if target.exists():
            # Keep unique names when archive already has same file.
            stamp = datetime.now().strftime("%H%M%S")
            target = dst / f"{item.stem}_{stamp}{item.suffix}"
        shutil.move(str(item), str(target))
        moved += 1
    return moved


def main() -> None:
    runs_root = Path("runs")
    saved_models = runs_root / "saved_models"
    training_logs = runs_root / "training_logs"

    reset_id = datetime.now().strftime("reset_%Y%m%d_%H%M%S")
    archive_root = runs_root / "archive" / reset_id

    moved_saved = move_contents(saved_models, archive_root / "saved_models")
    moved_logs = move_contents(training_logs, archive_root / "training_logs")

    # Recreate clean folders for fresh experiments.
    saved_models.mkdir(parents=True, exist_ok=True)
    training_logs.mkdir(parents=True, exist_ok=True)

    # Fresh best-model registry.
    (saved_models / "best_registry.json").write_text("{}\n", encoding="utf-8")

    print(f"[RESET] Archive: {archive_root.resolve()}")
    print(f"[RESET] Moved from saved_models: {moved_saved} item(s)")
    print(f"[RESET] Moved from training_logs: {moved_logs} item(s)")
    print("[RESET] Ready for new training runs.")


if __name__ == "__main__":
    main()
