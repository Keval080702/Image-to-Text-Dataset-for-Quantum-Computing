"""
Unified stage monitoring module.
Respects STAGE_MONITOR_DIR env; writes to events.log; creates per-stage folders.
"""
from __future__ import annotations

import os
import shutil
import datetime
from pathlib import Path
from typing import Dict

DEFAULT_SUBFOLDERS = [
    "stage0_accepted",
    "stage0_rejected",
    "stage1_ocr_accepted",
    "stage1_ocr_rejected",
    "stage1_text_accepted",
    "stage1_text_rejected",
    "stage2_accepted",
    "stage2_rejected",
    "final_accepted",
    "final_rejected",
]

class StageMonitor:
    def __init__(self, root: str | Path | None = None, subfolders = DEFAULT_SUBFOLDERS):
        base_dir = Path(__file__).resolve().parent
        env_dir = os.environ.get("STAGE_MONITOR_DIR")
        if env_dir:
            output_root = Path(env_dir)
        elif root is not None:
            output_root = Path(root)
        else:
            output_root = base_dir / "stage_monitoring"
        self.root = output_root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

        self.verbose = bool(os.environ.get("STAGE_MONITOR_VERBOSE"))
        self.events_log = self.root / "events.log"

        self.folders: Dict[str, Path] = {}
        for name in subfolders:
            p = self.root / name
            p.mkdir(parents=True, exist_ok=True)
            self.folders[name] = p

        print(f"[Monitor] Using stage monitoring root: {self.root}")

    def save(self, folder_key: str, arxiv_id: str, fig_num: int, image_path: str):
        src = Path(image_path)
        if not src.exists():
            return
        dst_dir = self.folders.get(folder_key, self.root)
        dst_dir.mkdir(parents=True, exist_ok=True)
        safe_id = (arxiv_id or "").replace("/", "_").replace(" ", "_")
        ext = src.suffix.lower() if src.suffix else ".png"
        dst = dst_dir / f"{safe_id}_fig{fig_num}{ext}"
        try:
            shutil.copy(str(src), str(dst))
            self._log_event("SAVED", folder_key, safe_id, fig_num, src, dst)
            if self.verbose:
                print(f"[Monitor] {folder_key}: {src} -> {dst}")
        except Exception as e:
            print(f"[Monitor] Copy failed to {folder_key}: {e}")
            self._log_event("ERROR", folder_key, safe_id, fig_num, src, dst, error=str(e))

    def _log_event(self, kind: str, folder_key: str, safe_id: str, fig_num: int, src: Path, dst: Path, error: str = ""):
        try:
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            with self.events_log.open("a", encoding="utf-8") as f:
                line = f"{ts} {kind} {folder_key} {safe_id} fig{fig_num} :: {src} -> {dst}"
                if error:
                    line += f" :: {error}"
                f.write(line + "\n")
        except Exception:
            pass
