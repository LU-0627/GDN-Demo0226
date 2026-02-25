import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    ZoneInfo = None


@dataclass
class ExperimentPaths:
    exp_dir: str
    log_file: str
    config_file: str
    metrics_file: str
    model_file: str


def _shanghai_tz():
    if ZoneInfo is not None:
        try:
            return ZoneInfo("Asia/Shanghai")
        except Exception:
            pass
    return timezone(timedelta(hours=8))


def shanghai_now() -> datetime:
    return datetime.now(_shanghai_tz())


def format_shanghai_timestamp(dt: Optional[datetime] = None) -> str:
    if dt is None:
        dt = shanghai_now()
    return dt.strftime("%Y%m%d_%H%M%S")


def get_experiment_dir(log_root: str, dataset: str, timestamp: str) -> str:
    return str(Path(log_root) / dataset / timestamp)


def get_latest_experiment_dir(log_root: str, dataset: str) -> Optional[str]:
    dataset_dir = Path(log_root) / dataset
    if not dataset_dir.exists():
        return None

    exp_dirs = [p for p in dataset_dir.iterdir() if p.is_dir()]
    if not exp_dirs:
        return None

    exp_dirs.sort(key=lambda p: p.name)
    return str(exp_dirs[-1])


def create_experiment_dir(log_root: str, dataset: str, resume_dir: Optional[str] = None) -> str:
    if resume_dir:
        exp_dir = Path(resume_dir)
    else:
        exp_dir = Path(log_root) / dataset / format_shanghai_timestamp()

    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir)


def build_experiment_paths(exp_dir: str, model_filename: str = "best_model.pt") -> ExperimentPaths:
    exp_path = Path(exp_dir)
    return ExperimentPaths(
        exp_dir=str(exp_path),
        log_file=str(exp_path / "train.log"),
        config_file=str(exp_path / "config.json"),
        metrics_file=str(exp_path / "metrics.csv"),
        model_file=str(exp_path / model_filename),
    )


def init_experiment_logger(
    dataset: str,
    log_root: str = "./logs",
    resume_dir: Optional[str] = None,
    logger_name: str = "mtsad",
    level: int = logging.INFO,
    rank: int = 0,
) -> Tuple[logging.Logger, ExperimentPaths]:
    exp_dir = create_experiment_dir(log_root, dataset, resume_dir=resume_dir)
    paths = build_experiment_paths(exp_dir)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(paths.log_file, mode="a", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger, paths


def save_config(config: Dict, config_file: str):
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def append_metrics(metrics_file: str, row: Dict):
    metric_path = Path(metrics_file)
    metric_path.parent.mkdir(parents=True, exist_ok=True)

    file_exists = metric_path.exists()
    fieldnames = list(row.keys())

    with open(metric_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
