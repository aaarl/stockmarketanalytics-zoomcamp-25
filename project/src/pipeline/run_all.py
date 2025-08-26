import argparse
from pathlib import Path
import yaml

from src.pipeline.step_01_download import run as step01
from src.pipeline.step_02_unify_dataset import run as step02
from src.pipeline.step_03_feature_engineering import run as step03
from src.pipeline.step_04_train import run as step04
from src.pipeline.step_05_predict import run as step05
from src.pipeline.step_06_simulate import run as step06


def main(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    Path(cfg["paths"]["interim_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["figures_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["backtests_dir"]).mkdir(parents=True, exist_ok=True)

    step01(cfg)
    step02(cfg)
    step03(cfg)
    step04(cfg)
    step05(cfg)
    step06(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)
