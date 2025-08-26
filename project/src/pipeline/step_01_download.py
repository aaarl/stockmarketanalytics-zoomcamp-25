from pathlib import Path

def run(cfg: dict):
    raw_dir = Path(cfg["paths"]["raw_dir"])
    assert (
        raw_dir / cfg["data"]["main_file"]
    ).exists(), "Missing portfolio_data.csv in data/raw/"
    assert (
        raw_dir / cfg["data"]["meta_file"]
    ).exists(), "Missing symbols_valid_meta.csv in data/raw/"
    # In offline mode, we just validate presence. For online mode, implement API download here.
    print("[step_01] Raw files are present.")
