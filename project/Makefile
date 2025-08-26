.PHONY: setup run all eda clean

setup:
\tpython -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

run:
\tpython -m src.pipeline.run_all --config config/config.yaml

eda:
\tjupyter lab

clean:
\trm -rf data/interim/* data/processed/* reports/*

