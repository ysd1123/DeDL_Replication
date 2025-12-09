import argparse
import pathlib
import yaml


def parse_config(path: str) -> dict:
    """Load experiment configuration from a YAML file.

    Parameters
    ----------
    path: str
        Path to the configuration YAML file.
    """
    config_path = pathlib.Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fp:
        config = yaml.safe_load(fp)
    return config


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DeDL experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    return parser

