"""The study configuration (config.yaml) and the measured cost table (scheduler/cost_model.json)."""

from __future__ import annotations

import json
import os

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_cost(path: str) -> dict:
    """The cost table, or {} before it has been measured."""
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def save_cost(path: str, cost: dict) -> None:
    with open(path, "w") as f:
        json.dump(cost, f, indent=1)
