"""Command-line entry point for the retained scientific experiments."""

import logging

from experiments import cli
from experiments.runner_exp7 import SPEC as EXP7
from experiments.runner_exp8 import SPEC as EXP8
from experiments.runner_exp9 import SPEC as EXP9
from experiments.runner_exp10 import SPEC as EXP10
from experiments.runner_exp11 import SPEC as EXP11

SPECS = [EXP7, EXP8, EXP9, EXP10, EXP11]


def main() -> None:
    """Dispatch an explicitly selected experiment; help never starts a run."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cli.main(SPECS)


if __name__ == "__main__":
    main()
