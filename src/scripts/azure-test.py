import os
from pathlib import Path
import sys
import time

library_path = str(Path(__file__).parent.parent.parent)
PYPATH = os.environ.get("PYTHONPATH", "").split(":")
if library_path not in PYPATH:
    PYPATH.append(library_path)
    os.environ["PYTHONPATH"] = ":".join(PYPATH)

from model_drift.data.padchest import PadChest
from model_drift.data.padchest import LABEL_MAP
from model_drift.drift.sampler import Sampler
from model_drift.drift.performance import ClassificationReportCalculator
from model_drift.drift.categorical import ChiSqDriftCalculator
from model_drift.drift.numeric import KSDriftCalculator, BasicDriftCalculator
from model_drift.drift.tabular import TabularDriftCalculator
from model_drift import settings, helpers
import warnings
import pandas as pd
import numpy as np
import logging
import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--run_azure", type=int, dest="run_azure", help="run in AzureML", default=0)
parser.add_argument("--input_dir", type=str)
parser.add_argument("--dt", type=float, default=.1)
parser.add_argument("--total", type=int, default=100)


args = parser.parse_args()

helpers.basic_logging()
warnings.filterwarnings("ignore")


print("~-" * 10)
print("Print Testing")
print("System:", sys.version_info)
print("Pandas Version:", pd.__version__)
print("Numpy Version:", np.__version__)
print("~-" * 10)

print("Test print to stderr", file=sys.stderr)
print("Test print to stdout", file=sys.stdout)

logger = logging.getLogger("logging tester")
logger.debug("Debug Test")
logger.info("Info Test")
logger.warning("warning Test")
logger.error("error Test")


for i in tqdm.tqdm(range(args.total)):
    time.sleep(args.dt)



