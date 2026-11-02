"""Single place that defines where data / results live.

supp_script/
  script/   <- all code (this file)
  data/     <- input data, written by preprocessing/ (Her2st/org, Her2st/preprocessed, Her2st/cross_validation)
  result/   <- training / evaluation outputs

Everything is resolved relative to this file, never to the current working directory,
so running from any directory will not touch data outside supp_script/.
Override with the env vars IDER_DATA_ROOT / IDER_RESULT_ROOT if needed.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("IDER_DATA_ROOT", os.path.join(PROJECT_ROOT, "data"))
RESULT_ROOT = os.environ.get("IDER_RESULT_ROOT", os.path.join(PROJECT_ROOT, "result"))
