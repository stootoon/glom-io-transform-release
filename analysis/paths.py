import os, sys
from pathlib import Path
proj_path  = os.environ["GLOM_IO"]
conn_models_path = os.environ["OB_IO_CONN_MODELS"]
data_root = os.path.join(proj_path, "data")
fits_root = os.path.join(proj_path, "model-fitting")
print(f"{proj_path=}")
print(f"{fits_root=}")
print(f"{data_root=}")
print(f"{conn_models_path=}")
