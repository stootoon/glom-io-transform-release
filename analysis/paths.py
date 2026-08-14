import os, sys
from pathlib import Path
proj_path  = os.environ["GLOM_IO"]
data_root = os.path.join(proj_path, "data")
fits_root = os.path.join(proj_path, "model_fitting")
print(f"{proj_path=}")
print(f"{fits_root=}")
print(f"{data_root=}")
