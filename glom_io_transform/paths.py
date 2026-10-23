import os, sys
from pathlib import Path
proj_path = os.environ["GLOM_IO"] 
data_root = os.environ["GLOM_IO_DATA"]
fits_root = os.path.join(proj_path, "model_fitting")
assert os.path.exists(proj_path), f"Path {proj_path} does not exist"
print(f"{proj_path=}")
assert os.path.exists(fits_root), f"Path {fits_root} does not exist"
print(f"{fits_root=}")
assert os.path.exists(data_root), f"Path {data_root} does not exist"
print(f"{data_root=}")
