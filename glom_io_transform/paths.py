import os, sys
from pathlib import Path
proj_path = os.path.join(os.environ["GLOM_IO"], "glom_io_transform") 
data_root = os.environ["GLOM_IO_DATA"]
fits_root = os.path.join(proj_path, "model_fitting")
print(f"{proj_path=}")
print(f"{fits_root=}")
print(f"{data_root=}")
