import os
from scipy.io import loadmat
from typing import List, NamedTuple

# odours MATLAB file
odours_mat_file = os.path.join(os.environ["GLOM_IO_DATA"], "odour_labels.mat")
assert os.path.exists(odours_mat_file), f"Odours file not found: {odours_mat_file}" 
odours_mat = loadmat(odours_mat_file)

class Odours(NamedTuple):
    names: List[str]
    classes: List[str]

odours  = Odours(names   = [str(n[0]) for n in odours_mat["odour_labels"][0]],
                 classes = [str(n[0]) for n in odours_mat["odour_labels"][1]])



