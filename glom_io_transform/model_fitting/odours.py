import os
from scipy.io import loadmat
from typing import List, NamedTuple
import pandas as pd

# odours MATLAB file
from glom_io_transform.analysis import paths
odours_mat_file = os.path.join(paths.data_root, "odour_labels.mat")
assert os.path.exists(odours_mat_file), f"Odours file not found: {odours_mat_file}" 
odours_mat = loadmat(odours_mat_file)

# Odour orders: name, chemical_sort, input, output
order_file = os.path.join(paths.data_root, "odour_orders.csv")
assert os.path.exists(order_file), f"Odour order file not found: {order_file}"
orders_df = pd.read_csv(order_file, delimiter=";")

def rename_odour(odour: str) -> str:
    """ Converts odours names to Tobias new list of odours names. """
    rename = {
        "2-methyl-4-butanol": "2-methyl-2-butanol",
        "cineol":"cineole",
        "1,4-cineol":"1,4-cineole"
    }
    return rename[odour] if odour in rename else odour

class Odours(NamedTuple):
    names: List[str]
    classes: List[str]
    
    def get_order(self, which_order: str) -> List[int]:
        if which_order == "default":
            return list(range(len(self.names)))
        elif which_order in ["chemical_class", "input", "output"]:
            sorted_odours = orders_df.sort_values(by=which_order).name
            return [self.names.index(n) for n in sorted_odours]
        else:
            raise ValueError(f"Unknown order: {which_order}. Must be one of 'default', 'chemical_class', 'input', 'output'.")
    
odours  = Odours(names   = [rename_odour(str(n[0]).lower()) for n in odours_mat["odour_labels"][0]],
                 classes = [str(n[0]) for n in odours_mat["odour_labels"][1]])


