# glom-io-transform-release
Released code for the glomerular input-output paper.

## Installation Instructions
First, download and unpack the [connectivity models repository](https://github.com/stootoon/ob_io_conn_models/)

Add path-related environment variables by update the paths in `add_paths.sh` and sourcing that file using
`source add_paths.sh`
This will
1. Add an environment variable GLOM_IO to point to the root of this repository.
2. Add an environment variable GLOM_IO_DATA to point to `data` directory of this repository.
3. Add an environment variable OB_IO_CONN_MODELS to the root directory where you installed the repository.

## Fitting Connectivity Models
This done in several steps.
1. A YAML file is created specifying the model and parameters. The files for the Diagonal and Free models used in the paper are provided.
2. **Creating model input files**
   The YAML files are used to generate input files for different parameter sweeps and different random samplings of the data. 
   These specifications must be expanded into input files for specific runs.
   To generate these inputs files, change to the `model-fitting` directory and run
   `python driver.py --gen [YAML_FILE]`
   This will create subfolders `model-fitting/fits` containing the required model input pickle files, named `in.XYZ.p`/
	   For example, for the provided `fit_diag.yaml`, this will create the directory
	   `glom-io-transform-release/model-fitting/fits/center=True/standardization=separate/normalization=odour_std/fit_diag`
	   and put the input files in it.
3. ** Running the model fits**
   To compute the fit, run
   `python driver.py --run [INPUT_FILE]`
   This will geneate a corresponding output file `out.XYZ.p` in the same folder as the input file.
4. ** Collecting the fits**
   Once all the output files have been generated, the results must be collected into a single file.
   To perform the collection, run
   `python driver.py --collect [PATH_TO_OUTPUTS]`
   This will produce a `collected.p` in the specified path.
   For example, to collect the results of the runs above, run
   `python driver.py --collect glom-io-transform-release/model-fitting/fits/center=True/standardization=separate/normalization=odour_std/fit_diag`
   
### Evaluating the Fits
Once the fits of a given model have been collected, the training, test and validation performance can be viewed over the range of regularization parameters by running the notebook `proc_fit_models.ipynb`. 

## Producing the figures
Once the fits to the Diagonal and Free models have been computed, and the data collected, the figures can be produced by running the notebook `analysis/fig_driver.ipynb`.
