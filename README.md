# SedCas: Sediment Cascade Model

Source code for SedCas. For a detailed model description, please see Hirschberg et al. (2021)

## How to install and run

Pyhton has to be installed to run the codes. To make sure it works correctly, it is easiest to install Anaconda and create an environment with the right packages from the `.yml` file. To this end, in a command-line interpreter, change the working directory to where you saved this project and run the following:

`$ conda env create -f sedcas.yml`

`$ conda activate sedcas` or `$ source activate sedcas`

To run an example:

`$ python run.py`

## How to modify the model

If you want to modify the model or adapt it to your case, there are the following options:

- change parameters in the `.par` file
- change the climate input in the `.met` file
- add or change a module in `module.py`
- implement new modules into the model structure in `SedCas.py`

## Related publications

Hirschberg, J., McArdell, B. W., Bennett, G. L., Molnar, P. (2022). Numerical Investigation of Sediment Yield Underestimation in Supply-Limited Mountain Basins with Short Records. Geophysical Research Letters, in review.

Hirschberg, J., Fatichi, S., Bennett, G. L., McArdell, B. W., Peleg, N., Lane, S. N., Schlunegger, F., Molnar, P. (2021). Climate Change Impacts on Sediment Yield and Debris-Flow Activity in an Alpine Catchment. Journal of Geophysical Research: Earth Surface, 126, e2020JF005739. https://doi.org/10.1029/2020JF005739

Bennett, G. L., P. Molnar, B. W. McArdell, and P. Burlando (2014), A probabilistic sediment cascade model of sediment transfer in the Illgraben, Water Resour. Res., 50, 1225– 1244, https://doi.org/10.1002/2013WR013806.

## License
GNU General Public License v3.0
