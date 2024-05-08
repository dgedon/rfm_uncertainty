# Uncertainty Estimation with Recursive Feature Machines

This is the official repository for the paper "Uncertainty Estimation with Recursive Feature Machines" by 
Daniel Gedon, Amirhesam Abedsoltan, Thomas B. Schön, and Mikhail Belkin which is presented at UAI 2024.

## Paper summary

We combine the Recursive Feature Machines (RFM), see [here](https://www.science.org/doi/epdf/10.1126/science.adi5639), with Gaussian Processes (GPs) to provide a novel, powerful method for uncertainty estimation.
The RFM is a novel, feature-learning kernel machine, which learns a Mahalanobis distance to re-weight covariates within a kernel machine.
We show that the RFM can be used as a kernel within GPs to provide a powerful method for uncertainty estimation.
Within extensive experiments, we show that the resulting GP-RFM provides a strong alternative to existing methods for a wide range of tabular regression tasks. 

## Install requirements

We utilise conda environments and provide a YAML file with the required packages. 
To install the environment, follow the instructions below.

1. Create a new conda environment from the `environment.yml` file:
   ```setup
   conda env create -f environment.yml
   ```
   
2. Activate the environment
   ```setup
   conda activate uncertainty_rfm
   ```

## Quick-start demo

We provide a simple demo in a Jupyter notebook to illustrate how to use the RFM as a kernel within GPs.
The notebook is named `demo_rfm_uncertainty.ipynb`. 
It does the following steps:

1. Load a dataset from the tabularbenchmark OpenMl repository.
2. Normalize the data.
3. Define the GP-RFM model.
4. Train the RFM model to obtain the Mahalanobis distance.
5. Train the GP-RFM model to obtain uncertainty estimates.
6. Evaluate the model on the test set.

Currently, it runs by default the [ISOLET](https://www.openml.org/search?type=data&status=active&id=44135) (Isolated Letter Speech Recognition) dataset from the OpenML repository.
The demo evaluates the RMSE and NLL and reproduces the results from Table 1 and 6 in the paper.


## Usage

To run the main experiments from the paper, you can use the `main.py` script.
This script uses one dataset and compute all models. We compare with

- GP-RBF
- GP-Laplace
- deep Kernel Learning
- GP-ARD-RBF
- GP-ARD-Laplace
- GP-ARD-Lapace with full Mahalanobis distance (trained with MLE)
- NGBoost
- CatBoost-Ensemble

To remove one comparison method, comment them out from the variable `methods_list`.
Each method has their own set of hyperparameters. You can either set them as default through the arguments. Note to set the `--config_folder=default` in that case.
Otherwise, the hyperparameters from the paper as stored in the `configs` folder will be loaded and used.




## Citation

If you use this code in your research, please consider citing the paper:
```bibtex
@inproceedings{gedon2024uncertainty,
  title={Uncertainty Estimation with Recursive Feature Machines},
  author={Gedon, Daniel and Abedsoltan, Amirhesam and Sch{\"o}n, Thomas B and Belkin, Mikhail},
  booktitle={Proceedings of the 40th Conference on Uncertainty in Artificial Intelligence},
  year={2024}
}
```
