MRIV
==============================

#### Introduction
This repository contains the code to our paper "Estimating individual treatment effects under unobserved confounding unsing binary instruments".


#### Requirements
The project is build with python 3.9.7 and uses the packages listed in the file `requirements.txt`. In particular the following packages need to be installed to reproduce our results:
1. [Pytorch 1.10.0, Pytorch lightning 1.5.1] - deep learning models
2. [Optuna 2.10.0] - hyperparameter tuning
4. Other: Pandas 1.3.4, numpy 1.21.5, scikit-learn 1.0.1

The calculation of the propensity score of the OHIE data (see Appendix D) is performed in the R script `data/propensity_score.R`. To run the script, the R package `BiasedUrn` needs to be installed.

#### Datasets
In our paper we used three datasets: Synthetic, real-world data from the Oregon health insurance experiment (OHIE), and semi-synthetic data. 

###### Synthetic data
The script for synthetic data generation is `data/sim.py`. Here, the data is simulated using Gaussian Processes according to Appendix C in the paper.

###### Real-world data
We use the data from the Oregon Health insurance experiment from Finkelstein et al (2012). The data is publicly available and can be downloaded together with a detailed documentation on the website `https://www.nber.org/programs-projects/projects-and-centers/oregon-health-insurance-experiment`. To run the experiments, the `.dta` files need to be copied into the folder `data/oregon_health_exp/OHIE_Data`.

###### Semi-synthetic data
The semi-synthetic data (Appendix H) is generated via the `data/sim_semi.py.py`. Note that the OHIE data needs to be downloaded before running the script.


#### Reproducing the experiments
The scripts running the experiments are contained in the `/experiments` folder. There are three directories, one for each dataset (synthetic = `/sim`, real-world = `/real`, and semi-synthetic = `/sim_semi`). Most experiments can be configured by a `.yaml` configuration file. Here, parameters for data generation (e.g., sample size, confounding level, smoothness) as well as the methods used may be adjusted. The following base methods are available (for details see Appendix E):

- `tsls`: Two-stage least squares, 
- `waldlinear`: Linear Wald estimator,
- `ncnet`: MRIV (network only),
- `tarnet`: TARNet,
- `dmliv`: DMLIV,
- `deepiv`: DeepIV,

The corresponindg .yaml configuration files can be found in `/experiments/conifg/`. Here, the "treat" parameter denotes the treatment configuration (setting) and takes values in {1,2,3}.

#### Reproducing hyperparameter tuning
The hyperparameters for the models trained from the /experiments folder are stored under `/hyperparame/parameters`. For reproducing hyperparameter tuning, one needs to run `hyperparam/hyperparameter.py` (synthetic + semi-synthetic data) or hyperparam/hyperparameter_backpain.py (real-world data). Again, the correct configuration files need to be specified, indicating the models and settings.
