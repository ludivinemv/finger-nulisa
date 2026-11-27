# Plasma proteomics reveals signatures of cognitive aging and response to the FINGER multidomain lifestyle intervention 
by Makrina Daniilidou Andrea L. Benedet, Ari Solomon1, Anna Rosenberg, Ludivine Morvan, Jenni Lehtisalo, Francesca Mangialasche, Esko Levälahti, Burak Arslan, Markku Peltonen, Ilaria Pola, Kübra Tan, Tiia Ngandu, Henrik Zetterberg*, Miia Kivipelto*, Anna Matton*

This repository provides the python and R code from the submitted paper --> [Link preprint](https://www.medrxiv.org/content/10.64898/2026.11.13.26367619v1)


## Abstract
The FINGER trial demonstrated that a multidomain lifestyle intervention benefits cognition in older adults at-risk of dementia, yet the underlying biological mechanisms remain unclear. We measured neurodegeneration-related proteins using the NULISAseq CNS disease panel in 1158 FINGER participants to investigate associations between circulating proteins, cognitive trajectories, and intervention response. Independent of the intervention, ten proteins spanning immune, neuronal, and metabolic processes were associated with poorer cognitive trajectories, with TREM2 among the strongest signals. The intervention led to changes in eight proteins, including CRH, IL-7 and MME. Eleven proteins modified the cognitive response to intervention, with KDR and CRH showing the strongest effect. Complementary Elastic Net analyses confirmed CRH, and further identified PTN and CCL2 as markers of intervention responsiveness. Additional proteins showed APOE ε4- and sex-specific associations with the intervention response. Together, these findings identify proteomic signatures of cognitive response that could inform precision dementia prevention. 

# Repo Contents
- R: R scripts
- python: Python scripts
- synthetic_data: synthetic dataset with the same columns that the orginal data and 100 rows. 


# System requirements

The scripts were tested on Windows 11 operating system with the following version:
- R version 4.6.0
- python version 3.11.13


Required Python packages and their tested versions are listed in python/requirements.txt.

Required packages are being installed at the beginning of the R scripts with the following lines:
```r
# Loop to install packages
for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) { # Install only if the package is not already installed
    install.packages(pkg)
  }
}
```
The following packages versions were used:

```
broom  1.0.12
broom.mixed 0.2.9.7
colorspace   2.1.2
dplyr   1.2.1
effectsize   1.0.3
emmeans   2.0.3
forcats   1.0.1
ggplot2   4.0.3
ggrepel   0.9.8
gtsummary   2.6.1
lme4   2.0.1
lmerTest   3.2.1
parameters  0.29.2
patchwork   1.3.2
pheatmap  1.0.13
purrr   1.2.2
ragg   1.5.2
readr   2.2.0
remotes   2.5.0
stringr   1.6.0
svglite   2.2.2
tidyr   1.3.2
tidyverse   2.0.0
```

# Installation guide

To run the analyses, ensure that the appropriate versions of **R** and **Python** are installed.

- **R:** Install the required R version and packages listed in the project dependencies.
- **Python:** Install the required Python version. The required Python packages and their tested versions are listed in `python/requirements.txt`.

For Python, install the dependencies by running:

```bash
pip install -r python/requirements.txt
```

## Running the analyses with the synthetic dataset

Once the required environments and dependencies are installed, run the scripts from the root directory of the repository. It should take less than one hour to run all the scripts for the synthetic dataset with the current settings.

Outputs from R scripts will appear in the folder outputs_R. Outputs from python scripts will appear in the folder outputs_python.

**R scripts**

- `R/total_cohort_analyses.R`: Performs analyses of the total cohort, without considering intervention group as a predictor (groups are only considered for adjustment).
- `R/analyses_by_group.R`: Performs analyses in which intervention group is included as a predictor.
- `R/responder_analysis.R`: Performs responder analyses.
- `R/correlations.R`: Performs correlation analyses.

**Python scripts**

- `python/demographic.py`: Performs demographic analyses.
- `python/bbm_trajectory_plots.py`: Generates BBM trajectory plots.
- `python/elasticnet.py`: Trains and evaluates ElasticNet models.



## Running the analyses with your data

To ensure the scripts work correctly with your data, we recommend using the same column names as those specified in synthetic_data/original/meta_data.csv.

To apply the scripts to your data, change the following with your paths and file names. 
- In R scripts, update paths under "## Path definition ----"
- In R scripts update the variables under section "# Arguments ----"
- In R script responder_analysis.R, define high adherence under "# Create adherence groups"
- In python scripts, update variables under section "#! Options ########"
- For python scripts, update python/projectspecific/config.py file with your paths




# Citation
Makrina Daniilidou, Andrea L. Benedet, Ari Solomon, Anna Rosenberg, Ludivine Morvan, Jenni Lehtisalo, Francesca Mangialasche, Esko Levälahti, Burak Arslan, Markku Peltonen, Ilaria Pola, Kübra Tan, Tiia Ngandu, Henrik Zetterberg, Miia Kivipelto, Anna Matton (November 2026), Plasma proteomics reveals signatures of cognitive aging and response to the FINGER multidomain lifestyle intervention, medRxiv 2026.11.13.26367619 (preprint); doi: https://doi.org/10.64898/2026.11.13.26367619
