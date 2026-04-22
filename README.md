# Random Forest with Biased Splitting

Implementation of the method proposed in the paper *"A New Approach to Handle Data Shift Based on Feature Importance Measurement"*.

This repository contains the code used to evaluate a biased splitting strategy for Random Forests under data shift scenarios, following the benchmark protocol proposed by Gardner et al. for the TableShift benchmark suite.

Repository: [https://github.com/ecostadelle/random_forest_with_biased_splitting](https://github.com/ecostadelle/random_forest_with_biased_splitting)

## Environment Setup

This project was developed using Python 3.9 on Ubuntu 24.04 LTS.

First, clone the repository:

```bash
git clone https://github.com/ecostadelle/random_forest_with_biased_splitting.git
cd random_forest_with_biased_splitting
```

Then install the required system dependencies:

```bash
sudo apt update
sudo apt install build-essential libgsl-dev python3.9 python3.9-venv
```

After that, create and activate a virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were executed on the following hardware:

* AMD Ryzen 9 3900X 12-Core Processor 
* 64 GB RAM

The code was not tested on Windows, although it may also work there with small adjustments.

## Datasets

The datasets should be downloaded following the instructions from Gardner et al. (2023), also available at:

* [https://tableshift.org/datasets.html](https://tableshift.org/datasets.html)

After downloading the datasets, update the cached dataset path in `run.py`.

By default, the repository uses:

```python
dataset_path = "../tableshift/tmp"
```

Adjust this variable according to the location where the datasets were stored on your machine.

### Available Datasets

* ASSISTments (`assistments`)
* Childhood Lead (`nhanes_lead`)
* College Scorecard (`college_scorecard`)
* Diabetes (`brfss_diabetes`)
* FICO HELOC (`heloc`)
* Food Stamps (`acsfoodstamps`)
* Hospital Readmission (`diabetes_readmission`)
* Hypertension (`brfss_blood_pressure`)
* ICU Length of Stay (`mimic_extract_los_3`)
* ICU Mortality (`mimic_extract_mort_hosp`)
* Income (`acsincome`)
* Public Health Insurance (`acspubcov`)
* Sepsis (`physionet`)
* Unemployment (`acsunemployment`)
* Voting (`anes`)

## Running Experiments

Once the environment and datasets are configured, the experiments can be executed with:

```bash
python run.py
```

## Outputs

The execution generates result files in multiple formats, including:

* CSV files
* Markdown reports
* LaTeX tables

## License

This project is distributed under the GNU General Public License v3.0 (GPLv3).

## Citation

This work is associated with the following forthcoming publication:

```bibtex
@inproceedings{costadelle2026,
  title={A New Approach to Handle Data Shift Based on Feature Importance Measurement},
  author={Costadelle, Ewerton Luiz and Maia, Marcelo Rodrigues de Holanda and Plastino, Alexandre and Freitas, Alex Alves},
  booktitle={To appear in Proceedings of the 21st Iberian Conference on Information Systems and Technologies (CISTI 2026)},
  year={2026}
}
```

## Acknowledgments

This project builds upon following repositories:

* TableShift benchmark by Gardner et al. (2023): [https://github.com/mlfoundations/tableshift](https://github.com/mlfoundations/tableshift)
* The interpretable ensembles implementation by Maia et al. (2023): [https://github.com/marcelorhmaia/interpretable-ensembles-for-ucd](https://github.com/marcelorhmaia/interpretable-ensembles-for-ucd)
* The IPMRF package by Epifânio (2017): [https://cran.r-project.org/web/packages/IPMRF/](https://cran.r-project.org/web/packages/IPMRF/)
