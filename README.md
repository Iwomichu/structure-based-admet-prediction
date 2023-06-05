# structure-based-admet-prediction

### Requirements

* `conda` https://docs.conda.io/en/latest/

### Environment

* Create new Conda environment `conda env create -f environment.yml `
* Run `conda activate structure-based-admet-prediction`

### Usage

Supported CLI commands:

* `dock` Ligand docking
* `train-model` Train a (optionally supplied) regressor model to predict ADMET features
* `predict` Predict ADMET features with supplied pipeline (trained with `train-model`)

Run `python -m sbap --help` for commands usage details.