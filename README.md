# structure-based-admet-prediction

The goal of this project is to apply machine learning to [Structural Interation Fingerprints](https://prolif.readthedocs.io/en/stable/) (or SIFt's in short) in order to predict those ADME properties of chemical compounds which depend heavily on interactions with some particular receptors. Here we focus on three examples:
- predicting ligand metabolism based on interactions with [CYP2C8](https://www.rcsb.org/structure/2NNI)
- predicting ligand matebolism based on interactions with [CYP2C9](https://www.rcsb.org/structure/4NZ2)
- predicting ligand distribution based on interactions with [human serum albumin](https://www.rcsb.org/structure/4LA0)

This repository contains code transforming the pair (``.sdf`` file with ligands, ``.pdb`` file with receptor) into structural interaction fingerprints, which can be used as input for machine learning models. This involves performing molecular docking via [smina](https://github.com/mwojcikowski/smina). Furthermore, example machine learning models were implemented and compared in the form of Jupyter Notebooks.

## Development
### Requirements
* `conda` https://docs.conda.io/en/latest/

### Environment
* Create new Conda environment `conda env create -f environment.yml `
* Run `conda activate structure-based-admet-prediction`
* Run `pip install -r requirements.txt`

### Usage

#### Python API

We need to supply two files - ``.sdf`` with ligands, ``.pdb`` with protein and a directory where docked ligands can be found (or should be placed, if molecular docking was not performed yet)
```
sdf_file = pathlib.Path("path/to/sdf/with/ligands")
protein_pdb_file = pathlib.Path("path/to/protein/pdb/file")
docked_ligands_target_directory = pathlib.Path("directory/with/docked/ligands")
```

To perform molecular docking, you should specify some parameters, see [Smina](https://github.com/mwojcikowski/smina) documentation for more details:
```
config = SminaConfig(
    center_x=48.254, center_y=11.175, center_z=-20.580, size_x=30, size_y=30, size_z=30, exhaustiveness=8,
)
```
Perform molecular docking and save results in ``docked_ligands_target_directory``:
```
persistent_docking_handler = SminaDockingPersistenceHandler.create(
    smina_config=config,
    docked_ligands_target_directory=str(docked_ligands_target_directory),
    logging_level=logging.INFO,
)
persistent_docking_handler.dock(protein_pdb_file, sdf_file, starting_batch=1, batch_size=15)
```
And transform this output into SIFt's.
```
fingerprint_featurizer = DockedProlifFingerprintFeaturizer.create(
    logging_level=logging.INFO,
)
fingerprint_featurizer.fit(protein_pdb_file, docked_ligands_target_directory)
X, y = fingerprint_featurizer.transform(protein_pdb_file, docked_ligands_target_directory)
```

#### CLI

A command-line tool, ``sbap``, is available on the branch``cli-predict``.

Supported CLI commands:

* `dock` Ligand docking
* `train-model` Train a (optionally supplied) regressor model to predict ADMET features
* `predict` Predict ADMET features with supplied pipeline (trained with `train-model`)

Run `python -m sbap --help` for commands usage details.
