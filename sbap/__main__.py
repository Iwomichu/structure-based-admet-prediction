import logging
import pathlib
import pickle
import shutil
from datetime import datetime, timezone
from typing import Optional, cast

import pandas as pd
import rich
import typer
from sklearn.ensemble import RandomForestRegressor

from sbap.docking import SminaConfig
from sbap.featurizers.prolif_smina import DockedProlifFingerprintFeaturizer, \
    SminaDockingPersistenceHandler
from sbap.pipeline import Regressor, Pipeline

app = typer.Typer()


@app.command()
def dock(
        sdf_file: pathlib.Path,
        protein_pdb_file: pathlib.Path,
        smina_config_file: pathlib.Path,
        docked_ligands_target_directory: Optional[pathlib.Path] = None,
        batch_size: int = 5,
        starting_batch: int = 0,
) -> None:
    logging.basicConfig()
    if docked_ligands_target_directory is None:
        protein_name = protein_pdb_file.name
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y_%m_%d_%H_%M_%S")
        docked_ligands_target_directory = pathlib.Path(f"{protein_name}_{timestamp}")
        logging.info(f"Docking results will be saved to {docked_ligands_target_directory.absolute()}")
    shutil.rmtree(docked_ligands_target_directory, ignore_errors=True)
    config = SminaConfig.parse_file(smina_config_file, content_type="yaml")
    handler = SminaDockingPersistenceHandler.create(config, str(docked_ligands_target_directory),
                                                    logging.INFO)  # pass unlabeled persistence here
    handler.dock(protein_pdb_file, sdf_file, batch_size=batch_size, starting_batch=starting_batch)


@app.command()
def train_model(
        sdf_file: pathlib.Path,
        protein_pdb_file: pathlib.Path,
        smina_config_file: pathlib.Path,
        output_model_path: pathlib.Path,
        pickled_regressor_path: Optional[pathlib.Path] = None,
) -> None:
    config = SminaConfig.parse_file(smina_config_file, content_type="yaml")
    fingerprint_featurizer = DockedProlifFingerprintFeaturizer.create(
        logging_level=logging.INFO,
    )
    if pickled_regressor_path is not None:
        with open(pickled_regressor_path, 'rb') as f:
            model: Regressor = pickle.load(f)
    else:
        model = cast(Regressor, RandomForestRegressor())
    pipeline = Pipeline(config, fingerprint_featurizer, model)
    rich.print(f"r2 score: {pipeline.fit(protein_pdb_file, sdf_file)}")
    with open(output_model_path, "wb") as f:
        pickle.dump(pipeline, f)


@app.command()
def predict(
        sdf_file: pathlib.Path,
        pickled_pipeline_path: pathlib.Path,
        output_path: pathlib.Path,
) -> None:
    with open(pickled_pipeline_path, "rb") as f:
        pipeline: Pipeline = pickle.load(f)
    output = pipeline.predict(sdf_file)
    pd.DataFrame(output).to_csv(output_path, header=False, index=False)


app()
