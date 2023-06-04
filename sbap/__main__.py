import logging
import pathlib
import shutil
from datetime import datetime, timezone
from typing import Optional

import typer

from sbap.docking import SminaConfig
from sbap.featurizers.prolif_smina import SminaDockingPersistenceHandler

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
    handler = SminaDockingPersistenceHandler.create(config, str(docked_ligands_target_directory), logging.INFO)
    handler.dock(protein_pdb_file, sdf_file, batch_size=batch_size, starting_batch=starting_batch)


app()
