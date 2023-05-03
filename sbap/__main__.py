import logging
import pathlib
import shutil

import typer

from sbap.docking import SminaConfig
from sbap.featurizers.prolif_smina import SminaDockingPersistenceHandler

app = typer.Typer()


@app.command()
def dock(
        sdf_file: pathlib.Path,
        protein_pdb_file: pathlib.Path,
        docked_ligands_target_directory: pathlib.Path,
        smina_config_file: pathlib.Path,
        batch_size: int = 25,
        starting_batch: int = 0,
) -> None:
    shutil.rmtree(docked_ligands_target_directory, ignore_errors=True)
    config = SminaConfig.parse_file(smina_config_file, content_type="yaml")
    handler = SminaDockingPersistenceHandler.create(config, str(docked_ligands_target_directory), logging.INFO)
    handler.dock(protein_pdb_file, sdf_file, batch_size=batch_size, starting_batch=starting_batch)


app()
