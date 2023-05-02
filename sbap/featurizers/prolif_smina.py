import logging
import pathlib
from typing import Optional

import numpy as np
import rdkit.Chem
from numpy import typing as npt

from sbap.featurizers.base import BaseFeaturizer
from sbap.docking import DockingConfig, SminaDockerizer
from sbap.fingerprint import ProlifInteractionFingerprintGenerator
from sbap.types import ReceptorInteractionCombination
from sbap.sdf import ChemblSdfRecord


class ProlifSminaFeaturizer(BaseFeaturizer):
    def __init__(
            self,
            smina_config: Optional[DockingConfig] = None,
            logging_level: int = logging.INFO,
            docked_ligands_directory: Optional[str] = None,
    ) -> None:
        super().__init__(logging_level)
        self.docked_ligands_directory = docked_ligands_directory
        self.smina_config = smina_config
        self.prolif_fingerprint_generator = ProlifInteractionFingerprintGenerator(logging_level)
        if self.docked_ligands_directory is None:
            if self.smina_config is None:
                raise RuntimeError("Either Smina Config or docker ligands location should be provided")
            self.smina_dockerizer = SminaDockerizer()

    def fit(
            self,
            protein: rdkit.Chem.Mol,
            ligands: list[ChemblSdfRecord],
    ) -> None:
        # TODO @mjuralowicz: It might be a good idea to do docking and fp here but only to obtain receptor interactions
        self.logger.debug(f"{self.__class__.__name__} does not need fitting")

    def transform(
            self,
            protein: rdkit.Chem.Mol,
            ligands: list[ChemblSdfRecord],
            allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = None,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        if allowed_receptor_interaction_combinations is None:
            raise RuntimeError(f"Allowed receptors cannot be empty for {self.__class__.__name__}")
        if self.docked_ligands_directory is not None:
            docked_mols = [
                rdkit.Chem.MolFromMol2File(file)
                for file in pathlib.Path(self.docked_ligands_directory).glob("*.mol")
            ]
        else:
            ligand_mols = [rdkit.Chem.MolFromMolBlock(record['mol']) for record in ligands]
            docked_mols = self.smina_dockerizer.dock(protein, ligand_mols, self.smina_config)

        fingerprints = self.prolif_fingerprint_generator.generate(
            protein,
            docked_mols,
            allowed_receptor_interaction_combinations,
        )
        standard_values = [float(record["standardValue"]) for record in ligands]

        return np.array(fingerprints), np.array(standard_values)
