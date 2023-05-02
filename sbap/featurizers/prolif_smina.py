from __future__ import annotations

import logging
import pathlib
from typing import Optional

import numpy as np
import rdkit.Chem
from numpy import typing as npt

from sbap._types import ReceptorInteractionCombination
from sbap.docking import SminaConfig, SminaDockerizer
from sbap.featurizers.base import BaseFeaturizer
from sbap.fingerprint import ProlifInteractionFingerprintGenerator
from sbap.sdf import ChemblSdfReader, ChemblSdfRecord


class ProlifSminaFeaturizer(BaseFeaturizer):
    def __init__(
            self,
            sdf_reader: ChemblSdfReader,
            prolif_fingerprint_generator: ProlifInteractionFingerprintGenerator,
            smina_dockerizer: Optional[SminaDockerizer] = None,
            smina_config: Optional[SminaConfig] = None,
            logging_level: int = logging.INFO,
            docked_ligands_directory: Optional[str] = None,
    ) -> None:
        super().__init__(logging_level)
        self.sdf_reader = sdf_reader
        self.docked_ligands_directory = docked_ligands_directory
        self.smina_config = smina_config
        self.prolif_fingerprint_generator = prolif_fingerprint_generator
        self.smina_dockerizer = smina_dockerizer
        self.allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = set()

    @staticmethod
    def create(
            smina_config: Optional[SminaConfig] = None,
            logging_level: int = logging.INFO,
            docked_ligands_directory: Optional[str] = None,
    ) -> ProlifSminaFeaturizer:
        sdf_reader = ChemblSdfReader(logging_level)
        prolif_fingerprint_generator = ProlifInteractionFingerprintGenerator(logging_level)
        smina_dockerizer = SminaDockerizer(smina_config) if smina_config is not None else None
        if docked_ligands_directory is None and smina_config is None:
            raise RuntimeError("Either Smina Config or docker ligands location should be provided")
        return ProlifSminaFeaturizer(
            sdf_reader=sdf_reader,
            prolif_fingerprint_generator=prolif_fingerprint_generator,
            smina_dockerizer=smina_dockerizer,
            smina_config=smina_config,
            logging_level=logging_level,
            docked_ligands_directory=docked_ligands_directory,
        )

    def fit(self, protein_pdb_file_path: pathlib.Path, ligands_sdf_file: pathlib.Path) -> None:
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)[:25]
        docked_mols = self._get_docked_mols(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        self.allowed_receptor_interaction_combinations = self.prolif_fingerprint_generator\
            .get_receptor_interaction_combinations(
                protein,
                docked_mols,
            )
        self.logger.debug(f"{self.allowed_receptor_interaction_combinations=}")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            ligands_sdf_file: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        if self.allowed_receptor_interaction_combinations is None:
            self.logger.warning("allowed_receptor_interaction_combinations list is unset. "
                                "All interactions will be allowed")
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)[:50]
        standard_values = [float(record["standardValue"]) for record in parsed_records]
        docked_mols = self._get_docked_mols(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        fingerprints = self.prolif_fingerprint_generator.generate(
            protein,
            docked_mols,
            self.allowed_receptor_interaction_combinations,
        )

        return np.array(fingerprints), np.array(standard_values)

    def _get_docked_mols(
            self,
            parsed_records: list[ChemblSdfRecord],
            protein_pdb_file_path: pathlib.Path,
    ) -> list[rdkit.Chem.Mol]:
        if self.docked_ligands_directory is not None:
            docked_mols = [
                rdkit.Chem.MolFromMol2File(file)
                for file in pathlib.Path(self.docked_ligands_directory).glob("*.mol")
            ]
        else:
            ligand_mols = [rdkit.Chem.MolFromMolBlock(record['mol']) for record in parsed_records]
            docked_mols = self.smina_dockerizer.dock(protein_pdb_file_path=protein_pdb_file_path, ligands=ligand_mols)
        return docked_mols
