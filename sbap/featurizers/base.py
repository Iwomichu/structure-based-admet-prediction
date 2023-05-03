from __future__ import annotations

import csv
import logging
import pathlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import rdkit.Chem
from numpy import typing as npt

from sbap._types import ReceptorInteractionCombination
from sbap.docking import Dockerizer
from sbap.fingerprint import InteractionFingerprintGenerator
from sbap.sdf import ChemblSdfReader, ChemblSdfRecord


class RawInputBaseFeaturizer(ABC):
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    @abstractmethod
    def fit(self, protein_pdb_file_path: pathlib.Path, ligands_sdf_file: pathlib.Path) -> None:
        pass

    @abstractmethod
    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            ligands_sdf_file: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        pass


@dataclass
class LabeledDockingResult:
    mol: rdkit.Chem.Mol
    label: float


class LabeledDockingResultHandler:
    def __init__(self, directory_path: pathlib.Path) -> None:
        self.directory_path = directory_path
        self.labels_file_path = self.directory_path.joinpath("labels.csv")
        self.directory_path.mkdir(exist_ok=True, parents=True)
        self.labels_file_path.touch(exist_ok=True)

    def save(self, result: LabeledDockingResult) -> None:
        result_id = self._save_to_unique_file(result)
        with open(self.labels_file_path, 'a', encoding='utf-8') as f:
            csv.writer(f).writerow((result_id, result.label))

    def save_many(self, results: Iterable[LabeledDockingResult]) -> None:
        with open(self.labels_file_path, 'a', encoding='utf-8') as f:
            csv.writer(f).writerows((self._save_to_unique_file(result), result.label) for result in results)

    def read(self) -> list[LabeledDockingResult]:
        with open(self.labels_file_path, 'r', encoding='utf-8') as f:
            return [
                LabeledDockingResult(
                    mol=rdkit.Chem.MolFromMolFile(self.directory_path.joinpath(f"{result_id}.mol")),
                    label=label,
                ) for result_id, label in csv.reader(f)
            ]

    def _save_to_unique_file(self, result: LabeledDockingResult) -> uuid.UUID:
        result_id = uuid.uuid4()
        rdkit.Chem.MolToMolFile(result.mol, str(self.directory_path.joinpath(f"{result_id}.mol")), kekulize=False)
        return result_id


class DockedInputBaseFeaturizer(ABC):
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    @abstractmethod
    def fit(self, protein_pdb_file_path: pathlib.Path, docked_ligands_directory_path: pathlib.Path) -> None:
        pass

    @abstractmethod
    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            docked_ligands_directory_path: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        pass


class LigandDockingFingerprintFeaturizer(RawInputBaseFeaturizer):
    def __init__(
            self,
            sdf_reader: ChemblSdfReader,
            fingerprint_generator: InteractionFingerprintGenerator,
            dockerizer: Dockerizer,
            logging_level: int = logging.INFO,
            labeled_docking_result_handler: Optional[LabeledDockingResultHandler] = None,
    ) -> None:
        super().__init__(logging_level)
        self.sdf_reader = sdf_reader
        self.labeled_docking_result_handler = labeled_docking_result_handler
        self.fingerprint_generator = fingerprint_generator
        self.dockerizer = dockerizer
        self.allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = set()

    def fit(self, protein_pdb_file_path: pathlib.Path, ligands_sdf_file: pathlib.Path) -> None:
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)
        docked_mols = self._get_docked_mols(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        self.allowed_receptor_interaction_combinations += self.fingerprint_generator \
            .get_receptor_interaction_combinations(protein, docked_mols)
        self.logger.debug(f"{self.allowed_receptor_interaction_combinations=}")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            ligands_sdf_file: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        if self.allowed_receptor_interaction_combinations is None:
            self.logger.warning("allowed_receptor_interaction_combinations list is unset. "
                                "All interactions will be allowed")
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)
        standard_values = [float(record["standardValue"]) for record in parsed_records]
        docked_mols = self._get_docked_mols(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        fingerprints = self.fingerprint_generator.generate(
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
        ligand_mols = [rdkit.Chem.MolFromMolBlock(record['mol']) for record in parsed_records]
        docked_mols = self.dockerizer.dock(protein_pdb_file_path=protein_pdb_file_path, ligands=ligand_mols)
        if self.labeled_docking_result_handler is not None:
            self.labeled_docking_result_handler.save_many(
                LabeledDockingResult(docked_mol, record['standardValue'])
                for record, docked_mol in zip(parsed_records, docked_mols)
                if docked_mol is not None
            )
        return docked_mols


class DockedLigandFingerprintFeaturizer(DockedInputBaseFeaturizer):
    def __init__(
            self,
            fingerprint_generator: InteractionFingerprintGenerator,
            logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(logging_level)
        self.fingerprint_generator = fingerprint_generator
        self.allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = set()

    def fit(self, protein_pdb_file_path: pathlib.Path, docked_ligands_directory_path: pathlib.Path) -> None:
        handler = LabeledDockingResultHandler(docked_ligands_directory_path)
        docking_results = handler.read()
        docked_mols = [result.mol for result in docking_results]
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        self.allowed_receptor_interaction_combinations += self.fingerprint_generator \
            .get_receptor_interaction_combinations(protein, docked_mols)
        self.logger.debug(f"{self.allowed_receptor_interaction_combinations=}")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            docked_ligands_directory_path: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        handler = LabeledDockingResultHandler(docked_ligands_directory_path)
        docking_results = handler.read()
        docked_mols = [result.mol for result in docking_results]
        standard_values = [result.label for result in docking_results]
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        fingerprints = self.fingerprint_generator.generate(
            protein,
            docked_mols,
            self.allowed_receptor_interaction_combinations,
        )

        return np.array(fingerprints), np.array(standard_values)
