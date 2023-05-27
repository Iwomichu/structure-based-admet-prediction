from __future__ import annotations

import csv
import logging
import pathlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import rdkit.Chem
from numpy import typing as npt
from tqdm import tqdm

from sbap._types import ReceptorInteractionCombination, DockingScore, InteractionFingerprint
from sbap.docking import Dockerizer, DockingResult
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
class LabeledDockingResult(DockingResult):
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
            csv.writer(f).writerow((result_id, result.score, result.label))

    def save_many(self, results: Iterable[LabeledDockingResult]) -> None:
        with open(self.labels_file_path, 'a', encoding='utf-8') as f:
            csv.writer(f).writerows(
                (self._save_to_unique_file(result), result.score, result.label)
                for result in results
            )

    def read(self) -> list[LabeledDockingResult]:
        with open(self.labels_file_path, 'r', encoding='utf-8') as f:
            return [
                LabeledDockingResult(
                    mol=rdkit.Chem.MolFromMolFile(str(self.directory_path.joinpath(f"{result_id}.mol")),
                                                  sanitize=False),
                    score=score,
                    label=label,
                ) for result_id, score, label in csv.reader(f)
            ]

    def get_docked_ligands_paths(self) -> Iterable[pathlib.Path]:
        with open(self.labels_file_path, 'r', encoding='utf-8') as f:
            return [self.directory_path.joinpath(f"{result_id}.mol") for result_id, _, label in csv.reader(f)]

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


def _prepare_features(
        docking_results: list[DockingResult],
        fingerprints: list[InteractionFingerprint],
        standard_values: list[float],
) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    assert len(docking_results) == len(fingerprints)
    fingerprints_np = np.array(fingerprints)
    docking_scores_np = np.array([result.score for result in docking_results]).reshape(-1, 1)
    features = np.hstack([docking_scores_np, fingerprints_np]).astype('float')
    return features, np.array(standard_values).astype('float')


class LigandDockingFingerprintFeaturizer(RawInputBaseFeaturizer):
    def __init__(
            self,
            sdf_reader: ChemblSdfReader,
            fingerprint_generator: InteractionFingerprintGenerator,
            dockerizer: Dockerizer,
            logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(logging_level)
        self.sdf_reader = sdf_reader
        self.fingerprint_generator = fingerprint_generator
        self.dockerizer = dockerizer
        self.allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = set()

    def fit(self, protein_pdb_file_path: pathlib.Path, ligands_sdf_file: pathlib.Path) -> None:
        """
        Fit the featurizer, i.e. perform the docking, generate the fingerprint
        and learn the combinations of receptor-interaction pairs that should be included in the fingerprint.

        For example, if the input to the fit method consist of a ligands that have the following interactions:
         ``[InteractionX ReceptorA, InteractionX ReceptorB, InteractionY ReceptorC]``
        and
         ``[InteractionX ReceptorB, InteractionX ReceptorC]``
        then the featurizer memorizes the
         ``[InteractionX ReceptorA, InteractionX ReceptorB, InteractionX ReceptorC, InteractionY ReceptorC]``
        pairs as allowed combinations. Then, if in the process of transforming a ligand with interactions
         ``[InteractionX ReceptorA, InteractionZ ReceptorA]``
        gets passed, the featurizer returns ``[1 0 0 0]`` as its fingerprint.

         This ensures
         1) strict order of columns (same column always mean the same receptor-interaction combination)
         2) only known interactions are included in the fingerprint

        :param protein_pdb_file_path:
        :param ligands_sdf_file:
        :return:
        """
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)
        docking_results = self._get_docking_results(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        self.allowed_receptor_interaction_combinations.update(
            self.fingerprint_generator.get_receptor_interaction_combinations(
                protein,
                [result.mol for result in docking_results],
            )
        )
        self.logger.debug(f"{self.allowed_receptor_interaction_combinations=}")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            ligands_sdf_file: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Perform docking with Smina and calculate Prolif fingerprints for docked ligands.

        Return features in the form of Numpy array in which first column (output[:, 0]) is the Docking score
        and next columns are either 0. if the ligand does not react with a fitted receptor or 1. if it does.

        :param protein_pdb_file_path:
        :param ligands_sdf_file:
        :return:
        """
        if self.allowed_receptor_interaction_combinations is None:
            self.logger.warning("allowed_receptor_interaction_combinations list is unset. "
                                "All interactions will be allowed")
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)
        standard_values = [float(record["standardValue"]) for record in parsed_records]
        docking_results = self._get_docking_results(parsed_records, protein_pdb_file_path)
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        fingerprints = self.fingerprint_generator.generate(
            protein,
            [result.mol for result in docking_results],
            self.allowed_receptor_interaction_combinations,
        )
        return _prepare_features(docking_results, fingerprints, standard_values)

    def _get_docking_results(
            self,
            parsed_records: list[ChemblSdfRecord],
            protein_pdb_file_path: pathlib.Path,
    ) -> list[DockingResult]:
        ligand_mols = [rdkit.Chem.MolFromMolBlock(record['mol']) for record in parsed_records]
        docking_results = self.dockerizer.dock(protein_pdb_file_path=protein_pdb_file_path, ligands=ligand_mols)
        return docking_results


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
        """
        Fit the featurizer, i.e. read docked ligands, generate the fingerprint
        and learn the combinations of receptor-interaction pairs that should be included in the fingerprint.

        For example, if the input to the fit method consist of a ligands that have the following interactions:
         ``[InteractionX ReceptorA, InteractionX ReceptorB, InteractionY ReceptorC]``
        and
         ``[InteractionX ReceptorB, InteractionX ReceptorC]``
        then the featurizer memorizes the
         ``[InteractionX ReceptorA, InteractionX ReceptorB, InteractionX ReceptorC, InteractionY ReceptorC]``
        pairs as allowed combinations. Then, if in the process of transforming a ligand with interactions
         ``[InteractionX ReceptorA, InteractionZ ReceptorA]``
        gets passed, the featurizer returns ``[1 0 0 0]`` as its fingerprint.

         This ensures
         1) strict order of columns (same column always mean the same receptor-interaction combination)
         2) only known interactions are included in the fingerprint

        :param protein_pdb_file_path:
        :param docked_ligands_directory_path:
        :return:
        """
        handler = LabeledDockingResultHandler(docked_ligands_directory_path)
        docking_results = handler.read()
        docked_mols = [result.mol for result in docking_results]
        protein = rdkit.Chem.MolFromPDBFile(str(protein_pdb_file_path))
        self.allowed_receptor_interaction_combinations.update(
            self.fingerprint_generator.get_receptor_interaction_combinations(protein, docked_mols)
        )
        self.logger.debug(f"{self.allowed_receptor_interaction_combinations=}")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            docked_ligands_directory_path: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        """
        Read docked ligands and calculate Prolif fingerprints for them.

        Return features in the form of Numpy array in which first column (output[:, 0]) is the Docking score
        and next columns are either 0. if the ligand does not react with a fitted receptor or 1. if it does.

        :param protein_pdb_file_path:
        :param docked_ligands_directory_path:
        :return:
        """
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
        return _prepare_features(docking_results, fingerprints, standard_values)


class DockingScoreFeaturizer(ABC):
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    def featurize(
            self,
            protein_pdb_file_path: pathlib.Path,
            docked_ligands_directory_path: pathlib.Path,
    ) -> npt.ArrayLike:
        """
        Read docked ligands and calculate docking score for them.

        Return features in the form of Numpy array

        :param protein_pdb_file_path:
        :param docked_ligands_directory_path:
        :return:
        """
        handler = LabeledDockingResultHandler(docked_ligands_directory_path)
        return np.array([
            self.calculate(protein_pdb_file_path, docked_ligand_path)
            for docked_ligand_path in tqdm(handler.get_docked_ligands_paths())
        ])

    @abstractmethod
    def calculate(
            self,
            protein_pdb_file_path: pathlib.Path,
            docked_ligand_path: pathlib.Path,
    ) -> DockingScore:
        pass
