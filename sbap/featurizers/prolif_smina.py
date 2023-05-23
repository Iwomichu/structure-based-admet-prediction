from __future__ import annotations

import logging
import pathlib

import rdkit.Chem

from sbap.docking import SminaConfig, SminaDockerizer
from sbap.featurizers.base import LabeledDockingResultHandler, \
    LigandDockingFingerprintFeaturizer, DockedLigandFingerprintFeaturizer, LabeledDockingResult
from sbap.fingerprint import ProlifInteractionFingerprintGenerator
from sbap.sdf import ChemblSdfReader
from sbap.utils import batched


class SminaDockingPersistenceHandler:
    """
    Class used to perform docking and save the results for future use
    """
    def __init__(
            self,
            sdf_reader: ChemblSdfReader,
            smina_dockerizer: SminaDockerizer,
            labeled_docking_result_handler: LabeledDockingResultHandler,
            logging_level: int = logging.INFO,
    ) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)
        self.sdf_reader = sdf_reader
        self.smina_dockerizer = smina_dockerizer
        self.labeled_docking_result_handler = labeled_docking_result_handler

    @staticmethod
    def create(
            smina_config: SminaConfig,
            docked_ligands_target_directory: str,
            logging_level: int = logging.INFO,
    ) -> SminaDockingPersistenceHandler:
        sdf_reader = ChemblSdfReader(logging_level)
        return SminaDockingPersistenceHandler(
            smina_dockerizer=SminaDockerizer(smina_config),
            labeled_docking_result_handler=LabeledDockingResultHandler(pathlib.Path(docked_ligands_target_directory)),
            sdf_reader=sdf_reader,
        )

    def dock(
        self,
        protein_pdb_file_path: pathlib.Path,
        ligands_sdf_file: pathlib.Path,
        batch_size: int = 25,
        starting_batch: int = 0,
    ) -> None:
        self.logger.info(f"Starting docking process with {batch_size=} and {starting_batch=}...")
        parsed_records = self.sdf_reader.parse(ligands_sdf_file)
        batch_generator = batched(parsed_records, batch_size)
        while starting_batch > 0:
            self.logger.info(f"Skipping batch. {starting_batch} skipped batches left...")
            next(batch_generator)
            starting_batch -= 1

        for i, batch in enumerate(batch_generator, start=starting_batch):
            try:
                self.logger.info(f"Batch {i} started...")
                ligand_mols = [rdkit.Chem.MolFromMolBlock(record['mol']) for record in batch]
                docked_mols = self.smina_dockerizer.dock(protein_pdb_file_path=protein_pdb_file_path, ligands=ligand_mols)
                standard_values = [float(record["standardValue"]) for record in batch]
                assert len(docked_mols) == len(standard_values)
                self.labeled_docking_result_handler.save_many(
                    LabeledDockingResult(mol, value) for mol, value in zip(docked_mols, standard_values)
                )
                self.logger.info(f"Batch {i} finished.")
            except ValueError as e:
                self.logger.error(f"Batch {i} encountered error: {e}")
                continue


class SminaDockingToProlifFingerprintFeaturizer(LigandDockingFingerprintFeaturizer):
    def __init__(
            self,
            sdf_reader: ChemblSdfReader,
            prolif_fingerprint_generator: ProlifInteractionFingerprintGenerator,
            smina_dockerizer: SminaDockerizer,
            logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(
            logging_level=logging_level,
            sdf_reader=sdf_reader,
            dockerizer=smina_dockerizer,
            fingerprint_generator=prolif_fingerprint_generator,
        )

    @staticmethod
    def create(
            smina_config: SminaConfig,
            logging_level: int = logging.INFO,
    ) -> SminaDockingToProlifFingerprintFeaturizer:
        sdf_reader = ChemblSdfReader(logging_level)
        prolif_fingerprint_generator = ProlifInteractionFingerprintGenerator(logging_level)
        smina_dockerizer = SminaDockerizer(smina_config)
        return SminaDockingToProlifFingerprintFeaturizer(
            sdf_reader=sdf_reader,
            prolif_fingerprint_generator=prolif_fingerprint_generator,
            smina_dockerizer=smina_dockerizer,
            logging_level=logging_level,
        )


class DockedProlifFingerprintFeaturizer(DockedLigandFingerprintFeaturizer):
    def __init__(
            self,
            prolif_fingerprint_generator: ProlifInteractionFingerprintGenerator,
            logging_level: int = logging.INFO,
    ) -> None:
        super().__init__(
            logging_level=logging_level,
            fingerprint_generator=prolif_fingerprint_generator,
        )

    @staticmethod
    def create(logging_level: int = logging.INFO) -> DockedProlifFingerprintFeaturizer:
        prolif_fingerprint_generator = ProlifInteractionFingerprintGenerator(logging_level)
        return DockedProlifFingerprintFeaturizer(
            prolif_fingerprint_generator=prolif_fingerprint_generator,
            logging_level=logging_level,
        )
