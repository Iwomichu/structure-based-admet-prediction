import logging
import pathlib
from abc import ABC, abstractmethod

import numpy.typing as npt


class BaseFeaturizer(ABC):
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
