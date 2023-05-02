import logging
from abc import ABC, abstractmethod

import numpy.typing as npt
import rdkit.Chem

from sbap._types import ReceptorInteractionCombination
from sbap.sdf import ChemblSdfRecord


class BaseFeaturizer(ABC):
    def __init__(self, logging_level: int = logging.INFO) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    @abstractmethod
    def fit(self, protein: rdkit.Chem.Mol, ligands: list[ChemblSdfRecord]) -> None:
        pass

    @abstractmethod
    def transform(
            self,
            protein: rdkit.Chem.Mol,
            ligands: list[ChemblSdfRecord],
            allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = None,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        pass
