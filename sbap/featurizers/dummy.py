import numpy as np
import rdkit
from numpy import typing as npt

from sbap.featurizers.base import BaseFeaturizer
from sbap._types import ReceptorInteractionCombination
from sbap.sdf import ChemblSdfRecord


class DummyFeaturizer(BaseFeaturizer):
    def fit(self, protein: rdkit.Chem.Mol, ligands: list[ChemblSdfRecord]) -> None:
        self.logger.debug(f"{self.__class__.__name__} does not need fitting")

    def transform(
            self,
            protein: rdkit.Chem.Mol,
            ligands: list[ChemblSdfRecord],
            allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination] = None,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        if allowed_receptor_interaction_combinations is None:
            allowed_receptor_interaction_combinations = {
                ReceptorInteractionCombination(("ABC", "Interaction1")),
                ReceptorInteractionCombination(("ABC", "Interaction2"))
            }
        return (
            np.random.random_integers(0, 1, size=(len(ligands), len(allowed_receptor_interaction_combinations))),
            np.asarray([record['standardValue'] for record in ligands]),
        )
