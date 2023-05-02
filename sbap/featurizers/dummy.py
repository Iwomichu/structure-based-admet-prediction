import pathlib

import numpy as np
from numpy import typing as npt

from sbap.featurizers.base import BaseFeaturizer


class DummyFeaturizer(BaseFeaturizer):
    def fit(self, protein_pdb_file_path: pathlib.Path, ligands_sdf_file: pathlib.Path) -> None:
        self.logger.debug(f"{self.__class__.__name__} does not need fitting")

    def transform(
            self,
            protein_pdb_file_path: pathlib.Path,
            ligands_sdf_file: pathlib.Path,
    ) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        return (
            np.random.random_integers(0, 1, size=(30, 10)),
            np.random.random(30),
        )
