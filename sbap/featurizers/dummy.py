import numpy as np
from numpy import typing as npt

from sbap.featurizers.base import BaseFeaturizer
from sbap.sdf import ChemblSdfRecord


class DummyFeaturizer(BaseFeaturizer):
    def fit(self, records: list[ChemblSdfRecord]) -> None:
        self.logger.debug(f"{self.__class__.__name__} does not need fitting")

    def transform(self, records: list[ChemblSdfRecord]) -> tuple[npt.ArrayLike, npt.ArrayLike]:
        return (
            np.random.random_integers(0, 1, size=(len(records), 128)),
            np.asarray([record['standardValue'] for record in records]),
        )
