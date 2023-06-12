import pathlib
import tempfile
from abc import abstractmethod
from typing import Protocol, Optional

import numpy.typing as npt
from sklearn.model_selection import train_test_split

from sbap.docking import SminaConfig
from sbap.featurizers.base import DockedInputBaseFeaturizer
from sbap.featurizers.prolif_smina import SminaDockingPersistenceHandler


class Regressor(Protocol):
    @abstractmethod
    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> None:
        pass

    @abstractmethod
    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        pass

    @abstractmethod
    def score(self, X: npt.ArrayLike, y: npt.ArrayLike) -> float:
        pass


class Pipeline:
    def __init__(
            self,
            config: SminaConfig,
            featurizer: DockedInputBaseFeaturizer,
            regressor: Regressor,
    ) -> None:
        self.config = config
        self.featurizer = featurizer
        self.regressor = regressor
        self.pdb_file_content: Optional[str] = None

    def fit(self, protein_pdb_file_path: pathlib.Path, sdf_file: pathlib.Path) -> float:
        with open(protein_pdb_file_path, 'r') as f:
            self.pdb_file_content = f.read()

        with tempfile.TemporaryDirectory() as temporary_directory_path:
            handler = SminaDockingPersistenceHandler.create(self.config, temporary_directory_path)
            temporary_directory = pathlib.Path(temporary_directory_path)
            handler.dock(protein_pdb_file_path, sdf_file, batch_size=5, starting_batch=0)
            self.featurizer.fit(protein_pdb_file_path, temporary_directory)
            x, y = self.featurizer.transform(protein_pdb_file_path, temporary_directory)
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
            self.regressor.fit(x_train, y_train)
            return self.regressor.score(x_test, y_test)

    def predict(self, sdf_file: pathlib.Path) -> npt.ArrayLike:
        with tempfile.TemporaryDirectory() as temporary_directory_path:
            handler = SminaDockingPersistenceHandler.create(self.config, temporary_directory_path)
            temporary_directory = pathlib.Path(temporary_directory_path)
            protein_pdb_file_path = temporary_directory / "protein.pdb"
            with open(protein_pdb_file_path, "w") as f:
                f.write(self.pdb_file_content)
            handler.dock(protein_pdb_file_path, sdf_file, batch_size=5, starting_batch=0)
            x, _ = self.featurizer.transform(protein_pdb_file_path,
                                             temporary_directory)  # TODO: featurizer should be able to handle unlabeled data
            return self.regressor.predict(x)
