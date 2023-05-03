import pathlib
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import rdkit.Chem
import rdkit.Chem.AllChem


@dataclass
class SminaConfig:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    exhaustiveness: int


class Dockerizer(ABC):
    @abstractmethod
    def dock(
            self,
            ligands: list[rdkit.Chem.Mol],
            protein: Optional[rdkit.Chem.Mol] = None,
            protein_pdb_file_path: Optional[pathlib.Path] = None,
    ) -> list[rdkit.Chem.Mol]:
        pass


class SminaDockerizer(Dockerizer):
    def __init__(self, smina_config: SminaConfig) -> None:
        self.config = smina_config

    def dock(
            self,
            ligands: list[rdkit.Chem.Mol],
            protein: Optional[rdkit.Chem.Mol] = None,
            protein_pdb_file_path: Optional[pathlib.Path] = None,
    ) -> list[rdkit.Chem.Mol]:
        if protein is None and protein_pdb_file_path is None:
            raise RuntimeError("Either protein or protein_pdb_file_path should be provided")
        with tempfile.TemporaryDirectory() as directory_str:
            directory = pathlib.Path(directory_str)
            if protein_pdb_file_path is None:
                protein_pdb_file_path = directory.joinpath("protein.pdb")
                rdkit.Chem.MolToPDBFile(protein, str(protein_pdb_file_path))
            output = [
                self._dock_single_ligand(ligand, directory, protein_pdb_file_path)
                for ligand in ligands
            ]
            assert len(output) == len(ligands)
            return output

    def _dock_single_ligand(
            self,
            ligand: rdkit.Chem.Mol,
            directory: pathlib.Path,
            protein_path: pathlib.Path,
    ) -> rdkit.Chem.Mol:
        ligand_path = directory.joinpath("ligand.mol")
        mol2_ligand_path = directory.joinpath("ligand.mol2")
        docked_molecule_path = directory.joinpath("docked.mol2")
        ligand_optimized = self._optimize_conformation(ligand)
        rdkit.Chem.MolToMolFile(ligand_optimized, str(ligand_path))
        self._run_obabel(ligand_path, mol2_ligand_path, directory)
        self._run_smina(protein_path, mol2_ligand_path, docked_molecule_path, directory)
        return rdkit.Chem.MolFromMol2File(str(docked_molecule_path), sanitize=False)

    def _run_obabel(self, ligand_path: pathlib.Path, mol2_ligand_path: pathlib.Path, directory: pathlib.Path) -> None:
        subprocess.run(
            f"obabel -imol {ligand_path.name} -omol2 -O {mol2_ligand_path.name}",
            check=True,
            cwd=str(directory),
            shell=True,
        )

    def _run_smina(
            self,
            protein_path: pathlib.Path,
            mol2_ligand_path: pathlib.Path,
            docked_molecule_path: pathlib.Path,
            directory: pathlib.Path,
    ) -> None:
        subprocess.run(
            " ".join([
                "smina",
                f"-r {str(protein_path.absolute())}",
                f"-l {str(mol2_ligand_path)}",
                f"--center_x {self.config.center_x}",
                f"--center_y {self.config.center_y}",
                f"--center_z {self.config.center_z}",
                f"--size_x {self.config.size_x}",
                f"--size_y {self.config.size_y}",
                f"--size_z {self.config.size_z}",
                f"--exhaustiveness {self.config.exhaustiveness}",
                f"--out {str(docked_molecule_path)}",
                "--quiet",
            ]),
            shell=True,
            cwd=str(directory),
        )

    def _optimize_conformation(self, mol: rdkit.Chem.Mol):
        mol = rdkit.Chem.AddHs(mol)  # Adds hydrogens to make optimization more accurate
        rdkit.Chem.AllChem.EmbedMolecule(mol)  # Adds 3D positions
        rdkit.Chem.AllChem.MMFFOptimizeMolecule(mol)  # Improves the 3D positions using a force-field method
        return mol
