import pathlib
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass

import rdkit.Chem
import rdkit.Chem.AllChem


@dataclass
class DockingConfig:
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
            protein: rdkit.Chem.Mol,
            ligands: list[rdkit.Chem.Mol],
            docking_config: DockingConfig,
    ) -> list[rdkit.Chem.Mol]:
        pass


class SminaDockerizer(Dockerizer):
    def dock(
            self,
            protein: rdkit.Chem.Mol,
            ligands: list[rdkit.Chem.Mol],
            docking_config: DockingConfig,
    ) -> list[rdkit.Chem.Mol]:
        with tempfile.TemporaryDirectory() as directory_str:
            directory = pathlib.Path(directory_str)
            protein_path = directory.joinpath("protein.pdb")
            rdkit.Chem.MolToPDBFile(protein, str(protein_path))
            return [self._dock_single_ligand(ligand, directory, docking_config, protein_path) for ligand in ligands]

    def _dock_single_ligand(
            self,
            ligand: rdkit.Chem.Mol,
            directory: pathlib.Path,
            docking_config: DockingConfig,
            protein_path: pathlib.Path,
    ) -> rdkit.Chem.Mol:
        ligand_path = directory.joinpath("ligand.mol")
        mol2_ligand_path = directory.joinpath("ligand.mol2")
        docked_molecule_path = directory.joinpath("docked.mol2")
        ligand_optimized = self._optimize_conformation(ligand)
        rdkit.Chem.MolToMolFile(ligand_optimized, str(ligand_path))
        self._run_obabel(ligand_path, mol2_ligand_path, directory)
        self._run_smina(protein_path, mol2_ligand_path, docking_config, docked_molecule_path, directory)
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
            docking_config: DockingConfig,
            docked_molecule_path: pathlib.Path,
            directory: pathlib.Path,
    ) -> None:
        subprocess.run(
            " ".join([
                "smina",
                f"-r {str(protein_path)}",
                f"-l {str(mol2_ligand_path)}",
                f"--center_x {docking_config.center_x}",
                f"--center_y {docking_config.center_y}",
                f"--center_z {docking_config.center_z}",
                f"--size_x {docking_config.size_x}",
                f"--size_y {docking_config.size_y}",
                f"--size_z {docking_config.size_z}",
                f"--exhaustiveness {docking_config.exhaustiveness}",
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
