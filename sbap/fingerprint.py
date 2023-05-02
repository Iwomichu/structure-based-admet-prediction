import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Optional

import pandas as pd
import prolif as plf

import rdkit.Chem
from MDAnalysis.topology.tables import vdwradii
from prolif.interactions import Interaction

from sbap._types import ReceptorInteractionCombination, InteractionFingerprint


class InteractionFingerprintGenerator(ABC):
    def __init__(self, logging_level: int = logging.INFO):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging_level)

    @abstractmethod
    def generate(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
            allowed_receptor_interaction_combinations: Optional[set[ReceptorInteractionCombination]] = None,
    ) -> list[InteractionFingerprint]:
        pass


class ProlifInteractionFingerprintGenerator(InteractionFingerprintGenerator):
    def generate(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
            allowed_receptor_interaction_combinations: Optional[set[ReceptorInteractionCombination]] = None,
    ) -> list[InteractionFingerprint]:
        # @mjuralowicz: type ignored due to typing error caused by invalid prolif type annotations
        plf_protein = plf.Molecule.from_rdkit(protein)  # type: ignore
        plf_ligands = [plf.Molecule.from_rdkit(ligand) for ligand in docked_ligands]  # type: ignore
        fp = plf.Fingerprint(interactions=[
            "Hydrophobic",
            "HBDonor",
            "HBAcceptor",
            "PiStacking",
            "Anionic",
            "Cationic",
            "CationPi",
            "PiCation",
            "CustomVdWContact",
        ])
        fp.run_from_iterable(plf_ligands, plf_protein)
        fp_df = fp.to_dataframe()
        fp_df.columns = [' '.join(col).strip() for col in [vs[1:] for vs in fp_df.columns.values]]
        if allowed_receptor_interaction_combinations is not None:
            if not set(fp_df.columns).issubset(allowed_receptor_interaction_combinations):
                unknown_receptor_interactions = set(fp_df.columns).difference(allowed_receptor_interaction_combinations)
                self.logger.warning(
                    f"Unknown interactions found in dataset: {', '.join(unknown_receptor_interactions)}"
                )
            self.logger.info(f"Receptor interactions found: {fp_df.columns}")
            base_df = pd.DataFrame(columns=sorted(list(allowed_receptor_interaction_combinations)))
            fp_df = pd.concat([base_df, fp_df])
        fingerprints = fp_df.to_dict('records')
        return [
            InteractionFingerprint(
                [int(receptor_interaction) for receptor_interaction in fingerprint.values()]
            ) for fingerprint in fingerprints
        ]


class CustomVdWContact(Interaction):
    """Patched version of plf.VdwContact"""

    def __init__(self, tolerance=0.0):
        if tolerance >= 0:
            self.tolerance = tolerance
        else:
            raise ValueError("`tolerance` must be 0 or positive")
        self._vdw_cache = {}

    def detect(self, ligand, residue):
        lxyz = ligand.GetConformer()
        rxyz = residue.GetConformer()
        for la, ra in product(ligand.GetAtoms(), residue.GetAtoms()):
            lig = la.GetSymbol().upper()
            res = ra.GetSymbol().upper()
            try:
                vdw = self._vdw_cache[frozenset((lig, res))]
            except KeyError:
                try:
                    vdw = vdwradii[lig] + vdwradii[res] + self.tolerance
                except KeyError:
                    continue
                self._vdw_cache[frozenset((lig, res))] = vdw
            dist = lxyz.GetAtomPosition(la.GetIdx()).Distance(
                rxyz.GetAtomPosition(ra.GetIdx())
            )
            if dist <= vdw:
                return True, la.GetIdx(), ra.GetIdx()
        return False, None, None
