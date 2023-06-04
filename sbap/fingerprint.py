import logging
from abc import ABC, abstractmethod
from itertools import product
from typing import Optional

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

    @abstractmethod
    def get_receptor_interaction_combinations(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
    ) -> set[ReceptorInteractionCombination]:
        pass


class ProlifInteractionFingerprintGenerator(InteractionFingerprintGenerator):
    def generate(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
            allowed_receptor_interaction_combinations: Optional[set[ReceptorInteractionCombination]] = None,
    ) -> list[InteractionFingerprint]:
        fingerprint = self._run_prolif(protein, docked_ligands)
        sanitized_fingerprints = self._sanitize_fingerprint(fingerprint, allowed_receptor_interaction_combinations)
        assert len(docked_ligands) == len(sanitized_fingerprints)
        return sanitized_fingerprints

    def get_receptor_interaction_combinations(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
    ) -> set[ReceptorInteractionCombination]:
        fingerprints = self._run_prolif(protein, docked_ligands)
        fp_df = fingerprints.to_dataframe()
        receptor_interactions = {
            ReceptorInteractionCombination((receptor, interaction_type))
            for (receptor, interaction_type) in (fingerprint[1:] for fingerprint in fp_df.columns.values)
        }
        receptor_any_interaction = {(receptor, "Any") for (receptor, _) in receptor_interactions}
        return receptor_interactions | receptor_any_interaction

    def _run_prolif(
            self,
            protein: rdkit.Chem.Mol,
            docked_ligands: list[rdkit.Chem.Mol],
    ) -> plf.Fingerprint:
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
        return fp

    def _sanitize_fingerprint(
            self,
            fingerprint: plf.Fingerprint,
            allowed_receptor_interaction_combinations: Optional[set[ReceptorInteractionCombination]] = None,
    ) -> list[InteractionFingerprint]:
        fp_df = fingerprint.to_dataframe()
        fp_df.columns = [col for col in [vs[1:] for vs in fp_df.columns.values]]
        receptor_interaction_types: set[ReceptorInteractionCombination] = {
            ReceptorInteractionCombination(receptor_interaction)
            for receptor_interaction in fp_df.columns
        }
        if allowed_receptor_interaction_combinations is not None:
            receptor_interaction_types = self._check_interaction_combinations(
                receptor_interaction_types,
                allowed_receptor_interaction_combinations,
            )
        interacted_receptors = {receptor for (receptor, interaction) in receptor_interaction_types}
        return [
            InteractionFingerprint([
                                       int(record.get(specific_interaction, False))
                                       for specific_interaction in allowed_receptor_interaction_combinations
                                   ] + [
                                       int(any(
                                           value and receptor == generalized_receptor for (receptor, interaction), value
                                           in record.items()
                                       ))
                                       for generalized_receptor in interacted_receptors
                                   ])
            for record in fp_df.to_dict('records')
        ]

    def _check_interaction_combinations(
            self,
            receptor_interaction_types: set[ReceptorInteractionCombination],
            allowed_receptor_interaction_combinations: set[ReceptorInteractionCombination],
    ) -> set[ReceptorInteractionCombination]:
        if not receptor_interaction_types.issubset(allowed_receptor_interaction_combinations):
            unknown_receptor_interactions = receptor_interaction_types.difference(
                allowed_receptor_interaction_combinations,
            )
            self.logger.warning(
                f"Unknown interactions found in dataset: "
                f"{', '.join(' '.join(pair) for pair in unknown_receptor_interactions)}"
            )
        return allowed_receptor_interaction_combinations


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
