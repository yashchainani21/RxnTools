from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Set,Tuple

class unmapped_reaction:
    """
    Parameters
    ----------
    rxn_str : str
        Unmapped reaction string of the form "A + B = C + D" or "A.B >> C.D"
    """
    def __init__(self, rxn_str: str):
        self.rxn_str = rxn_str

class mapped_reaction:
    """
    Parameters
    ----------
    rxn_smarts : str
        Mapped reaction smarts with complete atom mapping of all species in reaction
    """
    def __init__(self, rxn_smarts: str):
        self.rxn_smarts = rxn_smarts

    @staticmethod
    def _get_mapped_bonds(mol: Chem.rdchem.Mol) -> Set[Tuple[int, int, Chem.rdchem.BondType]]:
        """
        Parameters
        ----------
        mol : Chem.rdchem.Mol
            RDKit mol object of a single molecule represented with fully atom-mapped SMARTS.

        Returns
        -------
        bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            A set of mapped bonds for the input molecule.
            Each element in this set is a tuple that corresponds to a mapped bond.
            This tuple itself comprises three elements.
            The first two elements are integers indicating the start & end atom indices of a bond.
            Third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.
        """
        # initialize an empty set to store all bonds
        bonds = set()
        for bond in mol.GetBonds():
            starting_atom = bond.GetBeginAtom()
            ending_atom = bond.GetEndAtom()

            # only consider atoms with mapping when storing mapped bonds
            if starting_atom.GetAtomMapNum() > 0 and ending_atom.GetAtomMapNum() > 0:
                bond_tuple = (starting_atom.GetAtomMapNum(),
                              ending_atom.GetAtomMapNum(),
                              bond.GetBondType())
                bonds.add(bond_tuple)

        return bonds

    def _get_all_changed_atoms(self) -> Tuple[Set[int], Set[Tuple[int, int]], Set[Tuple[int, int]]]:

        # create an RDKit reaction object from a fully atom-mapped reaction smarts string
        rxn = AllChem.ReactionFromSmarts(self.rxn_smarts)
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()

        reactant_bonds = set()
        product_bonds = set()

        # collect all mapped bonds from all reactants involved
        for reactant_mol in reactants:
            reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

        # collect all mapped bonds from all products involved
        for product_mol in products:
            product_bonds |= self._get_mapped_bonds(mol = product_mol)

        # identify all broken bonds and formed bonds
        broken_bonds = reactant_bonds - product_bonds # bonds present in reactants but not products
        formed_bonds = product_bonds - reactant_bonds # bonds present in products but not reactants

        # collect atom mapped numbers involved in broken or formed bonds
        changed_atoms = set()
        for bond in broken_bonds.union(formed_bonds):
            changed_atoms.add(bond[0]) # atom map number of the first atom
            changed_atoms.add(bond[1]) # atom map number of the second atom

        return changed_atoms, broken_bonds, formed_bonds
