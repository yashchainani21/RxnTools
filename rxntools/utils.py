from rdkit import Chem
from rdkit.Chem import rdFMCS

def are_isomorphic(s1: str, s2) -> bool:
    """
    Check if two molecules are isomorphic, i.e. their bond and atom arrangements are identical

    Parameters
    ----------
    s1 : str
        Either a SMILES string or SMARTS representation of the first molecule.
        If SMARTS are provided, these may or may not be atom mapped.

    s2 : str
        Either a SMILES string or SMARTS representation of the second molecule.
        If SMARTS are provided, these may or may not be atom mapped.

    Returns
    -------
    is_isomorphic : bool
        True if both molecules are isomorphic, False otherwise.
    """

    # the function Chem.MolFromSmiles() can work on both SMARTS and SMILES
    mol1 = Chem.MolFromSmiles(s1)
    mol2 = Chem.MolFromSmiles(s2)

    is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

    return is_isomorphic