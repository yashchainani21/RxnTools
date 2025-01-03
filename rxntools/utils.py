from rdkit import Chem

def are_isomorphic(mol1: Chem.rdchem.Mol,
                   mol2: Chem.rdchem.Mol,
                   consider_stereo: bool = False) -> bool:
    """
    Check if two molecules are isomorphic, i.e. their bond and atom arrangements are identical

    Parameters
    ----------
    mol1 : Chem.rdchem.Mol
        RDKit mol object of first molecule generated from its SMILES or SMARTS representation.
        If SMARTS were used in creating mol object, these need not be atom mapped.

    mol2 : Chem.rdchem.Mol
        RDKit mol object of second molecule generated from its SMILES or SMARTS representation.
        If SMARTS were used in creating mol object, these need not be atom mapped.

    consider_stereo: bool
        Whether to consider stereochemistry or not when comparing two molecules.

    Returns
    -------
    is_isomorphic : bool
        True if both molecules are isomorphic, False otherwise.
    """

    if consider_stereo:
        is_isomorphic = mol1.HasSubstructMatch(mol2, useChirality = True) and mol2.HasSubstructMatch(mol1, useChirality = True)
    else:
        is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)

    return is_isomorphic

def is_cofactor(mol: Chem.rdchem.Mol,
                cofactors_list: list,
                consider_stereo: bool = False) -> bool:
    """
    Checks if a given query molecule is equivalent to a cofactor.
    A list of predetermined cofactors needs to be passed into this function.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit mol object of query molecule generated from its SMILES or SMARTS representation.
        If SMARTS were used in creating mol object, these need not be atom mapped.
    cofactors_list : list
        List of SMILES strings representing cofactors.
    consider_stereo: bool
        Whether to consider stereochemistry or not when comparing two molecules.

    Returns
    -------
    bool
        True if the input query mol is isomorphic to any cofactor in the list, False otherwise.
    """
    for cofactor in cofactors_list:
        if are_isomorphic(mol1 = mol,
                          mol2 = Chem.MolFromSmiles(cofactor),
                          consider_stereo = consider_stereo):
            return True

    return False

def remove_stereo(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """
    Remove stereochemistry from a molecule.

    Parameters
    ----------
    mol : Chem.rdchem.Mol
        RDKit mol object of query molecule generated from its SMILES or SMARTS representation with stereochemistry.

    Returns
    -------
    mol : Chem.rdchem.Mol
        RDKit mol object of input molecule without stereochemistry.
    """
    Chem.RemoveStereochemistry(mol)
    return mol




