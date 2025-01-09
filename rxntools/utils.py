from rdkit import Chem
from rdkit.Chem import rdChemReactions

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

def remove_stereo_frm_rxn(reaction_SMARTS: str) -> str:
    """
    Remove stereochemistry from a reaction SMARTS.

    Parameters
    ----------
    reaction_SMARTS : str
        Fully atom-mapped SMARTS string for a given input reaction with potentially chiral species.

    Returns
    -------
    no_stereo_reaction_SMARTS : str
        Fully atom-mapped SMARTS string but stereochemistry information is remove for any participating chiral species.
    """
    reaction = rdChemReactions.ReactionFromSmarts(reaction_SMARTS)
    if reaction is None:
        raise ValueError("Invalid reaction SMARTS string!")

    # process reactants, products, and agents
    reactants = [remove_stereo(mol) for mol in reaction.GetReactants()]
    products = [remove_stereo(mol) for mol in reaction.GetProducts()]
    agents = [remove_stereo(mol) for mol in reaction.GetAgents()]

    # reconstruct the reaction SMARTS
    reactant_smarts = '.'.join([Chem.MolToSmiles(mol) for mol in reactants])
    product_smarts = '.'.join([Chem.MolToSmiles(mol) for mol in products])
    agent_smarts = '.'.join([Chem.MolToSmiles(mol) for mol in agents])

    # Combine into a single reaction SMARTS string
    if agent_smarts:
        no_stereo_reaction_SMARTS = f"{reactant_smarts}.{agent_smarts}>>{product_smarts}"
    else:
        no_stereo_reaction_SMARTS = f"{reactant_smarts}>>{product_smarts}"

    return no_stereo_reaction_SMARTS

def reset_atom_map(SMARTS: str,
                   starting_atom_num: int = 0) -> str:
    """
    Reset the numbering of an extracted reaction template.
    Resetting atom maps of extracted templates can help duplicate duplicates.

    Parameters
    ----------
    SMARTS: str
        Fully-atom mapped SMARTS string of an input molecule, complete with stereochemistry

    starting_atom_num: int
        Starting atom number for the new atom mapping
    """

    # convert the input molecule's SMARTS to an RDKit mol object
    mol = Chem.MolFromSmarts(SMARTS)
    if not mol:
        raise ValueError("Invalid SMARTS string!")

    # map the existing atom indices to new atom map numbers starting from 1
    if starting_atom_num == 0:
        new_atom_map = {atom.GetIdx(): starting_atom_num + i + 1 for i, atom in enumerate(mol.GetAtoms())}
    else:
        new_atom_map = {atom.GetIdx(): starting_atom_num + i for i, atom in enumerate(mol.GetAtoms())}

    # update the existing atom map numbers in the current molecule
    for atom in mol.GetAtoms():
        if atom.HasProp('molAtomMapNumber'):
            atom.SetProp('molAtomMapNumber', str(new_atom_map[atom.GetIdx()]))

    # convert the RDKit mol object back to SMARTS once new atom map numbers have been set
    SMARTS_with_new_atom_mapping = Chem.MolToSmarts(mol)
    return SMARTS_with_new_atom_mapping


