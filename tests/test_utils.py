import pytest
from rdkit import Chem
from rxntools import utils
import json

# Define a pytest fixture for cofactors_list
@pytest.fixture
def cofactors_list():
    """
    Pytest fixture to load the cofactors list from the JSON file.
    """
    with open('../data/cofactors.json') as f:
        cofactors_dict = json.load(f)
    return [cofactors_dict[key] for key in cofactors_dict.keys()]

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3:1][CH2:2][OH:3]"),
                                mol2 = Chem.MolFromSmiles("CCO"))

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & non-canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3:1][CH2:2][OH:3]"),
                                mol2 = Chem.MolFromSmiles("OCC"))

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3][CH2][OH]"),
                                mol2 = Chem.MolFromSmiles("CCO"))

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & non-canonical SMILES.
    The consider_stereo parameter is not needed here since ethanol has no stereocenters.
    """
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts("[CH3][CH2][OH]"),
                                mol2 = Chem.MolFromSmiles("OCC"))

def test_is_isomorphic_nad_plus_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when NAD(+) is represented with atom-mapped SMARTS & canonical SMILES.
    """
    nad_plus_atom_mapped_smarts_with_stereo = "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"
    nad_plus_smiles = "[NH2]C(=O)c1[cH][cH][cH][n+]([C@@H]2O[C@H]([CH2]OP(=O)([OH])OP(=O)([OH])O[CH2][C@H]3O[C@@H](n4[cH]nc5c([NH2])n[cH]nc54)[C@H]([OH])[C@@H]3[OH])[C@@H]([OH])[C@H]2[OH])[cH]1"

    # check if both representations of NAD(+) are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(nad_plus_atom_mapped_smarts_with_stereo),
                                mol2 = Chem.MolFromSmiles(nad_plus_smiles),
                                consider_stereo = True)

def test_is_isomorphic_nadh_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when NADH is represented with atom-mapped SMARTS & canonical SMILES.
    """
    nadh_atom_mapped_smarts_with_stereo = "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"
    nadh_smiles = "[NH2]C(=O)C1=[CH]N([C@@H]2O[C@H]([CH2]OP(=O)([OH])OP(=O)([OH])O[CH2][C@H]3O[C@@H](n4[cH]nc5c([NH2])n[cH]nc54)[C@H]([OH])[C@@H]3[OH])[C@@H]([OH])[C@H]2[OH])[CH]=[CH][CH2]1"

    # check if both representations of NADH are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(nadh_atom_mapped_smarts_with_stereo),
                                mol2 = Chem.MolFromSmiles(nadh_smiles),
                                consider_stereo = True)

def test_is_isomorphic_CO2_with_atom_mapped_SMARTS_and_canonical_SMILES():
    CO2_atom_mapped_smarts = "[C:23](=[O:24])=[O:27]"
    CO2_smiles = "O=C=O"

    # check if both representations of NADH are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(CO2_atom_mapped_smarts),
                                mol2 = Chem.MolFromSmiles(CO2_smiles))

def test_is_cofactor_CO2_with_canonical_SMILES(cofactors_list):

    assert utils.is_cofactor(mol = Chem.MolFromSmiles("O=C=O"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_CO2_with_atom_mapped_SMARTS(cofactors_list):

    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[C:23](=[O:24])=[O:27]"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_nad_plus_with_atom_mapped_SMARTS(cofactors_list):

    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)

def test_is_cofactor_nadh_with_atom_mapped_SMARTS(cofactors_list):

    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)