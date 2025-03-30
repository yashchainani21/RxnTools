import pytest
import pandas as pd
from rdkit import Chem
from rxntools import utils
import json

# Define a pytest fixture for cofactors_list
@pytest.fixture
def cofactors_list():
    """
    Pytest fixture to load the cofactors list from the JSON file.
    """
    with open('../data/raw/cofactors.json') as f:
        cofactors_dict = json.load(f)
    return [cofactors_dict[key] for key in cofactors_dict.keys()]

@pytest.fixture
def cofactors_df():
    """
    Pytest fixture to load the cofactors dataframe from a tsv file.
    """
    with open('../data/raw/all_cofactors.csv') as f:
        cofactors_df = pd.read_csv(f, sep=',')
    return cofactors_df

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
    """
    Ensure isomorphic when CO2 is represented with atom-mapped SMARTS & canonical SMILES.
    """
    CO2_atom_mapped_smarts = "[C:23](=[O:24])=[O:27]"
    CO2_smiles = "O=C=O"

    # check if both representations of NADH are equivalent and consider stereochemistry
    assert utils.are_isomorphic(mol1 = Chem.MolFromSmarts(CO2_atom_mapped_smarts),
                                mol2 = Chem.MolFromSmiles(CO2_smiles))

def test_is_cofactor_CO2_with_canonical_SMILES(cofactors_list):
    """
    Ensure CO2 is flagged as a cofactor when represented with its canonical SMILES string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmiles("O=C=O"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_CO2_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure CO2 is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[C:23](=[O:24])=[O:27]"),
                             cofactors_list = cofactors_list)

def test_is_cofactor_nad_plus_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure NAD(+) is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)

def test_is_cofactor_nadh_with_atom_mapped_SMARTS(cofactors_list):
    """
    Ensure NADH is flagged as a cofactor when represented with its atom-mapped SMARTS string.
    """
    assert utils.is_cofactor(mol = Chem.MolFromSmarts("[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"),
                             cofactors_list = cofactors_list,
                             consider_stereo = True)

def test_remove_stereochemistry_frm_nad_plus():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from NAD(+).
    NAD(+) is represented with its atom-mapped SMILES here.
    We first create a mol object using Chem.MolFromSmarts and then convert the mol object back to SMILES for comparison.
    """
    nad_plus_SMARTS_w_stereo = "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1"
    nad_plus_mol_w_stereo = Chem.MolFromSmarts(nad_plus_SMARTS_w_stereo)

    nad_plus_mol_wo_stereo = utils.remove_stereo(nad_plus_mol_w_stereo)
    nad_plus_SMARTS_wo_stereo = Chem.MolToSmiles(nad_plus_mol_wo_stereo)
    assert nad_plus_SMARTS_wo_stereo == "[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[cH:47]1"

def test_remove_stereochemistry_frm_nadh():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from NADH.
    """
    nadh_SMARTS_w_stereo = "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"
    nadh_mol_w_stereo = Chem.MolFromSmarts(nadh_SMARTS_w_stereo)

    nadh_mol_wo_stereo = utils.remove_stereo(nadh_mol_w_stereo)
    nadh_SMARTS_wo_stereo = Chem.MolToSmiles(nadh_mol_wo_stereo)
    assert nadh_SMARTS_wo_stereo == "[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1"

def test_remove_stereochemistry_frm_mandelonitrile():
    """
    Ensure that the utils.remove_stereo(mol) function can successfully remove stereochemistry from mandelonitrile.
     """
    mandelonitrile_SMARTS_w_stereo = "[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"
    mandelonitrile_mol_w_stereo = Chem.MolFromSmarts(mandelonitrile_SMARTS_w_stereo)

    mandelonitrile_mol_wo_stereo = utils.remove_stereo(mandelonitrile_mol_w_stereo)
    mandelonitrile_SMARTS_wo_stereo = Chem.MolToSmiles(mandelonitrile_mol_wo_stereo)
    assert mandelonitrile_SMARTS_wo_stereo == "[N:1]#[C:2][CH:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1"

def test_remove_stereochemistry_frm_ethanol_AdH_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for the alcohol dehydrogenation reaction of ethanol to form ethanal.
    """
    rxn_SMARTS_w_stereo = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([CH:12]2[O:13][CH:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][CH:26]3[O:27][CH:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[CH:39]([OH:40])[CH:41]3[OH:42])[CH:43]([OH:44])[CH:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'

def test_remove_stereochemistry_frm_mandelonitrile_nitrilase_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for converting mandelonitrile to benzyldehyde via a nitrilase.
    """
    rxn_SMARTS_w_stereo = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[N:1]#[C:2][CH:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'

def test_remove_stereochemistry_frm_decarboxylation_rxn():
    """
    Ensure that the function utils.remove_stereo_frm_rxn() can remove stereochemistry across all species.
    This particular test is for a decarboxylation reaction.
    """
    rxn_SMARTS_w_stereo = '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@H:20]3[CH:21]=[O:22].[OH:25][CH3:26]'
    rxn_SMARTS_wo_stereo = utils.remove_stereo_frm_rxn(rxn_SMARTS_w_stereo)
    assert rxn_SMARTS_wo_stereo == '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[CH:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[CH:17]2[CH2:18][CH:19]1[C:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[CH:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[CH:17]2[CH2:18][CH:19]1[CH:20]3[CH:21]=[O:22].[OH:25][CH3:26]'

def test_creating_new_atommap_for_template_01_start_at_1():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C:7][C&H2:9][C&H2:10]'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 1)
    assert final_template == '[C:1][C&H2:2][C&H2:3]'

def test_creating_new_atommap_for_template_01_start_at_7():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C:7][C&H2:9][C&H2:10]'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 7)
    assert final_template == '[C:7][C&H2:8][C&H2:9]'

def test_creating_new_atommap_for_template_02_start_at_1():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10][C@@&H1:4]'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 1)
    assert final_template == '[C&H1:1]=[C:2]([C&H3:3])[C&H2:4][C&H2:5][C@@&H1:6]'

def test_creating_new_atommap_for_template_02_start_at_11():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10][C@@&H1:4]'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 11)
    assert final_template == '[C&H1:11]=[C:12]([C&H3:13])[C&H2:14][C&H2:15][C@@&H1:16]'

def test_creating_new_atommap_for_template_03_start_at_1():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C:2][C@&H1:4]1[C&H2:5][C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10]1'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 1)
    assert final_template == '[C:1][C@&H1:2]1[C&H2:3][C&H1:4]=[C:5]([C&H3:6])[C&H2:7][C&H2:8]1'

def test_creating_new_atommap_for_template_03_start_at_2():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C:2][C@&H1:4]1[C&H2:5][C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10]1'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 2)
    assert final_template == '[C:2][C@&H1:3]1[C&H2:4][C&H1:5]=[C:6]([C&H3:7])[C&H2:8][C&H2:9]1'

def test_creating_new_atommap_for_template_04_start_at_1():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C&H2:1]=[C:2]([C&H3:3])[C@&H1:4]1[C&H2:5][C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10]1'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 1)
    assert final_template == '[C&H2:1]=[C:2]([C&H3:3])[C@&H1:4]1[C&H2:5][C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10]1'

def test_creating_new_atommap_for_template_04_start_at_50():
    """
    Reset the numbering of atoms within an extracted template and being tracking atoms from a specified starting number.
    """
    extracted_template = '[C&H2:1]=[C:2]([C&H3:3])[C@&H1:4]1[C&H2:5][C&H1:6]=[C:7]([C&H3:8])[C&H2:9][C&H2:10]1'
    final_template = utils.reset_atom_map(extracted_template, starting_atom_num = 50)
    assert final_template == '[C&H2:50]=[C:51]([C&H3:52])[C@&H1:53]1[C&H2:54][C&H1:55]=[C:56]([C&H3:57])[C&H2:58][C&H2:59]1'

def test_get_pyrophosphate_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                                cofactors_df = cofactors_df) == "PYROPHOSPHATE_DONOR_CoF"

def test_get_pyrophosphate_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
                                       cofactors_df = cofactors_df) == "PYROPHOSPHATE_ACCEPTOR_CoF"

def test_get_FAD_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)c2cc1C",
                                       cofactors_df = cofactors_df) == "FAD_CoF"

def test_get_FADH2_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Cc1cc2c(cc1C)N(C[C@H](O)[C@H](O)[C@H](O)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n3cnc4c(N)ncnc43)[C@H](O)[C@@H]1O)c1[nH]c(=O)[nH]c(=O)c1N2",
                                       cofactors_df = cofactors_df) == "FADH2_CoF"

def test_get_phosphate_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                                       cofactors_df = cofactors_df) == "PHOSPHATE_ACCEPTOR_CoF"

def test_get_NAD_CoF_code_frm_NAD(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES="NC(c1c[n+]([C@@H]2O[C@@H]([C@H]([C@H]2O)O)COP(O)(OP(O)(OC[C@H]3O[C@H]([C@@H]([C@@H]3O)O)n(cn4)c5c4c(N)ncn5)=O)=O)ccc1)=O",
                                       cofactors_df=cofactors_df) == "NAD_CoF"

def test_get_NADH_CoF_code_frm_NADH(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "NC(C1=CN(C=CC1)[C@@H]2O[C@@H]([C@H]([C@H]2O)O)COP(O)(OP(O)(OC[C@H]3O[C@H]([C@@H]([C@@H]3O)O)n(cn4)c5c4c(N)ncn5)=O)=O)=O",
                                        cofactors_df=cofactors_df) == "NADH_CoF"

def test_get_sulfate_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OS(=O)(=O)O)[C@@H](OP(=O)(O)O)[C@H]1O",
                                       cofactors_df = cofactors_df) == "SULFATE_DONOR_CoF"

def test_get_sulfate_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](OP(=O)(O)O)[C@H]1O",
                                       cofactors_df = cofactors_df) == "SULFATE_ACCEPTOR_CoF"

def test_get_methyl_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "C[S+](CC[C@H](N)C(=O)O)C[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1O",
                                       cofactors_df = cofactors_df) == "METHYL_DONOR_CoF"

def test_get_methyl_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CSCC[C@H](N)C(=O)O)[C@@H](O)[C@H]1O",
                                       cofactors_df = cofactors_df) == "METHYL_ACCEPTOR_CoF"

def test_get_glucosyl_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
                                       cofactors_df = cofactors_df) == "GLUCOSYL_DONOR_CoF"

def test_get_glucosyl_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1",
                                       cofactors_df = cofactors_df)

def test_get_ubiquinols_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*c1c(*)c(O)c(*)c(*)c1O",
                                       cofactors_df = cofactors_df)

def test_get_ubiquinones_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*C1=C(*)C(=O)C(*)=C(*)C1=O",
                                       cofactors_df = cofactors_df)

def test_get_prenyl_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "CC(C)=CCOP(=O)(O)OP(=O)(O)O",
                                       cofactors_df = cofactors_df) == "PRENYL_DONOR_CoF"

def test_get_prenyl_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    # comes out as PPI even though this is a prenyl-acceptor, need to correct later
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=P(O)(O)OP(=O)(O)O",
                                       cofactors_df = cofactors_df) == "PPI"

def test_get_carbonyl_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=C(O)CCC(=O)C(=O)O",
                                       cofactors_df = cofactors_df) == "CARBONYL_CoF"

def test_get_amino_CoF(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "N[C@@H](CCC(=O)O)C(=O)O",
                                       cofactors_df = cofactors_df) == "AMINO_CoF"

def test_get_formyl_donor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*C(=O)CC[C@H](NC(=O)c1ccc(N(C=O)C[C@H]2CNc3nc(N)[nH]c(=O)c3N2)cc1)C(=O)O",
                                       cofactors_df = cofactors_df) == "FORMYL_DONOR_CoF"

def test_get_formly_acceptor_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*C(=O)CC[C@H](NC(=O)c1ccc(NC[C@H]2CNc3nc(N)[nH]c(=O)c3N2)cc1)C(=O)O",
                                       cofactors_df = cofactors_df) == "FORMYL_ACCEPTOR_CoF"

def test_get_ascorbate_radical_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "[O]C1=C(O)C(=O)O[C@@H]1[C@@H](O)CO",
                                       cofactors_df = cofactors_df) == "ASCORBATE_RADICAL_CoF"

def test_get_ascorbate_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=C1O[C@H]([C@@H](O)CO)C(O)=C1O",
                                       cofactors_df = cofactors_df) == "ASCORBATE_CoF"

def test_get_oxidized_factor_F420_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*C(=O)[C@H](C)OP(=O)(O)OC[C@@H](O)[C@@H](O)[C@@H](O)Cn1c2nc(=O)[nH]c(=O)c-2cc2ccc(O)cc21",
                                       cofactors_df = cofactors_df) == "Oxidized-Factor-F420_CoF"

def test_get_reduced_factor_F420_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "*C(=O)[C@H](C)OP(=O)(O)OC[C@@H](O)[C@@H](O)[C@@H](O)CN1c2cc(O)ccc2Cc2c1[nH]c(=O)[nH]c2=O",
                                       cofactors_df = cofactors_df) == "Reduced-Factor-F420_CoF"

def test_get_acetyl_coa_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "CC(=O)SCCNC(=O)CCNC(=O)[C@H](O)C(C)(C)COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O",
                                       cofactors_df = cofactors_df) == "ACETYL-COA"

def test_get_CO_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "C#[O+]",
                                       cofactors_df = cofactors_df) == "CO"

def test_get_CO2_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=C=O",
                                       cofactors_df = cofactors_df) == "CO2"

def test_get_CO3_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=C(O)O",
                                       cofactors_df = cofactors_df) == "CO3"

def test_get_CoA_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",
                                       cofactors_df = cofactors_df) == "CoA"

def test_get_H_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "[H+]",
                                       cofactors_df = cofactors_df) == "H+"

def test_get_H2O2_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "OO",
                                       cofactors_df = cofactors_df) == "H2O2"

def test_get_HBr_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Br",
                                       cofactors_df = cofactors_df) == "HBr"

def test_get_HCN_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "C#N",
                                       cofactors_df = cofactors_df) == "HCN"

def test_get_HCl_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "Cl",
                                       cofactors_df = cofactors_df) == "HCl"

def test_get_HF_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "F",
                                       cofactors_df = cofactors_df) == "HF"

def test_get_HI_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "I",
                                       cofactors_df = cofactors_df) == "HI"

def test_get_NH3_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "N",
                                       cofactors_df = cofactors_df) == "NH3"

def test_get_O2_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=O",
                                       cofactors_df = cofactors_df) == "O2"

def test_get_PPI_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=P(O)(O)OP(=O)(O)O",
                                       cofactors_df = cofactors_df) == "PPI"

def test_get_PI_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=P(O)(O)O",
                                       cofactors_df = cofactors_df) == "Pi"

def test_get_SULFATE_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=S(=O)(O)O",
                                       cofactors_df = cofactors_df) == "SULFATE"

def test_get_SULFITE_CoF_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O=S(O)O",
                                       cofactors_df = cofactors_df) == "SULFITE"

def test_get_WATER_code(cofactors_df):
    """
    Check if the correct JN1224MIN-type cofactor code can be obtained for the input cofactor SMILES.
    """
    assert utils.get_cofactor_CoF_code(query_SMILES = "O",
                                       cofactors_df = cofactors_df) == "WATER"

def test_are_rxn_descriptors_equal_01():
    """
    Test if two given reaction descriptors specify the same types of species even if lists are formatted differently.
    """
    rxn_descriptor_01 = [s for s in "Any;NAD_CoF".split(";")]
    rxn_descriptor_02 = ["Any","NAD_CoF"]
    assert utils.are_rxn_descriptors_equal(rxn_descriptor_01, rxn_descriptor_02)

def test_are_rxn_descriptors_equal_02():
    """
    Test if two given reaction descriptors specify the same types of species even if lists are formatted differently.
    """
    rxn_descriptor_01 = [s for s in "Any;NAD_CoF".split(";")]
    rxn_descriptor_02 = ["NAD_CoF","Any"] # try a different order than the test above
    assert utils.are_rxn_descriptors_equal(rxn_descriptor_01, rxn_descriptor_02)

def test_are_rxn_descriptors_equal_03():
    """
    Test if two given reaction descriptors specify the same types of species even if lists are formatted differently.
    """
    rxn_descriptor_01 = [s for s in "Any;NADH_CoF;CO2".split(";")]
    rxn_descriptor_02 = ["NADH_CoF","CO2","Any"]
    assert utils.are_rxn_descriptors_equal(rxn_descriptor_01, rxn_descriptor_02)

def test_are_rxn_descriptors_equal_04():
    """
    Test if two given reaction descriptors specify the same types of species even if lists are formatted differently.
    """
    rxn_descriptor_01 = [s for s in "Any;NADH_CoF;CO2".split(";")]
    rxn_descriptor_02 = ["CO2","Any","NADH_CoF",]
    assert utils.are_rxn_descriptors_equal(rxn_descriptor_01, rxn_descriptor_02)

def test_does_template_fit_01_using_unmapped_rxn_str_forward_dir():
    """
    Test if a given template assigned to the unmapped MetaCyc ALCOHOL-DEHYDROG-RXN (rxn idx 903) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_01_using_mapped_rxn_str_forward_dir():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc ALCOHOL-DEHYDROG-RXN (rxn idx 903) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_01_using_unmapped_rxn_str_reverse_dir():
    """
    Test if a given template assigned to the unmapped MetaCyc ALCOHOL-DEHYDROG-RXN (rxn idx 903) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    This reaction direction and the template are both in the opposite direction to the tests above.
    """
    rxn_str = 'CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]>>CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1'
    rxn_template = '[#6:7]=[#8:8].[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_01_using_mapped_rxn_str_reverse_dir():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc ALCOHOL-DEHYDROG-RXN (rxn idx 903) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    This reaction direction and the template are both in the opposite direction to the tests above.
    """
    rxn_str = '[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:8][N:9]([C@@H:10]2[O:11][C@H:12]([CH2:13][O:14][P:15](=[O:16])([OH:17])[O:18][P:19](=[O:20])([OH:21])[O:22][CH2:23][C@H:24]3[O:25][C@@H:26]([n:27]4[cH:28][n:29][c:30]5[c:31]([NH2:32])[n:33][cH:34][n:35][c:36]45)[C@H:37]([OH:38])[C@@H:39]3[OH:40])[C@@H:41]([OH:42])[C@H:43]2[OH:44])[CH:45]=[CH:46][CH2:47]1>>[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][n+:9]([C@@H:10]2[O:11][C@H:12]([CH2:13][O:14][P:15](=[O:16])([OH:17])[O:18][P:19](=[O:20])([OH:21])[O:22][CH2:23][C@H:24]3[O:25][C@@H:26]([n:27]4[cH:28][n:29][c:30]5[c:31]([NH2:32])[n:33][cH:34][n:35][c:36]45)[C@H:37]([OH:38])[C@@H:39]3[OH:40])[C@@H:41]([OH:42])[C@H:43]2[OH:44])[cH:45][cH:46][cH:47]1'
    rxn_template = '[#6:7]=[#8:8].[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_02_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc RXN-10781 (rxn idx 2286) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=CCc1c[nH]c2ccc(O)cc12.[H+]>>NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.OCCc1c[nH]c2ccc(O)cc12'
    rxn_template = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_02_using_mapped_rxn_str():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc RXN-10781 (rxn idx 2286) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[H+].[NH2:1][C:2](=[O:3])[C:4]1=[CH:5][N:6]([C@@H:7]2[O:8][C@H:9]([CH2:10][O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:18])[O:19][CH2:20][C@H:21]3[O:22][C@@H:23]([n:24]4[cH:25][n:26][c:27]5[c:28]([NH2:29])[n:30][cH:31][n:32][c:33]45)[C@H:34]([OH:35])[C@@H:36]3[OH:37])[C@@H:38]([OH:39])[C@H:40]2[OH:41])[CH:42]=[CH:43][CH2:44]1.[O:45]=[CH:46][CH2:47][c:48]1[cH:49][nH:50][c:51]2[cH:52][cH:53][c:54]([OH:55])[cH:56][c:57]12>>[NH2:1][C:2](=[O:3])[c:4]1[cH:5][n+:6]([C@@H:7]2[O:8][C@H:9]([CH2:10][O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:18])[O:19][CH2:20][C@H:21]3[O:22][C@@H:23]([n:24]4[cH:25][n:26][c:27]5[c:28]([NH2:29])[n:30][cH:31][n:32][c:33]45)[C@H:34]([OH:35])[C@@H:36]3[OH:37])[C@@H:38]([OH:39])[C@H:40]2[OH:41])[cH:42][cH:43][cH:44]1.[OH:45][CH2:46][CH2:47][c:48]1[cH:49][nH:50][c:51]2[cH:52][cH:53][c:54]([OH:55])[cH:56][c:57]12'
    rxn_template = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_03_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc RXN-10911 (rxn idx 2320) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C[C@H](O)c1ccc(O)c(O)c1.[H+]>>NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.OC[C@H](O)c1ccc(O)c(O)c1'
    rxn_template = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_03_using_mapped_rxn_str():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc RXN-10911 (rxn idx 2320) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[H+].[NH2:1][C:2](=[O:3])[C:4]1=[CH:5][N:6]([C@@H:7]2[O:8][C@H:9]([CH2:10][O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:18])[O:19][CH2:20][C@H:21]3[O:22][C@@H:23]([n:24]4[cH:25][n:26][c:27]5[c:28]([NH2:29])[n:30][cH:31][n:32][c:33]45)[C@H:34]([OH:35])[C@@H:36]3[OH:37])[C@@H:38]([OH:39])[C@H:40]2[OH:41])[CH:42]=[CH:43][CH2:44]1.[O:45]=[CH:46][C@H:47]([OH:48])[c:49]1[cH:50][cH:51][c:52]([OH:53])[c:54]([OH:55])[cH:56]1>>[NH2:1][C:2](=[O:3])[c:4]1[cH:5][n+:6]([C@@H:7]2[O:8][C@H:9]([CH2:10][O:11][P:12](=[O:13])([OH:14])[O:15][P:16](=[O:17])([OH:18])[O:19][CH2:20][C@H:21]3[O:22][C@@H:23]([n:24]4[cH:25][n:26][c:27]5[c:28]([NH2:29])[n:30][cH:31][n:32][c:33]45)[C@H:34]([OH:35])[C@@H:36]3[OH:37])[C@@H:38]([OH:39])[C@H:40]2[OH:41])[cH:42][cH:43][cH:44]1.[OH:45][CH2:46][C@H:47]([OH:48])[c:49]1[cH:50][cH:51][c:52]([OH:53])[c:54]([OH:55])[cH:56]1'
    rxn_template = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_04_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc RXN-10915 (rxn idx 2324) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'COc1cc([C@@H](O)CO)ccc1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>COc1cc([C@@H](O)C=O)ccc1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_04_using_mapped_rxn_str():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc RXN-10915 (rxn idx 2324) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[CH3:1][O:2][c:3]1[cH:4][c:5]([C@@H:6]([OH:7])[CH2:8][OH:9])[cH:10][cH:11][c:12]1[OH:13].[NH2:14][C:15](=[O:16])[c:17]1[cH:18][cH:19][cH:20][n+:21]([C@@H:22]2[O:23][C@H:24]([CH2:25][O:26][P:27](=[O:28])([OH:29])[O:30][P:31](=[O:32])([OH:33])[O:34][CH2:35][C@H:36]3[O:37][C@@H:38]([n:39]4[cH:40][n:41][c:42]5[c:43]([NH2:44])[n:45][cH:46][n:47][c:48]45)[C@H:49]([OH:50])[C@@H:51]3[OH:52])[C@@H:53]([OH:54])[C@H:55]2[OH:56])[cH:57]1>>[CH3:1][O:2][c:3]1[cH:4][c:5]([C@@H:6]([OH:7])[CH:8]=[O:9])[cH:10][cH:11][c:12]1[OH:13].[H+].[NH2:14][C:15](=[O:16])[C:17]1=[CH:57][N:21]([C@@H:22]2[O:23][C@H:24]([CH2:25][O:26][P:27](=[O:28])([OH:29])[O:30][P:31](=[O:32])([OH:33])[O:34][CH2:35][C@H:36]3[O:37][C@@H:38]([n:39]4[cH:40][n:41][c:42]5[c:43]([NH2:44])[n:45][cH:46][n:47][c:48]45)[C@H:49]([OH:50])[C@@H:51]3[OH:52])[C@@H:53]([OH:54])[C@H:55]2[OH:56])[CH:20]=[CH:19][CH2:18]1'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_05_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc RXN-7693 (rxn idx 5751) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'CC(C)CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC(C)CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_05_using_mapped_rxn_str():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc RXN-7693 (rxn idx 5751) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[CH3:1][CH:2]([CH3:3])[CH2:4][CH2:5][OH:6].[NH2:7][C:8](=[O:9])[c:10]1[cH:11][cH:12][cH:13][n+:14]([C@@H:15]2[O:16][C@H:17]([CH2:18][O:19][P:20](=[O:21])([OH:22])[O:23][P:24](=[O:25])([OH:26])[O:27][CH2:28][C@H:29]3[O:30][C@@H:31]([n:32]4[cH:33][n:34][c:35]5[c:36]([NH2:37])[n:38][cH:39][n:40][c:41]45)[C@H:42]([OH:43])[C@@H:44]3[OH:45])[C@@H:46]([OH:47])[C@H:48]2[OH:49])[cH:50]1>>[CH3:1][CH:2]([CH3:3])[CH2:4][CH:5]=[O:6].[H+].[NH2:7][C:8](=[O:9])[C:10]1=[CH:50][N:14]([C@@H:15]2[O:16][C@H:17]([CH2:18][O:19][P:20](=[O:21])([OH:22])[O:23][P:24](=[O:25])([OH:26])[O:27][CH2:28][C@H:29]3[O:30][C@@H:31]([n:32]4[cH:33][n:34][c:35]5[c:36]([NH2:37])[n:38][cH:39][n:40][c:41]45)[C@H:42]([OH:43])[C@@H:44]3[OH:45])[C@@H:46]([OH:47])[C@H:48]2[OH:49])[CH:13]=[CH:12][CH2:11]1'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_06_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc RXN-18376 (rxn idx 4258) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'C[N+](C)(C)CCO.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O.[H+]>>C[NH+](C)C.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O.O=CCO'
    rxn_template = '[#6:1]-[#7:2].[#6:3]1=[#6:4]-[#7:5]-[#6:6]=[#6:7]-[#6:8]-1.[#8:9]=[#8:10]>>[#6:3]1:[#6:8]:[#6:7]:[#6:6]:[#7+:5]:[#6:4]:1.[#6:1]=[#8:10].[#7:2].[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_06_using_mapped_rxn_str():
    """
    Test if a given template assigned to the fully atom-mapped MetaCyc RXN-18376 (rxn idx 4258) fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = '[CH3:1][N+:2]([CH3:3])([CH3:4])[CH2:5][CH2:6][OH:7].[H+].[NH2:8][C:9](=[O:10])[C:11]1=[CH:12][N:13]([C@@H:14]2[O:15][C@H:16]([CH2:17][O:18][P:19](=[O:20])([OH:21])[O:22][P:23](=[O:24])([OH:25])[O:26][CH2:27][C@H:28]3[O:29][C@@H:30]([n:31]4[cH:32][n:33][c:34]5[c:35]([NH2:36])[n:37][cH:38][n:39][c:40]45)[C@H:41]([OH:42])[C@@H:43]3[OH:44])[C@@H:45]([OH:46])[C@H:47]2[OH:48])[CH:49]=[CH:50][CH2:51]1.[O:52]=[O:53]>>[CH3:1][NH+:2]([CH3:3])[CH3:4].[CH:5]([CH2:6][OH:7])=[O:53].[NH2:8][C:9](=[O:10])[c:11]1[cH:12][n+:13]([C@@H:14]2[O:15][C@H:16]([CH2:17][O:18][P:19](=[O:20])([OH:21])[O:22][P:23](=[O:24])([OH:25])[O:26][CH2:27][C@H:28]3[O:29][C@@H:30]([n:31]4[cH:32][n:33][c:34]5[c:35]([NH2:36])[n:37][cH:38][n:39][c:40]45)[C@H:41]([OH:42])[C@@H:43]3[OH:44])[C@@H:45]([OH:46])[C@H:47]2[OH:48])[cH:49][cH:50][cH:51]1.[OH2:52]'
    rxn_template = '[#6:1]-[#7:2].[#6:3]1=[#6:4]-[#7:5]-[#6:6]=[#6:7]-[#6:8]-1.[#8:9]=[#8:10]>>[#6:3]1:[#6:8]:[#6:7]:[#6:6]:[#7+:5]:[#6:4]:1.[#6:1]=[#8:10].[#7:2].[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_07_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'CC1(C)C2CC[C@]3(C)C(CC=C4C5C[C@@](C)(CO)CC[C@]5(C)CC[C@]43C)[C@@]2(C)CC[C@@H]1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O>>CC1(C)C2CC[C@]3(C)C([C@H](O)C=C4C5C[C@@](C)(CO)CC[C@]5(C)CC[C@]43C)[C@@]2(C)CC[C@@H]1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O'
    rxn_template = '[#6:1].[#6:2]1=[#6:3]-[#7:4]-[#6:5]=[#6:6]-[#6:7]-1.[#8:8]=[#8:9]>>[#6:1]-[#8:8].[#6:2]1:[#6:7]:[#6:6]:[#6:5]:[#7+:4]:[#6:3]:1.[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_08_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'CC(C)O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O>>CC(O)CO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O'
    rxn_template = '[#6:1].[#6:2]1=[#6:3]-[#7:4]-[#6:5]=[#6:6]-[#6:7]-1.[#8:8]=[#8:9]>>[#6:1]-[#8:8].[#6:2]1:[#6:7]:[#6:6]:[#6:5]:[#7+:4]:[#6:3]:1.[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_09_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'O=C1c2c(O)cc(O)cc2O[C@H](c2ccc(O)cc2)[C@H]1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O>>O=C1c2c(O)cc(O)cc2O[C@H](c2ccc(O)c(O)c2)[C@H]1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O'
    rxn_template = '[#6:1].[#6:2]1=[#6:3]-[#7:4]-[#6:5]=[#6:6]-[#6:7]-1.[#8:8]=[#8:9]>>[#6:1]-[#8:8].[#6:2]1:[#6:7]:[#6:6]:[#6:5]:[#7+:4]:[#6:3]:1.[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_10_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'C=C[C@]1(C)CC[C@@H]2C(=CC[C@H]3[C@@](C)(C=O)CCC[C@]23C)C1.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O>>C=C[C@]1(C)CC[C@@H]2C(=CC[C@H]3[C@@](C)(C(=O)O)CCC[C@]23C)C1.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O'
    rxn_template = '[#6:1].[#6:2]1=[#6:3]-[#7:4]-[#6:5]=[#6:6]-[#6:7]-1.[#8:8]=[#8:9]>>[#6:1]-[#8:8].[#6:2]1:[#6:7]:[#6:6]:[#6:5]:[#7+:4]:[#6:3]:1.[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_11_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = 'CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/Cc1cc(O)ccc1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O>>CC(C)=CCC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/CC/C(C)=C/Cc1ccccc1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O'
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1.[#8:9]>>[#6:1].[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#8:2]=[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_12_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "CC(C)=CCC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](CO)[C@@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O>>CC(C)=CCC[C@@H](C)[C@H]1CC[C@H]2C3=CC[C@H]4[C@H](C)[C@@H](O)CC[C@]4(C)[C@H]3CC[C@@]21C.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O"
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1.[#8:9]>>[#6:1].[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#8:2]=[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_13_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "COc1cc2c(O)c3c(c(-c4ccc5c(c4)OCO5)c2cc1OC)C(=O)OC3.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O>>COc1cc2cc3c(c(-c4ccc5c(c4)OCO5)c2cc1OC)C(=O)OC3.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O"
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1.[#8:9]>>[#6:1].[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#8:2]=[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_14_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "COc1cc2c(O)c3c(c(-c4ccc5c(c4)OCO5)c2cc1OC)C(=O)OC3.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O>>COc1cc2cc3c(c(-c4ccc5c(c4)OCO5)c2cc1OC)C(=O)OC3.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=O"
    rxn_template = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1.[#8:9]>>[#6:1].[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#8:2]=[#8:9]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_15_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "CC(=O)N[C@H]1[C@H](C)O[C@H](O)[C@H](NC(C)=O)[C@H]1O.O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1>>CC(=O)N[C@H]1[C@H](C)O[C@H](OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)[C@H](NC(C)=O)[C@H]1O.O"
    rxn_template = '[#6:1]-[#8:2].[#8:3]>>[#6:1]-[#8:3].[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_16_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "CC(=O)N[C@H]1[C@H](C)O[C@H](OP(=O)(O)OP(=O)(O)OC[C@H]2O[C@@H](n3ccc(=O)[nH]c3=O)[C@H](O)[C@@H]2O)[C@H](NC(C)=O)[C@H]1O.O>>CC(=O)N[C@H]1[C@H](C)O[C@H](O)[C@H](NC(C)=O)[C@H]1O.O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1"
    rxn_template = '[#6:1]-[#8:2].[#8:3]>>[#6:1]-[#8:3].[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_17_using_unmapped_rxn_str():
    """
    Test if a given template assigned to the unmapped MetaCyc rxn fits.
    Note that on the LHS of a reaction, the order in which reactants appear MUST MATCH that within the template.
    On the RHS of a reaction, the order in which products appear does NOT need to match that within the template.
    """
    rxn_str = "CCCCCCCC/C=C\\CCCCCCCC(=O)OC[C@@H](O)CO.O>>OCC(O)CO.CCCCCCCC/C=C\\CCCCCCCC(=O)O"
    rxn_template = '[#6:1]-[#8:2].[#8:3]>>[#6:1]-[#8:3].[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True

def test_does_template_fit_18_using_unmapped_rxn_str():
    rxn_str = "OC[C@H]1O[C@@H](Oc2cc3c(O)cc(O)cc3[o+]c2-c2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O.O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]2O)c(=O)[nH]1>>O=c1ccn([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)O[C@H]3O[C@H](CO)[C@@H](O)[C@H](O)[C@H]3O)[C@@H](O)[C@H]2O)c(=O)[nH]1.Oc1ccc(-c2[o+]c3cc(O)cc(O)c3cc2O)cc1"
    rxn_template = '[#6:1]-[#8:2].[#8:3]>>[#6:1]-[#8:3].[#8:2]'
    assert utils.does_template_fit(rxn_str, rxn_template) is True