from rdkit import Chem
from rdkit.Chem import AllChem
from rxntools import reaction

def test_get_mapped_bonds_data_type():
    """
    Test to ensure the internal function _get_mapped_bonds returns a set.
    Each element in this set is a tuple that corresponds to a mapped bond.
    This tuple itself comprises three elements.
    The first two elements are integers indicating the start & end atom indices of a given bond.
    Meanwhile, the third element corresponds to the bond type, i.e. single, double or aromatic.
    This bond type is a Chem.rdchem.BondType data type.
    :return:
    """
    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')
    mapped_bonds = mapped_rxn._get_mapped_bonds(mol = Chem.MolFromSmarts('[CH3:1][CH2:2][OH:3]'))
    assert isinstance(mapped_bonds, set)

    for bond in mapped_bonds:
        assert isinstance(bond, tuple)

        # the index of the starting atom in a bond should be an integer
        atom_start_idx = bond[0]
        assert isinstance(atom_start_idx, int)

        # the index of the ending atom in a bond should also be an integer
        atom_end_idx = bond[1]
        assert isinstance(atom_end_idx, int)

        # the bond type should be an RDKit bond type
        bond_type = bond[2]
        assert isinstance(bond_type, Chem.rdchem.BondType)

def test_get_mapped_bonds_frm_ethanol_SMARTS():
    """
    Test to ensure the internal function _get_mapped_bonds returns all atom-mapped bonds.
    The start index, end index, as well as bond type should be correctly identified.
    This test checks for bonds in ethanol in an atom-mapped alcohol dehydrogenase reaction.
    """
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')
    mapped_bonds = mapped_rxn._get_mapped_bonds(mol=Chem.MolFromSmarts('[CH3:1][CH2:2][OH:3]'))
    mapped_bonds = list(mapped_bonds) # convert the set of mapped bonds to a list

    assert (1, 2, Chem.rdchem.BondType.SINGLE) in mapped_bonds # ensure C-C bond is present
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in mapped_bonds # ensure C-O bond is present

def test_get_mapped_bonds_frm_ethanal_SMARTS():
    """
    Test to ensure the internal function _get_mapped_bonds returns all atom-mapped bonds.
    The start index, end index, as well as bond type should be correctly identified.
    This test checks for bonds in ethanal in an atom-mapped alcohol dehydrogenase reaction.
    """
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')
    mapped_bonds = mapped_rxn._get_mapped_bonds(mol=Chem.MolFromSmarts('[CH3:1][CH:2]=[O:3]'))
    mapped_bonds = list(mapped_bonds) # convert the set of mapped bonds to a list

    assert (1, 2, Chem.rdchem.BondType.SINGLE) in mapped_bonds  # ensure C-C bond is present
    assert (2, 3, Chem.rdchem.BondType.DOUBLE) in mapped_bonds  # ensure C=O bond is present

def test_get_mapped_bonds_frm_nad_plus_SMARTS():
    """
    Test to ensure the internal function _get_mapped_bonds returns all atom-mapped bonds.
    The start index, end index, as well as bond type should be correctly identified.
    This test checks for bonds in NAD(+) in an atom-mapped alcohol dehydrogenase reaction.
    """
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')
    mapped_bonds = mapped_rxn._get_mapped_bonds(mol=Chem.MolFromSmarts('[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1'))
    mapped_bonds = list(mapped_bonds) # convert the set of mapped bonds to a list

    # check atom mappings of the R-C(=O)NH2 amide group
    assert (4, 5, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (5, 6, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (5, 7, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the pyridine aromatic ring boned to the R-C(=O)NH2 amide group
    assert (7, 8, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (8, 9, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (9, 10, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (10, 11, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (11, 47, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (47, 7, Chem.rdchem.BondType.AROMATIC) in mapped_bonds

    # check atom mapping of the first tetrahydrofuran ring bonded to the pyridine ring
    assert (11, 12, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (12, 13, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (13, 14, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (14, 43, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (43, 44, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (43, 45, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (45, 46, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the diphosphate bridge between the tetrahydrofuran rings
    assert (14, 15, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (15, 16, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (16, 17, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (17, 18, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (17, 19, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (17, 20, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (20, 21, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (21, 22, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (21, 23, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (21, 24, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (24, 25, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the second tetrahydrofuran ring bonded to the indole group
    assert (25, 26, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (26, 27, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (27, 28, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (28, 39, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (39, 40, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (39, 41, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (41, 42, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (41, 26, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the indole group
    assert (28, 29, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (29, 30, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (30, 31, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (30, 31, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (31, 32, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (38, 32, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (38, 29, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (32, 33, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (33, 34, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (33, 35, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (33, 35, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (35, 36, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (36, 37, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (37, 38, Chem.rdchem.BondType.AROMATIC) in mapped_bonds

def test_get_mapped_bonds_frm_nadh_plus_SMARTS():
    """
    Test to ensure the internal function _get_mapped_bonds returns all atom-mapped bonds.
    The start index, end index, as well as bond type should be correctly specified.
    This test checks for bonds in NADH in an atom-mapped alcohol dehydrogenase reaction.
    """
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')
    mapped_bonds = mapped_rxn._get_mapped_bonds(mol=Chem.MolFromSmarts('[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'))
    mapped_bonds = list(mapped_bonds) # convert the set of mapped bonds to a list

    # check atom mappings of the R-C(=O)NH2 amide group
    assert (4, 5, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (5, 6, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (5, 7, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the pyridine aromatic ring boned to the R-C(=O)NH2 amide group
    assert (8, 7, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (9, 8, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (10, 9, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (11, 10, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (47, 11, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (7, 47, Chem.rdchem.BondType.DOUBLE) in mapped_bonds

    # check atom mapping of the first tetrahydrofuran ring bonded to the pyridine ring
    assert (11, 12, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (12, 13, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (13, 14, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (14, 43, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (43, 44, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (43, 45, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (45, 46, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the diphosphate bridge between the tetrahydrofuran rings
    assert (14, 15, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (15, 16, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (16, 17, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (17, 18, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (17, 19, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (17, 20, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (20, 21, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (21, 22, Chem.rdchem.BondType.DOUBLE) in mapped_bonds
    assert (21, 23, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (21, 24, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (24, 25, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the second tetrahydrofuran ring bonded to the indole group
    assert (25, 26, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (26, 27, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (27, 28, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (28, 39, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (39, 40, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (39, 41, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (41, 42, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (41, 26, Chem.rdchem.BondType.SINGLE) in mapped_bonds

    # check atom mapping of the indole group
    assert (28, 29, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (29, 30, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (30, 31, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (30, 31, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (31, 32, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (38, 32, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (38, 29, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (32, 33, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (33, 34, Chem.rdchem.BondType.SINGLE) in mapped_bonds
    assert (33, 35, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (33, 35, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (35, 36, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (36, 37, Chem.rdchem.BondType.AROMATIC) in mapped_bonds
    assert (37, 38, Chem.rdchem.BondType.AROMATIC) in mapped_bonds

def test_get_all_changed_atoms_frm_ethanol_AdH_rxn_SMARTS():
    mapped_rxn = reaction.mapped_reaction(
        '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn._get_all_changed_atoms()

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [2, 3, 7, 8, 9, 10, 11, 47]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    # ensure that a C-O single bond has been broken within ethanol (CCO)
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in broken_bonds

    # ensure that a C=O double bond has been formed within ethanal (CC=O)
    assert (2, 3, Chem.rdchem.BondType.DOUBLE) in formed_bonds

    # ensure that the aromaticity of the pyridine is broken as expected in an AdH reaction
    assert (7, 8, Chem.rdchem.BondType.AROMATIC) in broken_bonds
    assert (8, 9, Chem.rdchem.BondType.AROMATIC) in broken_bonds
    assert (9, 10, Chem.rdchem.BondType.AROMATIC) in broken_bonds
    assert (10, 11, Chem.rdchem.BondType.AROMATIC) in broken_bonds
    assert (11, 47, Chem.rdchem.BondType.AROMATIC) in broken_bonds
    assert (47, 7, Chem.rdchem.BondType.AROMATIC) in broken_bonds

    # ensure that new single and double bonds are formed within the pyrimidine ring
    assert (8, 7, Chem.rdchem.BondType.SINGLE) in formed_bonds
    assert (9, 8, Chem.rdchem.BondType.SINGLE) in formed_bonds
    assert (10, 9, Chem.rdchem.BondType.DOUBLE) in formed_bonds
    assert (11, 10, Chem.rdchem.BondType.SINGLE) in formed_bonds
    assert (47, 11, Chem.rdchem.BondType.SINGLE) in formed_bonds
    assert (7, 47, Chem.rdchem.BondType.DOUBLE) in formed_bonds