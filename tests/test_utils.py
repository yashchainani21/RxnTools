from rxntools import utils

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & canonical SMILES
    """
    assert utils.are_isomorphic(s1 = "[CH3:1][CH2:2][OH:3]",
                                s2 = "CCO")

def test_is_isomorphic_ethanol_with_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphism when ethanol is represented with atom-mapped SMARTS & non-canonical SMILES
    """
    assert utils.are_isomorphic(s1="[CH3:1][CH2:2][OH:3]",
                                s2="OCC")

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & canonical SMILES
    """
    assert utils.are_isomorphic(s1="[CH3][CH2][OH]",
                                s2="CCO")

def test_is_isomorphic_ethanol_without_atom_mapped_SMARTS_and_non_canonical_SMILES():
    """
    Ensure isomorphic when ethanol is represented with non-atom-mapped SMARTS & canonical SMILES
    """
    assert utils.are_isomorphic(s1="[CH3][CH2][OH]",
                                s2="OCC")