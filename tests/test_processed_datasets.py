import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Tuple, List
from collections import Counter

def are_isomorphic(mol1: Chem.rdchem.Mol, mol2: Chem.rdchem.Mol, consider_stereo: bool = False) -> bool:
    """ 
    Check if two molecules are isomorphic, i.e. their bond and atom arrangements are identical

    Parameters ----------
    mol1 : Chem.rdchem.Mol
        RDKit mol object of first molecule generated from its SMILES or SMARTS representation. If SMARTS were used in creating mol object, these need not be atom mapped.
    mol2 : Chem.rdchem.Mol
        RDKit mol object of second molecule generated from its SMILES or SMARTS representation. If SMARTS were used in creating mol object, these need not be atom mapped.
    consider_stereo: bool
        Whether to consider stereochemistry or not when comparing two molecules.

    Returns -------
    is_isomorphic : bool
        True if both molecules are isomorphic, False otherwise.
    """
    if consider_stereo:
        is_isomorphic = mol1.HasSubstructMatch(mol2, useChirality=True) and mol2.HasSubstructMatch(mol1, useChirality=True)
    else:
        is_isomorphic = mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)
    return is_isomorphic

def are_mol_tuples_isomorphic(
    tup1: Tuple[Chem.rdchem.Mol, ...],
    tup2: Tuple[Chem.rdchem.Mol, ...],
    consider_stereo: bool = False,
) -> bool:
    """
    Check whether two tuples of RDKit molecules are identical as multisets
    under molecular isomorphism.
    """

    if len(tup1) != len(tup2):
        return False

    unmatched = list(tup2)

    for mol1 in tup1:
        match_idx = None
        for i, mol2 in enumerate(unmatched):
            if are_isomorphic(mol1, mol2, consider_stereo=consider_stereo):
                match_idx = i
                break

        if match_idx is None:
            return False

        # consume the matched molecule (preserves multiplicity!)
        unmatched.pop(match_idx)

    return True

@pytest.fixture()
def KEGG_df():
    return pd.read_parquet("../data/processed/enzymemap_Kegg_JN_mapped_non_unique.parquet")

@pytest.fixture()
def MetaCyc_df():
    return pd.read_parquet("../data/processed/enzymemap_MetaCyc_JN_mapped_non_unique.parquet")

@pytest.fixture()
def BRENDA_df():
    return pd.read_parquet("../data/processed/enzymemap_BRENDA_JN_mapped_non_unique.parquet")

@pytest.fixture()
def JN_rules_df():
    JN_rules_df = pd.read_csv("../data/raw/JN1224MIN_rules.tsv", delimiter='\t')

    all_num_substrates = []
    all_num_products = []
    all_num_lhs_cofactors = []
    all_num_rhs_cofactors = []
    all_lhs_cofactor_codes = []
    all_rhs_cofactor_codes = []

    # for each reaction rule in the generalized reaction operators' dataframe
    for _, rule_row in JN_rules_df.iterrows():
        num_substrates_in_rule = list(rule_row['Reactants'])[0].split(';').count('Any')
        num_products_in_rule = list(rule_row['Products'])[0].split(';').count('Any')
        
        # cofactors are also indicated by anything that is not 'Any' in the JN rule definition
        reactants = list(rule_row['Reactants'])[0].split(';')
        products = list(rule_row['Products'])[0].split(';')
        num_lhs_cofactors_in_rule = sum(1 for s in reactants if s.strip() != 'Any')
        num_rhs_cofactors_in_rule = sum(1 for s in products if s.strip() != 'Any')

        # finally, we check if the identifies of the cofactors match those in the JN rule definition
        lhs_cofactor_codes_in_rule = [s.strip() for s in reactants if s.strip() != 'Any']
        rhs_cofactor_codes_in_rule = [s.strip() for s in products if s.strip() != 'Any']

        all_num_substrates.append(num_substrates_in_rule)
        all_num_products.append(num_products_in_rule)
        all_num_lhs_cofactors.append(num_lhs_cofactors_in_rule)
        all_num_rhs_cofactors.append(num_rhs_cofactors_in_rule)
        all_lhs_cofactor_codes.append(lhs_cofactor_codes_in_rule)
        all_rhs_cofactor_codes.append(rhs_cofactor_codes_in_rule)

    JN_rules_df['num_substrates'] = all_num_substrates
    JN_rules_df['num_products'] = all_num_products
    JN_rules_df['num_lhs_cofactors'] = all_num_lhs_cofactors
    JN_rules_df['num_rhs_cofactors'] = all_num_rhs_cofactors
    JN_rules_df['lhs_cofactor_codes'] = all_lhs_cofactor_codes
    JN_rules_df['rhs_cofactor_codes'] = all_rhs_cofactor_codes

    return JN_rules_df

def test_processed_KEGG_rxns_not_empty(KEGG_df):
    assert KEGG_df.shape[0] == 7966 # confirmed against the original .csv file too
    assert 'top_mapped_operator' in KEGG_df.columns

def test_processed_MetaCyc_rxns_not_empty(MetaCyc_df):
    assert MetaCyc_df.shape[0] == 4580 # confirmed against the original .csv file too
    assert 'top_mapped_operator' in MetaCyc_df.columns

def test_processed_BRENDA_rxns_not_empty():
    BRENDA_df = pd.read_parquet("../data/processed/enzymemap_BRENDA_JN_mapped_non_unique.parquet")
    assert BRENDA_df.shape[0] > 0
    assert 'top_mapped_operator' in BRENDA_df.columns

def check_columns_in_KEGG_processed_df(KEGG_df):
    expected_columns = ['substrates', 'products', 'LHS_cofactors', 'RHS_cofactors', 'LHS_cofactor_codes', 'RHS_cofactor_codes', 'top_mapped_operator']
    for col in expected_columns:
        assert col in KEGG_df.columns

def check_columns_in_MetaCyc_processed_df(MetaCyc_df):
    expected_columns = ['substrates', 'products', 'LHS_cofactors', 'RHS_cofactors', 'LHS_cofactor_codes', 'RHS_cofactor_codes', 'top_mapped_operator']
    for col in expected_columns:
        assert col in MetaCyc_df.columns

# test alcohol dehydrogenase related rules (rule0002 & rule0003) for KEGG were mapped correctly
def test_processed_KEGG_rule0002_and_rule0003_rxns_count(KEGG_df):
    rule0002_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0002']
    rule0003_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0003']
    assert rule0002_df.shape[0] == 704
    assert rule0003_df.shape[0] == 125

    # the substrates column should contain a list with exactly on element each (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column should contain a list with exactly one element each as well (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column should contain exactly one element each
    assert rule0002_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column should contain exactly one element each as well
    assert rule0002_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactor codes column for rule0002 should contain exactly one element: NAD_CoF
    assert rule0002_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # the rhs cofactor codes column for rule0002 should contain exactly one element: NADH_CoF
    assert rule0002_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the lhs cofactor codes column for rule0003 should contain exactly one element: NADH_CoF
    assert rule0003_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the rhs cofactor codes column for rule0003 should contain exactly one element: NAD_CoF
    assert rule0003_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # finally, check that running the template with RDKit's RunReactants method gives the resulting products
    JN_rule0002_SMARTS = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    for _, rxn_row in rule0002_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0002_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

    # repeat the above with rule0003
    JN_rule0003_SMARTS = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    for _, rxn_row in rule0003_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0003_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                         tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

# test alcohol dehydrogenase related rules (rule0002 & rule0003) for MetaCyc were mapped correctly
def test_processed_MetaCyc_rule0002_and_rule0003_rxns_count(MetaCyc_df):
    rule0002_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0002']
    rule0003_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0003']
    assert rule0002_df.shape[0] == 404
    assert rule0003_df.shape[0] == 389

    # the substrates column should contain a list with exactly one element each (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column should contain a list with exactly one element each as well (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column should contain exactly one element each
    assert rule0002_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column should contain exactly one element each as well
    assert rule0002_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactor codes column for rule0002 should contain exactly one element: NAD_CoF
    assert rule0002_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # the rhs cofactor codes column for rule0002 should contain exactly one element: NADH_CoF
    assert rule0002_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the lhs cofactor codes column for rule0003 should contain exactly one element: NADH_CoF
    assert rule0003_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the rhs cofactor codes column for rule0003 should contain exactly one element: NAD_CoF
    assert rule0003_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # finally, check that running the template with RDKit's RunReactants method gives the resulting products
    JN_rule0002_SMARTS = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    for _, rxn_row in rule0002_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0002_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

    # repeat the above with rule0003
    JN_rule0003_SMARTS = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    for _, rxn_row in rule0003_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0003_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                         tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

# test alcohol dehydrogenase related rules (rule0002 & rule0003) for BRENDA were mapped correctly
def test_processed_BRENDA_rule0002_and_rule0003_rxns_count(BRENDA_df):
    rule0002_df = BRENDA_df[BRENDA_df['top_mapped_operator'] == 'rule0002']
    rule0003_df = BRENDA_df[BRENDA_df['top_mapped_operator'] == 'rule0003']
    assert rule0002_df.shape[0] > 0
    assert rule0003_df.shape[0] > 0

    # the substrates column should contain a list with exactly one element each (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column should contain a list with exactly one element each as well (the alcohol being oxidized or the aldehyde being reduced)
    assert rule0002_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column should contain exactly one element each
    assert rule0002_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column should contain exactly one element each as well
    assert rule0002_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactor codes column for rule0002 should contain exactly one element: NAD_CoF
    assert rule0002_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # the rhs cofactor codes column for rule0002 should contain exactly one element: NADH_CoF
    assert rule0002_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the lhs cofactor codes column for rule0003 should contain exactly one element: NADH_CoF
    assert rule0003_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the rhs cofactor codes column for rule0003 should contain exactly one element: NAD_CoF
    assert rule0003_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NAD_CoF').all()

    # finally, check that running the template with RDKit's RunReactants method gives the resulting products
    JN_rule0002_SMARTS = '[#6:1]-[#8:2].[#6:3]1:[#6:4]:[#6:5]:[#6:6]:[#7+:7]:[#6:8]:1>>[#6:3]1=[#6:8]-[#7+0:7]-[#6:6]=[#6:5]-[#6:4]-1.[#6:1]=[#8:2]'
    for _, rxn_row in rule0002_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0002_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

    # repeat the above with rule0003
    JN_rule0003_SMARTS = '[#6:1]1=[#6:2]-[#7:3]-[#6:4]=[#6:5]-[#6:6]-1.[#6:7]=[#8:8]>>[#6:7]-[#8:8].[#6:1]1:[#6:6]:[#6:5]:[#6:4]:[#7+:3]:[#6:2]:1'
    for _, rxn_row in rule0003_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0003_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                         tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found    

# test decarboxylase related rules (rule0023 & rule0024) for KEGG were mapped correctly
def test_processed_KEGG_rule0023_and_rule0024_rxns_count(KEGG_df):
    rule0023_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0023']
    rule0024_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0024']

    assert rule0023_df.shape[0] == 9  
    assert rule0024_df.shape[0] == 120

    # the substrates column for carboxylation rxns (rule0023) should contain a list with exactly one element each
    assert rule0023_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for carboxylation rxns (rule0023) should contain a list with exactly one element each
    assert rule0023_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the substrates column for decarboxylation rxns (rule0024) should contain a list with exactly one element each
    assert rule0024_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for decarboxylation rxns (rule0024) should contain a list with exactly one element each
    assert rule0024_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column for carboxylation rxns (rule0023) should contain exactly one element each
    assert rule0023_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column for carboxylation rxns (rule0023) should contain no cofactors
    assert rule0023_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the lhs cofactors column for decarboxylation rxns (rule0024) should contain no cofactors
    assert rule0024_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the rhs cofactors column for decarboxylation rxns (rule0024) should contain exactly one element each
    assert rule0024_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactor codes column for carboxylation rxns (rule0023) should contain exactly one element: CO2
    assert rule0023_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'CO2').all()

    # the rhs cofactor codes column for carboxylation rxns (rule0023) should contain no cofactors
    assert rule0023_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the lhs cofactor codes column for decarboxylation rxns (rule0024) should contain no cofactors
    assert rule0024_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the rhs cofactor codes column for decarboxylation rxns (rule0024) should contain exactly one element: CO2
    assert rule0024_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'CO2').all()

    # finally, check that running the template with RDKit's RunReactants method gives the resulting products
    JN_rule0023_SMARTS = '[#6:1].[#6:2]=[#8:3]>>[#6:1]-[#6:2]-[#8:3]'
    for _, rxn_row in rule0023_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0023_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

    # repeate the above with rule0024
    JN_rule0024_SMARTS = '[#6:1]-[#6:2]-[#8:3]>>[#6:1].[#6:2]=[#8:3]'
    for _, rxn_row in rule0024_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0024_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

# test decarboxylase related rules (rule0023 & rule0024) for MetaCyc were mapped correctly
def test_processed_MetaCyc_rule0023_and_rule0024_rxns_count(MetaCyc_df):
    rule0023_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0023']
    rule0024_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0024']

    # there are no carboxylation and decarboxylation reactions in MetaCyc
    assert rule0023_df.shape[0] == 0
    assert rule0024_df.shape[0] == 0

# test decarboxylase related rules (rule0023 & rule0024) for BRENDA were mapped correctly
def test_processed_BRENDA_rule0023_and_rule0024_rxns_count(BRENDA_df):
    rule0023_df = BRENDA_df[BRENDA_df['top_mapped_operator'] == 'rule0023']
    rule0024_df = BRENDA_df[BRENDA_df['top_mapped_operator'] == 'rule0024']

    # there are no carboxylation and decarboxylation reactions in BRENDA
    assert rule0023_df.shape[0] > 0
    assert rule0024_df.shape[0] > 0

    # the substrates column for carboxylation rxns (rule0023) should contain a list with exactly one element each
    assert rule0023_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for carboxylation rxns (rule0023) should contain a list with exactly one element each
    assert rule0023_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the substrates column for decarboxylation rxns (rule0024) should contain a list with exactly one element each
    assert rule0024_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for decarboxylation rxns (rule0024) should contain a list with exactly one element each
    assert rule0024_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column for carboxylation rxns (rule0023) should contain exactly one element each
    assert rule0023_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column for carboxylation rxns (rule0023) should contain no cofactors
    assert rule0023_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the lhs cofactors column for decarboxylation rxns (rule0024) should contain no cofactors
    assert rule0024_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the rhs cofactors column for decarboxylation rxns (rule0024) should contain exactly one element each
    assert rule0024_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactor codes column for carboxylation rxns (rule0023) should contain exactly one element: CO2
    assert rule0023_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'CO2').all()

    # the rhs cofactor codes column for carboxylation rxns (rule0023) should contain no cofactors
    assert rule0023_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the lhs cofactor codes column for decarboxylation rxns (rule0024) should contain no cofactors
    assert rule0024_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 0).all()

    # the rhs cofactor codes column for decarboxylation rxns (rule0024) should contain exactly one element: CO2
    assert rule0024_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'CO2').all()

    # finally, check that running the template with RDKit's RunReactants method gives the resulting products
    JN_rule0023_SMARTS = '[#6:1].[#6:2]=[#8:3]>>[#6:1]-[#6:2]-[#8:3]'
    for _, rxn_row in rule0023_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0023_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

    # repeate the above with rule0024
    JN_rule0024_SMARTS = '[#6:1]-[#6:2]-[#8:3]>>[#6:1].[#6:2]=[#8:3]'
    for _, rxn_row in rule0024_df.iterrows():
        reactants_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['substrates'])
        reactants_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['LHS_cofactors'])

        expected_products_tuple = tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['products'])
        expected_products_tuple += tuple(Chem.MolFromSmiles(smi) for smi in rxn_row['RHS_cofactors'])

        # use RDKit to run the reaction template on the reactants
        rxn = AllChem.ReactionFromSmarts(JN_rule0024_SMARTS)
        products_from_RDKit = rxn.RunReactants(reactants_tuple)

        # for each product set generated by RDKit, check if it matches the expected products
        expected_products_found = False
        for products in products_from_RDKit:
            if are_mol_tuples_isomorphic(tup1 = products,
                                             tup2 = expected_products_tuple,):
                expected_products_found = True
                break
        assert expected_products_found

# test aldehyde dehydrogenase related rules for KEGG (rule0025 & rule0026) were mapped correctly
def test_processed_KEGG_aldehyde_dehydrogenase_rules(KEGG_df):
    rule0025_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0025']
    rule0026_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0026']

    assert rule0025_df.shape[0] == 7
    assert rule0026_df.shape[0] == 150

    # the substrates column for rule0025 should contain exactly one element: the aldehyde being oxidized
    assert rule0025_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()   

    # the products column for rule0025 should contain exactly one element: the carboxylic acid produced
    assert rule0025_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the substrates column for rule0026 should contain exactly one element: the carboxylic acid being oxidized
    assert rule0026_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()     

    # the products column for rule0026 should contain exactly one element: the aldehyde produced
    assert rule0026_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column for rule0025 should contain exactly one element
    assert rule0025_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column for rule0025 should contain exactly two elements
    assert rule0025_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the lhs cofactors column for rule0026 should contain exactly two elements
    assert rule0026_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the rhs cofactors column for rule0026 should contain exactly one element
    assert rule0026_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    
    # the lhs cofactor codes column for rule0025 should contain exactly one element: NADH_CoF
    assert rule0025_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the rhs cofactor codes column for rule0025 should contain exactly two elements: NAD_CoF and WATER
    assert rule0025_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the lhs cofactor codes column for rule0026 should contain exactly two elements: NAD_CoF and WATER
    assert rule0026_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the rhs cofactor codes column for rule0026 should contain exactly one element: NADH_CoF
    assert rule0026_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()
    
# test aldehyde dehydrogenase related rules for MetaCyc (rule0025 & rule0026) were mapped correctly
def test_processed_MetaCyc_aldehyde_dehydrogenase_rules(MetaCyc_df):
    rule0025_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0025']
    rule0026_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0026']

    assert rule0025_df.shape[0] == 7
    assert rule0026_df.shape[0] == 1

    # the substrates column for rule0025 should contain exactly one element: the aldehyde being oxidized
    assert rule0025_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()   

    # the products column for rule0025 should contain exactly one element: the carboxylic acid produced
    assert rule0025_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the substrates column for rule0026 should contain exactly one element: the carboxylic acid being oxidized
    assert rule0026_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()     

    # the products column for rule0026 should contain exactly one element: the aldehyde produced
    assert rule0026_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column for rule0025 should contain exactly one element
    assert rule0025_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the rhs cofactors column for rule0025 should contain exactly two elements
    assert rule0025_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the lhs cofactors column for rule0026 should contain exactly two elements
    assert rule0026_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the rhs cofactors column for rule0026 should contain exactly one element
    assert rule0026_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    
    # the lhs cofactor codes column for rule0025 should contain exactly one element: NADH_CoF
    assert rule0025_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()

    # the rhs cofactor codes column for rule0025 should contain exactly two elements: NAD_CoF and WATER
    assert rule0025_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the lhs cofactor codes column for rule0026 should contain exactly two elements: NAD_CoF and WATER
    assert rule0026_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the rhs cofactor codes column for rule0026 should contain exactly one element: NADH_CoF
    assert rule0026_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1 and x[0] == 'NADH_CoF').all()
    
# test monooxygenase related rules (rule0004 & rule0005) for KEGG were mapped correctly
def test_processed_KEGG_monooxygenase_rules(KEGG_df):
    rule0004_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0004']
    rule0005_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0005']

    assert rule0004_df.shape[0] == 111
    assert rule0005_df.shape[0] == 1

    # the substrates column for rule0004 should contain exactly one element: the substrate being monooxygenated
    assert rule0004_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for rule0004 should contain exactly one element: the monooxygenated product
    assert rule0004_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the substrates column for rule0005 should contain exactly one element: the monooxygenated product
    assert rule0005_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the products column for rule0005 should contain exactly one element: the substrate being monooxygenated
    assert rule0005_df['products'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()

    # the lhs cofactors column for rule0004 should contain exactly two elements
    assert rule0004_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the rhs cofactors column for rule0004 should contain exactly two elements
    assert rule0004_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the lhs cofactors column for rule0005 should contain exactly two elements
    assert rule0005_df['LHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the rhs cofactors column for rule0005 should contain exactly two elements
    assert rule0005_df['RHS_cofactors'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2).all()

    # the lhs cofactor codes column for rule0004 should contain exactly two elements: NADH_CoF and OXYGEN
    assert rule0004_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NADH_CoF' in x and 'O2' in x).all()

    # the rhs cofactor codes column for rule0004 should contain exactly two elements: NAD_CoF and WATER
    assert rule0004_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the lhs cofactor codes column for rule0005 should contain exactly two elements: NAD_CoF and WATER
    assert rule0005_df['LHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NAD_CoF' in x and 'WATER' in x).all()

    # the rhs cofactor codes column for rule0005 should contain exactly two elements: NADH_CoF and OXYGEN
    assert rule0005_df['RHS_cofactor_codes'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 2 and 'NADH_CoF' in x and 'O2' in x).all()

# test mappings for all reactions in KEGG
def test_all_KEGG_rxns_mapped(KEGG_df, JN_rules_df):
    mapped_rxns_df = KEGG_df[(KEGG_df['top_mapped_operator'].notna()) & (KEGG_df['top_mapped_operator']!='None')]

    assert mapped_rxns_df.shape[0] > 0

    for _, row in mapped_rxns_df.iterrows():

        # check that the substrates, products, LHS_cofactors, RHS_cofactors, LHS_cofactor_codes, RHS_cofactor_codes are all numpy arrays
        assert isinstance(row['substrates'], np.ndarray)
        assert isinstance(row['products'], np.ndarray)
        assert isinstance(row['LHS_cofactors'], np.ndarray)
        assert isinstance(row['RHS_cofactors'], np.ndarray)
        assert isinstance(row['LHS_cofactor_codes'], np.ndarray)
        assert isinstance(row['RHS_cofactor_codes'], np.ndarray)

        # test that the number of substrates and products aligns with the top mapped operator
        JN_mapped_rule = row['top_mapped_operator']
        rule_row = JN_rules_df[JN_rules_df['Name'] == JN_mapped_rule]
        rule_SMARTS = rule_row['SMARTS']
        
        # substrates and products are indicated by 'Any' in the JN rule definition
        assert len(row['substrates']) == list(rule_row['Reactants'])[0].split(';').count('Any')
        assert len(row['products']) == list(rule_row['Products'])[0].split(';').count('Any')

        # cofactors are also indicated by anything that is not 'Any' in the JN rule definition
        reactants = list(rule_row['Reactants'])[0].split(';')
        products = list(rule_row['Products'])[0].split(';')
        lhs_non_any = sum(1 for s in reactants if s.strip() != 'Any')
        rhs_non_any = sum(1 for s in products if s.strip() != 'Any')
        assert len(row['LHS_cofactors']) == lhs_non_any
        assert len(row['RHS_cofactors']) == rhs_non_any

        # finally, we check if the identifies of the cofactors match those in the JN rule definition
        lhs_cofactor_codes_in_rule = [s.strip() for s in reactants if s.strip() != 'Any']
        rhs_cofactor_codes_in_rule = [s.strip() for s in products if s.strip() != 'Any']

        assert set(row['LHS_cofactor_codes']) == set(lhs_cofactor_codes_in_rule)
        assert set(row['RHS_cofactor_codes']) == set(rhs_cofactor_codes_in_rule)

def test_all_MetaCyc_rxns_mapped(MetaCyc_df, JN_rules_df):
    mapped_rxns_df = MetaCyc_df[(MetaCyc_df['top_mapped_operator'].notna()) & (MetaCyc_df['top_mapped_operator']!='None')]

    assert mapped_rxns_df.shape[0] > 0

    for _, row in mapped_rxns_df.iterrows():

        # check that the substrates, products, LHS_cofactors, RHS_cofactors, LHS_cofactor_codes, RHS_cofactor_codes are all numpy arrays
        assert isinstance(row['substrates'], np.ndarray)
        assert isinstance(row['products'], np.ndarray)
        assert isinstance(row['LHS_cofactors'], np.ndarray)
        assert isinstance(row['RHS_cofactors'], np.ndarray)
        assert isinstance(row['LHS_cofactor_codes'], np.ndarray)
        assert isinstance(row['RHS_cofactor_codes'], np.ndarray)

        # test that the number of substrates and products aligns with the top mapped operator
        JN_mapped_rule = row['top_mapped_operator']
        rule_row = JN_rules_df[JN_rules_df['Name'] == JN_mapped_rule]
        rule_SMARTS = rule_row['SMARTS']
        
        # substrates and products are indicated by 'Any' in the JN rule definition
        assert len(row['substrates']) == list(rule_row['Reactants'])[0].split(';').count('Any')
        assert len(row['products']) == list(rule_row['Products'])[0].split(';').count('Any')

        # cofactors are also indicated by anything that is not 'Any' in the JN rule definition
        reactants = list(rule_row['Reactants'])[0].split(';')
        products = list(rule_row['Products'])[0].split(';')
        lhs_non_any = sum(1 for s in reactants if s.strip() != 'Any')
        rhs_non_any = sum(1 for s in products if s.strip() != 'Any')
        assert len(row['LHS_cofactors']) == lhs_non_any
        assert len(row['RHS_cofactors']) == rhs_non_any

        # finally, we check if the identifies of the cofactors match those in the JN rule definition
        lhs_cofactor_codes_in_rule = [s.strip() for s in reactants if s.strip() != 'Any']
        rhs_cofactor_codes_in_rule = [s.strip() for s in products if s.strip() != 'Any']

        assert set(row['LHS_cofactor_codes']) == set(lhs_cofactor_codes_in_rule)
        assert set(row['RHS_cofactor_codes']) == set(rhs_cofactor_codes_in_rule)

def test_all_BRENDA_rxns_mapped(BRENDA_df, JN_rules_df):
    mapped_rxns_df = BRENDA_df[(BRENDA_df['top_mapped_operator'].notna()) & (BRENDA_df['top_mapped_operator']!='None')]

    assert mapped_rxns_df.shape[0] > 0

    for _, row in mapped_rxns_df.iterrows():

        # check that the substrates, products, LHS_cofactors, RHS_cofactors, LHS_cofactor_codes, RHS_cofactor_codes are all numpy arrays
        assert isinstance(row['substrates'], np.ndarray)
        assert isinstance(row['products'], np.ndarray)
        assert isinstance(row['LHS_cofactors'], np.ndarray)
        assert isinstance(row['RHS_cofactors'], np.ndarray)
        assert isinstance(row['LHS_cofactor_codes'], np.ndarray)
        assert isinstance(row['RHS_cofactor_codes'], np.ndarray)

        # test that the number of substrates and products aligns with the top mapped operator
        JN_mapped_rule = row['top_mapped_operator']
        rule_row = JN_rules_df[JN_rules_df['Name'] == JN_mapped_rule]
        rule_SMARTS = rule_row['SMARTS']
        
        # substrates and products are indicated by 'Any' in the JN rule definition
        assert len(row['substrates']) == list(rule_row['Reactants'])[0].split(';').count('Any')
        assert len(row['products']) == list(rule_row['Products'])[0].split(';').count('Any')

        # cofactors are also indicated by anything that is not 'Any' in the JN rule definition
        reactants = list(rule_row['Reactants'])[0].split(';')
        products = list(rule_row['Products'])[0].split(';')
        lhs_non_any = sum(1 for s in reactants if s.strip() != 'Any')
        rhs_non_any = sum(1 for s in products if s.strip() != 'Any')
        assert len(row['LHS_cofactors']) == lhs_non_any
        assert len(row['RHS_cofactors']) == rhs_non_any

        # finally, we check if the identifies of the cofactors match those in the JN rule definition
        lhs_cofactor_codes_in_rule = [s.strip() for s in reactants if s.strip() != 'Any']
        rhs_cofactor_codes_in_rule = [s.strip() for s in products if s.strip() != 'Any']

        assert set(row['LHS_cofactor_codes']) == set(lhs_cofactor_codes_in_rule)
        assert set(row['RHS_cofactor_codes']) == set(rhs_cofactor_codes_in_rule)

# ensure unmapped reactions have 'None' in the top_mapped_operator column
def test_unmapped_reactions_have_none_top_mapped_operator(KEGG_df, MetaCyc_df):
    KEGG_unmapped_df = KEGG_df[KEGG_df['top_mapped_operator'].isna() | (KEGG_df['top_mapped_operator']=='None')]
    MetaCyc_unmapped_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'].isna() | (MetaCyc_df['top_mapped_operator']=='None')]

    assert KEGG_unmapped_df.shape[0] == 1522
    assert MetaCyc_unmapped_df.shape[0] == 542

# ensure that unmapped reactions in KEGG are truly unmapped (i.e., they do not fit any JN rule)
def test_all_KEGG_rxns_unmapped(KEGG_df, JN_rules_df):
    KEGG_unmapped_df = KEGG_df[KEGG_df['top_mapped_operator'].isna() | (KEGG_df['top_mapped_operator']=='None')]

    assert KEGG_unmapped_df.shape[0] > 0

    for _, row in KEGG_unmapped_df.iterrows():

        # check that the substrates, products, LHS_cofactors, RHS_cofactors, LHS_cofactor_codes, RHS_cofactor_codes are all numpy arrays
        assert isinstance(row['substrates'], np.ndarray)
        assert isinstance(row['products'], np.ndarray)
        assert isinstance(row['LHS_cofactors'], np.ndarray)
        assert isinstance(row['RHS_cofactors'], np.ndarray)
        assert isinstance(row['LHS_cofactor_codes'], np.ndarray)
        assert isinstance(row['RHS_cofactor_codes'], np.ndarray)

        KEGG_rxn_num_substrates = len(row['substrates'])
        KEGG_rxn_num_products = len(row['products'])
        KEGG_rxn_num_lhs_cofactors = len(row['LHS_cofactors'])
        KEGG_rxn_num_rhs_cofactors = len(row['RHS_cofactors'])
        KEGG_rxn_lhs_cofactor_codes = Counter(row['LHS_cofactor_codes'])
        KEGG_rxn_rhs_cofactor_codes = Counter(row['RHS_cofactor_codes'])

        # ensure that no JN rule matches this reaction's characteristics
        assert JN_rules_df[(JN_rules_df["num_substrates"] == KEGG_rxn_num_substrates)
                    & (JN_rules_df["num_products"] == KEGG_rxn_num_products)
                    & (JN_rules_df["num_lhs_cofactors"] == KEGG_rxn_num_lhs_cofactors)
                    & (JN_rules_df["num_rhs_cofactors"] == KEGG_rxn_num_rhs_cofactors)
                    & (JN_rules_df["lhs_cofactor_codes"].apply(lambda x: Counter(x) == KEGG_rxn_lhs_cofactor_codes))
                    & (JN_rules_df["rhs_cofactor_codes"].apply(lambda x: Counter(x) == KEGG_rxn_rhs_cofactor_codes))
                   ].shape[0] == 0
        
def test_all_MetaCyc_rxns_unmapped(MetaCyc_df, JN_rules_df):
    MetaCyc_unmapped_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'].isna() | (MetaCyc_df['top_mapped_operator']=='None')]

    assert MetaCyc_unmapped_df.shape[0] > 0

    for _, row in MetaCyc_unmapped_df.iterrows():

        # check that the substrates, products, LHS_cofactors, RHS_cofactors, LHS_cofactor_codes, RHS_cofactor_codes are all numpy arrays
        assert isinstance(row['substrates'], np.ndarray)
        assert isinstance(row['products'], np.ndarray)
        assert isinstance(row['LHS_cofactors'], np.ndarray)
        assert isinstance(row['RHS_cofactors'], np.ndarray)
        assert isinstance(row['LHS_cofactor_codes'], np.ndarray)
        assert isinstance(row['RHS_cofactor_codes'], np.ndarray)

        MetaCyc_rxn_num_substrates = len(row['substrates'])
        MetaCyc_rxn_num_products = len(row['products'])
        MetaCyc_rxn_num_lhs_cofactors = len(row['LHS_cofactors'])
        MetaCyc_rxn_num_rhs_cofactors = len(row['RHS_cofactors'])
        MetaCyc_rxn_lhs_cofactor_codes = Counter(row['LHS_cofactor_codes'])
        MetaCyc_rxn_rhs_cofactor_codes = Counter(row['RHS_cofactor_codes'])

        # ensure that no JN rule matches this reaction's characteristics
        assert JN_rules_df[(JN_rules_df["num_substrates"] == MetaCyc_rxn_num_substrates)
                    & (JN_rules_df["num_products"] == MetaCyc_rxn_num_products)
                    & (JN_rules_df["num_lhs_cofactors"] == MetaCyc_rxn_num_lhs_cofactors)
                    & (JN_rules_df["num_rhs_cofactors"] == MetaCyc_rxn_num_rhs_cofactors)
                    & (JN_rules_df["lhs_cofactor_codes"].apply(lambda x: Counter(x) == MetaCyc_rxn_lhs_cofactor_codes))
                    & (JN_rules_df["rhs_cofactor_codes"].apply(lambda x: Counter(x) == MetaCyc_rxn_rhs_cofactor_codes))
                   ].shape[0] == 0
