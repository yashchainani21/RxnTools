import pytest
import pandas as pd
import numpy as np

@pytest.fixture()
def KEGG_df():
    return pd.read_parquet("../data/processed/enzymemap_Kegg_JN_mapped_non_unique.parquet")

@pytest.fixture()
def MetaCyc_df():
    return pd.read_parquet("../data/processed/enzymemap_MetaCyc_JN_mapped_non_unique.parquet")

def test_processed_KEGG_rxns_not_empty(KEGG_df):
    assert KEGG_df.shape[0] == 7966 # confirmed against the original .csv file too
    assert 'top_mapped_operator' in KEGG_df.columns

def test_processed_MetaCyc_rxns_not_empty(MetaCyc_df):
    assert MetaCyc_df.shape[0] == 4580 # confirmed against the original .csv file too
    assert 'top_mapped_operator' in MetaCyc_df.columns

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

# test decarboxylase related rules (rule0023 & rule0024) for MetaCyc were mapped correctly
def test_processed_MetaCyc_rule0023_and_rule0024_rxns_count(MetaCyc_df):
    rule0023_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0023']
    rule0024_df = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0024']

    # there are no carboxylation and decarboxylation reactions in MetaCyc
    assert rule0023_df.shape[0] == 0
    assert rule0024_df.shape[0] == 0

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