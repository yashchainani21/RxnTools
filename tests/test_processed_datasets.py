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
    assert KEGG_df.shape[0] == 7966 # c
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

def test_processed_KEGG_rule0002_and_rule0003_rxns_count(KEGG_df):
    rule0002_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0002']
    rule0003_df = KEGG_df[KEGG_df['top_mapped_operator'] == 'rule0003']
    rule_0002_count = rule0002_df.shape[0]
    rule_0003_count = rule0003_df.shape[0]
    assert rule_0002_count == 704
    assert rule_0003_count == 125

    # the substrates column should contain a list with exactly on element each
    assert rule0002_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()
    assert rule0003_df['substrates'].apply(lambda x: isinstance(x, np.ndarray) and len(x) == 1).all()


def test_processed_MetaCyc_rule0002_and_rule0003_rxns_count():
    MetaCyc_df = pd.read_parquet("../data/processed/enzymemap_MetaCyc_JN_mapped_non_unique.parquet")
    rule_0002_count = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0002'].shape[0]
    rule_0003_count = MetaCyc_df[MetaCyc_df['top_mapped_operator'] == 'rule0003'].shape[0]
    assert rule_0002_count == 404
    assert rule_0003_count == 389