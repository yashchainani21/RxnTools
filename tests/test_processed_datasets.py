import pytest
import pandas as pd

def test_processed_KEGG_rxns_not_empty():
    KEGG_df = pd.read_parquet("../data/processed/enzymemap_Kegg_JN_mapped_non_unique.parquet")
    assert KEGG_df.shape[0] > 0
    assert 'top_mapped_operator' in KEGG_df.columns

