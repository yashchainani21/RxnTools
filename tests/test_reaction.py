import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import pandas as pd
from rxntools import reaction

@pytest.fixture
def cofactors_list():
    """
    Pytest fixture to load the cofactors list from a JSON file.
    """
    with open('../data/cofactors.json') as f:
        cofactors_dict = json.load(f)
    return [cofactors_dict[key] for key in cofactors_dict.keys()]

@pytest.fixture
def cofactors_df():
    """
    Pytest fixture to load the cofactors dataframe from a tsv file.
    """
    with open('../data/all_cofactors.csv') as f:
        cofactors_df = pd.read_csv(f, sep=',')
    return cofactors_df

#### ----------------------- Tests for the unmapped reaction class -----------------------

# tests involving an alcohol dehydrogenase that oxidizes ethanol to ethanal
# EC 1.1.1.1, MetaCyc rxn idx 903 (ALCOHOL-DEHYDROG-RXN)
ethanol_AdH_rxn_str = "CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_01():
    rxn = reaction.unmapped_reaction(ethanol_AdH_rxn_str)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "CCO.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    assert products_str == "CC=O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]"

def test_extracting_substrates_from_unmapped_rxn_str_01(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanol_AdH_rxn_str)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["CCO"]

def test_extracting_products_from_unmapped_rxn_str_01(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanol_AdH_rxn_str)
    assert rxn.get_products(cofactors_list=cofactors_list,
                              consider_stereo=False) == ["CC=O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_01(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanol_AdH_rxn_str)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list, consider_stereo = False) == ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_01(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanol_AdH_rxn_str)
    assert rxn.get_rhs_cofactors(cofactors_list=cofactors_list, consider_stereo=False) == ["NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1","[H+]"]

def test_get_JN_rxn_descriptor_01(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = ethanol_AdH_rxn_str)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','NAD_CoF']
    assert RHS_descriptor == ['Any','NADH_CoF']

# tests involving an alcohol dehydrogenase that oxidizes cis-4-hydroxyproline to the corresponding aldehyde
# EC 1.1.1.104, MetaCyc rxn idx 693 (4-OXOPROLINE-REDUCTASE-RXN)
hydroxyproline_AdH_rxn_str = "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])[C@@H]1C[C@H](O)C[NH2+]1>>NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C1C[NH2+][C@H](C(=O)[O-])C1.[H+]"
def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_02():
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=C([O-])[C@@H]1C[C@H](O)C[NH2+]1"
    assert products_str == "NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.O=C1C[NH2+][C@H](C(=O)[O-])C1.[H+]"

def test_extracting_substrates_from_unmapped_rxn_str_02(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["O=C(O)[C@@H]1C[C@H](O)CN1"]

def test_extracting_products_from_unmapped_rxn_str_02(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)
    assert rxn.get_products(cofactors_list=cofactors_list, consider_stereo=False) == ["O=C1CN[C@H](C(=O)O)C1"]

def extracting_LHS_cofactors_from_unmapped_rxn_str_02(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)
    assert rxn.get_lhs_cofactors(cofactors_list=cofactors_list, consider_stereo=False) == ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"]

def extracting_RHS_cofactors_from_unmapped_rxn_str_02(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)
    assert rxn.get_rhs_cofactors(cofactors_list=cofactors_list, consider_stereo=False) == ["NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1","[H+]"]

def test_get_JN_rxn_descriptor_02(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = hydroxyproline_AdH_rxn_str)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','NAD_CoF',]
    assert RHS_descriptor == ['Any', 'NADH_CoF',]

# tests involving an alcohol dehydrogenase that oxidizes mannofuranose to the corresponding aldehyde
# EC 1.1.1.173, MetaCyc rxn idx 1652 (L-RHAMNOSE-1-DEHYDROGENASE-RXN)
mannofuranose_AdH_rxn_str = "C[C@H](O)[C@@H]1OC(O)[C@H](O)[C@@H]1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1>>C[C@H](O)[C@@H]1OC(=O)[C@H](O)[C@@H]1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]"
def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_03():
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "C[C@H](O)[C@@H]1OC(O)[C@H](O)[C@@H]1O.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"
    assert products_str == "C[C@H](O)[C@@H]1OC(=O)[C@H](O)[C@@H]1O.NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1.[H+]"

def test_extracting_substrates_from_unmapped_rxn_str_03(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)
    assert rxn.get_substrates(cofactors_list = cofactors_list, consider_stereo = False) == ["C[C@H](O)[C@@H]1OC(O)[C@H](O)[C@@H]1O"]

def test_extracting_products_from_unmapped_rxn_str_03(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)
    assert rxn.get_products(cofactors_list=cofactors_list, consider_stereo=False) == ["C[C@H](O)[C@@H]1OC(=O)[C@H](O)[C@@H]1O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_03(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)
    assert rxn.get_lhs_cofactors(cofactors_list=cofactors_list, consider_stereo=False) == ["NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)c1"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_03(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1","[H+]"]

def test_get_JN_rxn_descriptor_03(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = mannofuranose_AdH_rxn_str)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','NAD_CoF',]
    assert RHS_descriptor == ['Any', 'NADH_CoF',]

# tests involving the polyneuridine-aldehyde esterase reaction
# EC 3.1.1.78, MetaCyc rxn idx 583 (3.1.1.78-RXN)
esterase_rxn_str = "CC=C1CN2[C@H]3Cc4c([nH]c5ccccc45)[C@@H]2C[C@H]1[C@@]3(C=O)C(=O)OC.O>>CC=C1CN2[C@H]3C[C@H]1[C@H](C=O)[C@@H]2Cc1c3[nH]c2ccccc12.CO.O=C=O"
def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_04():
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "CC=C1CN2[C@H]3Cc4c([nH]c5ccccc45)[C@@H]2C[C@H]1[C@@]3(C=O)C(=O)OC.O"
    assert products_str == "CC=C1CN2[C@H]3C[C@H]1[C@H](C=O)[C@@H]2Cc1c3[nH]c2ccccc12.CO.O=C=O"

def test_extracting_substrates_from_unmapped_rxn_str_04(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)
    assert rxn.get_substrates(cofactors_list = cofactors_list, consider_stereo = False) == ["CC=C1CN2[C@H]3Cc4c([nH]c5ccccc45)[C@@H]2C[C@H]1[C@@]3(C=O)C(=O)OC"]

def test_extracting_products_from_unmapped_rxn_str_04(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)
    assert rxn.get_products(cofactors_list=cofactors_list, consider_stereo=False) == ["CC=C1CN2[C@H]3C[C@H]1[C@H](C=O)[C@@H]2Cc1c3[nH]c2ccccc12","CO"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_04(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_04(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O=C=O"]

def test_get_JN_rxn_descriptor_04(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = esterase_rxn_str)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','WATER',]
    assert RHS_descriptor == ['Any', 'Any', 'CO2']

# tests involving the glucopyranose-phosphatase reaction
# EC 3.1.3.10, MetaCyc rxn idx 1414 (GLUCOSE-1-PHOSPHAT-RXN)
glucopyranose_phosphatase_rxn = "O.O=P([O-])([O-])O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O>>O=P([O-])([O-])O.OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_05():
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "O.O=P([O-])([O-])O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"
    assert products_str == "O=P([O-])([O-])O.OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"

def test_extracting_substrates_from_unmapped_rxn_str_05(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)
    assert rxn.get_substrates(cofactors_list = cofactors_list, consider_stereo = False) == ["O=P(O)(O)O[C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O"]

def test_extracting_products_from_unmapped_rxn_str_05(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)
    assert rxn.get_products(cofactors_list=cofactors_list, consider_stereo=False) == ["OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_05(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_05(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O=P(O)(O)O"]

def test_get_JN_rxn_descriptor_05(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = glucopyranose_phosphatase_rxn)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','WATER',]
    assert RHS_descriptor == ['Any', 'Pi']

# tests involving the ethanolamine-phosphate phosphatase reaction
# EC 4.2.3.2, MetaCyc rxn idx 1309 (ETHANOLAMINE-PHOSPHATE-PHOSPHO-LYASE-RXN)
ethanolamine_phosphate_phosphatase_rxn = "O.[NH3+]CCOP(=O)([O-])[O-]>>CC=O.O=P([O-])([O-])O.[NH4+]"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_06():
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "O.[NH3+]CCOP(=O)([O-])[O-]"
    assert products_str == "CC=O.O=P([O-])([O-])O.[NH4+]"

def test_extracting_substrates_from_unmapped_rxn_str_06(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["NCCOP(=O)(O)O"]

def test_extracting_products_from_unmapped_rxn_str_06(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)
    assert rxn.get_products(cofactors_list = cofactors_list,
                            consider_stereo=False) == ["CC=O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_06(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_06(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["O=P(O)(O)O","N"]

def test_get_JN_rxn_descriptor_06(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = ethanolamine_phosphate_phosphatase_rxn)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any','WATER',]
    assert RHS_descriptor == ['Any', 'Pi', 'NH3']

# tests involving an epimerization reaction
# EC 5.1.3.38, MetaCyc rxn idx 4160 (RXN-17771)
epimerization_rxn = "O=C(COP(=O)([O-])[O-])[C@H](O)CO>>O=C(COP(=O)([O-])[O-])[C@@H](O)CO"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_07():
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "O=C(COP(=O)([O-])[O-])[C@H](O)CO"
    assert products_str == "O=C(COP(=O)([O-])[O-])[C@@H](O)CO"

def test_extracting_substrates_from_unmapped_rxn_str_07(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["O=C(COP(=O)(O)O)[C@H](O)CO"]

def test_extracting_products_from_unmapped_rxn_str_07(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)
    assert rxn.get_products(cofactors_list = cofactors_list,
                            consider_stereo=False) == ["O=C(COP(=O)(O)O)[C@@H](O)CO"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_07(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == []

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_07(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == []

def test_get_JN_rxn_descriptor_07(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = epimerization_rxn)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any']
    assert RHS_descriptor == ['Any']

# tests involving the succoasyn reaction
# EC 6.2.1.5, MetaCyc rxn idx 6847 (SUCCCOASYN-RXN)
succoasyn_rxn = "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCS.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=C([O-])CCC(=O)[O-]>>CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)[O-].Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])O"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_08():
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCS.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=C([O-])CCC(=O)[O-]"
    assert products_str == "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)[O-].Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])O"

def test_extracting_substrates_from_unmapped_rxn_str_08(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["O=C(O)CCC(=O)O"]

def test_extracting_products_from_unmapped_rxn_str_08(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)
    assert rxn.get_products(cofactors_list = cofactors_list,
                            consider_stereo=False) == ["CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCC(=O)O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_08(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",
                                                              "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_08(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                                                              "O=P(O)(O)O"]

def test_get_JN_rxn_descriptor_08(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = succoasyn_rxn)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any', 'CoA', 'PYROPHOSPHATE_DONOR_CoF']
    assert RHS_descriptor == ['Any', 'PHOSPHATE_ACCEPTOR_CoF', 'Pi']

# tests involving the biotin coa ligase reaction
# EC 6.2.1.11, MetaCyc rxn idx 1030 (BIOTIN--COA-LIGASE-RXN)
biotin_coa_ligase_rxn = "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCS.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=C([O-])CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21>>CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])OP(=O)([O-])O"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_09():
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCS.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=C([O-])CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21"
    assert products_str == "CC(C)(COP(=O)([O-])OP(=O)([O-])OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)([O-])[O-])[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])OP(=O)([O-])O"

def test_extracting_substrates_from_unmapped_rxn_str_09(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["O=C(O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21"]

def test_extracting_products_from_unmapped_rxn_str_09(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)
    assert rxn.get_products(cofactors_list = cofactors_list,
                            consider_stereo=False) == ["CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCSC(=O)CCCC[C@@H]1SC[C@@H]2NC(=O)N[C@@H]21"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_09(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                             consider_stereo = False) == ["CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS",
                                                          "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_09(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo = False) == ["Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)O)[C@@H](O)[C@H]1O",
                                                              "O=P(O)(O)OP(=O)(O)O"]

def test_get_JN_rxn_descriptor_09(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = biotin_coa_ligase_rxn)

    LHS_descriptor, RHS_descriptor = rxn.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
                                                               consider_stereo = False)

    assert LHS_descriptor == ['Any', 'CoA', 'PYROPHOSPHATE_DONOR_CoF']
    assert RHS_descriptor == ['Any', 'PYROPHOSPHATE_ACCEPTOR_CoF', 'PPI']

# tests involving RXN-14637
# EC 6.3.1.18, MetaCyc rxn idx 3419 (RXN-14637)
rxn_14637 = "Nc1ccccc1.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]>>Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])O.[NH3+][C@@H](CCC(=O)Nc1ccccc1)C(=O)[O-]"

def test_separating_unmapped_rxn_str_into_reactant_and_product_strs_10(cofactors_df):
    rxn = reaction.unmapped_reaction(rxn_str = rxn_14637)
    reactants_str, products_str = rxn._rxn_2_cpds()
    assert reactants_str == "Nc1ccccc1.Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.[NH3+][C@@H](CCC(=O)[O-])C(=O)[O-]"
    assert products_str == "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])[C@@H](O)[C@H]1O.O=P([O-])([O-])O.[NH3+][C@@H](CCC(=O)Nc1ccccc1)C(=O)[O-]"

def test_extracting_substrates_from_unmapped_rxn_str_10(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = rxn_14637)
    assert rxn.get_substrates(cofactors_list = cofactors_list,
                              consider_stereo = False) == ["Nc1ccccc1"]

def test_extracting_products_from_unmapped_rxn_str_10(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = rxn_14637)
    assert rxn.get_products(cofactors_list = cofactors_list,
                            consider_stereo=False) == ["N[C@@H](CCC(=O)Nc1ccccc1)C(=O)O"]

def test_extracting_LHS_cofactors_from_unmapped_rxn_str_10(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = rxn_14637)
    assert rxn.get_lhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo=False) == ["Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                                                            "N[C@@H](CCC(=O)O)C(=O)O"]

def test_extracting_RHS_cofactors_from_unmapped_rxn_str_10(cofactors_list):
    rxn = reaction.unmapped_reaction(rxn_str = rxn_14637)
    assert rxn.get_rhs_cofactors(cofactors_list = cofactors_list,
                                 consider_stereo=False) == ["Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O",
                                                            "O=P(O)(O)O"]


#### ----------------------- Tests for the mapped reaction class -----------------------

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
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for an alcohol dehydrogenase transforming ethanol to ethanal
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')

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

def test_get_all_changed_atoms_frm_nitrilase_rxn_SMARTS():
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for a nitrilase transforming mandelonitrile to benzaldehyde
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn._get_all_changed_atoms()

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [2, 3, 4]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    # ensure that the C-C bond between the cyanide group and the alcohol carbon is broken
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in broken_bonds

    # ensure that the C-O bond within the alcohol carbon is broken and reformed as C=O
    assert (3, 4, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (3, 4, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_decarb_rxn_SMARTS():
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@H:20]3[CH:21]=[O:22].[OH:25][CH3:26]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn._get_all_changed_atoms()

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [20, 23, 25, 27]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    assert (20, 23, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (23, 25, Chem.rdchem.BondType.SINGLE) in broken_bonds

    assert (23, 27, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_ethanol_AdH_rxn_SMARTS_including_cofactors(cofactors_list,
                                                                              consider_stereo = True,
                                                                              include_cofactors = True):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for an alcohol dehydrogenase transforming ethanol to ethanal
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = include_cofactors,
                                                                                 consider_stereo = consider_stereo,
                                                                                 cofactors_list = cofactors_list)

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

def test_get_all_changed_atoms_frm_ethanol_AdH_rxn_SMARTS_excluding_cofactors(cofactors_list,
                                                                              consider_stereo = True,
                                                                              include_cofactors = False):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for an alcohol dehydrogenase transforming ethanol to ethanal
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts='[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = include_cofactors,
                                                                                 consider_stereo = consider_stereo,
                                                                                 cofactors_list = cofactors_list)

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [2, 3]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    # ensure that a C-O single bond has been broken within ethanol (CCO)
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in broken_bonds

    # ensure that a C=O double bond has been formed within ethanal (CC=O)
    assert (2, 3, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_nitrilase_rxn_SMARTS_including_cofactors(cofactors_list,
                                                                            consider_stereo = True,
                                                                            include_cofactors = True):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for a nitrilase transforming mandelonitrile to benzaldehyde
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = include_cofactors,
                                                                                 consider_stereo = consider_stereo,
                                                                                 cofactors_list = cofactors_list)

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [2, 3, 4]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    # ensure that the C-C bond between the cyanide group and the alcohol carbon is broken
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in broken_bonds

    # ensure that the C-O bond within the alcohol carbon is broken and reformed as C=O
    assert (3, 4, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (3, 4, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_nitrilase_rxn_SMARTS_excluding_cofactors(cofactors_list,
                                                                            consider_stereo = True,
                                                                            include_cofactors = False):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    # fully atom mapped reaction SMARTS for a nitrilase transforming mandelonitrile to benzaldehyde
    # reaction SMARTS were obtained from the MetaCyc database released by the EnzymeMap paper
    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = include_cofactors,
                                                                                 consider_stereo = consider_stereo,
                                                                                 cofactors_list = cofactors_list)

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [2, 3, 4]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    # ensure that the C-C bond between the cyanide group and the alcohol carbon is broken
    assert (2, 3, Chem.rdchem.BondType.SINGLE) in broken_bonds

    # ensure that the C-O bond within the alcohol carbon is broken and reformed as C=O
    assert (3, 4, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (3, 4, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_decarb_rxn_SMARTS_including_cofactors(cofactors_list,
                                                                         consider_stereo = True,
                                                                         include_cofactors = True):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@H:20]3[CH:21]=[O:22].[OH:25][CH3:26]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(cofactors_list = cofactors_list,
                                                                                 consider_stereo = consider_stereo,
                                                                                 include_cofactors = include_cofactors)

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [20, 23, 25, 27]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    assert (20, 23, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (23, 25, Chem.rdchem.BondType.SINGLE) in broken_bonds

    assert (23, 27, Chem.rdchem.BondType.DOUBLE) in formed_bonds

def test_get_all_changed_atoms_frm_decarb_rxn_SMARTS_excluding_cofactors(cofactors_list,
                                                                         consider_stereo = True,
                                                                         include_cofactors = False):
    """
    Test to ensure ._get_all_changed_atoms returns the atom indices of all transformed atoms
    This should be a set of atom indices.
    The bonds broken and bonds formed throughout should also be returned as a set of tuples.
    Within the set of bonds broken and formed, each tuple should have three elements.
    First two elements are integers indicating start & end atom indices of the transformed bond
    Meanwhile, the third element is a Chem.rdchem.BondType bond type
    """

    mapped_rxn = reaction.mapped_reaction(rxn_smarts = '[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@:20]3([CH:21]=[O:22])[C:23](=[O:24])[O:25][CH3:26].[OH2:27]>>[C:23](=[O:24])=[O:27].[CH3:1][CH:2]=[C:3]1[CH2:4][N:5]2[C@H:6]3[CH2:7][c:8]4[c:9]([nH:10][c:11]5[cH:12][cH:13][cH:14][cH:15][c:16]45)[C@@H:17]2[CH2:18][C@H:19]1[C@@H:20]3[CH:21]=[O:22].[OH:25][CH3:26]')

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(cofactors_list = cofactors_list,
                                                                                 consider_stereo = consider_stereo,
                                                                                 include_cofactors = include_cofactors)

    # check first if the data types returned are as expected
    assert isinstance(changed_atoms, set)
    assert isinstance(broken_bonds, set)
    assert isinstance(formed_bonds, set)

    # ensure correct atom indices corresponding to transformed atoms are in changed_atoms
    expected_atoms_changed = [20, 25]
    assert all(item in changed_atoms for item in expected_atoms_changed)

    assert (20, 23, Chem.rdchem.BondType.SINGLE) in broken_bonds
    assert (23, 25, Chem.rdchem.BondType.SINGLE) in broken_bonds

def test_extract_rxn_template_radius1_frm_ethanol_wo_stereo_wo_cofactors(cofactors_list,
                                                                         radius = 1,
                                                                         include_stereo = False):

    atom_mapped_ethanol_AdH_rxn_smarts = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    AllChem.ReactionFromSmarts(atom_mapped_ethanol_AdH_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_ethanol_AdH_rxn_smarts)
    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = False,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    # check to ensure that only the C2 and O3 atoms are changed since the C2-O3 bond is oxidized
    assert changed_atoms == {2, 3}

    template = mapped_rxn.get_template_around_rxn_site(atom_mapped_substrate_smarts='[CH3:1][CH2:2][OH:3]',
                                                       reactive_atom_indices=list(changed_atoms),
                                                       radius = radius,
                                                       include_stereo = include_stereo)

    assert template == '[C&H3:1][C&H2:2][O&H1:3]'

def test_extract_rxn_template_radius2_frm_ethanol_wo_stereo_wo_cofactors(cofactors_list,
                                                                         radius = 2,
                                                                         include_stereo = False):

    atom_mapped_ethanol_AdH_rxn_smarts = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    AllChem.ReactionFromSmarts(atom_mapped_ethanol_AdH_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_ethanol_AdH_rxn_smarts)
    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = False,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    # check to ensure that only the C2 and O3 atoms are changed since the C2-O3 bond is oxidized
    assert changed_atoms == {2, 3}

    template = mapped_rxn.get_template_around_rxn_site(atom_mapped_substrate_smarts='[CH3:1][CH2:2][OH:3]',
                                                       reactive_atom_indices=list(changed_atoms),
                                                       radius = radius,
                                                       include_stereo = include_stereo)

    assert template == '[C&H3:1][C&H2:2][O&H1:3]'

def test_extract_rxn_template_radius3_frm_ethanol_wo_stereo_wo_cofactors(cofactors_list,
                                                                         radius = 3,
                                                                         include_stereo = False):

    atom_mapped_ethanol_AdH_rxn_smarts = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    AllChem.ReactionFromSmarts(atom_mapped_ethanol_AdH_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_ethanol_AdH_rxn_smarts)
    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = False,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    # check to ensure that only the C2 and O3 atoms are changed since the C2-O3 bond is oxidized
    assert changed_atoms == {2, 3}

    template = mapped_rxn.get_template_around_rxn_site(atom_mapped_substrate_smarts='[CH3:1][CH2:2][OH:3]',
                                                       reactive_atom_indices=list(changed_atoms),
                                                       radius = radius,
                                                       include_stereo = include_stereo)

    assert template == ''

def test_extract_rxn_template_radius4_frm_ethanol_wo_stereo_wo_cofactors(cofactors_list,
                                                                         radius = 4,
                                                                         include_stereo = False):

    atom_mapped_ethanol_AdH_rxn_smarts = '[CH3:1][CH2:2][OH:3].[NH2:4][C:5](=[O:6])[c:7]1[cH:8][cH:9][cH:10][n+:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[cH:47]1>>[CH3:1][CH:2]=[O:3].[H+].[NH2:4][C:5](=[O:6])[C:7]1=[CH:47][N:11]([C@@H:12]2[O:13][C@H:14]([CH2:15][O:16][P:17](=[O:18])([OH:19])[O:20][P:21](=[O:22])([OH:23])[O:24][CH2:25][C@H:26]3[O:27][C@@H:28]([n:29]4[cH:30][n:31][c:32]5[c:33]([NH2:34])[n:35][cH:36][n:37][c:38]45)[C@H:39]([OH:40])[C@@H:41]3[OH:42])[C@@H:43]([OH:44])[C@H:45]2[OH:46])[CH:10]=[CH:9][CH2:8]1'
    AllChem.ReactionFromSmarts(atom_mapped_ethanol_AdH_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_ethanol_AdH_rxn_smarts)
    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors = False,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    # check to ensure that only the C2 and O3 atoms are changed since the C2-O3 bond is oxidized
    assert changed_atoms == {2, 3}

    template = mapped_rxn.get_template_around_rxn_site(atom_mapped_substrate_smarts='[CH3:1][CH2:2][OH:3]',
                                                       reactive_atom_indices=list(changed_atoms),
                                                       radius = radius,
                                                       include_stereo = include_stereo)

    assert template == ''

def test_extract_rxn_template_radius1_frm_mandelonitrile_w_stereo_wo_cofactors(cofactors_list,
                                                                               radius = 1,
                                                                               include_stereo = True):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2,3,4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius1_w_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius = radius,
        include_stereo = include_stereo)

    assert template_radius1_w_stereo == '[N:1]#[C:2][C@&H1:3]([O&H1:4])[c:5]'

def test_extract_rxn_template_radius1_frm_mandelonitrile_wo_stereo_wo_cofactors(cofactors_list,
                                                                               radius = 1,
                                                                               include_stereo = False):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2,3,4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius1_wo_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius = radius,
        include_stereo = include_stereo)

    assert template_radius1_wo_stereo == '[N:1]#[C:2][C&H1:3]([O&H1:4])[c:5]'

def test_extract_rxn_template_radius2_frm_mandelonitrile_w_stereo_wo_cofactors(cofactors_list,
                                                                               radius = 2,
                                                                               include_stereo = True):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2,3,4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius2_w_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius = radius,
        include_stereo = include_stereo)

    assert template_radius2_w_stereo == '[N:1]#[C:2][C@&H1:3]([O&H1:4])[c:5]([c&H1:6])[c&H1:10]'

def test_extract_rxn_template_radius2_frm_mandelonitrile_wo_stereo_wo_cofactors(cofactors_list,
                                                                               radius = 2,
                                                                               include_stereo = False):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2,3,4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius2_wo_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius = radius,
        include_stereo = include_stereo)

    assert template_radius2_wo_stereo == '[N:1]#[C:2][C&H1:3]([O&H1:4])[c:5]([c&H1:6])[c&H1:10]'


def test_extract_rxn_template_radius3_frm_mandelonitrile_w_stereo_wo_cofactors(cofactors_list,
                                                                               radius=3,
                                                                               include_stereo=True):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2, 3, 4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius3_w_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius=radius,
        include_stereo=include_stereo)

    assert template_radius3_w_stereo == '[N:1]#[C:2][C@&H1:3]([O&H1:4])[c:5]([c&H1:6][c&H1:7])[c&H1:10][c&H1:9]'


def test_extract_rxn_template_radius3_frm_mandelonitrile_wo_stereo_wo_cofactors(cofactors_list,
                                                                                radius=3,
                                                                                include_stereo=False):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2, 3, 4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius3_wo_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius=radius,
        include_stereo=include_stereo)

    assert template_radius3_wo_stereo == '[N:1]#[C:2][C&H1:3]([O&H1:4])[c:5]([c&H1:6][c&H1:7])[c&H1:10][c&H1:9]'

def test_extract_rxn_template_radius4_frm_mandelonitrile_w_stereo_wo_cofactors(cofactors_list,
                                                                               radius=4,
                                                                               include_stereo=True):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2, 3, 4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius4_w_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius=radius,
        include_stereo=include_stereo)

    assert template_radius4_w_stereo == '[N:1]#[C:2][C@&H1:3]([O&H1:4])[c:5]1[c&H1:6][c&H1:7][c&H1:8][c&H1:9][c&H1:10]1'


def test_extract_rxn_template_radius4_frm_mandelonitrile_wo_stereo_wo_cofactors(cofactors_list,
                                                                                radius=4,
                                                                                include_stereo=False):
    atom_mapped_nitrilase_rxn_smarts = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1>>[CH:3](=[O:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1.[N:1]#[CH:2]'
    AllChem.ReactionFromSmarts(atom_mapped_nitrilase_rxn_smarts)

    mapped_rxn = reaction.mapped_reaction(rxn_smarts=atom_mapped_nitrilase_rxn_smarts)

    changed_atoms, broken_bonds, formed_bonds = mapped_rxn.get_all_changed_atoms(include_cofactors=True,
                                                                                 consider_stereo=True,
                                                                                 cofactors_list=cofactors_list)

    assert changed_atoms == {2, 3, 4}

    mandelonitrile_atom_mapped_SMARTS = '[N:1]#[C:2][C@H:3]([OH:4])[c:5]1[cH:6][cH:7][cH:8][cH:9][cH:10]1'

    template_radius4_wo_stereo = mapped_rxn.get_template_around_rxn_site(
        atom_mapped_substrate_smarts=mandelonitrile_atom_mapped_SMARTS,
        reactive_atom_indices=list(changed_atoms),
        radius=radius,
        include_stereo=include_stereo)

    assert template_radius4_wo_stereo == '[N:1]#[C:2][C&H1:3]([O&H1:4])[c:5]1[c&H1:6][c&H1:7][c&H1:8][c&H1:9][c&H1:10]1'
