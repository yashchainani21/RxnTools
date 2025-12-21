from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from itertools import permutations
from typing import List, Tuple

class ReactionTemplate:
    def __init__(self, rxn_smarts: str):
        self.rxn_smarts = rxn_smarts

    def RunReactantsChiral(self,
                           rule_ID: str,
                           rule_df: pd.DataFrame,
                           reactants_SMILES_list: List[str],
                           useChirality: bool = True):
        
        # using the rule_ID, get the reactant and product codes from rule_df
        # these codes describe the reactants, products, and cofactors involved in the reaction
        reactant_codes: str = rule_df[rule_df["Name"]==rule_ID]["Reactants"].values[0] # e.g., 'Any;NAD_CoF'
        product_codes: str = rule_df[rule_df["Name"]==rule_ID]["Products"].values[0] # e.g., 'Any;NADH_CoF'
        
        reactant_mols: List[Chem.Mol] = [Chem.MolFromSmiles(smiles) for smiles in reactants_SMILES_list]
        if None in reactant_mols:
            raise ValueError("One or more reactant SMILES could not be converted to RDKit Mol objects.")
        
        # generate all possible orderings of reactants
        all_reactant_combinations: List[Tuple[Chem.Mol, ...]] = list(permutations(reactant_mols, len(reactant_mols)))

        lhs_template, rhs_template = self.rxn_smarts.split('>>')
        lhs_templates_list: List[str] = lhs_template.split('.') # list like [Any_SMARTS, NAD_CoF_SMARTS]
        rhs_templates_list: List[str] = rhs_template.split('.') # list like [Any_SMARTS, NADH_CoF_SMARTS]

        reactant_codes_list: List[str] = reactant_codes.split(';') # list like ['Any', 'NAD_CoF']
        product_codes_list: List[str] = product_codes.split(';') # list like ['Any', 'NADH_CoF']

        # initialize a list to store the correct order in which reactants should be positioned based on their SMARTS
        # determining the correct order of reactants is necessary because RDKit's RunReactants method is order-dependent
        correct_reactant_combinations: List[Tuple[Chem.Mol, ...]] = []