from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from itertools import permutations
from typing import List, Tuple
from .utils import are_isomorphic

class ReactionTemplate:
    """
    Parameters
    ----------
    rxn_smarts : str
        SMARTS template of the form "A.B >> C.D" describing a chemical reaction.
    """
    def __init__(self, rxn_SMARTS: str):
        self.rxn_SMARTS = rxn_SMARTS

    def RunReactantsChiral(self,
                           rule_ID: str,
                           rule_df: pd.DataFrame,
                           cofactors_df: pd.DataFrame,
                           reactants_SMILES_list: List[str],
                           useChirality: bool = True) -> List[Tuple[Chem.Mol, ...]]:

        """Run the reaction template on the provided reactants.

        Parameters
        ----------
        self

        rule_ID : str
            Identifier for the reaction rule.

        rule_df : pd.DataFrame
            DataFrame containing reaction rules with reactant and product codes.

        reactants_SMILES_list : List[str]
            List of SMILES strings representing the reactants.

        useChirality : bool, optional
            Whether to consider chirality in substructure matching (default is True).

        Returns
        -------
        List[Tuple[Chem.Mol, ...]]
            List of tuples, each containing RDKit Mol objects representing the products formed.
        """
        # using the rule_ID, get the reactant and product codes from rule_df
        # these codes describe the reactants, products, and cofactors involved in the reaction
        reactant_codes: str = rule_df[rule_df["Name"]==rule_ID]["Reactants"].values[0] # e.g., 'Any;NAD_CoF'
        product_codes: str = rule_df[rule_df["Name"]==rule_ID]["Products"].values[0] # e.g., 'Any;NADH_CoF'
        
        reactant_mols: List[Chem.Mol] = [Chem.MolFromSmiles(smiles) for smiles in reactants_SMILES_list]
        if None in reactant_mols:
            raise ValueError("One or more reactant SMILES could not be converted to RDKit Mol objects.")
        
        # generate all possible orderings of reactants
        all_reactant_combinations: List[Tuple[Chem.Mol, ...]] = list(permutations(reactant_mols, len(reactant_mols)))

        lhs_template, rhs_template = self.rxn_SMARTS.split('>>')
        lhs_templates_list: List[str] = lhs_template.split('.') # list like [Any_SMARTS, NAD_CoF_SMARTS]
        rhs_templates_list: List[str] = rhs_template.split('.') # list like [Any_SMARTS, NADH_CoF_SMARTS]

        reactant_codes_list: List[str] = reactant_codes.split(';') # list like ['Any', 'NAD_CoF']
        product_codes_list: List[str] = product_codes.split(';') # list like ['Any', 'NADH_CoF']

        # initialize a list to store the correct order in which reactants should be positioned based on their SMARTS
        # determining the correct order of reactants is necessary because RDKit's RunReactants method is order-dependent
        correct_reactant_combinations: List[Tuple[Chem.Mol, ...]] = []

        # for each combination of reactant positionings, check if the reactants match their corresponding SMARTS templates
        for combination in all_reactant_combinations:

            num_matches = 0

            for (reactant_mol, lhs_template_SMARTS) in zip(combination, lhs_templates_list):
                if reactant_mol.HasSubstructMatch(Chem.MolFromSmarts(lhs_template_SMARTS, useChirality=useChirality)):
                    num_matches += 1

                if num_matches == len(lhs_templates_list):
                    correct_reactant_combinations.append(combination)
        
        # now, for each correct reactant combination, run the reaction and collect products
        all_products: List[Tuple[Chem.Mol, ...]] = []

        for correct_combination in correct_reactant_combinations:
            rxn = AllChem.ReactionFromSmarts(self.rxn_SMARTS)
            products: Tuple[Tuple[Chem.Mol, ...], ...] = rxn.RunReactants(correct_combination)
            for product_set in products:
                all_products.append(product_set)

        # initialize a list to store the correct order in which products should be positioned based on their SMARTS
        correct_product_combinations: List[Tuple[Chem.Mol, ...]] = []

        # for each combination of product positionings, check if the products match their corresponding SMARTS templates
        for product_combination in all_products:

            num_matches = 0

            for (product_mol, rhs_template_SMARTS, product_code) in zip(product_combination, rhs_templates_list, product_codes_list):

                # first, check to see if the product mol object fits its corresponding SMARTS template
                if product_mol.HasSubstructMatch(Chem.MolFromSmarts(rhs_template_SMARTS), useChirality=useChirality):

                    # then, if the product is a cofactor (i.e., not 'Any'), check to see that the correct cofactor has been produced
                    if product_code != "Any":
                        if are_isomorphic(product_mol,
                                          Chem.MolFromSmiles(cofactors_df[cofactors_df["#ID"]==product_code]["SMILES"].to_list()[0])):
                            num_matches += 1
                        else:
                            continue

                    # but if the product is not a cofactor (i.e., 'Any') and its structure fits the corresponding SMARTS template, count it as a match
                    elif product_code == "Any":
                        num_matches += 1

                    else:
                        continue

            if num_matches == len(rhs_templates_list):
                correct_product_combinations.append(product_combination)
        return correct_product_combinations
