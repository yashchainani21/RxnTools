from rdkit import Chem
from rdkit.Chem import AllChem

class ReactionTemplate:
    def __init__(self, smarts: str):
        self.smarts = smarts
        
    def RunReactantsChiral(self,
                           reactants_SMILES_list: List[str],)