from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from typing import Set, Tuple, List
from .utils import get_cofactor_CoF_code, is_cofactor, remove_stereo, neutralize_atoms, canonicalize_smiles, does_template_fit

class unmapped_reaction:
    """
    Parameters
    ----------
    rxn_str : str
        Unmapped reaction string of the form "A + B = C + D" or "A.B >> C.D"
    """
    def __init__(self, rxn_str: str):
        self.rxn_str = rxn_str

    def _rxn_2_cpds(self) -> Tuple[str, str]:
        """
        Parse a reaction string to return two lists: a reactants list and a products list.

        Parameters
        ----------
        self

        Returns
        -------
        reactants_str : str
            A string representation of all reactants involved in the reaction, e.g. "A + B" or "A.B"

        products_str : str
            A string representation of all products involved in the reaction ,e.g. "C + D" or "C.D"
        """
        rxn_str = self.rxn_str

        if " = " in rxn_str:
            reactants_str = rxn_str.split(" = ")[0]
            products_str = rxn_str.split(" = ")[1]
            return reactants_str, products_str

        if ">>" in rxn_str:
            reactants_str = rxn_str.split(">>")[0]
            products_str = rxn_str.split(">>")[1]
            return reactants_str, products_str

    def get_substrates(self,
                       cofactors_list: List[str],
                       consider_stereo: bool) -> List[str]:
        """
        For an input unmapped reaction string, identify the reactants and ignore cofactors.

        Parameters
        ----------
        self

        cofactors_list: List[str]
            List of cofactor SMILES strings

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        Returns
        -------
        reactants_list: List[str]
            List of reactants smiles strings
        """

        reactants_str, products_str = self._rxn_2_cpds()
        reactants_list = []

        if " + " in reactants_str:

            # for each reactant's SMILES string
            for reactant_smiles in reactants_str.split(" + "):

                reactant_smiles = canonicalize_smiles(reactant_smiles)
                reactant_smiles = neutralize_atoms(reactant_smiles)
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)

                # if this reactant is a cofactor, do not store
                if is_cofactor(mol = reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if this reactant is not a cofactor, however, store and return its SMILES
                else:
                    reactants_list.append(reactant_smiles)

        if "." in reactants_str:

            # for each reactant's SMILES string
            for reactant_smiles in reactants_str.split("."):

                reactant_smiles = canonicalize_smiles(reactant_smiles)
                reactant_smiles = neutralize_atoms(reactant_smiles)
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)

                # if this reactant is a cofactor, do not store
                if is_cofactor(mol = reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if this reactant is not a cofactor, however, store and return its SMILES
                else:
                    reactants_list.append(reactant_smiles)

        else:

            # if neither " + " nor "." has been used, then only one reactant is present on the LHS
            reactant_smiles = reactants_str
            reactant_smiles = canonicalize_smiles(reactant_smiles)
            reactant_smiles = neutralize_atoms(reactant_smiles)
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)

            # if this reactant is a cofactor, do not store
            if is_cofactor(mol=reactant_mol,
                           cofactors_list=cofactors_list,
                           consider_stereo=consider_stereo):
                pass

            # if this reactant is not a cofactor, however, store and return its SMILES
            else:
                reactants_list.append(reactant_smiles)

        return reactants_list

    def get_products(self,
                    cofactors_list: List[str],
                    consider_stereo: bool) -> List[str]:
        """
        For an input unmapped reaction string, identify the products and ignore cofactors.

        Parameters
        ----------
        self

        cofactors_list: List[str]
            List of cofactor SMILES strings

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        Returns
        -------
        products_list: List[str]
            List of product smiles strings
        """

        reactants_str, products_str = self._rxn_2_cpds()
        products_list = []

        if " + " in products_str:

            # for each reactant's SMILES string
            for product_smiles in products_str.split(" + "):

                product_smiles = canonicalize_smiles(product_smiles)
                product_smiles = neutralize_atoms(product_smiles)
                product_mol = Chem.MolFromSmiles(product_smiles)

                # if this reactant is a cofactor, do not store
                if is_cofactor(mol = product_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if this reactant is not a cofactor, however, store and return its SMILES
                else:
                    products_list.append(product_smiles)

        if "." in products_str:

            # for each reactant's SMILES string
            for product_smiles in products_str.split("."):

                product_smiles = canonicalize_smiles(product_smiles)
                product_smiles = neutralize_atoms(product_smiles)
                product_mol = Chem.MolFromSmiles(product_smiles)

                # if this reactant is a cofactor, do not store
                if is_cofactor(mol = product_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if this reactant is not a cofactor, however, store and return its SMILES
                else:
                    products_list.append(product_smiles)

        else:

            # if neither " + " nor "." has been used, then only one product is present on the RHS
            product_smiles = products_str
            product_smiles = canonicalize_smiles(product_smiles)
            product_smiles = neutralize_atoms(product_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)

            # if this reactant is a cofactor, do not store
            if is_cofactor(mol = product_mol,
                           cofactors_list = cofactors_list,
                           consider_stereo = consider_stereo):
                pass

            # if this reactant is not a cofactor, however, store and return its SMILES
            else:
                products_list.append(product_smiles)

        return products_list

    def get_lhs_cofactors(self,
                          cofactors_list: List[str],
                          consider_stereo: bool) -> List[str]:
        """
        For an input unmapped reaction string, identify the cofactors on the LHS of a reaction.

        Parameters
        ----------
        self

        cofactors_list: List[str]
            List of cofactor SMILES strings

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        Returns
        -------
        LHS_cofactors_list: List[str]
            List of cofactors smiles strings for cofactors involved on the LHS of a reaction
        """

        reactants_str, products_str = self._rxn_2_cpds()
        LHS_cofactors_list = []

        if " + " in reactants_str:

            # for each reactant's SMILES string
            for reactant_smiles in reactants_str.split(" + "):

                reactant_smiles = canonicalize_smiles(reactant_smiles)
                reactant_smiles = neutralize_atoms(reactant_smiles)
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)

                # if this reactant is a cofactor, store its SMILES
                if is_cofactor(mol = reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):

                    LHS_cofactors_list.append(reactant_smiles)

                # if this reactant is not a cofactor, however, then do nothing
                else:
                    pass

        if "." in reactants_str:

            # for each reactant's SMILES string
            for reactant_smiles in reactants_str.split("."):

                reactant_smiles = canonicalize_smiles(reactant_smiles)
                reactant_smiles = neutralize_atoms(reactant_smiles)
                reactant_mol = Chem.MolFromSmiles(reactant_smiles)

                # if this reactant is a cofactor, store its SMILES
                if is_cofactor(mol = reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):

                    LHS_cofactors_list.append(reactant_smiles)

                # if this reactant is not a cofactor, however, then do nothing
                else:
                    pass

        else:

            # if neither " + " nor "." has been used, then only one reactant is present on the LHS
            reactant_smiles = reactants_str
            reactant_smiles = canonicalize_smiles(reactant_smiles)
            reactant_smiles = neutralize_atoms(reactant_smiles)
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)

            if is_cofactor(mol = reactant_mol,
                           cofactors_list = cofactors_list,
                           consider_stereo = consider_stereo):
                LHS_cofactors_list.append(reactant_smiles)

            else:
                pass

        return LHS_cofactors_list

    def get_rhs_cofactors(self,
                          cofactors_list: List[str],
                          consider_stereo: bool) -> List[str]:
        """
        For an input unmapped reaction string, identify the cofactors on the RHS of a reaction.

        Parameters
        ----------
        self

        cofactors_list: List[str]
            List of cofactor SMILES strings

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        Returns
        -------
        reactants_list: List[str]
            List of reactants smiles strings
        """
        reactants_str, products_str = self._rxn_2_cpds()
        RHS_cofactors_list = []

        if " + " in products_str:

            # for each product's SMILES string
            for product_smiles in products_str.split(" + "):

                product_smiles = canonicalize_smiles(product_smiles)
                product_smiles = neutralize_atoms(product_smiles)
                product_mol = Chem.MolFromSmiles(product_smiles)

                # if this product is a cofactor, store its SMILES
                if is_cofactor(mol = product_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):

                    RHS_cofactors_list.append(product_smiles)

                # if this product is not a cofactor, however, then do nothing
                else:
                    pass

        if "." in products_str:

            # for each product's SMILES string
            for product_smiles in products_str.split("."):

                product_smiles = canonicalize_smiles(product_smiles)
                product_smiles = neutralize_atoms(product_smiles)
                product_mol = Chem.MolFromSmiles(product_smiles)

                # if this reactant is a cofactor, store its SMILES
                if is_cofactor(mol = product_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):

                    RHS_cofactors_list.append(product_smiles)

                # if this product is not a cofactor, however, then do nothing
                else:
                    pass

        # if neither " + " nor "." has been used, then only one product is present on the RHS
        product_smiles = products_str
        product_smiles = canonicalize_smiles(product_smiles)
        product_smiles = neutralize_atoms(product_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)

        if is_cofactor(mol = product_mol,
                       cofactors_list = cofactors_list,
                       consider_stereo = consider_stereo):

            RHS_cofactors_list.append(product_mol)

        else:
            pass



        return RHS_cofactors_list

    def get_JN_rxn_descriptor(self,
                             cofactors_df: pd.DataFrame,
                             consider_stereo: bool) -> Tuple[List[str], List[str]]:

        # create a list of cofactor SMILES strings with the input cofactors_dict
        cofactors_list = list(cofactors_df["SMILES"])

        substrates_list = self.get_substrates(cofactors_list, consider_stereo = consider_stereo)
        products_list = self.get_products(cofactors_list, consider_stereo = consider_stereo)
        lhs_cofactors_list = self.get_lhs_cofactors(cofactors_list, consider_stereo = consider_stereo)
        rhs_cofactors_list = self.get_rhs_cofactors(cofactors_list, consider_stereo = consider_stereo)

        LHS_descriptor = []
        RHS_descriptor = []

        ### populate the reaction descriptor for the LHS of the reaction

        # for any substrates that are not cofactors, we add "Any" to the reaction descriptor for the LHS
        for _ in substrates_list:
            LHS_descriptor.append("Any")

        # for any substrates that are cofactors, we find the CoF code for this particular cofactor on the LHS
        for lhs_cofactor_smiles in lhs_cofactors_list:
            lhs_cofactor_CoF_code = get_cofactor_CoF_code(query_SMILES = lhs_cofactor_smiles,
                                                          cofactors_df = cofactors_df)

            # ignore accounting for hydrogens in generating the reaction descriptor
            if lhs_cofactor_CoF_code != "H+":
                LHS_descriptor.append(lhs_cofactor_CoF_code)

        ### populate the reaction descriptor for the RHS of the reaction

        # for any products that are not cofactors, we add "Any" to the product descriptor for the RHS
        for _ in products_list:
            RHS_descriptor.append("Any")

        # for any products that are cofactors, we find the CoF code for this particular cofactor on the RHS
        for rhs_cofactor_smiles in rhs_cofactors_list:
            rhs_cofactor_CoF_code = get_cofactor_CoF_code(query_SMILES = rhs_cofactor_smiles,
                                                          cofactors_df = cofactors_df)

            # ignore accounting for hydrogens in generating the reaction descriptor
            if rhs_cofactor_CoF_code != "H+":
                RHS_descriptor.append(rhs_cofactor_CoF_code)

        return LHS_descriptor, RHS_descriptor

    # def map_rxn_to_rule(self,
    #                     cofactors_df: pd.DataFrame,
    #                     rule_type: str):
    #
    #     if rule_type == "generalized":
    #         LHS_descriptor, RHS_descriptor = self.get_JN_rxn_descriptor(cofactors_df = cofactors_df,
    #                                                                     consider_stereo = False)
    #
    #         query_signature = (LHS_descriptor, RHS_descriptor)
    #
    #         possible_rxn_templates = []
    #
    #         for i in range(0, cofactors_df.shape[0]):
    #             lhs_rxn_signature = cofactors_df.iloc[i, :]['Reactants']
    #             rhs_rxn_signature = cofactors_df.iloc[i, :]['Products']
    #             rxn_signature_rule_i = (lhs_rxn_signature, rhs_rxn_signature)
    #
    #             if


class mapped_reaction:
    """
    Parameters
    ----------
    rxn_smarts : str
        Mapped reaction SMARTS with complete atom mapping of all species in reaction.

    Attributes
    ----------
    rxn_smarts : str
        Mapped reaction SMARTS with complete atom mapping of all species in reaction.

    Methods
    ----------
    _get_mapped_bonds :
        Get the set of all atom-mapped bonds from a molecule's RDKit mol object.
        This is a static method.

    _get_all_changed_atoms:
        Get all transformed atoms as well as broken and formed bonds from a atom-mapped reaction.

    get_all_changed_atoms:
        Same as above but with the added option to include or exclude cofactors.

    get_template_around_rxn_site:
        Extract a SMARTS template for this reaction with varying radii around the reaction site.
        Templates can be extracted both with and without stereochemistry.
        This is a static method.
    """
    def __init__(self,
                 rxn_smarts: str,
                 include_stereo: bool = True):
        self.rxn_smarts = rxn_smarts

    @staticmethod
    def _get_mapped_bonds(mol: Chem.rdchem.Mol) -> Set[Tuple[int, int, Chem.rdchem.BondType]]:
        """
        Parameters
        ----------
        mol : Chem.rdchem.Mol
            RDKit mol object of a single molecule represented with fully atom-mapped SMARTS.

        Returns
        -------
        bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            A set of mapped bonds for the input molecule.
            Each element in this set is a tuple that corresponds to a mapped bond.
            This tuple itself comprises three elements.
            The first two elements are integers indicating the start & end atom indices of a bond.
            Third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.
        """
        # initialize an empty set to store all bonds
        bonds = set()
        for bond in mol.GetBonds():
            starting_atom = bond.GetBeginAtom()
            ending_atom = bond.GetEndAtom()

            # only consider atoms with mapping when storing mapped bonds
            if starting_atom.GetAtomMapNum() > 0 and ending_atom.GetAtomMapNum() > 0:
                bond_tuple = (starting_atom.GetAtomMapNum(),
                              ending_atom.GetAtomMapNum(),
                              bond.GetBondType())
                bonds.add(bond_tuple)

        return bonds

    def _get_all_changed_atoms(self) -> Tuple[ Set[int],
                                               Set[Tuple[int, int, Chem.rdchem.BondType]],
                                               Set[Tuple[int, int, Chem.rdchem.BondType]] ]:
        """
        Parameters
        ----------
        self

        Returns
        -------
        changed_atoms: Set[int]
            The set atom indices corresponding to all changed atoms across the reaction.

        broken_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            The set of all bonds that were broken across the reaction.
            Each element in this set is a tuple comprising three elements.
            The first two elements are integers indicating start & end atom indices of a bond.
            The third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.

        formed_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            Similar to broken_bonds, but for formed bonds.
        """
        # create an RDKit reaction object from a fully atom-mapped reaction smarts string
        rxn = AllChem.ReactionFromSmarts(self.rxn_smarts)
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()

        reactant_bonds = set()
        product_bonds = set()

        # collect all mapped bonds from all reactants involved
        for reactant_mol in reactants:
            reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

        # collect all mapped bonds from all products involved
        for product_mol in products:
            product_bonds |= self._get_mapped_bonds(mol = product_mol)

        # identify all broken bonds and formed bonds
        broken_bonds = reactant_bonds - product_bonds # bonds present in reactants but not products
        formed_bonds = product_bonds - reactant_bonds # bonds present in products but not reactants

        # collect atom mapped numbers involved in broken or formed bonds
        changed_atoms = set()
        for bond in broken_bonds.union(formed_bonds):
            changed_atoms.add(bond[0]) # atom map number of the first atom
            changed_atoms.add(bond[1]) # atom map number of the second atom

        return changed_atoms, broken_bonds, formed_bonds

    def get_all_changed_atoms(self,
                              include_cofactors: bool,
                              consider_stereo: bool,
                              cofactors_list: List[str]) -> Tuple[ Set[int],
                                                              Set[Tuple[int, int, Chem.rdchem.BondType]],
                                                              Set[Tuple[int, int, Chem.rdchem.BondType]] ]:
        """
        Parameters
        ----------
        self

        include_cofactors: bool
            Whether to include transformed cofactor atoms in tracking all changed atoms

        consider_stereo: bool
            Whether to consider stereochemistry of cofactors specifically

        cofactors_list: List[str]
            List of cofactor SMILES strings

        Returns
        -------
        changed_atoms: Set[int]
            The set atom indices corresponding to all changed atoms across the reaction.

        broken_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            The set of all bonds that were broken across the reaction.
            Each element in this set is a tuple comprising three elements.
            The first two elements are integers indicating start & end atom indices of a bond.
            The third element corresponds to the bond type, i.e. single, double or aromatic.
            This bond type is a Chem.rdchem.BondType data type.

        formed_bonds: Set[Tuple[int, int, Chem.rdchem.BondType]]
            Similar to broken_bonds, but for formed bonds.
        """
        # create an RDKit reaction object from a fully atom-mapped reaction smarts string
        rxn = AllChem.ReactionFromSmarts(self.rxn_smarts)
        reactants = rxn.GetReactants()
        products = rxn.GetProducts()

        reactant_bonds = set()
        product_bonds = set()

        # collect all mapped bonds from all reactants involved
        for reactant_mol in reactants:

            # if cofactors are being tracked too, store all mapped bonds for all reactants
            if include_cofactors:
                reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

            # if tracking only changed substrate and product atoms but not cofactor atoms,
            else:
                # check if a reactant is a cofactor first and do nothing if this is true
                if is_cofactor(reactant_mol,
                               cofactors_list = cofactors_list,
                               consider_stereo = consider_stereo):
                    pass

                # if a reactant is not a cofactor, store its mapped bonds
                else:
                    reactant_bonds |= self._get_mapped_bonds(mol = reactant_mol)

        # collect all mapped bonds from all products involved
        for product_mol in products:

            # if cofactors are being tracked too, store all mapped bonds for all products
            if include_cofactors:
                product_bonds |= self._get_mapped_bonds(mol = product_mol)

            # if tracking only changed substrate and product atoms but not cofactor atoms,
            else:
                # check if a product is a cofactor first and do nothing if this is true
                if is_cofactor(product_mol,
                                cofactors_list = cofactors_list,
                                consider_stereo = consider_stereo):
                    pass

                # if a product is not a cofactor, store its mapped bonds
                else:
                    product_bonds |= self._get_mapped_bonds(mol = product_mol)

        # identify all broken bonds and formed bonds
        broken_bonds = reactant_bonds - product_bonds # bonds present in reactants but not products
        formed_bonds = product_bonds - reactant_bonds # bonds present in products but not reactants

        # collect atom mapped numbers involved in broken or formed bonds
        changed_atoms = set()
        for bond in broken_bonds.union(formed_bonds):
            changed_atoms.add(bond[0]) # atom map number of the first atom
            changed_atoms.add(bond[1]) # atom map number of the second atom

        return changed_atoms, broken_bonds, formed_bonds

    @staticmethod
    def get_template_around_rxn_site(atom_mapped_substrate_smarts: str,
                                     reactive_atom_indices: List[int],
                                     radius: int,
                                     include_stereo: bool) -> str:
        """
        Extract the chemical environment for a given substrate around its specified reaction site.
        The reaction site could comprise a single or even multiple carbon atoms.
        Reaction sites must be specified as a list of atom indices of the transformed atoms.
        The substrate mol

        Parameters
        ----------
        atom_mapped_substrate_smarts: str
            Atom-mapped SMARTS string representing the substrate molecule.

        reactive_atom_indices: List[int]
            List of atom indices of the reactive atoms.

        radius: int
            Radius (in bonds) around each reactive atom to extract the environment

        include_stereo: bool
            Whether to include stereochemistry in the generated SMARTS.

        Returns
        -------
        env_smarts: str
            The SMARTS string for the chemical environment around the reaction site.
            If the input radius exceeds the input substrate molecule, then an empty string is returned
        """

        substrate_mol = Chem.MolFromSmarts(atom_mapped_substrate_smarts)

        if substrate_mol is None:
            raise ValueError("Invalid SMARTS string provided")

        if not include_stereo:
            substrate_mol = remove_stereo(substrate_mol)

        # initialize a set to store the union of all chemical environments
        combined_env = set()

        # iterate through each atom in the substrate molecule
        for atom in substrate_mol.GetAtoms():

            # extract atom index & atom map number for each atom
            atom_index_num = atom.GetIdx()
            atom_map_num = atom.GetAtomMapNum()

            # if atom map number is in the list of reactive atom indices
            if atom_map_num in reactive_atom_indices:
                # extract the chemical environment around this atom using its atom index
                # note that atom index is passed
                env = Chem.FindAtomEnvironmentOfRadiusN(substrate_mol,
                                                        radius=radius,
                                                        rootedAtAtom=atom_index_num)
                combined_env.update(env)

        # with the final combined environment, extract a submolecule
        combined_submol = Chem.PathToSubmol(substrate_mol, list(combined_env))
        return Chem.MolToSmarts(combined_submol)

