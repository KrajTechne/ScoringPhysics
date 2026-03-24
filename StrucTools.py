import torch
import gemmi
import py2Dmol
import numpy as np
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

def extract_atom_array(struc_file_path: str, ca_only = False):
    """ Extract atom array from either CIF File or PDB File"""
    if struc_file_path.endswith(".cif"):
        pdbx_file = pdbx.CIFFile.read(struc_file_path)
        atom_array = pdbx.get_structure(pdbx_file=pdbx_file, model = 1)
    elif struc_file_path.endswith(".pdb"):
        pdb_file = pdb.PDBFile.read(struc_file_path)
        atom_array = pdb_file.get_structure(model = 1)
    else:
        raise ValueError("File must be either a PDB or CIF file")
    if ca_only:
        atom_array = atom_array[atom_array.atom_name == "CA"]
    return atom_array

def extract_sidechain_atom_array(atom_array):
    """ Extract atom_array subset corresponding to all residues' side chain atoms (atoms attached to CB)
        For the case of Glycine, since only bound to Hydrogen. Extract its CA as a proxy
    """
    backbone_atoms = ['N', 'CA', 'C', 'O']
    atom_array_side_chains = atom_array[~np.isin(atom_array.atom_name, backbone_atoms) | 
                                            ((atom_array.res_name == 'GLY') & (atom_array.atom_name == 'CA'))]
    return atom_array_side_chains
    
#------------------------------ Identification of surface and core residues functions --------------------------------------------------------------------
def compute_sasa(atom_array, per_residue = True) -> np.array:
    """ Computes solvent accessible surface area (SASA)
        Biotite computes SASA per atom basis so need to aggregate (np.nansum) for per-residue basis
        Returns per-residue solvent accessible surface area (SASA) if per_residue = True, else returns per-atom SASA
    """
    # 1. Calculate Solvent Accessible Surface Area (computed for all atoms)
    sasa_atom = struc.sasa(array = atom_array)
    # 2. Compute Residue-wise Solvent Accessible Surface Area
    # 2.1 Takes in original atom array, per-atom calculations, and a function to aggregate to per-residue
    if per_residue:
        sasa = struc.apply_residue_wise(atom_array, sasa_atom, np.nansum)
    else:
        sasa = sasa_atom
    return sasa

def compute_relative_sasa(atom_array) -> np.array:
    """ Computes relative solvent accessible surface area (SASA)
        Biotite computes SASA per atom basis so need to aggregate (np.nansum) for per-residue basis and then normalize by max SASA
        Returns per-residue relative solvent accessible surface area (SASA)
    """
    # Source: Tien, M. Z., et al. (2013). "Maximum allowed solvent accessibilites of residues in proteins". PLoS ONE.
    # Scale: Theoretical (in square Angstroms)
    max_sasa = {"ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0, "GLN": 225.0, "GLU": 223.0, 
                "GLY": 104.0, "HIS": 224.0, "ILE": 197.0, "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, 
                "PRO": 159.0, "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0}
    
    # 1. Compute per-residue SASA
    sasa_residue = compute_sasa(atom_array = atom_array, per_residue = True)
    # 2.2 Extract chain residues
    res_indices, residues_3aa = struc.get_residues(atom_array)
    # 2.3 Conduct Calculation: divide per-residue sasa by max sasa for each residue
    sasa_relative = []
    for aa3, per_residue_sasa in zip(residues_3aa, sasa_residue):
        per_residue_sasa_max = max_sasa[aa3]
        per_residue_sasa_relative = per_residue_sasa / per_residue_sasa_max
        sasa_relative.append(per_residue_sasa_relative)
    
    return sasa_relative

def identify_surface_core_residues(atom_array, core_threshold: float = 0.25):
    """ 
    Using the relative surface area of each residue to identify core and surface residues
    Surface Residues have a relative surface area > 0.25
    Core Residues have a relative surface area <= 0.25

    Args:
        atom_array (np.array): Atom array of the Fab
        core_threshold (float): Threshold for identifying core residues
    Returns:
        surface_indices (np.array): Numpy array of 0-indexed indices of surface residues
        core_indices (np.array): Numpy array of 0-indexed indices of core residues
    """

    relative_sasa_per_residue = compute_relative_sasa(atom_array)
    # 1. Create boolean mask for core and surface residues
    mask_core = np.array(relative_sasa_per_residue) <= core_threshold
    mask_surface = ~mask_core
    # 2. Extract indices of core and surface residues
    indices_core = np.where(mask_core)[0]
    indices_surface = np.where(mask_surface)[0]
    # 3. Save both masks and indices as separate dictionaries
    mask_dictionary = {'surface': mask_surface, 'core': mask_core}
    indices_dictionary = {'surface': indices_surface, 'core': indices_core}
    return mask_dictionary, indices_dictionary

#------------------------------ Aligning Parent Fab (VH-VL) to Apo ScFv Design and Computing CDR & FR RMSD
def superimpose_align_parent_design(struc_parent_path: str, struc_design_path: str, mask_dictionary: dict, binder_chain_id: str = 'A'):
    """ Superimposes the design onto the parent structure and calculates the RMSD between the two structures.
        1st Superimpostion: Aligns the FRs of the parent and design structure using the transformation matrix
        2nd Superimposition: Aligns the CDRs of the parent and design structure using the transformation matrix
        Returns: 1st Superimposition RMSD (CDR RMSD) & 2nd Superimposition (FR RMSD) between the two structures
        Args:
            struc_parent_path (str): Path to the parent structure
            struc_design_path (str): Path to the design structure
            mask_dictionary (dict): Dictionary containing the FR, CDR regions boolean mask for the parent and design structure
                Structure: {'parent' : {'fr' : fr_mask_parent, 'cdr' : cdr_mask_parent, 'linker' : linker_mask_parent},
                            'design' : {'fr' : fr_mask_design, 'cdr' : cdr_mask_design, 'linker' : linker_mask_design}}
            binder_chain_id (str): Chain ID of the binder chain in the design structure
        Returns:
            cdr_rmsd (float): RMSD between the CDRs of the parent and design structure (aligned on FR residues)
            fr_rmsd (float): RMSD between the FRs of the parent and design structure (aligned on CDR residues)
            Saved as dictionary: {'rmsd_cdr': cdr_rmsd, 'rmsd_fr': fr_rmsd}
    """

    #--------------------------Extracted atom arrays
    atom_array_parent = extract_atom_array(struc_parent_path, ca_only = True)
    atom_array_design = extract_atom_array(struc_design_path, ca_only = True)
    atom_array_design = atom_array_design[atom_array_design.chain_id == binder_chain_id]

    #--------------Extract Coordinates of FRs and CDRs in both parent and design atom arrays
    fr_parent_coord = struc.coord(atom_array_parent[mask_dictionary['parent']['fr']])
    fr_design_coord = struc.coord(atom_array_design[mask_dictionary['design']['fr']])
    cdr_parent_coord = struc.coord(atom_array_parent[mask_dictionary['parent']['cdr']])
    cdr_design_coord = struc.coord(atom_array_design[mask_dictionary['design']['cdr']])
    #-------------Align on Framework Coordinates of respective Parent & Design to get a transformation mask
    aligned_design_on_parent, align_transformation = struc.superimpose(fixed = fr_parent_coord, mobile = fr_design_coord)
    #-------------Apply transformation mask on to original atom_array_design to get aligned_atom_array
    atom_array_design_aligned_fr = align_transformation.apply(atoms = atom_array_design)
    #-------------Extract CDR Coordinates of the Design
    cdr_aligned_design_coord = struc.coord(atom_array_design_aligned_fr[mask_dictionary['design']['cdr']])
    fr_aligned_design_coord = struc.coord(atom_array_design_aligned_fr[mask_dictionary['design']['fr']])
    #------------- Compute CDR RMSD of the Parent vs Design Chain in Binder-Target Complex
    cdr_rmsd = struc.rmsd(reference = cdr_parent_coord, subject = cdr_aligned_design_coord)
    #------------- Compute FR RMSD of the Parent vs Design Chain in Binder-Target Complex
    fr_rmsd = struc.rmsd(reference = fr_parent_coord, subject = fr_aligned_design_coord)
    print(f"After alignment of the design chain from the binder-target complex on to the parent Fab using just the FRs, ....")
    print(f"Computed CDR RMSD: {cdr_rmsd}")
    print(f"Computed FR RMSD: {fr_rmsd}")

    fr_cdr_rmsd = {'rmsd_cdr' : cdr_rmsd, 'rmsd_fr' : fr_rmsd}
    return fr_cdr_rmsd 

#------------------------------ Determine Binding Interface: Paratope, Epitope, Overlap on desired epitope residues --------------------------------------
def determine_binding_interface(pdb_file_path: str, desired_epitope_residues: list, binder_chain_id: str = "A", 
                                target_chain_id: str = "B", cutoff: float = 4.5) -> dict:
    """
        Answer and achieve these objectives:
        1. Which residues of the binder chain are in contact with the target chain? (Actual Paratope)
        2. Which residues of the target chain are in contact with the binder chain? (Actual Epitope)
        3. Calculate percent of desired epitope residues covered by actual epitope (Desired Epitope % Coverage)
    """
    obj_protein_seq = seq.ProteinSequence()
    # Make sure list of integers for set intersection
    desired_epitope_residues = [int(x) for x in desired_epitope_residues] 

    holo_atom_array = extract_atom_array(struc_file_path= pdb_file_path)
    
    # Return specific chain's heavy atoms' atom array
    binder_atom_array = holo_atom_array[(holo_atom_array.chain_id == binder_chain_id) & (holo_atom_array.element != "H")]
    target_atom_array = holo_atom_array[(holo_atom_array.chain_id == target_chain_id) & (holo_atom_array.element != "H")]

    # Create ROI Cell List: Based on target atom array
    # Create target_binder_adjacency_matrix (shape): (# of Target Heavy atoms, # of Binder Heavy atoms)
    roi_cell_list = struc.CellList(atom_array = target_atom_array, cell_size = cutoff)
    target_binder_adjacency_matrix = roi_cell_list.get_atoms(binder_atom_array.coord, radius = cutoff, as_mask= True)

    # Isolate Target's Heavy Contact Atoms Indices
    contact_atom_indices_target = np.any(target_binder_adjacency_matrix, axis = 0) # Collapse along axis 0 (target atoms)
    contact_atom_indices_binder = np.any(target_binder_adjacency_matrix, axis = 1) # Collapse along axis 1 (binder atoms)

    target_contact_atom_array = target_atom_array[contact_atom_indices_target]
    binder_contact_atom_array = binder_atom_array[contact_atom_indices_binder]
    

    epitope_indices, epitope_3aa = struc.get_residues(target_contact_atom_array)
    paratope_indices, paratope_3aa = struc.get_residues(binder_contact_atom_array)
        
    # Convert Residue Indices into Single String
    paratope_indices_str = ",".join(f"{res_index}" for res_index in paratope_indices)
    epitope_indices_str = ",".join(f"{res_index}" for res_index in epitope_indices)

    # Extract Paratope & Epitope 1-letter AA Strings
    paratope_1aa = "".join([obj_protein_seq.convert_letter_3to1(para_3aa) for para_3aa in paratope_3aa])
    epitope_1aa = "".join([obj_protein_seq.convert_letter_3to1(epi_3aa) for epi_3aa in epitope_3aa])

    # Count # of Residues
    paratope_length, epitope_length = len(paratope_indices), len(epitope_indices)
    
    # Determine Percent Coverage of Desired Epitope Residues in Actual Epitope Residues
    actual_epitope = set(epitope_indices)
    desired_epitope = set(desired_epitope_residues)
    
    # 1. Intersection: The "Good" Hits
    intersection = actual_epitope.intersection(desired_epitope)
    
    # 2. Union: The Total Footprint (Target + Spillover)
    union = actual_epitope.union(desired_epitope)
    
    # --- Compute Metrics ---
    # Avoid division by zero
    if (len(desired_epitope) == 0) or (len(actual_epitope) == 0):
        recall = 0
        precision = 0
        f1_score = 0
    else:
        # Recall: What percentage of the desired epitope residues were hit within pool of desired epitope residues?
        recall = len(intersection) / len(desired_epitope)
    
        # Precision: What percentage of the actual epitope residues were hits within pool of desired epitope residues?
        # Accounts for off-target hits since division by actual epitope residues rather than desire epitope residues
        precision = len(intersection) / len(actual_epitope)
    
        # Jaccard: The balanced F1 score (Best single metric). Accounts for both off-target hits and missed desired residues.
        f1_score = len(intersection) / len(union)

    contact_information = {
        "binder_chain": binder_chain_id,
        "target_chain": target_chain_id,
        "paratope_indices": paratope_indices_str,
        "paratope_length": paratope_length,
        "paratope_1aa": paratope_1aa,
        "epitope_indices": epitope_indices_str,
        "epitope_length": epitope_length,
        "epitope_1aa": epitope_1aa,
        "epitope_coverage_recall": recall,
        "epitope_coverage_precision": precision,
        "epitope_coverage_f1": f1_score
    }
    return contact_information

#---- Superimpose & Calculate RMSD ----------------------------------------------------------------------------------------------
def superimpose_and_calculate_specified_rmsd(atom_array_fixed: struc.AtomArray, 
                                             atom_array_mobile: struc.AtomArray,
                                             align_mask: np.ndarray, calculate_rmsd_mask: np.ndarray,
                                             ca_only: bool = True):
    """ 
        Superimpose two alpha-carbon only structures (binder-target & binder only) & Calculate RMSD at specified residues
        Approach:
            1. Extract only the alpha-carbons from both structures
            2. Superimpose the alpha carbons from both structures
            3. Calculate RMSD between desired regions of both structures
    """
    # 1. Extract only the alpha-carbons from both structures
    if ca_only:
        atom_array_fixed = atom_array_fixed[atom_array_fixed.atom_name == "CA"]
        atom_array_mobile = atom_array_mobile[atom_array_mobile.atom_name == "CA"]
    # 1.5: Assert both structures have the same number of atoms
    assert len(atom_array_fixed) == len(atom_array_mobile), "Both structures must have the same number of atoms"
    # 2. Superimpose the alpha carbons from both structures
    aligned_atom_array_mobile, align_transformation = struc.superimpose(fixed = atom_array_fixed,
                                                                           mobile = atom_array_mobile,
                                                                           atom_mask = align_mask)
    # 3. Calculate RMSD between desired regions of both structures
    rmsd = struc.rmsd(atom_array_fixed[calculate_rmsd_mask], aligned_atom_array_mobile[calculate_rmsd_mask])
    return rmsd, aligned_atom_array_mobile, align_transformation

def run_superimpose_and_calculate_rmsd_apo_holo_pipeline(pdb_file_path_holo: str, pdb_file_path_apo: str,
                                                         align_mask: np.ndarray, calculate_rmsd_mask: np.ndarray,
                                                         binder_chain_id:str = "A"):
    """ Run superimpose and calculate RMSD pipeline for two structures
        - pdb_file_path_binder_target (holo = binder & target): PDB File Path of the Binder Target Structure
        - pdb_file_path_binder (apo = binder only): PDB File Path of the Binder Structure
        - align_mask: Mask for atoms in both binder_target structure and binder_structure to align
        - calculate_rmsd_mask: Mask for atoms in both binder_target structure and binder_structure to be considered for RMSD calculation    
    """
    #1. Extract atom arrays from both structures
    atom_array_binder_target = extract_atom_array(pdb_file_path_holo)
    atom_array_binder = extract_atom_array(pdb_file_path_apo)
    #2. Extract only the binder atom atom arrays from both structures
    atom_array_relaxed_binder = atom_array_binder_target[atom_array_binder_target.chain_id == binder_chain_id]
    #3. Run superimpose and calculate RMSD pipeline for two structures while restricting only to alpha-carbon atoms
    rmsd, aligned_alpha_carbons_binder, align_transformation = superimpose_and_calculate_specified_rmsd(atom_array_fixed = atom_array_relaxed_binder,
                                                                                                        atom_array_mobile = atom_array_binder, 
                                                                                                        align_mask=align_mask, 
                                                                                                        calculate_rmsd_mask= calculate_rmsd_mask,
                                                                                                        ca_only = True)
    return rmsd, aligned_alpha_carbons_binder, align_transformation

#----------------------------------------- Predicted Structure Metrics----------------------------------------------------------
def calculate_ipsae_complex(pae_matrix, len_binder: int, pae_cutoff: int = 15, verbose = True): 
    """
        Calculate the ipsae score for a given pae matrix
        Assumption: 
            - PAE matrix is square matrix of shape (L, L) where L = Length of binder + target
            - PAE matrix derived with binder being first chain and target being second chain
        Args:
            pae_matrix (torch.tensor): PAE matrix of shape (L, L) where L = Length of binder + target
            len_binder (int): Length of binder
            pae_cutoff (int): PAE cutoff to use for ipsae calculation (Recommended Values: 10 or 15)
        Returns:
            ipsae_score (float): ipsae score for the given binder-target pae matrix
        
        Works for Chai & Boltz2 (load from pae_jobname.npz with key: "pae") Protein Structure Prediction
        Need to know order of sequences inserted into pae matrix (i.e Binder (1) -> Target (2) or Target (1) -> Binder (2)
        Only Functional for Binder (1) -> Target (2)
    
    """
    pae_numpy = pae_matrix if isinstance(pae_matrix, np.ndarray) else pae_matrix.detach().cpu().numpy()
    
    # Verifications: PAE matrix is a 2D Square Matrix
    # Check if pae_matrix is a 2D matrix:
    if len(pae_numpy.shape) != 2:
        raise ValueError(f"pae_matrix must be a 2-D matrix. Instead received a matrix of shape: {pae_numpy.shape}")
    # Check if pae_matrix is square matrix:
    if pae_numpy.shape[0] != pae_numpy.shape[1]:
        raise ValueError(f"pae_matrix must be a square matrix. Instead received a matrix of shape: {pae_numpy.shape}")
    
    len_target = pae_numpy.shape[0] - len_binder
    pae_indices = {"A": list(range(len_binder)), "B": list(range(len_binder, len_binder + len_target))}

    def calculate_d0(num_residues_pass_pae: int):
        """ Calculate d0 for a given number of residues passing PAE cutoff """
        if num_residues_pass_pae < 27:
            d0 = 1
        elif num_residues_pass_pae >= 27:
            d0 = (1.24 * ((num_residues_pass_pae - 15) ** (1 / 3))) - 1.8
        return d0

    def calculate_ipsae(chain_aligned: str, chain_measured: str):
        """ Calculate ipsae when aligned on chain_aligned and measured on chain_measured """
        chain_aligned_indices = pae_indices[chain_aligned]
        chain_measured_indices = pae_indices[chain_measured]
        extracted_indices = np.ix_(chain_aligned_indices, chain_measured_indices)
        pae_subset = pae_numpy[extracted_indices]
        pae_pass_mask = pae_subset < pae_cutoff
        n0_res_vals = np.sum(pae_pass_mask, axis=1)
        # If chains are not predicted to interact when aligning on the aligned chain, return ipSAE of 0
        if sum(n0_res_vals) == 0:
            return 0.0
        d0_vals = np.array([calculate_d0(n0) for n0 in n0_res_vals])
        d0_vals = d0_vals.reshape((len(d0_vals), 1))
        pae_subset[~pae_pass_mask] = -1000
        pae_mask = np.ma.masked_values(pae_subset, -1000)
        aligned_residues_ipsae = np.mean((1 / (1 + (pae_mask / d0_vals) ** 2)), axis=1)
        residue_aligned_highest_ipsae = np.max(aligned_residues_ipsae)
        return residue_aligned_highest_ipsae

    ipsae_a_b = calculate_ipsae(chain_aligned="A", chain_measured="B")
    ipsae_b_a = calculate_ipsae(chain_aligned="B", chain_measured="A")
    ipsae_min = min(ipsae_a_b, ipsae_b_a)
    ipsae_max = max(ipsae_a_b, ipsae_b_a)
    if verbose:
        print("ipsae_a_b: ", ipsae_a_b)
        print("ipsae_b_a: ", ipsae_b_a)
    return ipsae_min, ipsae_max

#--------------------------------------------------------------------------- Visualize Structures-------------------------------------
def visualize_structure(structure_path: str):
    """
    Args:
        - structure_path (str): Path to the structure to be visualized
    Returns:
        - None
     """
    viewer = py2Dmol.view()
    viewer.add_pdb(structure_path)
    viewer.show()
    
#------------------------------------------------------------------------- Convert PDB Files to CIF Files-------------------------------
def convert_pdb_to_cif(input_pdb_path):
    """Adapts the logic from Boltz's parse_pdb to convert a PDB file to mmCIF."""
        
    print(f"Reading PDB: {input_pdb_path}")
    output_cif_path = input_pdb_path.replace('.pdb', '.cif')
        
    # 1. Read the structure using Gemmi
    structure = gemmi.read_structure(input_pdb_path)
    structure.setup_entities()
        
    # 2. Apply the subchain renaming logic (Copied from Boltz source)
    # This ensures chains are correctly named for mmCIF format (e.g., handling multiple segments)
    subchain_counts, subchain_renaming = {}, {}
    for chain in structure[0]:
        subchain_counts[chain.name] = 0
        for res in chain:
            if res.subchain not in subchain_renaming:
                subchain_renaming[res.subchain] = chain.name + str(subchain_counts[chain.name] + 1)
                subchain_renaming[res.subchain] = str(subchain_counts[chain.name] + 1) # Simplified renaming
                subchain_counts[chain.name] += 1
            res.subchain = subchain_renaming[res.subchain]
            
    # Update entities with new subchain names
    for entity in structure.entities:
        entity.subchains = [subchain_renaming.get(subchain, subchain) for subchain in entity.subchains]

    # 3. Create mmCIF document and write to file
    doc = structure.make_mmcif_document()
    doc.write_file(output_cif_path)
    
    print(f"✅ Converted to CIF: {output_cif_path}")
    return output_cif_path