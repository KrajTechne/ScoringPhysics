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
    #------------- Compute CDR RMSD of the Parent vs Design Chain in Binder-Target Complex
    cdr_rmsd = struc.rmsd(reference = cdr_parent_coord, subject = cdr_aligned_design_coord)
    print(f"After alignment of the design chain from the binder-target complex on to the parent Fab using just the FRs, ....")
    print(f"Computed CDR RMSD: {cdr_rmsd}")

    #-------------Align on CDR Coordinates of respective Parent & Design to get a transformation mask
    aligned_design_on_parent, align_transformation = struc.superimpose(fixed = cdr_parent_coord, mobile = cdr_design_coord)
    #-------------Apply transformation mask on to original atom_array_design to get aligned_atom_array
    atom_array_design_aligned_cdr = align_transformation.apply(atoms = atom_array_design)
    #-------------Extract CDR Coordinates of the Design
    fr_aligned_design_coord = struc.coord(atom_array_design_aligned_cdr[mask_dictionary['design']['fr']])
    #------------- Compute CDR RMSD of the Parent vs Design Chain in Binder-Target Complex
    fr_rmsd = struc.rmsd(reference = fr_parent_coord, subject = fr_aligned_design_coord)
    print(f"After alignment of the design chain from the binder-target complex on to the parent Fab using just the CDRs, ....")
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
    
    # Avoid division by zero
    if (len(desired_epitope) == 0) or (len(actual_epitope) == 0):
        recall = 0
        precision = 0
        f1_score = 0

    # --- METRICS ---
    
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