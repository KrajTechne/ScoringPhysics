import numpy as np
import biotite.sequence as seq
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx

def extract_atom_array(struc_file_path: str):
    """ Extract atom array from either CIF File or PDB File"""
    if struc_file_path.endswith(".cif"):
        pdbx_file = pdbx.CIFFile.read(struc_file_path)
        atom_array = pdbx.get_structure(pdbx_file=pdbx_file, model = 1)
    elif struc_file_path.endswith(".pdb"):
        pdb_file = pdb.PDBFile.read(struc_file_path)
        atom_array = pdb_file.get_structure(model = 1)
    else:
        raise ValueError("File must be either a PDB or CIF file")
    return atom_array
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
        desired_epitope_coverage = 0

    # --- METRICS ---
    
    # Recall: What percentage of the desired epitope residues were hit within pool of desired epitope residues?
    recall = len(intersection) / len(desired_epitope)
    
    # Precision: What percentage of the actual epitope residues were hits within pool of desired epitope residues?
    # Accounts for off-target hits since division by actual epitope residues rather than desire epitope residues
    precision = len(intersection) / len(actual_epitope)
    
    # Jaccard: The balanced score (Best single metric). Accounts for both off-target hits and missed desired residues.
    desired_epitope_coverage = len(intersection) / len(union)

    contact_information = {
        "binder_chain": binder_chain_id,
        "target_chain": target_chain_id,
        "paratope_indices": paratope_indices_str,
        "paratope_length": paratope_length,
        "paratope_1aa": paratope_1aa,
        "epitope_indices": epitope_indices_str,
        "epitope_length": epitope_length,
        "epitope_1aa": epitope_1aa,
        "desired_epitope_coverage": desired_epitope_coverage
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