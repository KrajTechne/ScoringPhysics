import os
import biotite
import pandas as pd
import numpy as np
import pyrosetta as pr
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
import biotite.sequence as seq
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import AlignChainMover
from pyrosetta.rosetta.core.kinematics import MoveMap
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from StrucTools import run_superimpose_and_calculate_rmsd_apo_holo_pipeline, determine_binding_interface

# Initialize PyRosetta
# We use the same flags to ensure the scoring physics matches the original pipeline
dalphaball_path = os.path.join(os.path.abspath("."), "DAlphaBall.gcc")
pr.init(
    f"-ignore_unrecognized_res -ignore_zero_occupancy -mute all "
    f"-holes:dalphaball {dalphaball_path} -corrections::beta_nov16 true -relax:default_repeats 2"
)

def extract_pdb_designs(pdb_folder_path: str) -> list:
    """ Generate a list of the PDB files in the user-provided folder path containing PDBs of promising designs"""
    pdb_file_paths = []
    # Check if PDB folder exists and is a directory, prior to capturing full PBD filepaths
    if os.path.exists(pdb_folder_path):
        for file_name in os.listdir(pdb_folder_path):
            if file_name.endswith(".pdb"):
                file_path = os.path.join(pdb_folder_path, file_name)
                pdb_file_paths.append(file_path)
    
    return pdb_file_paths

def relax_pdb(pdb_file_path: str, save_relaxed: bool = True) -> pr.Pose:
    """ Relaxes a PDB file using PyRosetta's Fast Relax Protocol """

    try:
        pose = pr.pose_from_file(pdb_file_path)
        og_pose = pose.clone()
    except Exception as e:
        raise ValueError(f"Error loading PDB file: {e}")

    # Create MoveMap for FastRelax Protocol
    mmf = MoveMap()
    mmf.set_chi(True) # Allow sidechains to move
    mmf.set_bb(True) # Allow backbone atoms to move
    mmf.set_jump(False) # Do not allow chains to move relative to each other

    # Configure FastRelax
    fastrelax = FastRelax()
    scorefunc = pr.get_fa_scorefxn()
    fastrelax.set_scorefxn(scorefunc)
    fastrelax.set_movemap(mmf)
    fastrelax.max_iter(200)
    fastrelax.min_type("lbfgs_armijo_nonmonotone")
    fastrelax.constrain_relax_to_start_coords(True) # Keep it close to input

    # Running FastRelax
    print("Running FastRelax Relaxation Protocol...")
    fastrelax.apply(pose)

    # Align Relaxed Pose to Original Pose
    aligner = AlignChainMover()
    aligner.source_chain(0)
    aligner.target_chain(0)
    aligner.pose(og_pose)
    aligner.apply(pose)

    # Copy B-factors (Confidence scores) from og_pose back to pose
    for i in range(1, pose.total_residue() + 1):
        if pose.residue(i).is_protein():
            # Copy from first atom of residue i
            bf = og_pose.pdb_info().bfactor(i, 1)
            for atom_i in range(1, pose.residue(i).natoms() + 1):
                pose.pdb_info().bfactor(i, atom_i, bf)
    # Save Relaxed Pose to Subfolder: relaxed_pdbs
    if save_relaxed:
        # Create Relaxed PDB Folder
        relaxed_pdb_folder_path = os.path.join(os.path.abspath("."), "relaxed_pdbs")
        if not os.path.exists(relaxed_pdb_folder_path):
            os.makedirs(relaxed_pdb_folder_path)
            print(f"Created relaxed PDB folder: {relaxed_pdb_folder_path}")
        # Save Relaxed PDB
        relaxed_pdb_file_path = os.path.join(relaxed_pdb_folder_path, os.path.basename(pdb_file_path))
        pose.dump_pdb(relaxed_pdb_file_path)
        print(f"Saved relaxed PDB file: {relaxed_pdb_file_path}")
        
    return pose

def analyze_interface(pose: pr.Pose, epi_residues: list,temp_pdb_file_path: str = "/tmp/temp.pdb", binder_chain_id: str = "A", target_chain_id: str = "B",
                      cutoff: float = 4.5) -> dict:
    """ Analyzes the interface of a PyRosetta Relaxed pose using PyRosetta's InterfaceAnalyzerMover """

    contact_information = determine_binding_interface(pdb_file_path = temp_pdb_file_path, desired_epitope_residues = epi_residues,
                                                      binder_chain_id = binder_chain_id, target_chain_id = target_chain_id,
                                                      cutoff= cutoff)

    # Initialize Interface Analyzer
    interface_analyzer = InterfaceAnalyzerMover()
    interface_string = f"{binder_chain_id}_{target_chain_id}"
    interface_analyzer.set_interface(interface_string)
    scorefunc = pr.get_fa_scorefxn()
    interface_analyzer.set_scorefunction(scorefunc)
    # Run Interface Analysis
    interface_analyzer.set_compute_packstat(True)
    interface_analyzer.set_compute_interface_energy(True)
    interface_analyzer.set_calc_dSASA(True)
    interface_analyzer.set_calc_hbond_sasaE(True)
    interface_analyzer.set_compute_interface_sc(True)
    interface_analyzer.set_pack_separated(True)
    interface_analyzer.apply(pose)

    # Store interface metrics
    interface_score = interface_analyzer.get_all_data()
    interface_sc = interface_score.sc_value
    binding_interface_hbonds = interface_score.interface_hbonds
    interface_dG = interface_analyzer.get_interface_dG()
    interface_dSASA = interface_analyzer.get_interface_delta_sasa()
    interface_packstat = interface_analyzer.get_interface_packstat()
    interface_dG_SASA_ratio = interface_score.dG_dSASA_ratio * 100

    # Save output in a dictionary
    interface_metrics = {
        "interface_sc": interface_sc,
        "binding_interface_hbonds": binding_interface_hbonds,
        "interface_dG": interface_dG,
        "interface_dSASA": interface_dSASA,
        "interface_packstat": interface_packstat,
        "interface_dG_SASA_ratio": interface_dG_SASA_ratio
    }
    # Combine both contact information and interface metrics dictionaries into one
    full_interface_metrics = {**contact_information, **interface_metrics}
    return full_interface_metrics

def run_relaxation_and_physics_scoring_single_pdb(pdb_file_path: str, epi_residues: list,rmsd_inputs: dict = {},
                                                  binder_chain_id: str = "A", target_chain_id: str = "B", 
                                                  save_relaxed: bool = False, cutoff: float = 4.5):
    """ Runs relaxation and physics scoring on a single PDB File containing a promising design from structure-based design workflow """
    print(f"Processing PDB file: {pdb_file_path}")
    # 1. Relax PDB File
    relaxed_pose = relax_pdb(pdb_file_path= pdb_file_path, save_relaxed= save_relaxed) 
    # 2. Convert relaxed pose into PDB file for extracting contact information & calculating RMSD between relaxed holo and apo structures
    temp_pdb_file_path = "/tmp/temp.pdb"
    relaxed_pose.dump_pdb(temp_pdb_file_path)
    # 3. Calculate Interface Metrics
    interface_metrics = analyze_interface(pose= relaxed_pose, temp_pdb_file_path= temp_pdb_file_path, 
                                          epi_residues= epi_residues, binder_chain_id= binder_chain_id, target_chain_id= target_chain_id)
    # 4. Calculate RMSD between relaxed holo and apo structures (only if rmsd_inputs is provided)
    required_keys = {'pdb_file_path_apo', 'align_mask', 'measure_mask'}
    if required_keys.issubset(rmsd_inputs.keys()): # Returns True if required_keys are present, even if others exist
        interface_rmsd, _, _ = run_superimpose_and_calculate_rmsd_apo_holo_pipeline(pdb_file_path_holo = temp_pdb_file_path,
                                                                                    pdb_file_path_apo = rmsd_inputs['pdb_file_path_apo'],
                                                                                    align_mask = rmsd_inputs['align_mask'],
                                                                                    calculate_rmsd_mask = rmsd_inputs['measure_mask'],
                                                                                    binder_chain_id = binder_chain_id,
                                                                                )
        interface_metrics['interface_holo_apo_rmsd'] = interface_rmsd
    interface_metrics['file_path_holo'] = pdb_file_path
    interface_metrics['pdb_filename'] = os.path.basename(pdb_file_path)
    return interface_metrics


def run_relaxation_and_physics_scoring_over_folder(pdb_folder_path: str, epi_residues: list, binder_chain_id: str = "A", target_chain_id: str = "B", 
                                       save_relaxed: bool = False, cutoff: float = 4.5):
    """ Runs relaxation and physics scoring on a set of PDB Files containing promising designs from structure-based design workflow """
    # Create initial list of dictionaries to store results
    all_interface_metrics = []
    pdb_file_paths = extract_pdb_designs(pdb_folder_path)
    for pdb_file_path in pdb_file_paths:
        interface_metrics = run_relaxation_and_physics_scoring_single_pdb(pdb_file_path= pdb_file_path, epi_residues= epi_residues, 
                                                                          binder_chain_id= binder_chain_id, target_chain_id= target_chain_id, 
                                                                          save_relaxed= save_relaxed, cutoff= cutoff)
        all_interface_metrics.append(interface_metrics)
    
    # Convert list of dictionaries to Pandas DataFrame
    df_interface_metrics = pd.DataFrame(all_interface_metrics)
        
    return df_interface_metrics
            








