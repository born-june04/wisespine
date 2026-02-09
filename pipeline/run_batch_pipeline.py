#!/usr/bin/env python3
"""
WiseSpine Batch Pipeline Orchestrator
-------------------------------------
Runs the full physics-based simulation pipeline for multiple patients.
Stages:
1. Scoliosis Simulation (Physics-based warping)
2. Tumor Simulation (Lytic/Blastic)
3. Hardware Placement (Physics-based insertion)
4. Artifact Synthesis (Beam Hardening)
5. Surgery Process (Laminectomy/Graft)
6. Causal Response (Hematoma/Edema)

Usage:
    python run_batch_pipeline.py --input_dir VerSe/dataset-01training --output_dir outputs/batch_v1 --n 5
"""

import os
import sys
import glob
import argparse
import subprocess
import time
from pathlib import Path
import logging

# Configure Logging
LOG_FILE = Path(__file__).resolve().parent.parent / "batch_pipeline.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE))
    ]
)
logger = logging.getLogger(__name__)

# All pipeline scripts live in this directory (pipeline/)
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
SCOLIOSIS_SCRIPT = SCRIPT_DIR / "simulate_scoliosis.py"
TUMOR_SCRIPT = SCRIPT_DIR / "simulate_tumors.py"
HARDWARE_SCRIPT = SCRIPT_DIR / "place_hardware_physics.py"
ARTIFACT_SCRIPT = SCRIPT_DIR / "synthesize_artifacts_simple.py"
SURGERY_SCRIPT = SCRIPT_DIR / "simulate_surgery_process.py"
CAUSAL_SCRIPT = SCRIPT_DIR / "simulate_causal_response.py"

def run_step(script_path, args, step_name):
    """Run a python script as a subprocess."""
    cmd = [sys.executable, str(script_path)] + args
    logger.info(f"[{step_name}] Running: {' '.join(cmd)}")
    
    try:
        # Check if script exists
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False

        # Run
        t0 = time.time()
        # We wrap in a python one-liner to call the specific run function?
        # No, we modified the scripts to have a main() that uses args?
        # WAIT: I refactored them to have `run_...` functions but didn't update `main` to parse CLI args!
        # The `main` functions in the scripts still have HARDCODED paths!
        # I need to call the functions directly from Python wrapper OR update the scripts to parse CLI.
        
        # Updating 6 scripts to parse CLI is tedious.
        # Better: Create a small wrapper string here and run it via `python -c`.
        
        env = os.environ.copy()
        env["PYTHONPATH"] = str(SCRIPT_DIR) + ":" + str(PROJECT_ROOT)
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True,
            env=env
        )
        
        if result.returncode != 0:
            logger.error(f"[{step_name}] Failed:\n{result.stderr}")
            return False
            
        logger.info(f"[{step_name}] Completed in {time.time()-t0:.1f}s")
        return True
        
    except Exception as e:
        logger.error(f"[{step_name}] Exception: {e}")
        return False

def run_step_via_import_wrapper(script_path, function_name, kwargs, step_name):
    """
    Constructs a python script string to import and run the function.
    This avoids modifying the original scripts to add argparse.
    """
    logger.info(f"[{step_name}] Starting...")
    
    # Construct args string
    args_str = ", ".join([f"{k}={repr(v)}" for k, v in kwargs.items()])
    
    # We need to handle imports. 
    # If script is `spine-rl-sim/simulate_scoliosis.py`, we import `spine-rl-sim.simulate_scoliosis`?
    # Or just load source?
    
    module_name = script_path.stem
    parent_dir = script_path.parent
    
    python_code = f"""
import sys
from pathlib import Path
sys.path.append("{parent_dir}")
sys.path.append("{SCRIPT_DIR}") # For shared modules

# Import module
try:
    import {module_name} as mod
except ImportError as e:
    # If module name has hyphens etc, use importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("{module_name}", "{script_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["{module_name}"] = mod
    spec.loader.exec_module(mod)

# Run function
try:
    mod.{function_name}({args_str})
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    
    cmd = [sys.executable, "-c", python_code]
    
    t0 = time.time()
    env = os.environ.copy()
    # Add project root to PYTHONPATH
    env["PYTHONPATH"] = str(SCRIPT_DIR) + ":" + str(PROJECT_ROOT) + ":" + env.get("PYTHONPATH", "")
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env
    )
    
    if result.returncode != 0:
        logger.error(f"[{step_name}] FAILED:\n{result.stdout}\n{result.stderr}")
        return False
        
    # logger.info(f"[{step_name}] Output:\n{result.stdout}")
    logger.info(f"[{step_name}] Completed in {time.time()-t0:.1f}s")
    return True


def process_patient(patient_dir, output_root):
    """Process a single patient."""
    patient_id = patient_dir.name
    logger.info(f"=== Processing {patient_id} ===")
    
    # 1. Input Paths
    # VerSe structure: rawdata/sub-verse.../sub-verse..._ct.nii.gz
    # The input `patient_dir` is expected to be `sub-verseXXX`
    
    # Try finding CT
    ct_files = list(patient_dir.glob("*_ct.nii.gz"))
    if not ct_files:
        # Try rawdata structure if patient_dir is top level
        ct_files = list(patient_dir.glob(f"rawdata/{patient_id}/*_ct.nii.gz"))
        
    if not ct_files:
        logger.warning(f"No CT found for {patient_id}, skipping.")
        return
        
    ct_path = ct_files[0]
    
    # Try finding Mask
    # VerSe layout:
    #   dataset-XXX/rawdata/sub-verseYYY/sub-verseYYY_dir-ax_ct.nii.gz
    #   dataset-XXX/derivatives/sub-verseYYY/sub-verseYYY_dir-ax_seg-vert_msk.nii.gz
    # patient_dir = rawdata/sub-verseYYY/
    # patient_dir.parent = rawdata/
    # Replace "rawdata" → "derivatives" to find the mask sibling folder.
    
    mask_files = []
    
    # Strategy 1: rawdata → derivatives sibling
    rawdata_dir = patient_dir.parent  # e.g. .../dataset-01training/rawdata/
    if rawdata_dir.name == "rawdata":
        deriv_patient_dir = rawdata_dir.parent / "derivatives" / patient_id
        mask_files = list(deriv_patient_dir.glob("*_seg-vert_msk.nii.gz"))
        logger.info(f"  Mask search: {deriv_patient_dir} → found {len(mask_files)} files")
    
    # Strategy 2: Flat structure fallback
    if not mask_files:
        mask_files = list(patient_dir.glob("*_seg-vert_msk.nii.gz"))
    
    # Strategy 3: Recursive search in dataset root
    if not mask_files:
        dataset_root = rawdata_dir.parent  # e.g. .../dataset-01training/
        mask_files = list(dataset_root.glob(f"**/{patient_id}/*_seg-vert_msk.nii.gz"))
    
    if not mask_files:
        logger.warning(f"No Mask found for {patient_id}, skipping.")
        return
        
    mask_path = mask_files[0]
    
    logger.info(f"  CT: {ct_path}")
    logger.info(f"  Mask: {mask_path}")
    
    # 2. Output Dir
    out_dir = output_root / patient_id
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Pipeline Execution
    
    # Step 1: Scoliosis (Cobb 20, 40, 60)
    # Allows generating data
    angles = [40] # Reduced for batch speed, or do [20, 60]
    
    if not run_step_via_import_wrapper(
        SCOLIOSIS_SCRIPT, 
        "run_scoliosis_simulation", 
        {"ct_path_in": str(ct_path), "mask_path_in": str(mask_path), "out_dir": str(out_dir), "angles": angles, "subject_id": patient_id},
        "Scoliosis"
    ): return
    
    # Loop over angles for subsequent steps
    for angle in angles:
        logger.info(f"  -- Processing Angle {angle} --")
        
        # Step 2: Pose/Hardware
        # run_hardware_placement(ct_path, mask_path, pose_path, out_dir, angle=60)
        pose_path = out_dir / "scoliosis_pose.json"
        scoli_ct = out_dir / f"scoliosis_cobb{angle}.nii.gz"
        
        if not run_step_via_import_wrapper(
            HARDWARE_SCRIPT,
            "run_hardware_placement",
            {"ct_path_in": str(scoli_ct), "mask_path_in": str(mask_path), "pose_path_in": str(pose_path), "out_dir": str(out_dir), "angle": angle},
            f"Hardware_{angle}"
        ): continue
        
        # Step 3: Artifacts
        hw_path = out_dir / f"scoliosis_cobb{angle}_hardware.nii.gz"
        artifact_out = out_dir / f"scoliosis_cobb{angle}_artifacts.nii.gz"
        
        if not run_step_via_import_wrapper(
            ARTIFACT_SCRIPT,
            "run_artifact_synthesis",
            {"ct_path_in": str(scoli_ct), "hw_path_in": str(hw_path), "out_path_in": str(artifact_out)},
            f"Artifacts_{angle}"
        ): continue
        
        # Step 4: Tumors (Optional branch, let's run it)
        tumor_out = out_dir / f"scoliosis_cobb{angle}_tumors.nii.gz"
        # Tumor script needs segmentation. We don't have warped segmentation efficiently.
        # It uses HW mask as proxy if seg not found.
        # Let's pass None for seg, and let it use HW from `hw_path`.
        if not run_step_via_import_wrapper(
            TUMOR_SCRIPT,
            "run_tumor_simulation",
            {"ct_path": str(artifact_out), "seg_path": None, "hw_path": str(hw_path), "out_path": str(tumor_out), "angle": angle},
            f"Tumors_{angle}"
        ): continue
        
        # Step 5: Surgery
        # Inputs: Tumor output (or Artifact output) + HW
        # `run_surgery_simulation(ct_path, hw_path, out_path)`
        postop_out = out_dir / f"scoliosis_cobb{angle}_postop.nii.gz"
        
        if not run_step_via_import_wrapper(
            SURGERY_SCRIPT,
            "run_surgery_simulation",
            {"ct_path_in": str(tumor_out), "hw_path_in": str(hw_path), "out_path_in": str(postop_out)},
            f"Surgery_{angle}"
        ): continue
        
        # Step 6: Causal Response
        # Inputs: Postop + Preop (Scoli CT) + HW -> Causal
        causal_out = out_dir / f"scoliosis_cobb{angle}_causal.nii.gz"
        
        if not run_step_via_import_wrapper(
            CAUSAL_SCRIPT,
            "run_causal_simulation",
            {"post_op_path": str(postop_out), "pre_op_path": str(scoli_ct), "hw_path": str(hw_path), "out_path": str(causal_out)},
            f"Causal_{angle}"
        ): continue
        
    logger.info(f"=== Processed {patient_id} ===")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default="/gscratch/scrubbed/june0604/wisespine/VerSe/dataset-01training/rawdata", help="Path to VerSe rawdata")
    parser.add_argument("--output_root", type=str, default="/gscratch/scrubbed/june0604/wisespine/outputs/batch_run", help="Output directory")
    parser.add_argument("--n_patients", type=int, default=1, help="Number of patients to process")
    args = parser.parse_args()
    
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    # Scan for patients
    patients = sorted([d for d in input_root.iterdir() if d.is_dir()])
    logger.info(f"Found {len(patients)} patients in {input_root}")
    
    to_process = patients[:args.n_patients]
    
    for pat in to_process:
        process_patient(pat, output_root)
        
    logger.info("Batch Processing Complete.")

if __name__ == "__main__":
    main()
