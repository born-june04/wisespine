"""
WiseSpine Project Configuration
Loads settings from .env file for consistent path management across all scripts.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
PROJECT_ROOT = Path(__file__).parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
    print(f"✓ Loaded configuration from {ENV_FILE}")
else:
    # Try .env.example as fallback
    ENV_EXAMPLE = PROJECT_ROOT / ".env.example"
    if ENV_EXAMPLE.exists():
        load_dotenv(ENV_EXAMPLE)
        print(f"⚠ Using .env.example (copy to .env for custom settings)")
    else:
        print(f"⚠ No .env file found, using defaults")

# ============================================================================
# Directory Paths
# ============================================================================

WISESPINE_ROOT = Path(os.getenv("WISESPINE_ROOT", PROJECT_ROOT))
VERSE_DATA_DIR = Path(os.getenv("VERSE_DATA_DIR", WISESPINE_ROOT / "VerSe"))
OUTPUTS_DIR = Path(os.getenv("OUTPUTS_DIR", WISESPINE_ROOT / "outputs"))
MESHES_DIR = Path(os.getenv("MESHES_DIR", OUTPUTS_DIR / "meshes"))

# Phase-specific directories
PHASE3_OUTPUT_DIR = Path(os.getenv("PHASE3_OUTPUT_DIR", OUTPUTS_DIR / "phase3_physics_fracture"))
PHASE4_OUTPUT_DIR = Path(os.getenv("PHASE4_OUTPUT_DIR", OUTPUTS_DIR / "phase4_surgical_artifacts"))

# TotalSegmentator
TOTALSEG_EVAL_DIR = Path(os.getenv("TOTALSEG_EVAL_DIR", WISESPINE_ROOT / "totalseg_eval"))

# ============================================================================
# Default Parameters
# ============================================================================

DEFAULT_SUBJECT = os.getenv("DEFAULT_SUBJECT", "sub-verse563")
DEFAULT_VERTEBRA = os.getenv("DEFAULT_VERTEBRA", "L1")

# CT imaging
CT_SPACING = tuple(map(float, os.getenv("CT_SPACING", "1.5,1.5,1.5").split(",")))

# PyBullet physics
PYBULLET_TIMESTEP = float(os.getenv("PYBULLET_TIMESTEP", "0.01"))
PYBULLET_GRAVITY = float(os.getenv("PYBULLET_GRAVITY", "0.0"))

# RL training
RL_TIMESTEPS = int(os.getenv("RL_TIMESTEPS", "50000"))
RL_LEARNING_RATE = float(os.getenv("RL_LEARNING_RATE", "0.0003"))

# Surgical artifacts (moderate settings)
STREAK_INTENSITY = float(os.getenv("STREAK_INTENSITY", "1000"))
BLOOMING_SIGMA = float(os.getenv("BLOOMING_SIGMA", "2.0"))
CORRUPTION_RADIUS = float(os.getenv("CORRUPTION_RADIUS", "10"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# ============================================================================
# Helper Functions
# ============================================================================

def get_verse_ct_path(subject: str) -> Path:
    """Get path to VerSe CT image."""
    return VERSE_DATA_DIR / "rawdata" / subject / f"{subject}_ct.nii.gz"

def get_verse_seg_path(subject: str) -> Path:
    """Get path to VerSe segmentation mask."""
    return VERSE_DATA_DIR / "derivatives" / subject / f"{subject}_msk.nii.gz"

def get_mesh_path(subject: str, source: str = "ts") -> Path:
    """Get path to mesh directory (gt or ts)."""
    return MESHES_DIR / subject / source

def get_phase3_output_path(subdir: str = "") -> Path:
    """Get Phase 3 output path with optional subdirectory."""
    path = PHASE3_OUTPUT_DIR / subdir if subdir else PHASE3_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_phase4_output_path(subdir: str = "") -> Path:
    """Get Phase 4 output path with optional subdirectory."""
    path = PHASE4_OUTPUT_DIR / subdir if subdir else PHASE4_OUTPUT_DIR
    path.mkdir(parents=True, exist_ok=True)
    return path

def ensure_dir(path: Path) -> Path:
    """Ensure directory exists."""
    path.mkdir(parents=True, exist_ok=True)
    return path

# ============================================================================
# Validate Configuration
# ============================================================================

def validate_config():
    """Validate that required directories exist."""
    errors = []
    
    if not WISESPINE_ROOT.exists():
        errors.append(f"WISESPINE_ROOT not found: {WISESPINE_ROOT}")
    
    if not VERSE_DATA_DIR.exists():
        errors.append(f"VERSE_DATA_DIR not found: {VERSE_DATA_DIR}")
    
    # Create output directories if they don't exist
    for dir_path in [OUTPUTS_DIR, PHASE3_OUTPUT_DIR, PHASE4_OUTPUT_DIR]:
        ensure_dir(dir_path)
    
    if errors:
        print("⚠ Configuration Warnings:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    print("✓ Configuration validated successfully")
    return True

# ============================================================================
# Display Configuration
# ============================================================================

def print_config():
    """Print current configuration."""
    print("\n" + "="*60)
    print("WiseSpine Configuration")
    print("="*60)
    print(f"Project Root:     {WISESPINE_ROOT}")
    print(f"VerSe Data:       {VERSE_DATA_DIR}")
    print(f"Outputs:          {OUTPUTS_DIR}")
    print(f"Phase 3 Output:   {PHASE3_OUTPUT_DIR}")
    print(f"Phase 4 Output:   {PHASE4_OUTPUT_DIR}")
    print(f"\nDefaults:")
    print(f"  Subject:        {DEFAULT_SUBJECT}")
    print(f"  Vertebra:       {DEFAULT_VERTEBRA}")
    print(f"  CT Spacing:     {CT_SPACING}")
    print("="*60 + "\n")

if __name__ == "__main__":
    print_config()
    validate_config()

