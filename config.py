# config.py
SECTION_WEIGHTS = {
    "experience": 1.6,
    "projects": 1.4,
    "skills": 1.2,
    "education": 0.6,
    "general": 1.0
}

# Model constraints
MODEL_NAME = 'all-MiniLM-L6-v2'
MAX_SEQ_LENGTH = 256  # Actual model limit
CHUNK_OVERLAP_TOKENS = 32

# Scoring calibration
SIGMOID_K = 10   # Steepness of the curve
SIGMOID_X0 = 0.5 # Midpoint of the curve