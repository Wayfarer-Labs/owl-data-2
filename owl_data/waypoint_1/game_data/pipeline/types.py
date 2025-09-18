import numpy as np
from dataclasses import dataclass


@dataclass
class ExtractedData:
    """Holds only the essential, lightweight data for quality checks."""
    s3_key: str
    metadata: dict
    sampled_frames: dict[str, np.ndarray]
