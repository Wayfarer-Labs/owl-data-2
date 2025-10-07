from dataclasses import dataclass


@dataclass
class ExtractedData:
    """Holds only the essential, lightweight data for quality checks."""
    s3_key: str
    downsampled_video_bytes: list[bytes]
    controls_csv_str: str
    session_metadata: dict
    in_video_metadata: dict
    out_video_metadata: list[dict]