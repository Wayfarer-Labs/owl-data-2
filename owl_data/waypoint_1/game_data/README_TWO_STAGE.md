# Two-Stage Game Data Pipeline

This document describes the new two-stage pipeline architecture for processing game data TAR files and generating quality manifests.

## Overview

The original single-stage pipeline required re-downloading and re-extracting all TAR files whenever you wanted to experiment with different quality check methods. The new two-stage approach separates the expensive extraction process from the quality checks, enabling faster iteration.

## Architecture

### Stage 1: Extraction Pipeline
- **Input**: TAR files from `game-data` bucket
- **Process**: Download TAR → Extract video data → Save as .pt files
- **Output**: .pt files in `game-data-manifest` bucket
- **Purpose**: One-time expensive extraction and preprocessing

### Stage 2: Manifest Pipeline  
- **Input**: .pt files from `game-data-manifest` bucket
- **Process**: Load ExtractedData → Run quality checks → Generate manifest
- **Output**: Parquet manifest file
- **Purpose**: Fast iteration on quality check methods

## Benefits

1. **Faster Iteration**: Modify quality checks without re-downloading/re-extracting TAR files
2. **Cost Savings**: Reduced bandwidth and compute costs for experimentation
3. **Modularity**: Each stage can be run independently
4. **Resumability**: Skip already processed files in extraction stage
5. **Parallel Development**: Multiple researchers can experiment with quality checks using the same extracted data

## File Structure

```
pipeline/
├── extraction_pipeline.py      # Stage 1: TAR → .pt files
├── manifest_pipeline.py        # Stage 2: .pt files → manifest
├── two_stage_orchestrator.py   # Main orchestrator script
├── pt_utils.py                 # Serialization utilities
├── s3_utils.py                 # S3 upload/download utilities
├── pipeline.py                 # Original pipeline + quality checks
├── processor.py                # Original processor logic
├── manifest_utils.py           # Manifest creation utilities
└── README_TWO_STAGE.md         # This file
```

## Usage

### Run Both Stages (Full Pipeline)
```bash
python two_stage_orchestrator.py --stage both --task-list task_list.txt
```

### Run Only Extraction Stage
```bash
python two_stage_orchestrator.py --stage extraction --task-list task_list.txt
```

### Run Only Manifest Stage
```bash
python two_stage_orchestrator.py --stage manifest --output-path /path/to/manifest.parquet
```

### Command Line Options

- `--stage`: Which stage to run (`extraction`, `manifest`, or `both`)
- `--source-bucket`: Source bucket for TAR files (default: `game-data`)
- `--manifest-bucket`: Bucket for .pt files (default: `game-data-manifest`)
- `--task-list`: File with TAR S3 keys to process (default: `task_list.txt`)
- `--output-path`: Output path for parquet manifest
- `--prefix`: Filter .pt files by prefix in manifest stage
- `--no-skip-existing`: Don't skip existing .pt files in extraction
- `--log-level`: Logging verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`)

## Data Format

### ExtractedData Structure
The .pt files contain serialized `ExtractedData` objects with:
```python
@dataclass
class ExtractedData:
    s3_key: str                           # Original TAR S3 key
    video_id: str                         # Video identifier
    video_metadata: dict                  # FFmpeg metadata
    session_metadata: dict                # Game session metadata
    sampled_frames: dict[str, np.ndarray] # Strided frame samples
```

### S3 Key Mapping
- TAR file: `path/to/video.tar`
- PT file: `path/to/video.pt`

## Quality Checks

The current quality checks include:
- `is_video_mostly_dark`: Brightness analysis
- `is_dpi_scale_issue`: Letterboxing detection  
- `video_menu_percent`: Menu time estimation
- `is_video_mostly_menu`: Menu threshold check

To add new quality checks:
1. Add check function to `checks.py`
2. Update `_run_all_quality_checks()` in `pipeline.py`
3. Update manifest schema in `manifest_utils.py` if needed

## Error Handling

- **Extraction Stage**: Failed extractions create error logs but don't stop pipeline
- **Manifest Stage**: Failed quality checks create error records in manifest
- **Resumability**: Both stages can resume from where they left off
- **Logging**: Comprehensive logging to files and console

## Performance Considerations

### Extraction Stage
- CPU intensive (video processing)
- Network intensive (TAR downloads)
- Storage intensive (.pt file uploads)
- Runs once per TAR file

### Manifest Stage  
- CPU intensive (quality checks)
- Network intensive (.pt downloads)
- Much faster than extraction stage
- Can be run multiple times with different quality check logic

## Monitoring

Check logs for:
- Processing rates and throughput
- Error rates and failure patterns
- Queue sizes and thread utilization
- S3 upload/download performance

## Migration from Original Pipeline

1. **Existing Workflow**: Continue using `pipeline.py` for small-scale work
2. **New Workflow**: Use two-stage approach for large-scale processing
3. **Compatibility**: Both approaches produce identical manifest schemas
4. **Gradual Migration**: Start with extraction stage, then switch to manifest stage

## Example Workflows

### Initial Setup (Extract All Data)
```bash
# Extract all TAR files to .pt files (run once)
python two_stage_orchestrator.py --stage extraction --task-list all_tars.txt
```

### Experiment with Quality Checks
```bash
# Modify quality check logic in checks.py or pipeline.py
# Then regenerate manifest quickly:
python two_stage_orchestrator.py --stage manifest --output-path experiment_v2.parquet
```

### Process Specific Subset
```bash
# Extract only recent data
python two_stage_orchestrator.py --stage extraction --task-list recent_tars.txt

# Generate manifest for specific date range
python two_stage_orchestrator.py --stage manifest --prefix "2024/01/" --output-path jan_2024.parquet
```

## Troubleshooting

### Common Issues
1. **Missing .pt files**: Run extraction stage first
2. **S3 permissions**: Ensure access to both buckets
3. **Memory issues**: Reduce batch sizes or processor count
4. **Slow performance**: Check network connectivity and CPU utilization

### Recovery
- **Failed extraction**: Re-run with same task list (skips existing)
- **Failed manifest**: Delete partial manifest and re-run
- **Corrupted .pt files**: Delete specific files and re-extract

## Future Enhancements

- **Caching**: Local disk caching for frequently accessed .pt files
- **Compression**: Compress .pt files to reduce storage costs
- **Metadata**: Add extraction timestamps and version info
- **Validation**: Integrity checks for .pt files
- **Monitoring**: Metrics and alerting for production deployments 