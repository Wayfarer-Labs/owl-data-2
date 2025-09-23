"""
Two-Stage Pipeline Orchestrator

This script orchestrates the two-stage pipeline:
1. Extraction Pipeline: TAR files → ExtractedData → .pt files
2. Manifest Pipeline: .pt files → Quality checks → Parquet manifest

Usage examples:
- Run only extraction: python two_stage_orchestrator.py --stage extraction
- Run only manifest: python two_stage_orchestrator.py --stage manifest
- Run both stages: python two_stage_orchestrator.py --stage both
"""

import argparse
import logging
import os
from typing import List

from owl_data.waypoint_1.game_data.pipeline.tar_to_pt_pipeline import run_extraction_pipeline
from owl_data.waypoint_1.game_data.pipeline.pt_to_manifest_pipeline import run_manifest_pipeline


def setup_logging(log_level: str = "INFO"):
    """Set up logging configuration."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('two_stage_pipeline.log')
        ]
    )


def load_task_list(task_list_file: str) -> List[str]:
    """Load TAR file S3 keys from a task list file."""
    if not os.path.exists(task_list_file):
        raise FileNotFoundError(f"Task list file not found: {task_list_file}")
    
    with open(task_list_file, 'r') as f:
        tasks = [line.strip() for line in f if line.strip()]
    
    logging.info(f"Loaded {len(tasks)} tasks from {task_list_file}")
    return tasks


def run_extraction_stage(
    source_bucket: str,
    manifest_bucket: str,
    task_list_file: str,
    skip_existing: bool = True
):
    """Run the extraction pipeline stage."""
    logging.info("=" * 60)
    logging.info("STARTING EXTRACTION PIPELINE")
    logging.info("=" * 60)
    
    tasks = load_task_list(task_list_file)
    
    run_extraction_pipeline(
        source_bucket=source_bucket,
        manifest_bucket=manifest_bucket,
        master_task_list=tasks,
        skip_existing=skip_existing
    )
    
    logging.info("EXTRACTION PIPELINE COMPLETED")


def run_manifest_stage(
    manifest_bucket: str,
    output_path: str,
    prefix: str = ""
):
    """Run the manifest generation pipeline stage."""
    logging.info("=" * 60)
    logging.info("STARTING MANIFEST PIPELINE")
    logging.info("=" * 60)
    
    run_manifest_pipeline(
        manifest_bucket=manifest_bucket,
        output_path=output_path,
        pt_s3_keys=None,  # Process all .pt files
        prefix=prefix
    )
    
    logging.info("MANIFEST PIPELINE COMPLETED")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage pipeline for processing game data TAR files"
    )
    
    parser.add_argument(
        "--stage",
        choices=["extraction", "manifest", "both"],
        required=True,
        help="Which pipeline stage(s) to run"
    )
    
    parser.add_argument(
        "--source-bucket",
        default="game-data",
        help="Source bucket containing TAR files (default: game-data)"
    )
    
    parser.add_argument(
        "--manifest-bucket",
        default="game-data-manifest",
        help="Manifest bucket for .pt files (default: game-data-manifest)"
    )
    
    parser.add_argument(
        "--task-list",
        default="task_list.txt",
        help="File containing list of TAR S3 keys to process (default: task_list.txt)"
    )
    
    parser.add_argument(
        "--output-path",
        default="/mnt/data/sami/manifests/gamedata_quality_manifest.parquet",
        help="Output path for the parquet manifest file"
    )
    
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix filter for .pt files in manifest stage (default: no filter)"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Don't skip existing .pt files in extraction stage"
    )
    
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    try:
        if args.stage in ["extraction", "both"]:
            run_extraction_stage(
                source_bucket=args.source_bucket,
                manifest_bucket=args.manifest_bucket,
                task_list_file=args.task_list,
                skip_existing=not args.no_skip_existing
            )
        
        if args.stage in ["manifest", "both"]:
            run_manifest_stage(
                manifest_bucket=args.manifest_bucket,
                output_path=args.output_path,
                prefix=args.prefix
            )
        
        logging.info("=" * 60)
        logging.info("TWO-STAGE PIPELINE COMPLETED SUCCESSFULLY")
        logging.info("=" * 60)
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main() 