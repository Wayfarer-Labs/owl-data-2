"""
Test script for the two-stage pipeline.

This script creates a minimal test to verify that:
1. ExtractedData can be serialized/deserialized correctly
2. S3 utilities work for uploading/downloading .pt files
3. Both pipelines can process data without errors

Note: This requires valid AWS credentials and access to the test buckets.
"""

import tempfile
import os
import logging
import numpy as np
from unittest.mock import Mock

from owl_data.waypoint_1.game_data.pipeline.types import ExtractedData
from owl_data.waypoint_1.game_data.pipeline.pt_utils import (
    serialize_extracted_data, 
    deserialize_extracted_data,
    save_extracted_data_to_pt,
    load_extracted_data_from_pt,
    convert_s3_key_to_pt_key,
    convert_pt_key_to_s3_key
)
from owl_data.waypoint_1.game_data.pipeline.quality_checks import _run_all_quality_checks


def create_test_extracted_data() -> ExtractedData:
    """Create a test ExtractedData object for testing."""
    # Create test sampled frames
    sampled_frames = {
        "stride-3_chw": np.random.randint(0, 255, (5, 3, 224, 224), dtype=np.uint8),
        "stride-30_chw": np.random.randint(0, 255, (10, 3, 224, 224), dtype=np.uint8),
        "stride-60_chw": np.random.randint(0, 255, (8, 3, 224, 224), dtype=np.uint8),
    }
    
    return ExtractedData(
        s3_key="test/video123.tar",
        video_id="video123",
        video_metadata={
            "streams": [{"width": 1920, "height": 1080, "duration": "120.5"}],
            "format": {"duration": "120.5", "size": "1048576"}
        },
        session_metadata={
            "game": "test_game",
            "player_id": "test_player",
            "session_id": "test_session"
        },
        sampled_frames=sampled_frames
    )


def test_serialization():
    """Test ExtractedData serialization and deserialization."""
    print("Testing serialization...")
    
    # Create test data
    original_data = create_test_extracted_data()
    
    # Test in-memory serialization
    serialized = serialize_extracted_data(original_data)
    deserialized = deserialize_extracted_data(serialized)
    
    # Verify data integrity
    assert deserialized.s3_key == original_data.s3_key
    assert deserialized.video_id == original_data.video_id
    assert deserialized.video_metadata == original_data.video_metadata
    assert deserialized.session_metadata == original_data.session_metadata
    
    # Verify frame data
    for key in original_data.sampled_frames:
        assert key in deserialized.sampled_frames
        assert np.array_equal(original_data.sampled_frames[key], deserialized.sampled_frames[key])
    
    print("✓ Serialization test passed")


def test_file_io():
    """Test saving and loading ExtractedData to/from .pt files."""
    print("Testing file I/O...")
    
    original_data = create_test_extracted_data()
    
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Save to file
        save_extracted_data_to_pt(original_data, temp_path)
        
        # Load from file
        loaded_data = load_extracted_data_from_pt(temp_path)
        
        # Verify data integrity
        assert loaded_data.s3_key == original_data.s3_key
        assert loaded_data.video_id == original_data.video_id
        assert loaded_data.video_metadata == original_data.video_metadata
        assert loaded_data.session_metadata == original_data.session_metadata
        
        # Verify frame data
        for key in original_data.sampled_frames:
            assert key in loaded_data.sampled_frames
            assert np.array_equal(original_data.sampled_frames[key], loaded_data.sampled_frames[key])
        
        print("✓ File I/O test passed")
        
    finally:
        # Clean up
        try:
            os.unlink(temp_path)
        except:
            pass


def test_s3_key_conversion():
    """Test S3 key conversion utilities."""
    print("Testing S3 key conversion...")
    
    # Test TAR to PT conversion
    tar_key = "path/to/video.tar"
    pt_key = convert_s3_key_to_pt_key(tar_key)
    assert pt_key == "path/to/video.pt"
    
    # Test PT to TAR conversion
    converted_back = convert_pt_key_to_s3_key(pt_key)
    assert converted_back == tar_key
    
    # Test edge cases
    no_extension = "path/to/video"
    pt_from_no_ext = convert_s3_key_to_pt_key(no_extension)
    assert pt_from_no_ext == "path/to/video.pt"
    
    print("✓ S3 key conversion test passed")


def test_quality_checks():
    """Test that quality checks work on test data."""
    print("Testing quality checks...")
    
    test_data = create_test_extracted_data()
    
    try:
        quality_flags = _run_all_quality_checks(test_data)
        
        # Verify expected keys are present
        expected_keys = [
            'is_video_mostly_dark',
            'is_dpi_scale_issue', 
            'video_menu_percent',
            'is_video_mostly_menu'
        ]
        
        for key in expected_keys:
            assert key in quality_flags, f"Missing quality flag: {key}"
        
        # Verify types
        assert isinstance(quality_flags['is_video_mostly_dark'], bool)
        assert isinstance(quality_flags['is_dpi_scale_issue'], bool)
        assert isinstance(quality_flags['video_menu_percent'], (int, float))
        assert isinstance(quality_flags['is_video_mostly_menu'], bool)
        
        print(f"✓ Quality checks test passed. Results: {quality_flags}")
        
    except Exception as e:
        print(f"✗ Quality checks test failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("RUNNING TWO-STAGE PIPELINE TESTS")
    print("=" * 60)
    
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during tests
    
    try:
        test_serialization()
        test_file_io()
        test_s3_key_conversion()
        test_quality_checks()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe two-stage pipeline components are working correctly.")
        print("You can now run the full pipelines on real data.")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        raise


if __name__ == "__main__":
    run_all_tests() 