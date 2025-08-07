#!/bin/bash

set -e

# Configuration
VIDEO_PATH="/Users/ericsheen/Downloads/ironfleet.mp4"
TEST_DIR="test_complete_pipeline"

echo "=== Testing Complete Caption Pipeline ==="
echo "Video: $VIDEO_PATH"
echo "Test directory: $TEST_DIR"
echo ""

# Step 1: Create test directory structure
echo "1. Creating test directory structure..."
mkdir -p "$TEST_DIR"/{videos,tensors,captions}
echo "✓ Created directories"

# Step 2: Copy video to test directory
echo ""
echo "2. Copying video to test directory..."
cp "$VIDEO_PATH" "$TEST_DIR/videos/"
echo "✓ Copied video to $TEST_DIR/videos/"

# Step 3: Convert video to RGB tensors
echo ""
echo "3. Converting video to RGB tensors..."
python3 -m owl_data.videos_to_rgb_tensors \
    --input_dir "$TEST_DIR/videos" \
    --output_dir "$TEST_DIR/tensors" \
    --frame_skip 30 \
    --max_frames 50 \
    --resize_to 256 256 \
    --num_gpus 1

echo "✓ Converted video to RGB tensors"

# Step 4: Generate captions from tensors
echo ""
echo "4. Generating captions from RGB tensors..."
python3 -m owl_data.captions_from_tensors \
    --input_dir "$TEST_DIR/tensors" \
    --kernel_size 5 \
    --dilation 2 \
    --stride 3 \
    --num_gpus 1 \
    --overwrite

echo "✓ Generated captions"

# Step 5: Show results
echo ""
echo "5. Results:"
echo "=== Generated Files ==="
echo "RGB Tensor files:"
find "$TEST_DIR/tensors" -name "*_rgb.pt" -exec echo "  {}" \;

echo ""
echo "Caption files:"
find "$TEST_DIR/tensors" -name "*_captions.txt" -exec echo "  {}" \;

echo ""
echo "=== Sample Captions ==="
for caption_file in "$TEST_DIR/tensors"/*_captions.txt; do
    if [ -f "$caption_file" ]; then
        echo "File: $(basename "$caption_file")"
        head -5 "$caption_file"
        echo ""
    fi
done

echo ""
echo "=== Pipeline Complete ==="
echo "Test directory: $TEST_DIR"
echo "You can inspect the results in:"
echo "  - Videos: $TEST_DIR/videos/"
echo "  - RGB Tensors: $TEST_DIR/tensors/*_rgb.pt"
echo "  - Captions: $TEST_DIR/tensors/*_captions.txt"
echo ""
echo "This demonstrates the complete pipeline:"
echo "  Video → RGB Tensors → Captions" 