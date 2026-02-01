#!/bin/bash

# Script to build ARC dataset with reduced augmentation
# This creates arc1concept-aug-500 (half the augmentation of baseline)

set -e  # Exit on error

echo "================================================"
echo "Building ARC-AGI-1 Dataset with Reduced Augmentation"
echo "================================================"
echo ""

# Configuration
NUM_AUG=${1:-500}  # Default 500, can override with argument
OUTPUT_DIR="data/arc1concept-aug-${NUM_AUG}"

echo "Configuration:"
echo "  - Augmentations per puzzle: ${NUM_AUG}"
echo "  - Output directory: ${OUTPUT_DIR}"
echo ""

# Check if input data exists
if [ ! -d "kaggle/combined" ]; then
    echo "❌ Error: kaggle/combined directory not found!"
    echo "Please ensure ARC-AGI data is in kaggle/combined/"
    exit 1
fi

echo "✓ Input data found"
echo ""

# Build dataset
echo "Building dataset (this may take a few minutes)..."
python -m dataset.build_arc_dataset \
  --input-file-prefix kaggle/combined/arc-agi \
  --output-dir "${OUTPUT_DIR}" \
  --subsets training evaluation concept \
  --test-set-name evaluation \
  --num-aug ${NUM_AUG}

echo ""
echo "================================================"
echo "✅ Dataset created successfully!"
echo "================================================"
echo ""
echo "Dataset location: ${OUTPUT_DIR}"
echo ""

# Show dataset statistics
echo "Dataset statistics:"
if [ -f "${OUTPUT_DIR}/train/dataset.json" ]; then
    python3 << EOF
import json
with open("${OUTPUT_DIR}/train/dataset.json") as f:
    meta = json.load(f)
    print(f"  - Total puzzles: {meta['total_puzzles']}")
    print(f"  - Total groups: {meta['total_groups']}")
    print(f"  - Mean examples per puzzle: {meta['mean_puzzle_examples']:.1f}")
    print(f"  - Estimated total examples: ~{int(meta['total_puzzles'] * meta['mean_puzzle_examples'])}")
EOF
else
    echo "  - Unable to read metadata"
fi

echo ""
echo "Next steps:"
echo "  1. Upload ${OUTPUT_DIR} to Kaggle as a dataset"
echo "  2. Use in training: data_paths=\"[/kaggle/input/arc1concept-aug-${NUM_AUG}]\""
echo ""
echo "To create different augmentation levels, run:"
echo "  ./scripts/build_arc_reduced_aug.sh 300   # 300× augmentation"
echo "  ./scripts/build_arc_reduced_aug.sh 750   # 750× augmentation"
echo ""
