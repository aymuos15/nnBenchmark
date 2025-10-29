#!/bin/bash
# Integration test for nnBenchmark
# Note: Only tests on Dataset001_Hippo (small dataset, ~260 images)

set -e  # Exit on error

echo "========================================"
echo "nnBenchmark Integration Test"
echo "========================================"

CONFIG_PATH="/tmp/dataset001_hippo_auto.yaml"

# Test 1: Automatic planning (nnU-Net-style with automatic dataset preparation)
echo ""
echo "Test 1/7: Automatic experiment planning..."
nnBench.plan --dataset Dataset001_Hippo --output "$CONFIG_PATH"
echo "✓ Planning completed"

# Test 2: Modify config for quick testing (reduce epochs from 200 to 3, enable mixed precision and caching)
echo ""
echo "Test 2/7: Adapting config for quick testing..."
sed -i 's/epochs: 200/epochs: 3/' "$CONFIG_PATH"
sed -i 's/val_interval: 5/val_interval: 2/' "$CONFIG_PATH"
# Enable mixed precision for faster training and memory efficiency
sed -i 's/mixed_precision: false/mixed_precision: true/' "$CONFIG_PATH"
# Enable caching for faster training (default is disabled for stability)
sed -i 's/enabled: false/enabled: true/' "$CONFIG_PATH"
sed -i 's/cache_rate: false/cache_rate: 0.15/' "$CONFIG_PATH"
echo "✓ Config adapted (epochs: 200 -> 3, val_interval: 5 -> 2, mixed_precision: true, caching: enabled)"
echo "  Note: Defaults are conservative (1 worker, cache disabled) for stability"

# Test 3: Initial Training (first 3 epochs)
echo ""
echo "Test 3/7: Training model (first 3 epochs with mixed precision)..."
nnBench.train --config "$CONFIG_PATH"
echo "✓ Initial training completed (3 epochs)"

# Test 4: Modify config to extend training to 6 epochs total
echo ""
echo "Test 4/7: Extending training target to 6 epochs..."
sed -i 's/epochs: 3/epochs: 6/' "$CONFIG_PATH"
echo "✓ Config updated (target epochs: 3 -> 6)"

# Test 5: Resume Training (continue from epoch 3 to epoch 6)
echo ""
echo "Test 5/7: Resuming training from checkpoint (epochs 4-6)..."
nnBench.train --config "$CONFIG_PATH" --continue
echo "✓ Resumed training completed (total: 6 epochs)"

# Test 6: Testing
echo ""
echo "Test 6/7: Testing model..."
# Use validation split since this dataset doesn't have a dedicated test set
nnBench.test --config "$CONFIG_PATH" --use-val-split
echo "✓ Testing completed"

# Test 7: Plotting
echo ""
echo "Test 7/7: Generating plots..."
# Plot both training and test results (run after test to include test plots)
nnBench.plot --config "$CONFIG_PATH"
echo "✓ Plotting completed"

echo ""
echo "========================================"
echo "All integration tests passed! ✓"
echo "========================================"
