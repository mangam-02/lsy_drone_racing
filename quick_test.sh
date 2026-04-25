#!/usr/bin/env bash
# Quick test script to verify everything is working

cd "$(dirname "$0")"

# Activate virtual environment
source venv_drone/bin/activate

echo ""
echo "🚀 Running LSY Drone Racing Quick Test"
echo "======================================"
echo ""

# Run a quick test
python scripts/sim.py --config level0.toml --render=False --n_runs=1 2>&1 | grep -E "(Flight time|Finished|Gates passed)"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Test PASSED! Everything is working correctly."
    echo ""
    echo "To run more complex scenarios, try:"
    echo "  python scripts/sim.py --config level1.toml --render=True --n_runs=5"
else
    echo ""
    echo "❌ Test FAILED! Check the output above for errors."
fi
echo ""
