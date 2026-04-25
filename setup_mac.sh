#!/usr/bin/env bash
set -euo pipefail

echo "🚀 LSY Drone Racing - macOS Setup Script"
echo "========================================"
echo ""

# Determine Python version
PYTHON_VERSION=$(python3.12 --version 2>/dev/null || python3.13 --version 2>/dev/null || python3 --version 2>/dev/null)
echo "✓ Found Python: $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "📦 Creating Python virtual environment..."
if [ ! -d venv_drone ]; then
    python3.12 -m venv venv_drone 2>/dev/null || python3 -m venv venv_drone
    echo "✓ Virtual environment created: venv_drone/"
else
    echo "✓ Virtual environment already exists: venv_drone/"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv_drone/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip and friends
echo "📌 Upgrading pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1
echo "✓ Upgraded pip, setuptools, wheel"
echo ""

# Install base dependencies
echo "📥 Installing base dependencies..."
pip install fire numpy toml > /dev/null 2>&1
echo "✓ Installed: fire, numpy, toml"
echo ""

# Install simulation dependencies
echo "📥 Installing simulation dependencies (this may take a few minutes)..."
pip install 'gymnasium[array-api]>=1.2.0' 'ml-collections>=1.0' 'packaging>=24.0' 'jax[cpu]>=0.7' \
    drone-models drone-controllers crazyflow warp-lang > /dev/null 2>&1
echo "✓ Installed simulation packages"
echo ""

# Install acados_template
echo "📥 Installing ACADOS template (Python interface)..."
pip install -e acados/interfaces/acados_template > /dev/null 2>&1
echo "✓ Installed ACADOS template"
echo ""

# Install the project itself
echo "📥 Installing lsy_drone_racing package..."
pip install -e . > /dev/null 2>&1
echo "✓ Installed lsy_drone_racing package"
echo ""

# Run a quick test
echo "🧪 Running simulation test..."
python scripts/sim.py --config level0.toml --render=False --n_runs=1 > /tmp/sim_test.log 2>&1
if [ $? -eq 0 ]; then
    echo "✓ Simulation test PASSED!"
    echo ""
    FLIGHT_TIME=$(grep "Flight time" /tmp/sim_test.log | head -1)
    GATES_PASSED=$(grep "Gates passed" /tmp/sim_test.log | head -1)
    echo "  📊 Test Results:"
    echo "     $FLIGHT_TIME"
    echo "     $GATES_PASSED"
else
    echo "✗ Simulation test FAILED"
    echo "  Check /tmp/sim_test.log for details"
    exit 1
fi
echo ""

# Summary
echo "========================================"
echo "✅ Installation complete!"
echo "========================================"
echo ""
echo "To run simulations, use:"
echo "  source venv_drone/bin/activate"
echo "  python scripts/sim.py --config level0.toml --render=True"
echo ""
echo "Available configs:"
echo "  - level0.toml (Easy: Perfect knowledge)"
echo "  - level1.toml (Medium: Adaptive control)"
echo "  - level2.toml (Hard: Re-planning)"
echo "  - level3.toml (Expert: Online planning)"
echo ""
echo "For more options, run:"
echo "  python scripts/sim.py --help"
echo ""
