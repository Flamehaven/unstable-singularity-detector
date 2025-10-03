#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Unstable Singularity Detector - Setup Script
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

set -e  # Exit on error

echo "🔬 Unstable Singularity Detector - Setup Script"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. Environment File Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env

    # Generate random password for web interface (optional)
    if command -v openssl &> /dev/null; then
        RANDOM_PASSWORD=$(openssl rand -base64 16)

        # macOS and Linux compatible sed
        if [[ "$OSTYPE" == "darwin"* ]]; then
            sed -i '' "s/# WEB_PASSWORD=changeme_in_production/WEB_PASSWORD=$RANDOM_PASSWORD/" .env
        else
            sed -i "s/# WEB_PASSWORD=changeme_in_production/WEB_PASSWORD=$RANDOM_PASSWORD/" .env
        fi

        echo "✅ .env file created with random web password"
    else
        echo "✅ .env file created (manual password setup required)"
    fi

    echo "⚠️  Please review and update .env before running!"
else
    echo "⚠️  .env file already exists, skipping..."
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Directory Structure
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "📁 Creating required directories..."
mkdir -p logs checkpoints data outputs configs results
mkdir -p data/reference data/experiments
mkdir -p outputs/plots outputs/checkpoints outputs/reports

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Permission Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🔒 Setting secure permissions..."
if [ -f .env ]; then
    chmod 600 .env
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Python Environment Check
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo "🔍 Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found! Please install Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "✅ Found Python $PYTHON_VERSION"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Virtual Environment (Optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
read -p "Create Python virtual environment? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if [ ! -d "venv" ]; then
        echo "🐍 Creating virtual environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
        echo "   Activate with: source venv/bin/activate"
    else
        echo "⚠️  Virtual environment already exists"
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Dependencies Installation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
read -p "Install Python dependencies now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📦 Installing dependencies..."

    # Activate venv if it exists
    if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi

    pip install --upgrade pip
    pip install -r requirements.txt
    pip install -e .

    echo "✅ Dependencies installed"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. Security Tools (Optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
read -p "Install security scanning tools? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🛡️  Installing security tools..."
    pip install safety bandit

    echo "🔍 Running initial security scan..."
    safety check || echo "⚠️  Some vulnerabilities found, please review"

    echo "✅ Security tools installed"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 8. Git Hooks (Optional)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if [ -d ".git" ]; then
    read -p "Install pre-commit hooks? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            echo "✅ Pre-commit hooks installed"
        else
            echo "⚠️  pre-commit not found, skipping..."
        fi
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 9. Test Suite
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
read -p "Run test suite to verify installation? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running tests..."
    pytest tests/ -v --tb=short || echo "⚠️  Some tests failed, please review"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 10. Summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo ""
echo "✅ Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Next steps:"
echo "  1. Review and update .env file"
echo "  2. Activate virtual environment (if created):"
echo "     source venv/bin/activate"
echo "  3. Run examples:"
echo "     python examples/basic_detection_demo.py"
echo "  4. Start web interface:"
echo "     python src/web_interface.py"
echo "  5. Run with Docker:"
echo "     docker-compose up --build"
echo ""

if [ -f .env ]; then
    echo "🔐 Configuration file: .env"
    if grep -q "WEB_PASSWORD" .env; then
        echo "   Web password has been set (check .env file)"
    fi
fi

echo ""
echo "📚 Documentation:"
echo "  - README.md"
echo "  - docs/REPRODUCTION.md"
echo "  - docs/FUNNEL_INFERENCE_GUIDE.md"
echo ""
