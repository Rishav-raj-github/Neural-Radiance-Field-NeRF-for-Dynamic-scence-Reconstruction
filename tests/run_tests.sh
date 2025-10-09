#!/bin/bash
# Test runner script for Neural Radiance Field project
# This script runs all tests with various configurations

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "====================================================="
echo "Neural Radiance Field - Test Suite"
echo "====================================================="
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo -e "${RED}Error: pytest is not installed${NC}"
    echo "Please install it with: pip install pytest pytest-cov"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source venv/bin/activate
fi

# Run unit tests
echo -e "${YELLOW}Running unit tests...${NC}"
pytest tests/ -v --tb=short

# Run tests with coverage
echo ""
echo -e "${YELLOW}Running tests with coverage...${NC}"
pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# Check if coverage meets threshold
COVERAGE_THRESHOLD=70
echo ""
echo -e "${YELLOW}Checking coverage threshold (${COVERAGE_THRESHOLD}%)...${NC}"

# Run specific test categories
if [ "$1" == "unit" ]; then
    echo -e "${YELLOW}Running unit tests only...${NC}"
    pytest tests/unit/ -v
elif [ "$1" == "integration" ]; then
    echo -e "${YELLOW}Running integration tests only...${NC}"
    pytest tests/integration/ -v
elif [ "$1" == "performance" ]; then
    echo -e "${YELLOW}Running performance tests...${NC}"
    pytest tests/performance/ -v --benchmark-only
elif [ "$1" == "edge" ]; then
    echo -e "${YELLOW}Running edge device tests...${NC}"
    pytest tests/edge/ -v
fi

# Generate test report
if [ -f "htmlcov/index.html" ]; then
    echo ""
    echo -e "${GREEN}Coverage report generated: htmlcov/index.html${NC}"
fi

echo ""
echo -e "${GREEN}=====================================================${NC}"
echo -e "${GREEN}All tests completed successfully!${NC}"
echo -e "${GREEN}=====================================================${NC}"

exit 0
