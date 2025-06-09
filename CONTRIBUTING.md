# Contributing to HierRAGMed

Thank you for your interest in contributing to HierRAGMed! This document provides guidelines and instructions for contributing.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/hierragmed.git
   cd hierragmed
   ```
3. Create a conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate hierragmed
   ```
4. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
5. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

## Development Workflow

1. Create a new branch for your feature/fix:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Make your changes
3. Run tests:
   ```bash
   pytest
   ```
4. Run linting:
   ```bash
   pre-commit run --all-files
   ```
5. Commit your changes:
   ```bash
   git commit -m "Description of changes"
   ```
6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
7. Create a pull request

## Code Style

- Follow PEP 8 style guide
- Use type hints
- Write docstrings for all functions and classes
- Keep functions small and focused
- Write unit tests for new functionality

## Testing

- Write tests for all new functionality
- Ensure all tests pass before submitting PR
- Maintain or improve test coverage

## Documentation

- Update README.md if needed
- Add docstrings to new functions/classes
- Update configuration documentation if adding new options

## Pull Request Process

1. Update the README.md with details of changes if needed
2. Update the documentation if needed
3. The PR will be merged once you have the sign-off of at least one other developer

## Questions?

Feel free to open an issue for any questions or concerns. 