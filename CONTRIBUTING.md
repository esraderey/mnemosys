# Contributing to MNEME

Thank you for your interest in contributing to MNEME! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git
- Basic understanding of PyTorch and neural networks

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/MNEME---Motor-de-Memoria-Neural-M-rfica.git
   cd MNEME---Motor-de-Memoria-Neural-M-rfica
   ```

2. **Install development dependencies**
   ```bash
   make dev-setup
   # or manually:
   pip install -e .[dev,all]
   pre-commit install
   ```

3. **Run tests to ensure everything works**
   ```bash
   make test
   ```

## 📝 Contribution Guidelines

### Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Write docstrings for all public functions
- Keep functions small and focused
- Use meaningful variable names

### Commit Messages

Use conventional commit format:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Test additions/changes
- `chore`: Maintenance tasks

Examples:
```
feat(core): add context deduplication system
fix(security): resolve HMAC verification issue
docs(api): update serialization documentation
```

### Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following our style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Run checks**
   ```bash
   make check-all
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat(scope): your commit message"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
make test

# Run specific test categories
make test-gpu      # GPU tests
make test-security # Security tests

# Run with coverage
pytest tests/ -v --cov=src/mneme --cov-report=html
```

### Writing Tests

- Place tests in the `tests/` directory
- Use descriptive test names
- Test both success and failure cases
- Include edge cases and error conditions
- Use fixtures for common test data

Example test structure:
```python
import pytest
import torch
from mneme import ZSpace, MnemeConfig

class TestZSpace:
    def test_basic_operations(self):
        """Test basic ZSpace operations."""
        config = MnemeConfig()
        with ZSpace(config) as zspace:
            tensor = torch.randn(10, 10)
            desc = zspace.register("test", tensor)
            loaded = zspace.load("test")
            assert torch.allclose(tensor, loaded)
    
    def test_error_handling(self):
        """Test error handling."""
        config = MnemeConfig()
        with ZSpace(config) as zspace:
            with pytest.raises(ValueError):
                zspace.load("nonexistent")
```

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings
- Include type hints
- Document all public APIs
- Provide usage examples

Example:
```python
def process_context_for_deduplication(
    self, 
    context_id: str, 
    tensor: torch.Tensor, 
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Process context for deduplication.
    
    Args:
        context_id: Unique identifier for the context
        tensor: Input tensor to process
        metadata: Optional metadata dictionary
        
    Returns:
        Dictionary containing deduplication results
        
    Example:
        >>> result = zspace.process_context_for_deduplication(
        ...     "my_context", tensor, {"type": "feature_map"}
        ... )
        >>> print(result["deduplicated"])
    """
```

### Documentation Updates

- Update README.md for major changes
- Add examples for new features
- Update API documentation
- Include performance benchmarks

## 🔒 Security

### Security Guidelines

- Never commit secrets or API keys
- Use secure coding practices
- Validate all inputs
- Handle errors gracefully
- Follow the principle of least privilege

### Security Testing

```bash
# Run security checks
make security

# Check for vulnerabilities
safety check

# Run bandit security linter
bandit -r src/ -f json -o bandit-report.json
```

## 🚀 Performance

### Performance Guidelines

- Profile code before optimizing
- Use appropriate data structures
- Minimize memory allocations
- Leverage vectorization when possible
- Consider GPU acceleration

### Benchmarking

```bash
# Run performance benchmarks
make benchmark

# Profile specific functions
python -m cProfile examples/example_mneme.py
```

## 🐛 Bug Reports

### Reporting Bugs

When reporting bugs, please include:

1. **Environment information**
   - Python version
   - Operating system
   - MNEME version
   - Dependencies versions

2. **Reproduction steps**
   - Minimal code example
   - Expected behavior
   - Actual behavior
   - Error messages

3. **Additional context**
   - Screenshots if applicable
   - Related issues
   - Workarounds if any

### Bug Report Template

```markdown
**Bug Description**
Brief description of the bug.

**Environment**
- Python: 3.11.0
- OS: Windows 10
- MNEME: 2.0.0
- PyTorch: 2.0.0

**Reproduction Steps**
1. Run the following code:
```python
# Your code here
```
2. Observe the error

**Expected Behavior**
What should happen.

**Actual Behavior**
What actually happens.

**Error Message**
```
Traceback (most recent call last):
  File "...", line ..., in ...
    ...
Error: ...
```
```

## 💡 Feature Requests

### Suggesting Features

When suggesting features, please include:

1. **Use case description**
   - What problem does it solve?
   - Who would benefit from it?
   - How would it be used?

2. **Proposed solution**
   - High-level design
   - API considerations
   - Implementation approach

3. **Alternatives considered**
   - Other solutions you've considered
   - Why this approach is better

### Feature Request Template

```markdown
**Feature Description**
Brief description of the feature.

**Use Case**
Describe the use case and problem it solves.

**Proposed Solution**
Describe your proposed solution.

**Alternatives**
Describe alternatives you've considered.

**Additional Context**
Any other relevant information.
```

## 🏷️ Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests pass
- [ ] Documentation updated
- [ ] Changelog updated
- [ ] Version bumped
- [ ] Security review completed
- [ ] Performance benchmarks updated

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: msc.framework@gmail.com

### Code Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Testing** in different environments
4. **Documentation** review
5. **Security** review if applicable

## 🏆 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation
- GitHub contributors list

## 📄 License

By contributing to MNEME, you agree that your contributions will be licensed under the same license as the project (BUSL-1.1).

## 🙏 Thank You

Thank you for contributing to MNEME! Your contributions help make this project better for everyone.

---

*"La mejor compresión no es guardar los datos, sino guardar la receta para recrearlos."* – Esraderey y Raul Cruz Acosta
