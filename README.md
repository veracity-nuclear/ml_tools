# Veracity ML Tools

## Table of Contents
1. [Installation Instructions](#installation-instructions)
2. [Developer Tools](#developer-tools)
3. [Contributing](#contributing)

## Installation Instructions
Installing Veracity ML Tools using the following instructions will install the package along with the required dependencies.

### End User Installation
> This will change to PyPI installation once Veracity ML Tools is open source.
```bash
python -m pip install .
```
### Developer Installation
```bash
python3 -m pip install -e .[dev]
```
### Testing Installation (Primarily for CI)
```bash
python -m pip install .[test]
```

## Developer Tools
The configuration settings for the developer tools can be found in `~/ml_tools/pyproject.toml`.

### Linting Python code with pylint
Execute this line from the `~/ml_tools` directory to lint the code with [pylint](https://pypi.org/project/pylint/):
```bash
pylint ./ml_tools
```

### Formatting code with black
Execute this line from the `~/ml_tools` directory to automatically format the code to PEP8 standard using [black](https://pypi.org/project/black/):
```bash
black ./ml_tools
```

## Contributing
Veracity ML Tools is distributed under the terms of the **BEN TO PROVIDE LICENSE!!!!!!!**. If you propose or make a contribution to this repository, you hereby license your contribution to anyone under this same license.
