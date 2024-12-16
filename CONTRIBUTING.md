## Table of Contents

- [Contributing to TemporalScope](#contributing-to-temporalscope)
  - [Contribution Guidelines](#contribution-guidelines)
- [How to Contribute to TemporalScope](#how-to-contribute-to-temporalscope)
- [Issue Tracking](#issue-tracking)
- [Contributing Code](#contributing-code)
  - [Fork the Repository](#fork-the-repository)
  - [Setup Development Environment](#setup-development-environment)
  - [Install Pre-commit Hooks](#install-pre-commit-hooks)
  - [Create a New Branch](#create-a-new-branch)
  - [Make Your Changes](#make-your-changes)
  - [Ensure Code Quality](#ensure-code-quality)
  - [Commit Your Changes](#commit-your-changes)
  - [Submit a Pull Request](#submit-a-pull-request)
  - [After Submitting](#after-submitting)
  - [Documentation](#documentation)
  - [Test Policy](#test-policy)
  - [Development Roadmap \& Changelog](#development-roadmap--changelog)
  - [Workflow for Releasing New Versions](#workflow-for-releasing-new-versions)
  - [Code Style](#code-style)
  - [Reporting Issues \& Requesting Features](#reporting-issues--requesting-features)
  <!-- --8<-- [start:CONTRIBUTING] -->

# Contributing to TemporalScope ⌨️

Thank you for your interest in contributing to TemporalScope! We welcome and appreciate contributions of all types. This guide is designed to help you get started with the contribution process.

---
## Contribution Guidelines

> WARNING: **Important**
> By contributing to this project, you are agreeing to the following conditions

- **Adherence to the Apache License 2.0:**
    - All contributions must comply with the Apache License 2.0.
    - You must ensure that all work contributed is your own original creation, fully independent, or obtained through publicly available academic literature or properly licensed third-party sources.
- **Conflict of Interest and Independence**:
    - By contributing, you affirm that your contributions do not conflict with any other proprietary agreements, employment contracts, or other legal obligations you may have.
    - You agree that all work contributed does not infringe on any third-party rights or violate any agreements you are bound to (such as non-compete or non-disclosure agreements).
- **Academic Integrity and Citation:**
    - If your contributions are based on academic literature or third-party software, you must properly cite and reference these sources in compliance with the principles of academic integrity and the terms of the Apache License.
- **Workflow Compliance:**
    - All contributors must comply with the project's defined workflow protocols, including adhering to test, security, and deployment workflows, as outlined in the [OpenSSF Best Practices](https://openssf.org/).
- **Liability and Responsibility:**
    - Contributors assume full responsibility for the originality and legality of their contributions and will hold harmless the project maintainers from any claims arising from legal conflicts, breaches, or intellectual property issues related to their contributions.

Thank you for your understanding and cooperation. We look forward to your contributions!

---
## Issue Tracking

We use [GitHub Issues](https://github.com/philip-ndikum/TemporalScope/issues) to track bugs, enhancements, features, and refactoring suggestions. To propose something new:

**Open an Issue:**

- Describe the issue (expected behavior, actual behavior, context and reproduction steps)
- Provide a rough implementation or pseudo code if necessary
- Include any relevant information you've collected

**After submission:**

- The project team will label the issue
- A team member will attempt to reproduce the issue (If reproduction steps are unclear, you may be asked for more details)
- Reproducible issues will be scheduled for a fix, or left open for community implementation

**Guidelines for effective issue reports:**

- Be as specific as possible
- Include code samples when relevant
- Describe your environment (OS, python version, etc.) if applicable
- Use clear, concise language

---
## Contributing Code

### Fork the Repository

1. Create your own [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of the TemporalScope repository.
2. Clone a local copy of your fork:

    ```console
    git clone git@github.com:YOUR-USERNAME/TemporalScope.git
    ```
    
    or
    
    ```console
    git clone https://github.com/YOUR-USERNAME/TemporalScope.git
    ```

### Setup Development Environment

TemporalScope uses [Hatch](https://hatch.pypa.io/latest/), a Python project manager, for managing virtual environments, building the project, and publishing packages.

1. Install Hatch by following the [installation instructions](https://hatch.pypa.io/latest/install/) for your operating system.
2. Verify the installation (your version number may differ):

    ```console
    $ cd TemporalScope
    $ hatch version
    0.1.0
    ```

### Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency. Set up the git hook scripts:

```console
pre-commit install --hook-type commit-msg --hook-type pre-push
```

### Create a New Branch

Create a new branch with a descriptive name for your changes:

```console
git switch -c <descriptive-branch-name>
```

### Make Your Changes

1. Make the necessary code changes in your local repository.
2. Write or update tests as needed.
3. Update documentation if you're introducing new features or changing existing functionality.

### Ensure Code Quality

TemporalScope employs various tools to maintain consistent code style, quality, and static type checking. While the CI pipeline tests code quality, running these checks locally expedites the review cycle.

Before submitting your changes, perform the following steps:

1. Run the test suite:

    ```console
    hatch run test:unit
    ```

    ```console
    hatch run test:integration
    ```

2. Check your code format:

    ```console
    hatch run format-check
    ```

3. Format your code (if needed):

    ```console
    hatch run format
    ```

4. Check your code style according to linting rules:

    ```console
    hatch run check
    ```

5. Automatically fix some errors (when possible):

    ```console
    hatch run fix
    ```


> NOTE:
> Running these checks locally will help identify and resolve issues before submitting your changes, streamlining the review process.

### Commit Your Changes

1. Stage your changes:

    ```console
    git add [args]
    ```

2. Commit your changes with a descriptive commit message. Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:

    ```console
    git commit -m "<type>[optional scope]: <description>"
    ```

   Example commit messages:

   - `feat: add user authentication`
   - `fix(api): resolve data parsing error`
   - `docs: update README with new configuration options`

   Alternatively, [Commitizen](https://commitizen-tools.github.io/commitizen/) is a handy tool for crafting commit messages that follow the Conventional Commit format. Simply run the command:

```console
cz commit
```

   and follow the interactive prompts. Commitizen will generate a commit message that complies with the required standards.

> NOTE:
> If you've set up pre-commit hooks as recommended, they will automatically run various checks before finalizing your commit. This helps ensure code quality and consistency.

### Submit a Pull Request

1. Push your changes to your fork:

    ```console
    git push origin <your-branch-name>
    ```

2. Go to the [TemporalScope repository](https://github.com/philip-ndikum/TemporalScope) on GitHub.
3. Click on "Pull requests" and then "New pull request".
4. Choose your fork and the branch containing your changes.
5. Fill out the pull request template with details about your changes.
6. Submit the pull request.

> SUCCESS: **Tip**
> To ease the review process, please follow the instructions:
>
> - For the title, use the [conventional commit convention](https://www.conventionalcommits.org/en/v1.0.0/).
> - For the body, follow the existing pull request template. Describe and document your changes.
> - Ensure test, formatting, and linting pass locally before submitting your pull request.
> - Include any relevant information that will help reviewers understand your changes.

### After Submitting

- Respond to any feedback or questions from reviewers.
- Make additional changes if requested.
- Once approved, your changes will be merged into the main repository.

Thank you for contributing to TemporalScope!

---
## Documentation

TemporalScope utilizes [Mkdocs](https://www.mkdocs.org/) for its documentation.

To build the docs locally run the following command

```console
hatch run docs:build
```

To view the docs locally run the following command

```console
hatch run docs:serve
```

API documentation is automatically generated using the [mkdocstrings](https://mkdocstrings.github.io/) extension. It extracts and builds the documentation directly from the code’s docstrings, which are written in Numpy format.

- **Content**: The documentation is written in Markdown and stored in the `docs` directory.
- **API Documentation**: API references are automatically generated using the [mkdocstrings](https://mkdocstrings.github.io/) extension, which extracts information directly from the code's docstrings.
- **Reference Pages**: The script `scripts/gen_ref_pages.py` dynamically generates API reference pages during each build, enabling fully automated and hands-off API documentation.

Documentation is hosted on [Read the Docs](https://readthedocs.org/projects/temporalscope/) and is automatically updated with each commit to the main branch.
PRs also provide a preview of the documentation changes.

---

## Testing

TemporalScope prioritizes code quality, security, and stability. To uphold these standards:

- New functionality should include corresponding tests.
- [PyTest](https://docs.pytest.org/en/stable/) is our primary testing framework.
- Required test types:
    - Unit tests for all new features
    - Integration tests where applicable
- Test coverage should include:
    - Main functionality
    - Edge cases
    - Boundary conditions
- Before pushing code:
    - Ensure all test pass locally
    - Run formatting and linting checks
- Test guidelines:
    - Keep tests simple and easy to understand
    - Follow PyTest [best practices](https://emimartin.me/pytest_best_practices)

> TIP:
> If you are unfamiliar with PyTest, the [official documentation](https://docs.pytest.org/en/stable/) provides a comprehensive guide to writing and running tests.
> Additionally, the [PyTest Quick Start Guide](https://docs.pytest.org/en/stable/getting-started.html) offers a quick introduction to the framework.
---
## Coverage

Coverage reports are generated with [pytest-cov](https://github.com/pytest-dev/pytest-cov) and are available for viewing on [Coveralls](https://coveralls.io/github/philip-ndikum/TemporalScope).
Coveralls provides comments on pull requests, indicating changes in coverage and highlighting areas that need additional testing.

---
## Development Roadmap & Changelog

**TemporalScope** follows **[Semantic Versioning (SemVer)](https://semver.org/)**, a versioning system that conveys the scope of changes introduced in each new release. Each version is represented in the form **MAJOR.MINOR.PATCH**, where major releases introduce significant or breaking changes, minor releases add backward-compatible functionality, and patch releases are used for bug fixes or minor improvements. Below is the planned roadmap outlining feature development and milestones over the next 12–18 months. This roadmap is subject to change based on user feedback, emerging research, and community contributions.

| **Version** | **Status**  | **Description**                                                                                                                                                                                      |
| ----------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0.1.0**   | Pre-release | The initial release version featuring basic end-to-end functionality. This version will be the first public release to both PyPI and Conda, offering the foundational capabilities of TemporalScope. |
| **0.2.0**   | Planned     | This version will include more comprehensive end-to-end examples and tutorials designed to assist users in integrating the software into practical use cases across various domains.                 |
| **0.3.0**   | Planned     | Introduction of the Clara LLM module to analyze SHAP values and provide robotic model validation, offering deeper insights as a defensive layer of model verification.                               |
| **0.5.0**   | Planned     | Focused on achieving a stable release. This version will include extensive user testing, bug fixes, and performance optimizations after several months of use in diverse environments.               |
| **1.0.0**   | Stable      | The first fully stable release, with robust documentation, thorough testing, and any feedback-driven refinements. This version will be ready for broader production use and long-term support.       |

---
## Workflow for Releasing New Versions

In order to maintain consistency and clarity across different distribution platforms like **PyPI**, **Conda**, and **GitHub**, we follow a structured workflow for releasing new versions:

1. Update the `CHANGELOG.md` File:
    - Ensure that all the changes (new features, bug fixes, deprecations, and breaking changes) are accurately recorded in the `CHANGELOG.md`.
    - Each release should include a brief summary of changes, structured by categories like **Features**, **Fixes**, and **Breaking Changes**.
2. Generate Release Notes:
    - Use the information from the `CHANGELOG.md` to create consistent release notes.
    - Ensure that the release notes are in a uniform format for each platform:
    - PyPI: Include a summary of the changes in the release description.
    - Conda: Similar release notes can be included when publishing to Conda.
    - GitHub: Publish the release notes in the GitHub Releases section.
3. Distribute to Each Platform:
    - PyPI: Push the package using hatch after running the necessary build commands.
    - Conda: Ensure the package is properly built for Conda and distributed to the Conda package manager.
    - GitHub: Create a GitHub Release, attaching the release notes, and tagging the release in the repository.
4. Verify the Release:
    - Ensure all distribution platforms (PyPI, Conda, GitHub) reflect the new release.
    - Test the installation via `pip install temporalscope` and `conda install temporalscope` to ensure everything works as expected.

---
## Code Style

We strictly enforce code quality and style to ensure the stability and maintainability of the project.

- **[Ruff](https://docs.astral.sh/ruff)** formatting and linting.
- **[Mypy](https://mypy.readthedocs.io/en/stable/)** type checking.
- **[PEP 8](https://peps.python.org/pep-0008/)** guidelines are followed for Python code style.
- **[Numpy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html)** with type hints are required to conform to the project's documentation standards.
- Write clear and concise commit messages. Adhere to [conventional commit convention](https://www.conventionalcommits.org/en/v1.0.0/).
- Include comments and docstrings where necessary to improve code readability.

Once Hatch and pre-commit are installed, checks run automatically before each commit, ensuring your code meets project standards. The CI pipeline also verifies these checks before merging. 

> TIP:
Most IDEs and text editors have plugins to help adhere these standards.

---
## Reporting Issues & Requesting Features

If you encounter any bugs or issues, please read our `SECURITY.md` for instructions on managing security issues. Alternatively, utilize Github issues to report a bug or potential long term feature request.

<!-- --8<-- [end:CONTRIBUTING] -->
