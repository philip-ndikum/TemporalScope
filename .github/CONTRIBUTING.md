# Contributing to TemporalScope

Thank you for your interest in contributing to TemporalScope! Contributions of all kinds are welcome and appreciated.

## Contribution Guidelines

By contributing to this project, you are agreeing to the following conditions:

1. **Adherence to the Apache License 2.0**:
   - All contributions must comply with the [Apache License 2.0](LICENSE).
   - You must ensure that all work contributed is your own original creation, fully independent, or obtained through publicly available academic literature or properly licensed third-party sources.
2. **Conflict of Interest and Independence**:
   - By contributing, you affirm that your contributions do not conflict with any other proprietary agreements, employment contracts, or other legal obligations you may have.
   - You agree that all work contributed does not infringe on any third-party rights or violate any agreements you are bound to (such as non-compete or non-disclosure agreements).
3. **Academic Integrity and Citation**:
   - If your contributions are based on academic literature or third-party software, you must properly cite and reference these sources in compliance with the principles of academic integrity and the terms of the Apache License.
4. **Workflow Compliance**:
   - All contributors must comply with the project's defined workflow protocols, including adhering to test, security, and deployment workflows, as outlined in the [OpenSSF Best Practices](https://openssf.org/).
5. **Liability and Responsibility**:
   - Contributors assume full responsibility for the originality and legality of their contributions and will hold harmless the project maintainers from any claims arising from legal conflicts, breaches, or intellectual property issues related to their contributions.

We are excited to have your contributions but ask that you follow these guidelines to ensure the project remains legally sound and academically rigorous.

## How to Contribute

1. Fork the repository.
2. Create a new branch with a descriptive name (e.g., `feature/your-feature`).
3. Make your changes.
4. Ensure that all tests pass and that your code follows the project’s style guidelines.
    - Use the `precommit_quality_assurance.sh` script to run tests and checks before submitting code. This script runs:
        - **Black** for code formatting
        - **Flake8** for linting
        - **MyPy** for type checking
        - **Bandit** for security issues
        - **PyTest** for unit tests
5. Submit a pull request to the main repository.

## Test Policy

To maintain a high standard of code quality, security, and stability, TemporalScope requires all new functionality to include corresponding tests. Tests help ensure that your code integrates well with the existing codebase and that future changes don’t introduce bugs or regressions.

- **PyTest** is the primary testing tool, and **all new features** must be accompanied by unit tests, and integration tests where applicable.
- Tests should cover edge cases and boundary conditions to prevent unexpected issues during production use.
- **Minimal Lines of Code (LOC)**: Contributions should focus on keeping the codebase lean by avoiding unnecessary complexity.
- Before pushing code, run the `precommit_quality_assurance.sh` script to ensure that **Black**, **Flake8**, **MyPy**, **Bandit**, and **PyTest** pass successfully.

## Development Roadmap & Changelog

**TemporalScope** follows **Semantic Versioning (SemVer)**, a versioning system that conveys the scope of changes introduced in each new release. Each version is represented in the form **MAJOR.MINOR.PATCH**, where major releases introduce significant or breaking changes, minor releases add backward-compatible functionality, and patch releases are used for bug fixes or minor improvements. Below is the planned roadmap outlining feature development and milestones over the next 12–18 months. This roadmap is subject to change based on user feedback, emerging research, and community contributions.

| **Version** | **Status**     | **Description**                                                                                                                                                                   |
|-------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **0.1.0**   | Pre-release    | The initial release version featuring basic end-to-end functionality. This version will be the first public release to both PyPI and Conda, offering the foundational capabilities of TemporalScope.  |
| **0.2.0**   | Planned        | This version will include more comprehensive end-to-end examples and tutorials designed to assist users in integrating the software into practical use cases across various domains.   |
| **0.3.0**   | Planned        | Introduction of the Clara LLM module to analyze SHAP values and provide robotic model validation, offering deeper insights as a defensive layer of model verification.                |
| **0.5.0**   | Planned        | Focused on achieving a stable release. This version will include extensive user testing, bug fixes, and performance optimizations after several months of use in diverse environments. |
| **1.0.0**   | Stable         | The first fully stable release, with robust documentation, thorough testing, and any feedback-driven refinements. This version will be ready for broader production use and long-term support.  |


## Workflow for Releasing New Versions

In order to maintain consistency and clarity across different distribution platforms like **PyPI**, **Conda**, and **GitHub**, we follow a structured workflow for releasing new versions:

1. **Update the `CHANGELOG.md` File**:
    - Ensure that all the changes (new features, bug fixes, deprecations, and breaking changes) are accurately recorded in the `CHANGELOG.md`.
    - Each release should include a brief summary of changes, structured by categories like **Features**, **Fixes**, and **Breaking Changes**.
2. **Generate Release Notes**:
    - Use the information from the `CHANGELOG.md` to create consistent **release notes**.
    - Ensure that the release notes are in a uniform format for each platform:
        - **PyPI**: Include a summary of the changes in the release description.
        - **Conda**: Similar release notes can be included when publishing to **Conda**.
        - **GitHub**: Publish the release notes in the **GitHub Releases** section.
3. **Distribute to Each Platform**:
    - **PyPI**: Push the package using `twine` after running the necessary build commands.
    - **Conda**: Ensure the package is properly built for **Conda** and distributed to the Conda package manager.
    - **GitHub**: Create a **GitHub Release**, attaching the release notes, and tagging the release in the repository.
4. **Verify the Release**:
    - Ensure all distribution platforms (PyPI, Conda, GitHub) reflect the new release.
    - Test the installation via `pip install temporalscope` and `conda install temporalscope` to ensure everything works as expected.

By following this workflow, we ensure a consistent and smooth release process across all distribution channels, providing users with clear updates and robust software. We use HTTPS and SSH to protect against **man-in-the-middle (MITM) attacks** during the delivery process. However, **digital signatures** are not currently implemented, as the current security measures are sufficient for the project's scope. We will revisit this as the project scales and requires additional security layers.

## Code Style

We strictly enforce code quality and style to ensure the stability and maintainability of the project.

- **Black** is used for consistent code formatting.
- **PEP 8** guidelines are followed for Python code style.
- **Sphinx-style docstrings** with type hints are required to conform to **MyPy** standards, enabling early error detection.
- Write clear and concise commit messages.
- Include comments and docstrings where necessary to improve code readability.

## Reporting Issues & Requesting Features

If you encounter any bugs or issues, please read our `SECURITY.md` for instructions on managing security issues. Alternatively, utilize the Github Discussions to raise issues or potential long term features. 

