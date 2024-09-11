# Contributing to TemporalScope

Thank you for your interest in contributing to TemporalScope! Contributions of all kinds are welcome and appreciated.

## Contribution Guidelines

> [!IMPORTANT]
> By contributing to this project, you are agreeing to the following conditions:

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

# How to Contribute to TemporalScope

We welcome contributions to TemporalScope! This guide will help you get started with the contribution process.

# Issue Tracking

We use [GitHub Issues](https://github.com/philip-ndikum/TemporalScope/issues) to track bugs, enhancements, features, and refactoring suggestions. To propose something new:

1. Open an Issue
2. Describe the issue:

   - Expected behavior
   - Actual behavior
   - Context and reproduction steps

3. Provide a rough implementation or pseudo code if necessary
4. Include any relevant information you've collected

After submission:

1. The project team will label the issue
2. A team member will attempt to reproduce the issue
   - If reproduction steps are unclear, you may be asked for more details
3. Reproducible issues will be:
   - Scheduled for a fix, or left open for community implementation

Guidelines for effective issue reports:

- Be as specific as possible
- Include code samples when relevant
- Describe your environment (OS, python version, etc.) if applicable
- Use clear, concise language

# Contributing Code

## Fork the Repository

1. Create your own [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) of the TemporalScope repository.
2. Clone a local copy of your fork:

   ```console
   $ git clone git@github.com:YOUR-USERNAME/TemporalScope.git
   ```

   or

   ```console
   $ git clone https://github.com/YOUR-USERNAME/TemporalScope.git
   ```

## Setup Development Environment

TemporalScope uses [Hatch](https://hatch.pypa.io/latest/), a Python project manager, for managing virtual environments, building the project, and publishing packages.

1. Install Hatch by following the [installation instructions](https://hatch.pypa.io/latest/install/) for your operating system.
2. Verify the installation (your version number may differ):

   ```console
   $ cd TemporalScope
   $ hatch version
   0.1.0
   ```

## Install Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) hooks to ensure code quality and consistency. Set up the git hook scripts:

```console
$ pre-commit install
```

## Create a New Branch

Create a new branch with a descriptive name for your changes:

```console
$ git switch -c <descriptive-branch-name>
```

## Make Your Changes

1. Make the necessary code changes in your local repository.
2. Write or update tests as needed.
3. Update documentation if you're introducing new features or changing existing functionality.

## Ensure Code Quality

TemporalScope employs various tools to maintain consistent code style, quality, and static type checking. While the CI pipeline tests code quality, running these checks locally expedites the review cycle.

Before submitting your changes, perform the following steps:

1. Run the test suite and type checking:

```console
$ hatch run test:type
```

2. Check your code format:

```console
$ hatch run format-check
```

3. Format your code (if needed):

```console
$ hatch run format
```

4. Check your code style according to linting rules:

```console
$ hatch run check
```

5. Automatically fix some errors (when possible):

```console
$ hatch run fix
```

> [!NOTE]
> Running these checks locally will help identify and resolve issues before submitting your changes, streamlining the review process.

Here's the revised version of that section:

## Commit Your Changes

1. Stage your changes:

   ```console
   $ git add [args]
   ```

   This command allows you to review and selectively stage parts of your changes.

2. Commit your changes with a descriptive commit message. Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification:

   ```console
   $ git commit -m "<type>[optional scope]: <description>"
   ```

   Example commit messages:

   - `feat: add user authentication`
   - `fix(api): resolve data parsing error`
   - `docs: update README with new configuration options`

> [!NOTE]
> If you've set up pre-commit hooks as recommended, they will automatically run various checks before finalizing your commit. This helps ensure code quality and consistency.

## Submit a Pull Request

1. Push your changes to your fork:

   ```console
   $ git push origin <your-branch-name>
   ```

2. Go to the [TemporalScope repository](https://github.com/philip-ndikum/TemporalScope) on GitHub.
3. Click on "Pull requests" and then "New pull request".
4. Choose your fork and the branch containing your changes.
5. Fill out the pull request template with details about your changes.
6. Submit the pull request.

> [!TIP]
> To ease the review process, please follow the instructions:
>
> - For the title, use the [conventional commit convention](https://www.conventionalcommits.org/en/v1.0.0/).
> - For the body, follow the existing [pull request template](<(https://github.com/philip-ndikum/TemporalScope/blob/main/.github/pull_request_template.md)>). Describe and document your changes.

## After Submitting

- Respond to any feedback or questions from reviewers.
- Make additional changes if requested.
- Once approved, your changes will be merged into the main repository.

Thank you for contributing to TemporalScope!

## Test Policy

TemporalScope prioritizes code quality, security, and stability. To uphold these standards:

1. New functionality should include corresponding tests.
2. [PyTest](https://docs.pytest.org/en/stable/) is our primary testing framework.
3. Required test types:

   - Unit tests for all new features
   - Integration tests where applicable

4. Test coverage should include:

   - Main functionality
   - Edge cases
   - Boundary conditions

5. Before pushing code:

   - Ensure all test pass locally

6. Test guidelines:
   - Keep tests simple and easy to understand
   - Follow PyTest [best practices](https://emimartin.me/pytest_best_practices)

This policy helps ensure code integrates well with the existing codebase and prevents future bugs or regressions.

## Development Roadmap & Changelog

**TemporalScope** follows **[Semantic Versioning (SemVer)](https://semver.org/)**, a versioning system that conveys the scope of changes introduced in each new release. Each version is represented in the form **MAJOR.MINOR.PATCH**, where major releases introduce significant or breaking changes, minor releases add backward-compatible functionality, and patch releases are used for bug fixes or minor improvements. Below is the planned roadmap outlining feature development and milestones over the next 12–18 months. This roadmap is subject to change based on user feedback, emerging research, and community contributions.

| **Version** | **Status**  | **Description**                                                                                                                                                                                      |
| ----------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0.1.0**   | Pre-release | The initial release version featuring basic end-to-end functionality. This version will be the first public release to both PyPI and Conda, offering the foundational capabilities of TemporalScope. |
| **0.2.0**   | Planned     | This version will include more comprehensive end-to-end examples and tutorials designed to assist users in integrating the software into practical use cases across various domains.                 |
| **0.3.0**   | Planned     | Introduction of the Clara LLM module to analyze SHAP values and provide robotic model validation, offering deeper insights as a defensive layer of model verification.                               |
| **0.5.0**   | Planned     | Focused on achieving a stable release. This version will include extensive user testing, bug fixes, and performance optimizations after several months of use in diverse environments.               |
| **1.0.0**   | Stable      | The first fully stable release, with robust documentation, thorough testing, and any feedback-driven refinements. This version will be ready for broader production use and long-term support.       |

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
   - **PyPI**: Push the package using `hatch` after running the necessary build commands.
   - **Conda**: Ensure the package is properly built for **Conda** and distributed to the Conda package manager.
   - **GitHub**: Create a **GitHub Release**, attaching the release notes, and tagging the release in the repository.
4. **Verify the Release**:
   - Ensure all distribution platforms (PyPI, Conda, GitHub) reflect the new release.
   - Test the installation via `pip install temporalscope` and `conda install temporalscope` to ensure everything works as expected.

By following this workflow, we ensure a consistent and smooth release process across all distribution channels, providing users with clear updates and robust software. We use HTTPS and SSH to protect against **man-in-the-middle (MITM) attacks** during the delivery process. However, **digital signatures** are not currently implemented, as the current security measures are sufficient for the project's scope. We will revisit this as the project scales and requires additional security layers.

## Code Style

We strictly enforce code quality and style to ensure the stability and maintainability of the project.

- **[Ruff](https://docs.astral.sh/ruff)** formatting and linting
- **[Mypy](https://mypy.readthedocs.io/en/stable/)** type checking
- **[PEP 8](https://peps.python.org/pep-0008/)** guidelines are followed for Python code style.
- **[Sphinx-style docstrings](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/docstrings.html)** with type hints are required to conform to **MyPy** standards, enabling early error detection.
- Write clear and concise commit messages. Adhere to [conventional commit convention](https://www.conventionalcommits.org/en/v1.0.0/).
- Include comments and docstrings where necessary to improve code readability.

## Reporting Issues & Requesting Features

If you encounter any bugs or issues, please read our `SECURITY.md` for instructions on managing security issues. Alternatively, utilize the Github Discussions to raise issues or potential long term features.
