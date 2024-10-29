<p align="center">
  <img src="assets/temporalscope_github_banner.svg" alt="TemporalScope Logo" >
</p>

<h3 align="center">Scientifically driven Model-Agnostic Temporal Feature Importance Analysis</h3>

<p align="center">
  <!-- Twitter Share Button -->
  <a href="https://twitter.com/intent/tweet?text=Explore%20TemporalScope%20-%20Model-Agnostic%20Temporal%20Feature%20Importance%20Analysis!&url=https://github.com/philip-ndikum/TemporalScope&via=philip_ndikum&hashtags=MachineLearning,TemporalAnalysis,OpenSource" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Tweet" alt="Tweet" style="height: 28px;">
  </a>
  &nbsp;
  <!-- LinkedIn Share Button -->
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/philip-ndikum/TemporalScope" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Share%20on%20LinkedIn-0077B5?&logo=linkedin&logoColor=white&style=flat-square" alt="Share on LinkedIn" style="height: 28px;">
  </a>
  &nbsp;
  <!-- Reddit Share Button -->
  <a href="https://www.reddit.com/submit?url=https://github.com/philip-ndikum/TemporalScope&title=Explore%20TemporalScope%20-%20Model-Agnostic%20Temporal%20Feature%20Importance%20Analysis" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Share%20on%20Reddit-FF4500?&logo=reddit&logoColor=white&style=flat-square" alt="Share on Reddit" style="height: 28px;">
  </a>
  &nbsp;
  <!-- GitHub Discussions Link -->
  <a href="https://github.com/philip-ndikum/TemporalScope/discussions" target="_blank" style="text-decoration: none;">
    <img src="https://img.shields.io/badge/Discuss-GitHub%20Discussions-blue?style=flat-square&logo=github" alt="GitHub Discussions" style="height: 28px;">
  </a>
</p>

---

<!-- SPHINX-START -->

<div align="center">
  <table>
    <thead>
      <tr>
        <th>Compatibility</th>
        <th>License</th>
        <th>Code Quality</th>
        <th>Build Tools</th>
        <th>CI/CD</th>
        <th>Security</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>
          <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version"><br>
          <img src="https://img.shields.io/badge/OS-Linux-blue" alt="Linux Compatible">
        </td>
        <td>
          <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
        </td>
        <td>
          <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" alt="Ruff"></a><br>
          <img src="https://www.mypy-lang.org/static/mypy_badge.svg" alt="Checked with mypy">
        </td>
        <td>
          <a href="https://hatch.pypa.io/latest/"><img src="https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg" alt="Hatch project"></a>
        </td>
        <td>
          <a href="https://results.pre-commit.ci/latest/github/philip-ndikum/TemporalScope/main"><img src="https://results.pre-commit.ci/badge/github/philip-ndikum/TemporalScope/main.svg" alt="pre-commit.ci status"></a><br>
          <a href="https://codecov.io/gh/philip-ndikum/TemporalScope"><img src="https://codecov.io/gh/philip-ndikum/TemporalScope/branch/main/graph/badge.svg" alt="codecov"></a>
          <a href="https://github.com/philip-ndikum/TemporalScope/actions/workflows/test.yml"> <img src="https://github.com/philip-ndikum/TemporalScope/actions/workflows/test.yml/badge.svg"></a>
        </td>
        <td>
          <a href="https://www.bestpractices.dev/projects/9424"><img src="https://www.bestpractices.dev/projects/9424/badge" alt="OpenSSF Best Practices"></a><br>
          <a href="https://github.com/PyCQA/bandit"><img src="https://img.shields.io/badge/security-bandit-yellow.svg" alt="Security: Bandit"></a>
        </td>
      </tr>
    </tbody>
  </table>
</div>

---

**TemporalScope** is an open-source Python package designed to bridge the gap between scientific research and practical industry applications for analyzing the temporal dynamics of feature importance in AI & ML time series models. Developed in alignment with Linux Foundation standards and licensed under Apache 2.0, it builds on tools such as Boruta-SHAP and SHAP, using modern window partitioning algorithms to tackle challenges like non-stationarity and concept drift.  This package is flexible and extensible, supporting frameworks like **Pandas, Polars, Modin, Dask, and PyArrow** via **native Narwhals compatibility**. Additionally, the optional _Clara LLM_ modules (etymology from the word _Clarity_) are intended to serve as a model-validation tool to support explainability efforts (XAI). **Note**: TemporalScope is currently in **beta and pre-release** phase, so some installation methods may not work as expected on all platforms. Please check the `CONTRIBUTIONS.md` for the full roadmap.


<!-- SPHINX-END -->

### **Table of Contents**

- [**Installation**](#installation)
- [**Usage**](#usage)
  - [**Industrial Academic Applications**](#industrial-academic-applications)
- [Development Roadmap \& Changelog](#development-roadmap--changelog)
- [**Contributing**](#contributing)
  - [**Contributors 💠**](#contributors-)
- [Cite this Project](#cite-this-project)
- [**License, Limitations, and Legal Notice**](#license-limitations-and-legal-notice)

## **Installation**

**Note**: TemporalScope is currently in **beta**, so some installation methods may not work as expected on all platforms.

1. **Basic Installation using pip**: You can install the core package using pip:
   ```console
   $ pip install temporalscope
   ```
2. **Installation with conda**: For conda users, install via conda-forge:
   ```console
   $ conda install -c conda-forge temporalscope
   ```
3. **System-level Dependencies**: To view generated documentation locally, you may need `xdg-open`:
   ```console
   $ sudo apt install xdg-utils
   ```
4. **Git Clone and Setup**: For security reasons, we minimize system-level dependencies. If you prefer the latest development version, follow these steps to clone the repository and set up the project using Hatch:

   ```console
   $ git clone https://github.com/philip-ndikum/TemporalScope.git
   $ cd TemporalScope
   $ hatch shell
   ```

   This process clones the repository, navigates to the project directory, and uses Hatch to create and activate a virtual environment with the project installed in development mode.

## **Usage**

You can use TemporalScope with the following steps:

1. **Import TemporalScope**: Start by importing the package.
2. **Select Backend (Optional)**: TemporalScope defaults to using Pandas as the backend. However, you can specify other backends like Dask, Modin, or CuDF.
3. **Load Data**: Load your time series data into the `TimeSeriesData` class, specifying the `time_col` and optionally the `id_col`.
4. **Apply a Feature Importance Method**: TemporalScope defaults to using a Random Forest model from scikit-learn if no model is specified. You can either:
   - **A. Use a pre-trained model**: Pass a pre-trained model to the method.
   - **B. Train a Random Forest model within the method**: TemporalScope handles model training and application automatically.
5. **Analyze and Visualize Results**: Interpret the results to understand how feature importance evolves over time or across different phases.

Now, let's refine the code example using a random forest model and an academic dataset. We'll use the California housing dataset as a simple example since it's well-known and accessible.

```python
import polars as pl
import pandas as pd
from statsmodels.datasets import macrodata
from temporalscope.core.temporal_data_loader import TimeFrame
from temporalscope.partitioning.naive_partitioner import NaivePartitioner
from temporalscope.core.temporal_model_trainer import TemporalModelTrainer

# 1. Load the dataset using Pandas (or convert to Polars)
macro_df = macrodata.load_pandas().data
macro_df['time'] = pd.date_range(start='1959-01-01', periods=len(macro_df), freq='Q')

# Convert the Pandas DataFrame to a Polars DataFrame
macro_df_polars = pl.DataFrame(macro_df)

# 2. Initialize the TimeFrame object with the data
economic_tf = TimeFrame(
    df=macro_df_polars,
    time_col='time',
    target_col='realgdp',
    backend='pl',  # Using Polars as the backend
)

# 3. Apply Partitioning Strategy (like sklearn's train_test_split)
partitioner = NaivePartitioner(economic_tf)
partitioned_data = partitioner.apply()  # Returns a list of partitioned dataframes

# 4. Train and evaluate the model using the partitioned data
model_trainer = TemporalModelTrainer(
    partitioned_data=partitioned_data,  # Directly passing the partitioned data
    model=None,  # Use the default model (LightGBM)
    model_params={
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'rmse',
        'verbosity': -1
    }
)

# 5. Execute the training and evaluate
results = model_trainer.train_and_evaluate()

# Output predictions and metrics
for partition_name, predictions in results.items():
    print(f"Predictions for {partition_name}:")
    print(predictions[:5])  # Display first 5 predictions
```

### **Industrial Academic Applications**

**DISCLAIMER**: The following use cases are provided for academic and informational purposes only. TemporalScope is intended to support research and development in understanding temporal dynamics in feature importance. These examples are not intended as guidance for industrial applications without further validation and expert consultation. The use of TemporalScope in any industrial or production environment is at the user's own risk, and the developers disclaim any liability for such use. Please refer to the [License and Legal Notice](#license-and-legal-notice) for further details.

| **Sector**     | **Use Case**                                                                                                                                                                                                      | **Impact**                                                                                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Finance**    | **Quantitative Investing:** Understanding how the importance of financial indicators, such as interest rates or stock volatility, changes across different market cycles.                                         | Enhances quantitative investment strategies by identifying which indicators are most predictive in various market conditions, thereby improving risk management and investment decisions. |
| **Healthcare** | **Patient Outcome Prediction:** Tracking how various patient data features, such as age, vital signs, and medical history, contribute to health outcomes at different stages of treatment or disease progression. | Facilitates the creation of personalized treatment plans by identifying critical factors at specific stages, leading to improved patient outcomes and optimized healthcare resources.     |
| **Energy**     | **Load Forecasting:** Analyzing how the importance of variables such as weather conditions, time of day, and historical consumption data evolves in predicting energy demand.                                     | Improves energy load forecasting by adapting to changing conditions, leading to more efficient energy distribution and reduced operational costs.                                         |
| **Retail**     | **Customer Behavior Analysis:** Understanding how customer preferences and purchasing behaviors change over time, influenced by factors such as seasonal trends, promotions, and economic conditions.             | Enables retailers to optimize inventory management, marketing strategies, and pricing models by identifying the most influential factors driving sales in different periods.              |

For more detailed examples from sectors like engineering and other scientific applications, please refer to the [SCIENTIFIC_LITERATURE.md](SCIENTIFIC_LITERATURE.md).

## Development Roadmap & Changelog

For detailed test, security, and deployment workflows as defined by OpenSSF Best Practices, please refer to [CONTRIBUTING.md](CONTRIBUTING.md). **TemporalScope** follows **Semantic Versioning (SemVer)**, a versioning system that conveys the scope of changes introduced in each new release. Each version is represented in the form **MAJOR.MINOR.PATCH**, where major releases introduce significant or breaking changes, minor releases add backward-compatible functionality, and patch releases are used for bug fixes or minor improvements. Below is the planned roadmap outlining feature development and milestones over the next 12–18 months. This roadmap is subject to change based on user feedback, emerging research, and community contributions.

## **Contributing**

TemporalScope was conceived independently by [Philip Ndikum](https://github.com/philip-ndikum), [Serge Ndikum](https://github.com/serge-ndikum), and [Kane Norman](https://github.com/kanenorman) and has since been open-sourced to the broader academic and developer community. As the software continues to grow and evolve, it relies heavily on the active participation and contributions of its users. We encourage contributions from developers, researchers, and data scientists who are passionate about advancing open-source tools. Whether you are interested in extending the package’s functionality, fixing bugs, or improving documentation, your contributions are vital to the project’s ongoing success.

For detailed guidelines on how to contribute, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md). By working together, we can ensure that TemporalScope remains an innovative and reliable tool, continuously refined through community collaboration.

### **Contributors 💠**

Thanks to these wonderful people:

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/philip-ndikum"><img src="https://avatars.githubusercontent.com/u/125715876?v=4&s=100" width="100px;" alt="Philip Ndikum"/><br /><sub><b>Philip Ndikum</b></sub></a><br /><a href="https://github.com/philip-ndikum/TemporalScope/commits?author=philip-ndikum" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/serge-ndikum"><img src="https://avatars.githubusercontent.com/u/180889799?v=4&s=100" width="100px;" alt="Serge Ndikum"/><br /><sub><b>Serge Ndikum</b></sub></a><br /><a href="https://github.com/serge-ndikum" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kanenorman"><img src="https://avatars.githubusercontent.com/u/51185594?v=4&s=100" width="100px;" alt="Kane Norman"/><br /><sub><b>Kane Norman</b></sub></a><br /><a href="https://github.com/kanenorman" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Cite this Project

If you use **TemporalScope** in your research, please consider citing it:

```bibtex
@software{ndikum2024temporalscope,
  author = {Ndikum, Philip and Ndikum, Serge and Norman, Kane},
  title = {TemporalScope: Model-Agnostic Temporal Feature Importance Analysis},
  year = {2024},
  version = {0.1.0},
  publisher = {GitHub},
  url = {https://github.com/philip-ndikum/TemporalScope}
}
```

## **License, Limitations, and Legal Notice**

**TemporalScope** is primarily an academic tool designed for research and informational purposes. Practitioners and users of this software are strongly encouraged to consult the accompanying [SCIENTIFIC_LITERATURE.md](SCIENTIFIC_LITERATURE.md) document to fully understand the theoretical limitations, assumptions, and context of the techniques implemented within this package. Furthermore, use of this software falls under "as-is" software as defined by the [Apache License 2.0](LICENSE) provided in this repository and outlined below.

By using this package, you agree to comply with the terms and conditions set forth in the **Apache License 2.0**.

**LEGAL NOTICE**: THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.
