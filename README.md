<p align="center">
  <img src="assets/temporalscope_github_banner.svg" alt="TemporalScope Logo" >
</p>

<h3 align="center">Scientificially driven Model-Agnostic Temporal Feature Importance Analysis</h3>

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
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen" alt="Contributions Welcome">
  <img src="https://img.shields.io/badge/OS-Linux-blue" alt="Linux Compatible">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code Style: Black"></a>
  <a href="https://github.com/PyCQA/bandit"><img src="https://img.shields.io/badge/security-bandit-yellow.svg" alt="Security: Bandit"></a>
</p>


**TemporalScope** is an open-source Python package that provides flexible tools for analyzing and visualizing the temporal dynamics of feature importance in Artificial Intelligence (AI) and Machine Learning (ML) models. Licensed under the Apache 2.0 License and developed in accordance with Linux Foundation standards, TemporalScope is grounded in cutting-edge academic research. The package integrates advanced methods such as Boruta-SHAP, SHAP, and other feature selection techniques, offering a scientifically driven framework for understanding how feature importance evolves over time.


### **Table of Contents**

- [Installation](#installation)
- [Usage](#usage)
- [Why use TemporalScope?](#Why-use-TemporalScope?)
- [Technical Methods](#technical-methods)
- [Development Roadmap & Changelog](#development-roadmap--changelog)
- [Contributing](#contributing)
- [Citation](#cite-this-project)
- [License, Limitations and Legal Notice](#license-limitations-and-legal-notice)

## **Installation**

You can install TemporalScope from [PyPI](https://pypi.org/project/temporalscope/) with:

1. **Basic Installation using pip**:

   `pip install temporalscope`

2. **Installation with Optional Dependencies**: If you want to use additional backends like Dask, Modin, CUDF, or Polars, you can install them as follows:

   `pip install temporalscope[dask,modin,cudf,polars]`


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
from temporalscope.partioning.naive_partitioner import NaivePartitioner
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

## **Why use TemporalScope?**

| **Feature**                     | **Description**                                                                                                                                       |
|----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Enterprise-Grade**             | Adheres to Linux Foundation Open-Source Security standards, encouraging contributors to reference academic papers and use quality assurance scripts to maintain software reliability.  |
| **Modularity**                   | Incorporates Pandas, Polars, Modin, SHAP, Boruta-SHAP, and partitioning algorithms from the literature, making the software flexible and expandable as research advances. |
| **Explainability**               | Designed to spur further research into temporal-driven explainability, providing a framework that can evolve with contributions from the global AI community. |
| **Boruta-SHAP Integration**      | Combines the Boruta feature selection method with SHAP for robust, interpretable feature importance across time.                                         |
| **Time-Aware Feature Tracking**  | Tracks feature importance over time, accounting for changes in data distribution and offering insights into temporal shifts.                             |
| **Scalability**                  | Built to efficiently handle large datasets, with scalability in mind for both academic and industrial applications.                                     |
| **Extensible Framework**         | Easily integrates with custom feature importance methods and workflows, allowing users to extend the package as needed.                                |
| **Detailed Visualizations**      | Provides tools for visualizing changes in feature importance over time, aiding in model interpretability and data analysis.                             |
| **Robust Documentation**         | Offers comprehensive documentation with examples and tutorials, adhering to Linux Foundation development standards for high-quality open-source software. |


### **Industrial Academic Applications**

**DISCLAIMER**: The following use cases are provided for academic and informational purposes only. TemporalScope is intended to support research and development in understanding temporal dynamics in feature importance. These examples are not intended as guidance for industrial applications without further validation and expert consultation. The use of TemporalScope in any industrial or production environment is at the user's own risk, and the developers disclaim any liability for such use. Please refer to the [License and Legal Notice](#license-and-legal-notice) for further details.

### **Example Academic Use Cases**

| **Sector**    | **Use Case**                                                                                  | **Impact**                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Finance**   | **Quantitative Investing:** Understanding how the importance of financial indicators, such as interest rates or stock volatility, changes across different market cycles.   | Enhances quantitative investment strategies by identifying which indicators are most predictive in various market conditions, thereby improving risk management and investment decisions. |
| **Finance**   | **Credit Risk Modelling:** Analyzing how the contribution of variables like income, credit history, or employment status fluctuates over time in credit scoring models.   | Supports the development of more robust credit scoring models by highlighting how feature relevance changes, potentially leading to more accurate risk assessments. |
| **Healthcare**| **Patient Outcome Prediction:** Tracking how various patient data features, such as age, vital signs, and medical history, contribute to health outcomes at different stages of treatment or disease progression. | Facilitates the creation of personalized treatment plans by identifying critical factors at specific stages, leading to improved patient outcomes and optimized healthcare resources. |
| **Engineering**| **Predictive Maintenance for Machinery:** Examining how factors like temperature, vibration, and usage patterns affect machinery lifecycle over time.        | Reduces machinery downtime and maintenance costs by providing insights into when and why certain components are likely to fail, enabling more effective maintenance scheduling. |
| **Energy**    | **Load Forecasting:** Analyzing how the importance of variables such as weather conditions, time of day, and historical consumption data evolves in predicting energy demand.   | Improves energy load forecasting by adapting to changing conditions, leading to more efficient energy distribution and reduced operational costs. |
| **Retail**    | **Customer Behavior Analysis:** Understanding how customer preferences and purchasing behaviors change over time, influenced by factors such as seasonal trends, promotions, and economic conditions. | Enables retailers to optimize inventory management, marketing strategies, and pricing models by identifying the most influential factors driving sales in different periods. |

## Development Roadmap & Changelog

**TemporalScope** follows **Semantic Versioning (SemVer)**, a versioning system that conveys the scope of changes introduced in each new release. Each version is represented in the form **MAJOR.MINOR.PATCH**, where major releases introduce significant or breaking changes, minor releases add backward-compatible functionality, and patch releases are used for bug fixes or minor improvements. Below is the planned roadmap outlining feature development and milestones over the next 12–18 months. This roadmap is subject to change based on user feedback, emerging research, and community contributions.

| **Version** | **Status**     | **Description**                                                                                                                                                                   |
|-------------|----------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **0.1.0**   | Pre-release    | The initial release version featuring basic end-to-end functionality. This version will be the first public release to both PyPI and Conda, offering the foundational capabilities of TemporalScope.  |
| **0.2.0**   | Planned        | This version will include more comprehensive end-to-end examples and tutorials designed to assist users in integrating the software into practical use cases across various domains.   |
| **0.3.0**   | Planned        | Introduction of advanced features, including the integration of the Clara LLM module for generating SHAP output, along with enhanced visualization tools for deeper temporal analysis.  |
| **0.5.0**   | Planned        | Focused on achieving a stable release. This version will include extensive user testing, bug fixes, and performance optimizations after several months of use in diverse environments. |
| **1.0.0**   | Stable         | The first fully stable release, with robust documentation, thorough testing, and any feedback-driven refinements. This version will be ready for broader production use and long-term support.  |

**Contributor Guidelines**: All contributors are required to update the changelog to document any new features, bug fixes, or performance improvements. These updates must be reflected in both the version history and changelog, in accordance with the [Apache License 2.0](LICENSE) and Linux Foundation standards. Please ensure that all contributions are tested and well-documented to maintain the stability and reliability of the software.


## **Contributing**

TemporalScope was conceived by [Philip Ndikum](https://github.com/philip-ndikum), Serge Ndikum and [Kane Norman](https://github.com/kanenorman) and has since been open-sourced to the broader academic and developer community. As the software continues to grow and evolve, it relies heavily on the active participation and contributions of its users.

We encourage contributions from developers, researchers, and data scientists who are passionate about advancing open-source tools. Whether you are interested in extending the package’s functionality, fixing bugs, or improving documentation, your contributions are vital to the project’s ongoing success.

We are also looking for contributors who are excited about taking on more significant roles in the development of TemporalScope. Those interested in leading future implementations or contributing in a more substantial capacity are welcome to get involved.

For detailed guidelines on how to contribute, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md). By working together, we can ensure that TemporalScope remains an innovative and reliable tool, continuously refined through community collaboration.

### **Contributors 💠**

Thanks to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/philip-ndikum"><img src="https://avatars.githubusercontent.com/u/125715876?v=4&s=100" width="100px;" alt="Philip Ndikum"/><br /><sub><b>Philip Ndikum</b></sub></a><br /><a href="https://github.com/philip-ndikum/TemporalScope/commits?author=philip-ndikum" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/kanenorman"><img src="https://avatars.githubusercontent.com/u/51185594?v=4&s=100" width="100px;" alt="Kane Norman"/><br /><sub><b>Kane Norman</b></sub></a><br /><a href="https://github.com/kanenorman" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->



## Cite this Project

If you use **TemporalScope** in your research, please consider citing it:
```
{
  @software{ndikum2024temporalscope,
  author = {Philip Ndikum},
  title = {TemporalScope: Model-Agnostic Temporal Feature Importance Analysis},
  year = 2024,
  version = {1.0.0},
  publisher = {GitHub},
  url = {https://github.com/philip-ndikum/TemporalScope}
}
```

## **License, Limitations, and Legal Notice**

**TemporalScope** is primarily an academic tool designed for research and informational purposes. Practitioners and users of this software are strongly encouraged to consult the accompanying [SCIENTIFIC_LITERATURE.md](SCIENTIFIC_LITERATURE.md) document to fully understand the theoretical limitations, assumptions, and context of the techniques implemented within this package. Furthermore, use of this software falls under "as-is" software as defined by the [Apache License 2.0](LICENSE) provided in this repository and outlined below.

By using this package, you agree to comply with the terms and conditions set forth in the **Apache License 2.0**.

**LEGAL NOTICE**: THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.


