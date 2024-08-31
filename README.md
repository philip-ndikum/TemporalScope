<p align="center">
  <img src="assets/temporalscope_github_banner.svg" alt="TemporalScope Logo" >
</p>

<h3 align="center">Model-Agnostic Temporal Feature Importance Analysis</h3>

<p align="center">
  <a href="https://twitter.com/intent/tweet?text=Explore%20TemporalScope%20-%20Model-Agnostic%20Temporal%20Feature%20Importance%20Analysis!&url=https://github.com/philip-ndikum/TemporalScope&via=philip_ndikum&hashtags=MachineLearning,TemporalAnalysis,OpenSource" target="_blank">
    <img src="https://img.shields.io/twitter/url/http/shields.io.svg?style=social" alt="Tweet">
  </a>
  &nbsp;
  <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/philip-ndikum/TemporalScope" target="_blank">
    <img src="https://img.shields.io/badge/Share%20on%20LinkedIn-0077B5?&logo=linkedin&logoColor=white&style=for-the-badge" alt="Share on LinkedIn">
  </a>
</p>

---
<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/License-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/status-in%20development-yellow" alt="Development Status">
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen" alt="Contributions Welcome">
  <img src="https://img.shields.io/badge/OS-Linux-blue" alt="Linux Compatible">
</p>

**TemporalScope** is a lightweight, open-source software (OSS) Python package designed to analyze and understand the temporal dynamics of feature importance in machine learning models. Licensed under the Apache 2.0 License and developed with Linux Foundation standards, TemporalScope provides a straightforward API to track and visualize how feature importance evolves over time, enabling deeper insights into temporal data patterns. This tool is ideal for researchers and practitioners who need to account for temporal variations in feature importance in their predictive models.


### **Table of Contents**

- [Why use TemporalScope?](#Why-use-TemporalScope?)
- [Installation](#installation)
- [Usage](#usage)
- [Industrial Academic Applications](#industrial-academic-applications)
- [Technical Methods](#technical-methods)
- [Development Roadmap](#development-roadmap)
- [Contributing](#contributing)
- [Citation](#cite-this-project)
- [License and Legal Notice](#license-and-legal-notice)

## **Why use TemporalScope?**

1. **Addressing Non-Stationarity**: Research has shown that using SHAP or causal methods on a single dataset can be insufficient due to non-stationarity—changes in data distribution over time. **TemporalScope** directly addresses this challenge, offering a more nuanced and accurate analysis of feature importance.

2. **Built on Cutting-Edge Research**: **TemporalScope** is grounded in the latest scientific literature, designed to solve a critical problem faced by analysts, scientists, and quantitative professionals worldwide. It allows for a deeper understanding of how features influence model predictions over time, helping to uncover hidden patterns and insights.

3. **Standards and Scalability**: Developed according to Linux Foundation standards, **TemporalScope** offers robust documentation and a lean, scalable approach. Driven by the scientific community we hope to attract developers to help us improve the software through time.


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

## **Industrial Academic Applications**

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


## **Technical Methods**

TemporalScope leverages advanced methodologies to provide a comprehensive analysis of temporal feature importance. Each method is designed to tackle specific aspects of non-stationarity and data segmentation:

| **Method**                         | **Description**                                                                                          |
|------------------------------------|----------------------------------------------------------------------------------------------------------|
| **Reverse Expanding Window**       | Analyzes feature importance by incrementally expanding the data window, tracking how importance evolves over time. |
| **Partitioning Approach**          | Segments data into distinct phases (e.g., time-based) and examines feature importance within each segment.|
| **WinIT Approach**                 | Captures temporal dependencies by summarizing feature importance across a moving window of past data points.|

## **Development Roadmap**

We have a clear vision for the future development of TemporalScope. Our roadmap outlines the planned features and the stages of development:

| **Version** | **Status**    | **Features**                                                                                     |
|-------------|---------------|-------------------------------------------------------------------------------------------------|
| **0.0.1**   | Current       | Initial setup with basic structure and functionality.                                            |
| **0.0.2**   | In Progress   | Implement the Reverse Expanding Window method with thorough scientific documentation and testing.|
| **0.0.3**   | Planned       | Integrate additional methods with comprehensive documentation, attribution, and testing.         |
| **0.0.5**   | Planned       | Develop benchmarking tools and unit tests across various datasets to validate performance.       |
| **0.8.0**   | Pre-release   | Perform extensive testing with selected users to identify potential improvements.                |
| **1.0.0**   | Stable        | Release a fully stable version with extensive documentation, testing, and user feedback integration.|

## **Contributing**

TemporalScope was created and developed by [Philip Ndikum](https://github.com/philip-ndikum) and has since been open-sourced to the broader academic and developer community. As the software continues to grow and evolve, it relies heavily on the active participation and contributions of its users.

We encourage contributions from developers, researchers, and data scientists who are passionate about advancing open-source tools. Whether you are interested in extending the package’s functionality, fixing bugs, or improving documentation, your contributions are vital to the project’s ongoing success.

We are also looking for contributors who are excited about taking on more significant roles in the development of TemporalScope. Those interested in leading future implementations or contributing in a more substantial capacity are welcome to get involved.

For detailed guidelines on how to contribute, please refer to our [CONTRIBUTING.md](CONTRIBUTING.md). By working together, we can ensure that TemporalScope remains an innovative and reliable tool, continuously refined through community collaboration.

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

## **License and Legal Notice**

TemporalScope is licensed under the [Apache License 2.0](LICENSE). By using this package, you agree to comply with the terms and conditions set forth in this license.

**LEGAL NOTICE**: THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.

