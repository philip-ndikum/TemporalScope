<p align="center">
  <img src="assets/temporalscope_github_banner.svg" alt="TemporalScope Logo" >
</p>

<h3 align="center">Model-Agnostic Temporal Feature Importance Analysis</h3>

---

![License](https://img.shields.io/github/license/philip-ndikum/TemporalScope)
![Tests](https://github.com/philip-ndikum/TemporalScope/actions/workflows/run_tests.yml/badge.svg)
![Development Status](https://img.shields.io/badge/status-in%20development-yellow)
![GitHub Issues](https://img.shields.io/github/issues/philip-ndikum/TemporalScope)

**TemporalScope** is an innovative, open-source software (OSS) Python package licensed under the Apache 2.0 License. It is designed to analyze and understand the temporal dynamics of feature importance across various machine learning models. Traditional methods often treat feature importance as static, but **TemporalScope** enables you to explore how feature importance evolves over time or across different data segments.

1. **Addressing Non-Stationarity**: Research has shown that using SHAP or causal methods on a single dataset can be insufficient due to non-stationarity—changes in data distribution over time. **TemporalScope** directly addresses this challenge, offering a more nuanced and accurate analysis of feature importance.

2. **Built on Cutting-Edge Research**: **TemporalScope** is grounded in the latest scientific literature, designed to solve a critical problem faced by analysts, scientists, and quantitative professionals worldwide. It allows for a deeper understanding of how features influence model predictions over time, helping to uncover hidden patterns and insights.

3. **Standards and Scalability**: Developed according to Linux Foundation standards, **TemporalScope** offers robust documentation and a lean, scalable approach. It includes easy-to-use benchmarks for comparing different techniques, making it a valuable tool for both research and production environments.

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
# Step 1: Import TemporalScope and other necessary libraries
import temporalscope as ts
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

# Step 2: Load your data (using Pandas backend by default)
housing = fetch_california_housing(as_frame=True)
df = pd.concat([housing.data, housing.target.rename('MedHouseVal')], axis=1)
df['timestamp'] = pd.date_range(start='1/1/2000', periods=len(df), freq='M')
data = ts.TimeSeriesData(data=df, time_col='timestamp')

# Step 3A: Apply MASV method with a pre-trained model (Random Forest in this case)
model = RandomForestRegressor(n_estimators=100)
model.fit(df[housing.feature_names], df['MedHouseVal'])
masv_results = data.calculate_masv(model=model, phases=[(0, 100), (100, 200)])

# Step 3B: Alternatively, let TemporalScope train a Random Forest model
# masv_results = data.calculate_masv(model=None, phases=[(0, 100), (100, 200)])

# Step 4: Analyze and Visualize the results
# masv_results will be a dictionary with feature names as keys and MASV scores as values
for feature, masv_scores in masv_results.items():
    plt.plot(masv_scores, label=f'{feature}')
    
plt.title('Mean Absolute SHAP Values Across Phases')
plt.xlabel('Phase')
plt.ylabel('MASV Score')
plt.legend()
plt.show()
```

### **Example Use Cases**

| **Sector**    | **Use Case**                                                                                  | **Impact**                                                                                         |
|---------------|-----------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **Finance**   | Analyzing how the importance of financial indicators changes across different market cycles.   | Improves investment strategies by understanding how market conditions affect the relevance of indicators. |
| **Healthcare**| Tracking how patient data features contribute to health outcomes at various stages of treatment.| Enables personalized treatment plans by identifying critical features at different treatment phases.  |
| **Engineering**| Examining the impact of factors on machinery lifecycle, guiding maintenance schedules.        | Reduces downtime and maintenance costs by understanding how wear and tear affect feature importance.  |

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


## **License and Legal Notice**

TemporalScope is licensed under the [Apache License 2.0](LICENSE). By using this package, you agree to comply with the terms and conditions set forth in this license.

**LEGAL NOTICE**: THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.

