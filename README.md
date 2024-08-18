<p align="center">
  <img src="assets/temporal_scope_logo.png" alt="TemporalScope Logo" width="400">
</p>

<h3 align="center">Model-Agnostic Temporal Feature Importance Analysis</h3>

---

![License](https://img.shields.io/github/license/philip-ndikum/TemporalScope)
![Tests](https://github.com/philip-ndikum/TemporalScope/actions/workflows/run_tests.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/temporalscope)
![Downloads](https://img.shields.io/pypi/dm/temporalscope)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/temporalscope)

**TemporalScope** is an innovative Python package designed to analyze and understand the temporal dynamics of feature importance across various machine learning models. Unlike traditional methods that treat feature importance as static, TemporalScope allows you to explore how feature importance evolves over time or across different data segments.

## **Installation**

You can install TemporalScope from [PyPI](https://pypi.org/project/temporalscope/) with:

- **Install using pip**: `pip install temporalscope`

## **Usage**

Here’s how you can use TemporalScope:

- **Import the package**: `import temporalscope`
- **Analyze temporal feature importance**: Use the package's functions to explore how feature importance changes over time.

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

We welcome contributions from developers, researchers, and data scientists. Whether you're interested in extending the package, fixing bugs, or improving documentation, your input is valuable to us. Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

### **Acknowledgment and Consulting**

TemporalScope was created and developed by [Philip Ndikum](https://github.com/philip-ndikum). For advisory or consulting services related to advanced and proprietary implementations of this open-source software, please reach out directly.

## **License and Legal Notice**

TemporalScope is licensed under the [Apache License 2.0](LICENSE). By using this package, you agree to comply with the terms and conditions set forth in this license.

**LEGAL NOTICE**: THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE. 

THIS SOFTWARE IS INTENDED FOR ACADEMIC AND INFORMATIONAL PURPOSES ONLY. IT SHOULD NOT BE USED IN PRODUCTION ENVIRONMENTS OR FOR CRITICAL DECISION-MAKING WITHOUT PROPER VALIDATION. ANY USE OF THIS SOFTWARE IS AT THE USER'S OWN RISK.

