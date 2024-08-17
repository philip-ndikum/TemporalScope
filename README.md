# PlatformsLanguage
<p align="center">
  <strong><font size="6">TemporalScope</font></strong><br>
  <strong><font size="4">Model-Agnostic Temporal Feature Importance Analysis</font></strong>
</p>

---

![License](https://img.shields.io/github/license/philip-ndikum/TemporalScope)
![Tests](https://github.com/philip-ndikum/TemporalScope/actions/workflows/run_tests.yml/badge.svg)
![PyPI](https://img.shields.io/pypi/v/temporalscope)
![Downloads](https://img.shields.io/pypi/dm/temporalscope)
![PyPI pyversions](https://img.shields.io/pypi/pyversions/temporalscope)

## **Introduction**

**TemporalScope** is a cutting-edge Python package developed to address one of the most challenging issues in industrial and academic machine learning: the non-stationarity of data. Traditional feature importance methods often assume that the importance of features remains static over time, which is rarely the case in real-world applications. TemporalScope introduces scientifically driven methods and novel techniques to analyze how feature importance evolves across different temporal or contextual segments of your data, providing deeper insights into your models and their behavior.

### **Purpose and Motivation**

Non-stationarity is a pervasive issue in many industries where the relationship between input features and outcomes can change over time due to evolving conditions, market dynamics, patient responses, or machinery wear and tear. TemporalScope was created to offer a robust solution for analyzing these temporal dynamics, making it an essential tool for data scientists and researchers aiming to build more accurate, reliable, and interpretable models.

### **Key Features**

- **Temporal Feature Importance Analysis**: Track and quantify how the importance of features changes over time or across different data partitions.
- **Model-Agnostic Design**: Compatible with any machine learning model, providing flexibility across diverse applications.
- **Advanced Scientific Methods**: Incorporates state-of-the-art techniques such as Reverse Expanding Window, Partitioning, and the WinIT approach.
- **Integration with BorutaSHAP**: Enhance your feature selection process with the powerful BorutaSHAP methodology.

## **Installation**

TemporalScope can be installed easily via [PyPI](https://pypi.org/project/temporalscope/):

- **Install using pip**: `pip install temporalscope`

Alternatively, if you're working on a local machine and need to set up the environment manually:

1. Clone the repository: `git clone https://github.com/philip-ndikum/TemporalScope.git`
2. Navigate to the directory: `cd TemporalScope`
3. Run the setup script: `bash setup.sh`

This script will ensure that all dependencies are installed, and the environment is correctly configured.

## **Usage**

TemporalScope is designed to be user-friendly and straightforward to integrate into your existing workflows. Below is a basic usage example:

- **Import the package**: `import temporalscope`
- **Analyze temporal feature importance**: Utilize the provided methods to explore how feature importance varies over time or across data segments.

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

After reaching version 1.0.0, future developments will be guided by user feedback and the needs of the community, ensuring that TemporalScope remains robust, efficient, and relevant.

## **Contributing**

We welcome contributions from developers, researchers, and data scientists. Whether you're interested in extending the package, fixing bugs, or improving documentation, your input is valuable to us. Please refer to our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute.

### **Acknowledgment and Consulting**

TemporalScope was created and developed by [Philip Ndikum](https://github.com/philip-ndikum). For advisory or consulting services related to advanced and proprietary implementations of this open-source software, please reach out directly.

## **License**

TemporalScope is licensed under the [Apache License 2.0](LICENSE). By using this package, you agree to comply with the terms and conditions set forth in this license.

---
*If you find TemporalScope useful, please consider starring the repository on GitHub and sharing it with your network to help others discover this tool.*
