# TRACE
### Performance-Aware Adaptive Ensemble Learning for Concept-Drift-Resilient Network Intrusion Detection

> An intelligent Network Intrusion Detection System designed to evaluate streaming machine-learning models, detect concept drift, investigate model adaptation, and provide explainable intrusion-monitoring insights.

---

## 📌 Overview

Network traffic patterns continuously evolve due to changes in network infrastructure, user behavior, attack techniques, and emerging cybersecurity threats. Traditional machine-learning-based Network Intrusion Detection Systems (NIDS) often rely on static models trained on historical data. These models may become less effective when the underlying traffic distribution changes, a phenomenon known as **concept drift**.

TRACE investigates the performance of online machine-learning models and adaptive ensemble-learning approaches for network intrusion detection under changing data distributions.

The project evaluates baseline models, drift-detection mechanisms, adaptive ensemble strategies, selective model replacement, and explainability techniques. It also provides an interactive dashboard for examining network traffic, intrusion alerts, model performance, and explainability results.

---

## 🎯 Problem Statement

Traditional Network Intrusion Detection Systems commonly use machine-learning models trained on fixed datasets. However, real-world network environments are dynamic, and the statistical characteristics of network traffic may change over time.

When these changes occur, a trained model may experience performance degradation, resulting in increased false positives, missed attacks, and reduced detection reliability.

The problem addressed by TRACE is:

> **How can a machine-learning-based Network Intrusion Detection System maintain reliable detection performance under changing network traffic distributions while adapting efficiently and providing interpretable insights into its predictions?**

The project investigates online learning, ensemble learning, concept drift detection, adaptive model strategies, and explainable AI as potential approaches to addressing this challenge.

---

## 🎯 Objectives

The major objectives of TRACE are:

- Evaluate multiple machine-learning models for network intrusion detection.
- Establish baseline performance for online-learning algorithms.
- Investigate concept drift in streaming network traffic.
- Evaluate the effect of distributional changes on model performance.
- Develop and investigate adaptive ensemble-learning strategies.
- Analyze model adaptation and recovery after drift.
- Investigate selective model replacement strategies.
- Evaluate performance using accuracy, precision, recall, F1-score, and computational metrics.
- Analyze model predictions using explainable AI techniques.
- Develop an interactive dashboard for network monitoring and experimental visualization.

---

## 🧠 Proposed Approach

TRACE follows an experimental pipeline consisting of data preprocessing, baseline model evaluation, drift analysis, adaptive learning experiments, explainability, and dashboard visualization.

### 1. Data Preparation

The project uses the NSL-KDD dataset for network intrusion detection experiments.

The dataset is preprocessed to prepare features for machine-learning models. Categorical network features are encoded, and the resulting feature representation is aligned between training and testing data.

### 2. Baseline Model Evaluation

Multiple online-learning algorithms are evaluated to establish baseline performance.

The models investigated include:

- Hoeffding Tree
- Hoeffding Adaptive Tree
- Adaptive Random Forest
- Online Naive Bayes
- Streaming k-Nearest Neighbors

The models are evaluated using classification metrics and computational performance measurements.

### 3. Concept Drift Analysis

Concept drift experiments investigate how model performance changes when the distribution of incoming network traffic changes.

The experiments examine:

- Performance before drift
- Performance during distributional changes
- Performance degradation following drift
- Drift detection behavior
- Model recovery and adaptation

### 4. Adaptive Ensemble Learning

Adaptive ensemble strategies combine predictions from multiple online-learning models.

The project investigates mechanisms including:

- Model performance tracking
- Dynamic ensemble weighting
- Weighted prediction strategies
- Drift-aware adaptation
- Model replacement
- Selective replacement strategies

The objective is to investigate whether adaptation can improve detection performance and recovery under changing traffic conditions.

### 5. Explainable AI

Explainability techniques are used to investigate model predictions and feature contributions.

The project incorporates:

- SHAP-based feature importance analysis
- LIME-based local prediction explanations
- Feature importance comparison before and after distributional changes
- Prediction-level analysis

These methods help investigate which input features influence the model's predictions.

### 6. Interactive Dashboard

The project includes an interactive dashboard designed to visualize network monitoring information, intrusion alerts, model performance, and explainability outputs.

The dashboard supports two primary modes:

- **Dataset Replay:** Processes dataset samples incrementally and displays model and monitoring information.
- **Live Packet Capture:** Captures network packets using packet-capture functionality and displays traffic statistics and heuristic-based alerts.

---

## 🏗️ System Architecture

```text
                  Network Traffic / NSL-KDD Dataset
                               │
                               ▼
                     Data Preprocessing
                               │
                               ▼
                  Feature Encoding & Preparation
                               │
                               ▼
                 Online Machine Learning Models
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Hoeffding Tree   Adaptive Models   Ensemble Models
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                    Drift Monitoring
                               │
                               ▼
                  Adaptive Model Strategies
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
          Model Adaptation             Model Replacement
                 │                           │
                 └─────────────┬─────────────┘
                               ▼
                     Performance Evaluation
                               │
                               ▼
                       Explainable AI
                         SHAP / LIME
                               │
                               ▼
                     Interactive Dashboard
```

---

## 📊 Dataset

### NSL-KDD

TRACE uses the NSL-KDD intrusion-detection dataset.

The dataset contains network connection records represented using traffic-related features and labels describing normal or attack traffic.

Files used:

```text
data/
├── KDDTrain+.txt
└── KDDTest+.txt
```

The dataset is used for:

- Model training
- Baseline evaluation
- Streaming experiments
- Concept drift experiments
- Explainability analysis
- Dataset replay within the dashboard

> **Dataset Note:** NSL-KDD is a benchmark dataset and does not represent all characteristics of modern production network traffic. Experimental findings should therefore be interpreted in the context of the dataset and experimental setup.

---

## 🤖 Machine Learning Models

TRACE evaluates the following models:

| Model | Description |
|---|---|
| Hoeffding Tree | Incremental decision-tree learning model |
| Hoeffding Adaptive Tree | Adaptive tree model designed to respond to changing data distributions |
| Adaptive Random Forest | Ensemble of adaptive online decision trees |
| Online Naive Bayes | Incremental probabilistic classifier |
| Streaming k-NN | Instance-based learning model for streaming data |

---

## 📈 Evaluation Metrics

The models are evaluated using the following metrics:

### Classification Metrics

- **Accuracy:** Overall proportion of correctly classified instances.
- **Precision:** Proportion of predicted positive instances that are correctly classified.
- **Recall:** Proportion of actual positive instances correctly detected.
- **F1-Score:** Harmonic mean of precision and recall.

### Computational Metrics

- **Prediction Latency:** Time required to generate predictions.
- **Model Size:** Memory footprint or estimated model size.

### Drift-Related Metrics

- Performance before drift
- Performance after drift
- Performance after adaptation
- Detection delay
- False alarms
- Model replacement attempts
- Replacement success

---

## 🧪 Experiments

The project contains multiple experimental stages.

### Experiment 1 — Baseline Model Evaluation

The baseline experiment evaluates individual online-learning models using the prepared intrusion-detection dataset.

The objective is to establish comparative performance across the evaluated models.

### Experiment 2 — Drift Detection

This experiment investigates concept drift detection and analyzes the behavior of models when the incoming data distribution changes.

### Experiment 3 — Adaptive Ensemble Learning

This experiment investigates ensemble-learning strategies that dynamically incorporate model performance information.

### Experiment 4 — Drift-Aware Adaptive Ensemble

This experiment investigates adaptive ensemble behavior under drift and evaluates model adaptation strategies.

### Experiment 5 — Corrected Multi-Seed Evaluation

A corrected experimental implementation is evaluated across multiple seeds to investigate the consistency and reliability of the observed results.

### Experiment 6 — Explainability Under Distributional Change

Explainability techniques are applied to investigate feature importance and prediction behavior before and after distributional changes.

### Experiment 7 — Multi-Seed Drift Analysis

Multiple experimental runs are used to evaluate the consistency of adaptation behavior and model performance.

---

## 📊 Baseline Results

The following results represent the baseline experiments conducted using the current implementation.

| Model | Accuracy | Precision | Recall | F1-Score | Latency (ms) | Model Size (KB) |
|---|---:|---:|---:|---:|---:|---:|
| Hoeffding Tree | 0.9182 | 0.8838 | 0.9497 | 0.9156 | 0.024 | 119.3 |
| Hoeffding Adaptive Tree | 0.9412 | 0.9411 | 0.9323 | 0.9367 | 0.472 | 8794.8 |
| Adaptive Random Forest | 0.9750 | 0.9894 | 0.9566 | 0.9727 | 0.245 | 649.8 |
| Online Naive Bayes | 0.5758 | 0.9977 | 0.0913 | 0.1674 | 0.511 | 917.0 |
| Streaming k-NN | 0.9767 | 0.9745 | 0.9756 | 0.9751 | 3.032 | 1326.3 |

### Baseline Observations

- Adaptive Random Forest achieved strong overall baseline performance.
- Streaming k-NN achieved the highest baseline accuracy and F1-score in the reported results.
- Hoeffding Tree provided comparatively low prediction latency and a small model footprint.
- Hoeffding Adaptive Tree demonstrated higher computational and memory costs.
- Online Naive Bayes exhibited high precision but comparatively low recall in the evaluated configuration.

These observations highlight the trade-offs between predictive performance, latency, and model complexity.

---

## 🔄 Concept Drift and Adaptation Results

Controlled drift experiments were conducted to investigate performance changes and model recovery following distributional changes.

### Multi-Seed Results

| Evaluation Stage | Accuracy | F1-Score |
|---|---:|---:|
| Pre-Drift | 0.998 ± 0.0027 | 0.9980 ± 0.0027 |
| Post-Drift, Before Adaptation | 0.784 ± 0.0147 | 0.7256 ± 0.0234 |
| Post-Adaptation | 0.993 ± 0.0057 | 0.9931 ± 0.0056 |

### Key Observations

- The controlled distributional change caused a substantial reduction in model performance.
- Accuracy and F1-score decreased before adaptation.
- The evaluated adaptation mechanism recovered a substantial portion of the lost performance.
- Multi-seed evaluation demonstrated that recovery behavior was observed across multiple experimental runs.

### Performance Recovery

The reported average recovery was approximately:

- **20.90 percentage points in accuracy**
- **26.75 percentage points in F1-score**

> These results are specific to the controlled experimental setup. They should not be interpreted as proof of equivalent recovery performance in real-world production networks.

---

## 🧩 Adaptive Ensemble Analysis

The project also evaluates adaptive ensemble approaches using multiple online-learning models.

Ensemble mechanisms investigate:

- Performance-based model weighting
- Rolling model accuracy
- Weighted voting
- Dynamic adaptation
- Drift-related model performance
- Replacement strategies

Experimental results are retained in the `results/` directory to support reproducibility and analysis.

---

## 🔍 Explainability

TRACE incorporates explainability techniques to investigate the decisions made by machine-learning models.

### SHAP

SHAP is used to analyze feature contributions and global feature importance.

Features identified as influential in the evaluated analysis include:

- `dst_host_srv_serror_rate`
- `same_srv_rate`
- `src_bytes`
- `dst_bytes`
- `count`
- `flag_S0`
- `protocol_type_icmp`
- `dst_host_srv_count`
- `dst_host_same_srv_rate`

Feature importance may change following distributional changes, providing insights into how model behavior and feature reliance evolve.

### LIME

LIME is used to explain individual predictions.

The explainability experiments include examples involving normal traffic and attack traffic.

The explanations provide local insights into the features contributing to individual model predictions.

> Explainability outputs indicate model behavior and feature contributions. They should not be interpreted as establishing causal relationships between network features and intrusion activity.

---

## 🖥️ Dashboard

TRACE includes a Streamlit dashboard for monitoring and visualization.

### Dashboard Components

#### 1. Live Network Monitor

Displays information such as:

- Packet counts
- Protocol distribution
- Network traffic statistics
- Flow-related information
- Recent traffic activity

#### 2. Intrusion Alerts

Displays alert information generated by the monitoring system.

Features include:

- Alert feed
- Alert severity or category information
- Recent events
- Alert explanations

#### 3. Model Performance

Displays model evaluation and performance information.

Depending on the selected monitoring mode, visualizations may include:

- Accuracy
- Precision
- Recall
- F1-score
- Performance timelines
- Model comparisons

#### 4. Explainability

Displays available feature importance and prediction explanations.

---

## ⚠️ Live Monitoring Limitations

The live packet-capture mode and dataset-based model evaluation are currently separate components.

The NSL-KDD-trained models use a feature representation that is not directly equivalent to raw packets captured from a network interface.

Therefore:

- Dataset Replay evaluates the online machine-learning pipeline using dataset records.
- Live Packet Capture captures packets and analyzes traffic using the current monitoring and heuristic-based logic.
- Raw captured packets are not currently transformed into the complete NSL-KDD feature representation and passed directly through the trained experimental models.

The live monitoring functionality should therefore be understood as a packet-monitoring and heuristic-alert component rather than a fully integrated production-grade machine-learning intrusion detector.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data processing |
| NumPy | Numerical computation |
| Scikit-learn | Machine-learning utilities |
| River | Online and streaming machine learning |
| Streamlit | Interactive dashboard |
| Plotly | Interactive visualizations |
| Scapy | Network packet capture and analysis |
| SHAP | Explainability |
| LIME | Local prediction explanations |
| Matplotlib | Experimental plotting |

---

## 📁 Project Structure

```text
TRACE/
│
├── app.py
├── dashboard.py
├── live_traffic.py
├── requirements.txt
│
├── 01_baseline_models.py
├── 02_drift_detection.py
├── 03_adaptive_ensemble.py
├── 04_drift_adaptive_ensemble.py
├── 04_experiment_1B_corrected.py
├── 05_explainability_drift.py
├── 05_explainability_drift_multiseed.py
│
├── run_explainability.py
├── run_live.py
│
├── data/
│   ├── KDDTrain+.txt
│   └── KDDTest+.txt
│
├── plots/
│   ├── adaptive_ensemble_weights.png
│   ├── drift_aware_weights.png
│   └── independent_ensemble_accuracy.png
│
└── results/
    ├── baseline_summary.csv
    ├── adaptive_ensemble_results.csv
    ├── drift_adaptive_summary.csv
    ├── drift_detection_summary.csv
    ├── experiment_1B_corrected/
    ├── explainability/
    ├── explainability_drift/
    └── explainability_drift_multiseed_corrected/
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or 3.11 recommended
- Git
- Windows, Linux, or macOS
- Npcap on Windows for packet capture functionality

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/TRACE.git
cd TRACE
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Linux/macOS

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Project

### Run the Dashboard

```bash
streamlit run dashboard.py
```

The dashboard will open in your browser.

### Run Baseline Experiments

```bash
python 01_baseline_models.py
```

### Run Drift Detection Experiments

```bash
python 02_drift_detection.py
```

### Run Adaptive Ensemble Experiments

```bash
python 03_adaptive_ensemble.py
```

### Run Drift-Aware Ensemble Experiments

```bash
python 04_drift_adaptive_ensemble.py
```

### Run Corrected Experiment

```bash
python 04_experiment_1B_corrected.py
```

### Run Explainability Experiments

```bash
python run_explainability.py
```

### Run Live Monitoring

```bash
python run_live.py
```

---

## 📂 Results and Reproducibility

Experimental outputs are stored in the `results/` and `plots/` directories.

These include:

- CSV files containing experimental metrics
- Statistical summaries
- Individual run results
- Drift detection results
- Replacement experiment results
- Model prediction comparisons
- Ensemble weight histories
- Explainability outputs
- SHAP visualizations
- LIME explanations
- Performance plots

The experimental scripts can be executed to regenerate results, subject to the availability of the dataset, dependencies, and compatible execution environment.

---

## ⚠️ Limitations

The current project has the following limitations:

1. Experiments primarily rely on the NSL-KDD benchmark dataset.
2. Controlled drift experiments do not fully represent chronological drift in real production networks.
3. Live packet capture and dataset-trained machine-learning inference are not yet fully integrated.
4. Live traffic alerts currently rely on monitoring logic and heuristics.
5. Explainability results are dependent on the evaluated model, dataset, and sampling procedure.
6. Experimental performance may vary with model parameters, dataset composition, random seeds, and drift scenarios.
7. Benchmark results should not be interpreted as production cybersecurity guarantees.

---

## 🔮 Future Work

Potential future improvements include:

- Integrating a complete live-flow feature extraction pipeline.
- Converting captured packets into model-compatible feature representations.
- Integrating live traffic directly with online-learning models.
- Evaluating on modern network intrusion datasets.
- Implementing realistic chronological concept drift scenarios.
- Improving drift detection and adaptation mechanisms.
- Optimizing ensemble weighting and model replacement.
- Reducing prediction latency and memory overhead.
- Adding additional explainability methods.
- Improving alert prioritization and severity classification.
- Conducting broader multi-seed and statistical evaluations.
- Implementing automated model monitoring and retraining strategies.

---

## 👩‍💻 Authors

This project was developed by:

1. **Kavya Rajput**
2. **Anjali Kashyap**
3. **Ishika Nadar**
4. **Md. Tausif Haider**

---

## 🎓 Project

**Project Title:** Performance-Aware Adaptive Ensemble Learning for Concept-Drift-Resilient Network Intrusion Detection

**Domain:** Machine Learning · Cybersecurity · Network Intrusion Detection · Online Learning · Concept Drift · Explainable AI

---

## 📜 License

A license has not yet been specified for this repository.

If this project is intended to be publicly distributed or reused, consider adding an appropriate open-source license.

---

## ⭐ Acknowledgements

The project builds upon research and development in online machine learning, ensemble learning, concept drift detection, network intrusion detection, and explainable artificial intelligence.

The NSL-KDD dataset is used as the benchmark dataset for the experimental evaluation.
