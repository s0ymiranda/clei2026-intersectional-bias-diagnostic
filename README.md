# Intersectional Bias Diagnostic Mechanism for Supervised Machine Learning Models

A comprehensive system designed to detect, monitor, and mitigate bias in Machine Learning models, with a specific focus on **intersectional bias** that emerges from the overlap of multiple protected attributes.

## Key Features

### **Intersectional Analysis**
- Detects complex forms of discrimination at the intersections of gender, race, age, and other attributes.
- Computes specific group-level metrics (SPD, DI, EOD, HFI) for a multidimensional assessment.
- Identifies bias patterns that traditional, single-attribute methods often overlook.

### **Real-Time Monitoring**
- Continuous monitoring system for deployed models.
- Sliding prediction buffer (`PREDICTION_BUFFER_SIZE = 150`) for statistical analysis.
- Periodic metric calculation (`MONITORING_BATCH_SIZE = 150`) for proactive bias detection.
- Dashboard with trend visualization and automated alerts.

### **Decision Engine via Ciao Prolog**
- Expert system based on logic rules for optimal mitigation technique selection.
- 5+ implemented techniques: reweighting, resampling, feature augmentation, NSGA-II, and heuristic threshold adjustment.
- Explainable decisions with full logical justification.
- Seamless integration between Python (analysis) and Prolog (reasoning).

### **Modular Architecture**
- Support for binary classification, multiclass classification, and continuous prediction (regression) tasks.
- Adaptability to different datasets: Adult Census Income, German Credit Data, COMPAS, and Communities and Crime.
- Complete pipeline: Detection → Decision → Execution → Evaluation.
- Flexible configuration of fairness thresholds and utility weights.

---

## File Structure

The scripts are organized according to the type of Machine Learning task they address:

| File | Scenario | Main Description |
| :--- | :--- | :--- |
| `ClassificationMultiClass.ipynb` | Binary Classification | System implementation for tasks with **two output classes**. Uses the **Adult Census Income** and **German Credit Data** datasets. |
| `BinaryClassification.ipynb` | Multiclass Classification | System implementation for tasks with **multiple output classes**. By default, configured for the **COMPAS** dataset. |
| `Prediction.ipynb` | Prediction / Real-Time Monitoring | Implementation focused on **real-time monitoring** of biases and autonomous mitigation in regression tasks. Uses the **Communities and Crime** dataset. |

---

## Prerequisites

The scripts are designed to run in environments that allow the execution of system commands, such as **Google Colab** or **Linux/Unix-based** systems (like Jupyter Notebooks with shell access), as they require compiling and installing **Ciao Prolog** and other system dependencies.

### 1. System Dependencies

The scripts automatically handle the installation of:

*   `build-essential`, `wget`, `curl`.
*   **Ciao Prolog**: Download, compilation, and local installation.

### 2. Python Dependencies

*   **imbalanced-learn**: Required for applying certain data balancing techniques. This is installed automatically within the script.

---

## Usage Guide

Executing any of the scripts follows a complete autonomous cycle:

1.  **Run the Script:**
    Open and run the desired file in a compatible environment (Colab, Jupyter, etc.).

2.  **Configuration Phase (Automated):**
    The script first handles the installation and configuration of **Ciao Prolog** and necessary libraries. This phase is critical and may take some time during the first execution.

3.  **System Initialization Phase:**
    The main `BiasMitigationSystem` class is initialized, loading the dataset and setting up the logic environment.

4.  **Monitoring and Mitigation Phase:**
    A complete **monitoring** cycle of predictions is simulated. The system detects bias alerts and, based on Prolog logic, applies **mitigation decisions**.

5.  **Report Generation:**
    Upon completing the monitoring cycle, a **Final Executive Report** is generated, including a summary of alerts, mitigation decisions, and visualizations of their effectiveness.

---

## Parameter Modification

Key parameters controlling the simulation are found in the main execution section of each script.

### 1. Number of Predictions to Monitor

This parameter defines the length of the real-time monitoring simulation. Locate this value within the `main` function.

| File | Parameter | Default Value (Example) | Description |
| :--- | :--- | :--- | :--- |
| `clasificacionbinaria.ipynb` | `NUM_PREDICTIONS` | `600` | Controls the number of iterations in the monitoring cycle. |
| `clasificacionmulticlase.ipynb` | `NUM_PREDICTIONS` | `300` | Controls the number of iterations in the monitoring cycle. |
| `prediccion.ipynb` | `num_predictions` | `600` | Controls the number of iterations in the monitoring cycle. |

**How to Modify:** Change the value assigned to the `NUM_PREDICTIONS` or `num_predictions` variable in the main execution block.

### 2. Dataset to Use

This parameter mainly applies to `clasificacionbinaria.ipynb` to change the data source. Locate this value within the `main` function.

| File | Parameter | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `clasificacionbinaria.ipynb` | `DATASET_TO_RUN` | `'adult'` | Name of the dataset to load for the simulation. |

**How to Modify:** Change the string value of `DATASET_TO_RUN` to the name of the dataset you wish to use (if supported by the system).

### 3. Deep Modifications in the `Config` Class

The `Config` class acts as the system's "control panel" and is located near the beginning of all three scripts. Here you can adjust the system's sensitivity.

#### Modifiable Variables in `Config`:

*   **Fairness Thresholds:** Determine how strict the system is in considering the presence of bias.
    *   `SPD_THRESHOLD = 0.1`: If the statistical parity difference exceeds this value, an alert is triggered. *To be stricter, lower this to 0.05.*
    *   `DI_THRESHOLD = 0.8`: The 80% (or four-fifths) rule. If the disparate impact falls below 0.8, it indicates bias.
    *   `EOD_THRESHOLD = 0.1`: Threshold for Equal Opportunity Difference. Absolute values > 0.1 indicate disparity in TPR.
    *   `HFI_THRESHOLD = 0.8`: Threshold for the global harmonic index. *To force the system to seek greater fairness, raise this to 0.9.*

*   **System Operational Parameters:**
    *   `PREDICTION_BUFFER_SIZE = 150`: Number of samples used to calculate fairness metrics in real-time. Defines how many recent predictions the system keeps in memory for statistical analysis.
    *   `MONITORING_BATCH_SIZE = 150`: Number of predictions between metric calculations. Determines how frequently the system evaluates and detects bias based on the current buffer.

*   **Utility Weights:** The system uses these weights to decide whether to accept a new mitigated model or stick with the previous one.
    *   `UTILITY_WEIGHT_PS`: Weight of the performance (Accuracy/R2).
    *   `UTILITY_WEIGHT_HFI`: Weight of the fairness component.
    *   *Modification:* In the scripts, it is configured as **0.3 (PS) / 0.7 (HFI)**. This prioritizes fairness over accuracy.

*   **Performance Score Weights:**
    *   `PS_WEIGHT_F1 = 0.7`: Weight of the F1-score in the performance calculation.
    *   `PS_WEIGHT_ACC = 0.3`: Weight of the accuracy in the performance calculation.

---

## Known Limitations

### 1. Specific Datasets

Each script is pre-configured for a specific dataset. If a new dataset is required, the loading logic, preprocessing, and definition of sensitive columns must be updated in the system initialization function.

### 2. Initial Setup Time

Downloading and compiling Ciao Prolog can significantly increase startup time during the first execution in a new environment.

### 3. Dataset Capacity and Limitations

The datasets have intrinsic limits defined by their original sources.

#### A. Adult Census Dataset (`clasificacionbinaria.ipynb`)
*   **Source:** UCI Machine Learning Repository.
*   **Approximate Size:** ~48,842 rows (instances) and 14 columns.
*   **Limitation:** The script loads the entire dataset. If `NUM_PREDICTIONS` is set higher than ~10,000, the monitoring cycle might slow down, but the dataset supports up to ~45k unique predictions before repeating data (if sampling without replacement).
*   **Protected Attributes:** Race and Sex.

#### B. German Credit Dataset (`clasificacionbinaria.ipynb`)
*   **Source:** UCI Machine Learning Repository (Statlog).
*   **Approximate Size:** **Only 1,000 rows** and 20 columns.
*   **Critical Limitation:** Given only 1,000 instances, setting `NUM_PREDICTIONS` high (e.g., 800+) will consume almost all available test data. This can cause *overfitting* in monitoring or errors if the sampling attempts to draw more data than exists in the test set (usually 30% of the total, i.e., only ~300 rows). **Recommendation:** Keep `NUM_PREDICTIONS` low for German Credit.

#### C. COMPAS Dataset (`clasificacionmulticlase.ipynb`)
*   **Source:** ProPublica Analysis.
*   **Approximate Size:** ~7,214 rows.
*   **Limitation:** The script filters rows with null values in critical columns (`decile_score`, `race`, `sex`), reducing the effective size to about 6,000 usable records. The test set will be approx. 1,800 records.

#### D. Communities and Crime Dataset (`prediccion.ipynb`)
*   **Source:** UCI Machine Learning Repository.
*   **Approximate Size:** ~1,994 rows and 128 columns.
*   **Limitation:** High dimensionality but few rows. Like German Credit, the test set is small (approx. 600 rows). **Do not set `num_predictions` above 500** for this dataset, or the system will run out of new data to simulate the real-time flow.

---
