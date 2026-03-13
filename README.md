# Machine Learning-Aided Joint Resource Allocation and Signal Detection in OFDM

## Overview

Orthogonal Frequency Division Multiplexing (OFDM) is widely used in modern wireless communication systems due to its high spectral efficiency and robustness against multipath fading. However, OFDM performance degrades in scenarios with limited pilot symbols and short cyclic prefixes, leading to poor signal detection and increased inter-subcarrier interference.

This project implements a machine learning–aided framework to enhance signal detection and resource allocation in an OFDM system. Different machine learning models such as Support Vector Machine (SVM), Random Forest, Decision Tree, and Gradient Descent are trained to improve detection accuracy under varying channel conditions.

The project evaluates system performance in terms of Bit Error Rate (BER) versus Signal-to-Noise Ratio (SNR), detection accuracy, and model efficiency.

---

## Objectives

* Simulate a conventional OFDM communication system.
* Generate datasets from OFDM transmissions under different SNR conditions.
* Train machine learning models for signal detection.
* Compare ML-based detection with conventional OFDM detection.
* Analyze system performance using BER and accuracy metrics.

---

## System Architecture

```
Binary Data
    ↓
QPSK Modulation
    ↓
OFDM Transmitter (IFFT)
    ↓
Add Cyclic Prefix
    ↓
Wireless Channel (AWGN / Multipath)
    ↓
Remove Cyclic Prefix
    ↓
FFT (OFDM Receiver)
    ↓
ML-Based Signal Detection
    ↓
Bit Error Rate Evaluation
```

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Scikit-learn
* Pandas

---

## Machine Learning Models Implemented

* Support Vector Machine (SVM)
* Random Forest
* Decision Tree
* Logistic Regression (Gradient Descent)

These models are trained on features extracted from received OFDM signals.

---

## Repository Structure

```
ML_OFDM_Project
│
├── data
│   └── dataset.csv
│
├── src
│   ├── ofdm_system.py
│   ├── dataset_generator.py
│   ├── ml_models.py
│   └── evaluation.py
│
├── results
│   ├── ber_vs_snr.png
│   ├── ml_accuracy.png
│   └── confusion_matrix.png
│
├── report
│   └── project_report.pdf
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository

```
git clone https://github.com/yourusername/ML_OFDM_Project.git
```

Install required libraries

```
pip install -r requirements.txt
```

---

## Running the OFDM Simulation

Run the baseline OFDM system:

```
python src/ofdm_system.py
```

This generates the BER vs SNR graph for the conventional OFDM system.

---

## Dataset Generation

Generate the dataset used for training machine learning models:

```
python src/dataset_generator.py
```

The generated dataset will be saved in the `data/` directory.

---

## Training Machine Learning Models

Train ML models for signal detection:

```
python src/ml_models.py
```

The script trains SVM, Random Forest, Decision Tree, and Logistic Regression models and evaluates their performance.

---

## Results

The following performance metrics are analyzed:

* Bit Error Rate (BER) vs SNR
* Machine Learning Model Accuracy
* Confusion Matrix
* Training Time Comparison

Example output graphs:

* BER vs SNR for conventional OFDM
* ML detection accuracy comparison
* Confusion matrix for signal classification

---

## Sample Result

Typical results show that machine learning-based detection improves signal detection accuracy and reduces BER compared to conventional detection techniques, especially in low-SNR environments.

---

## Future Work

* Extend the model to multipath fading channels
* Implement deep learning models such as CNN or RNN
* Apply the approach to MIMO-OFDM systems
* Optimize pilot allocation using reinforcement learning

---

## Author

Komlaksh Sharma
B.Tech Project – Machine Learning in Wireless Communication

---

## License

This project is developed for academic and research purposes.
