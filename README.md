Federated Learning with Adaptive Sensitivity Hybrid Differential Privacy for Health Internet of Things Systems
This repository contains the implementation of the Adaptive Sensitivity Hybrid Differential Privacy (AS-HDP) framework for healthcare federated learning.
The project focuses on improving privacy protection in:
•	Healthcare machine learning
•	Health Internet of Things (HIoT) systems
•	Distributed healthcare analytics
while preserving predictive utility under:
•	Non-IID federated learning,
•	Client heterogeneity,
•	Dynamic training behaviour,
•	and class imbalance.
The framework integrates:
•	Federated Learning (FL),
•	Adaptive Sensitivity Estimation,
•	Adaptive Gradient Clipping,
•	Hybrid Differential Privacy (Local + Central DP),
•	Dynamic Privacy Budget Allocation,
•	and Rényi Differential Privacy (RDP) Accounting.
The framework was evaluated using:
•	Breast Cancer Wisconsin Diagnostic Dataset,
•	Diabetes Dataset,
•	and Glioma Grading Dataset.
Research Motivation
The rapid growth of the Health Internet of Things (HIoT) has enabled continuous patient monitoring and data-driven healthcare. However, healthcare data remains highly sensitive and vulnerable to privacy leakage.
Although Federated Learning allows institutions to collaboratively train models without sharing raw patient data, exchanged gradients and model updates can still leak sensitive information through:
•	Membership inference attacks,
•	Gradient leakage,
•	and reconstruction attacks.
Traditional Differential Privacy (DP) and Hybrid Differential Privacy (HDP) methods typically rely on:
•	Fixed clipping thresholds,
•	Static sensitivity assumptions,
•	and fixed noise calibration.
These assumptions often degrade model utility in heterogeneous healthcare environments.
This project introduces an adaptive privacy-preserving federated learning framework that dynamically aligns privacy protection with:
•	observed gradient behaviour,
•	sensitivity dynamics,
•	update magnitudes,
•	and training progress.
Main Contributions
The AS-HDP framework introduces:
1. Adaptive Sensitivity Estimation
The framework estimates client sensitivity dynamically instead of relying on fixed worst-case assumptions.
2. Adaptive Gradient Clipping
Gradient clipping bounds are updated dynamically using:
•	exponential moving averages,
•	and quantile-based update statistics.
3. Hybrid Differential Privacy
The framework combines:
•	Local Differential Privacy (LDP),
•	and Central Differential Privacy (CDP)
to strengthen privacy protection.
4. Dynamic Privacy Budget Scheduling
Privacy budgets are allocated adaptively across communication rounds based on learning behaviour.
5. Rényi Differential Privacy (RDP) Accounting
The framework tracks cumulative privacy expenditure across training rounds using Rényi Differential Privacy accounting.

Framework Architecture
The AS-HDP framework contains the following main components:
Component	Function
Local Model Training	Client-side federated learning
Adaptive Clipping	Dynamic update norm control
Client-Level Sensitivity Estimation	Dynamic privacy calibration
Local DP Noise Injection	Client-side privacy protection
Secure Aggregation	Aggregation of protected updates
Adaptive Budget Allocation	Dynamic privacy scheduling
Central DP Noise Injection	Server-side privacy protection
RDP Accounting	Multi-round privacy tracking
The architecture aligns privacy enforcement with real gradient behaviour during training.

Datasets Used
The framework was evaluated using publicly available healthcare datasets.
Dataset	Task
Breast Cancer Wisconsin Diagnostic	Binary cancer classification
Diabetes Dataset	Diabetes prediction
Glioma Grading Dataset	Brain tumour grading classification
The datasets simulate realistic heterogeneous healthcare federated learning environments.

Repository Structure
├── Breast_Cancer_AS_HDP.py
├── Diabetes_AS_HDP.py
├── Glioma_AS_HDP.py
├── README.md
├── requirements.txt
└── outputs/

Key Features of the Codebase
The implementation supports:
•	Federated learning simulation
•	Non-IID Dirichlet client partitioning
•	Multi-seed experimentation
•	Adaptive clipping
•	Adaptive sensitivity estimation
•	Hybrid Differential Privacy
•	Dynamic privacy scheduling
•	Rényi DP accounting
•	Contribution-aware aggregation
•	Statistical evaluation
•	Confidence intervals
•	Privacy fairness verification
•	Validation AUPRC tracking
The codes are designed for:
•	reproducibility,
•	healthcare privacy research,
•	and experimental benchmarking.

Experimental Configuration
Typical experimental settings include:
Parameter	Value
Number of Clients	8
Communication Rounds	30
Client Participation	75%
Local Epochs	2–3
Dirichlet Alpha	0.5–0.6
Random Seeds	Multi-seed evaluation
Differential Privacy δ	1e−5
The implementation also performs:
•	fairness checks,
•	sequential privacy composition,
•	and confidence interval analysis.

Ablation Studies
The framework includes several ablation configurations.
Ablation	Description
NO_DP	Federated Learning without Differential Privacy
FIXED_HDP	Hybrid DP with fixed sensitivity assumptions
A1	Adaptive clipping only
A2	Adaptive clipping + adaptive budgeting
A3	Adaptive clipping + adaptive sensitivity
A4	Full AS-HDP framework
The ablation studies isolate the contribution of each framework component.

Evaluation Metrics
The framework evaluates:
•	Accuracy
•	Precision
•	Recall
•	F1-score
•	AUPRC (Area Under Precision Recall Curve)
AUPRC is emphasized because healthcare datasets are highly imbalanced.

Core Methodology
Adaptive Clipping
The clipping bound is updated dynamically using:
•	Exponential Moving Averages (EMA),
•	and quantile-based update norm estimation.
Sensitivity Estimation
Sensitivity estimation combines:
•	feature variability,
•	update magnitudes,
•	and client heterogeneity.
Hybrid Differential Privacy
The framework applies:
•	Local DP noise at the client side,
•	and Central DP noise during server aggregation.
Privacy Accounting
The framework tracks cumulative privacy expenditure using:
•	Rényi Differential Privacy (RDP),
•	and converts accumulated privacy cost into final ((\epsilon, \delta))-DP guarantees.

Key Findings
The experiments show that:
•	Static privacy mechanisms are often suboptimal in healthcare federated learning.
•	Adaptive sensitivity improves privacy calibration.
•	Adaptive clipping improves convergence stability.
•	Hybrid DP reduces privacy leakage risks.
•	Privacy–utility behaviour is highly dataset-dependent.
•	Sensitivity dynamics strongly influence model utility.
•	AS-HDP reduces seed variability compared with fixed HDP.
•	AS-HDP preserves clinically meaningful performance under privacy constraints.

Installation
Install Dependencies
pip install numpy pandas scikit-learn matplotlib torch scipy

Running the Experiments
Breast Cancer Experiments
python Breast_Cancer_AS_HDP.py
Diabetes Experiments
python Diabetes_AS_HDP.py
Glioma Experiments
python Glioma_AS_HDP.py

Outputs Generated
The framework automatically generates:
•	Validation histories
•	Test performance metrics
•	Privacy ledgers
•	Mean ± standard deviation tables
•	95% confidence intervals
•	Privacy fairness statistics
•	Convergence curves
•	AUPRC plots

Intended Applications
This repository is intended for:
•	Healthcare AI research
•	Federated Learning research
•	Differential Privacy research
•	Health Internet of Things (HIoT) systems
•	Privacy-preserving machine learning
•	Graduate and PhD research
•	Experimental benchmarking

Citation
If you use this repository in your research, please cite:
Rebecca Adhiambo Okaka, Evanson Mwangi Karanja, Isaac Nyabisa Oteyo. Federated Learning with Adaptive Sensitivity Hybrid Differential Privacy for Health Internet of Things Systems.2026.

Author
Rebecca Adhiambo Okaka
Research Areas:
•	Federated Learning
•	Differential Privacy
•	Healthcare AI
•	Privacy-Preserving Machine Learning
•	Adaptive Privacy Mechanisms

Acknowledgements
This work builds upon research in:
•	Federated Learning,
•	Differential Privacy,
•	Adaptive Clipping,
•	Rényi Differential Privacy,
•	and Healthcare AI.
Special thanks to the open-source research community for foundational contributions to privacy-preserving machine learning.

