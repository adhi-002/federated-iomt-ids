# Federated IoMT IDS

An undergraduate capstone prototype investigating unseen-attack detection in federated intrusion detection for Internet of Medical Things networks.

## Current prototype

Three simulated hospital clients train an MLP with 64 and 32 hidden neurons. A central FedAvg step combines their learned weights and biases in proportion to local sample counts. A softmax-confidence threshold, calibrated on known validation examples, rejects potentially unknown traffic.

The implementation runs in a single process using scikit-learn. It includes a Streamlit viewer, reproducible configurations, saved checkpoints and protocol tests. It does not yet implement Flower, differential privacy, adaptive aggregation or the planned Mahalanobis/energy detector comparison.

**The default run uses synthetic data for a pipeline demonstration. Its results are not CICIoMT2024 research findings.** CICIoMT2024 integration requires checking the real files, label mapping and capture splits. Datasets and generated results are not distributed in this repository.

## Quick start

Use Python 3.10 or later:

```powershell
cd "Project Work/prototype"
python -m pip install -r requirements.txt
python -m unittest -v
python pipeline.py
python -m streamlit run app.py
```

See [the prototype guide](Project%20Work/prototype/README.md) for labelled CSV input, withholding an attack class, metrics and limitations.

## Research direction

The proposed study compares confidence, Mahalanobis and energy detection under heterogeneous client distributions using a fixed federated classifier. This is a working experimental direction; novelty and performance are not established.

Research papers, review submissions, private working notes, datasets and generated model files stay outside the public source history.
