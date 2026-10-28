# Small federated IDS prototype

This is a three-client FedAvg simulation with a 64/32-neuron MLP and softmax-confidence unknown rejection. Synthetic mode is a pipeline demonstration, never evidence about CICIoMT2024. Clients are simulated sequentially in one process; there is no hospital deployment, network transport, differential privacy or secure aggregation. Sample counts weight the aggregation of corresponding weights and biases.

## Run on Windows

Open a terminal in this folder. Python 3.10+ is recommended.

```powershell
python -m pip install -r requirements.txt
python -m unittest -v
python pipeline.py
python -m streamlit run app.py
```

The viewer opens in a browser. It reads saved results, rather than rerunning training. Results include the checkpoint, configuration, metrics, per-round validation scores, test decisions, client class counts and a real parameter-payload excerpt.

## Use your friend's dataset

Provide TWO labelled CSV files: original training data and original test data. Each must contain the same numeric feature columns and a label column. Preserve original capture splits. If the release has one CSV per class without labels, add class labels from the documented filenames before combining files WITHIN each split. Do not simply combine train and test and randomly split them again.

```powershell
python pipeline.py --train-csv "D:\data\train.csv" --test-csv "D:\data\test.csv" --label label --unknown "EXACT_WITHHELD_LABEL" --output results/real
```

Enter the absolute path of `results/real` in the viewer's Results folder field. The unknown label must match exactly. Its records are excluded from all training and calibration, and retained only in test. Any other test labels absent from training trigger an error rather than being silently reclassified. Missing labels trigger an error. Only numeric feature columns are used: inspect the schema and remove numeric IDs, targets, capture identifiers or leakage-prone columns before running. A numeric class label is excluded by name.

Start with a manageable, documented subset selected separately inside the original training and test sets, preserving the withheld class in test. Sampling and the interpretation of filenames must be verified against the actual dataset. No CICIoMT2024 download is included. The CSV loader has not yet been verified against the actual release files.

## Protocol and limitations

- Seed 42 by default. Three clients use Dirichlet label partitioning, alpha 0.5. Partitions contain known training data only.
- A stratified 20% of known training records is reserved for calibration/validation. Median imputation and scaling fit training records only. Exact feature duplicates crossing training/test are removed from test; training/validation duplicates are removed from training. This does not guarantee capture/session independence.
- MLP uses ReLU, SGD at learning rate 0.01, zero momentum and L2 alpha 0.0001. This uses installed scikit-learn, not PyTorch/Flower. All clients share the global initialization each round. No optimizer momentum is averaged. The saved global estimator is for prediction; clients perform the actual updates.
- Default: 15 rounds, two local epochs. Architecture and training settings are provisional. There is no early stopping or selection using unknown test data.
- Threshold is the lower 5th percentile of known validation confidence. Unknown is predicted when confidence is strictly below it. The test false unknown rate can differ from the validation budget.
- AUROC uses unknown as positive and `1 - confidence` as its score. Known macro-F1 is reported before and after rejection, alongside unknown recall and false unknown rates.
- This initial prototype implements ONLY softmax rejection. Mahalanobis and energy comparisons remain future work. Unexpected or poor unknown performance is reported honestly.
- Per-client class quality, privacy protections, adaptive aggregation, personalization, multi-seed uncertainty and full-scale benchmarking are not yet implemented.

Never load a checkpoint from an untrusted source: joblib uses pickle serialization.
