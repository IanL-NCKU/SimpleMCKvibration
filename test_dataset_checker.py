"""
Example script to test the dataset checker with input-output relationship plots.

Usage:
    python test_dataset_checker.py
"""

from Exp_dataset import load_exponential_data

# Load data with normalization
print("Loading data...")
train_loader, val_loader, test_loader, inputs_normalizer, outputs_normalizer = \
    load_exponential_data(filepath='exponential_trainval_data.npz', normalize=True)

print("\nDataset checker will run automatically with the following options:")
print("  - Error threshold: 10%")
print("  - Verbose: False (only summary shown)")
print("  - Plot I/O relations: Enabled by default in load_exponential_data()")
print("  - Plot sample rate: 1 (all samples plotted)")
print("\nTo customize plotting, you can manually call check_all_datasets():")
print("Example:")
print("  from Exp_dataset import check_all_datasets")
print("  results = check_all_datasets(")
print("      train_loader, val_loader, test_loader,")
print("      inputs_normalizer, outputs_normalizer,")
print("      error_threshold=0.10,")
print("      verbose=False,")
print("      plot_io_relation=True,")
print("      save_dir='./IOrelation',")
print("      plot_sample_rate=10  # Plot every 10th sample")
print("  )")
