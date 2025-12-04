"""
Test script to verify the data distribution plotting function
"""
from Exp_dataset import load_exponential_data
from Exp_train import plot_loader_data_distribution

# Data paths
Train_Val_data_source = r'E:\Ian\PINNexample\exponential_trainval_data.npz'
Test_data_source = r'E:\Ian\PINNexample\exponential_test_data.npz'
data_normalize = True

# Load the dataset
print("Loading data...")
train_loader, val_loader, _, train_val_inputs_normalizer, train_val_targets_normalizer = load_exponential_data(
    filepath=Train_Val_data_source,
    batch_size=512,
    normalize=data_normalize,
    shuffle_train=True
)

test_loader, _, _, test_inputs_normalizer, test_targets_normalizer = load_exponential_data(
    filepath=Test_data_source,
    batch_size=512,
    normalize=data_normalize,
    shuffle_train=False
)

print("Data loaders created successfully!")

# Plot data distribution
results_figure_folder = './test_data_distributions'
plot_loader_data_distribution(train_loader, val_loader, test_loader, output_dir=results_figure_folder)

print("\nTest completed! Check the plots in:", results_figure_folder)
