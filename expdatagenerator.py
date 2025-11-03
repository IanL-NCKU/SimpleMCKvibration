import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

def analytical_solution_exp(a, b, t):
    """
    Analytical solution for exponential function:
    x(t) = b * exp(a * t)
    v(t) = dx/dt = b * a * exp(a * t)
    a(t) = dv/dt = b * a^2 * exp(a * t)

    Args:
        a: exponential rate parameter, range [-10, 10]
        b: coefficient parameter, range [-1000, 1000]
        t: time, range [1e-3, 10]

    Returns:
        x_t: displacement at time t
        v_t: velocity at time t (first derivative)
        acc_t: acceleration at time t (second derivative)
    """
    # Calculate the exponent
    exponent = a * t

    # Check for potential overflow/underflow
    # For float32: overflow at ~88.73, underflow at ~-104
    # For float64: overflow at ~709.79, underflow at ~-746
    # We'll use conservative limits for safety
    MAX_EXP = 50.0  # Conservative limit
    MIN_EXP = -50.0

    if exponent > MAX_EXP or exponent < MIN_EXP:
        # This will trigger overflow/underflow, return NaN
        return np.nan, np.nan, np.nan

    # Calculate exp(a*t)
    exp_at = np.exp(exponent)

    # Calculate position, velocity, acceleration
    x_t = b * exp_at
    v_t = b * a * exp_at
    acc_t = b * a * a * exp_at

    return x_t, v_t, acc_t


def function_save_datadistribution(data, output_dir='exp_data_distributions'):
    """
    Generates and saves histograms for each column of the provided data array.
    Each variable uses its own data-driven axis limits.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    column_names = ['a', 'b', 't', 'x_t', 'v_t', 'a_t']

    print(f"\nGenerating and saving data distribution plots to '{output_dir}'...")

    for i, name in enumerate(column_names):
        # Filter out non-finite values
        column_data = data[:, i]
        finite_data = column_data[np.isfinite(column_data)]

        if finite_data.size == 0:
            print(f"  Skipping '{name}' as it contains no finite data.")
            continue

        # Special handling for time 't'
        if name == 't':
            # Real-scale plot
            plt.figure(figsize=(10, 6))
            plt.hist(finite_data, bins=100, density=False, color='blue', alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {name} (Real Scale)\n(100.0% positive, n={len(finite_data)})')
            plt.xlabel('Time Value (Real Scale)')
            plt.ylabel('Count')
            plt.xlim(left=0)
            plt.grid(axis='y', alpha=0.75)
            filename_real = os.path.join(output_dir, f'distribution_{name}_real.png')
            plt.savefig(filename_real, dpi=100)
            plt.close()

            # Log-scale plot
            plt.figure(figsize=(10, 6))
            plt.hist(finite_data, bins=100, density=False, color='orange', alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {name} (Log Scale)\n(100.0% positive, n={len(finite_data)})')
            plt.xlabel('Time Value (Log Scale)')
            plt.ylabel('Count')
            plt.xscale('log')
            plt.grid(axis='y', alpha=0.75)
            filename_log = os.path.join(output_dir, f'distribution_{name}_log.png')
            plt.savefig(filename_log, dpi=100)
            plt.close()

            print(f"  Saved {name}: 100.0% positive (real and log scale), n={len(finite_data)}")
            continue

        # Check if data contains negative values
        has_negative = np.any(finite_data < 0)

        if has_negative:
            # Separate positive, negative, zero
            positive_data = finite_data[finite_data > 0]
            negative_data = finite_data[finite_data < 0]
            zero_data = finite_data[finite_data == 0]

            total_count = len(finite_data)
            positive_pct = (len(positive_data) / total_count) * 100
            negative_pct = (len(negative_data) / total_count) * 100
            zero_pct = (len(zero_data) / total_count) * 100

            # Determine if we should create both log and linear scale plots
            create_both_scales = name in ['x_t', 'v_t', 'a_t']

            if create_both_scales:
                # ==== LOG SCALE VERSION ====
                fig, axes = plt.subplots(3, 1, figsize=(10, 15))
                threshold = 1e-100

                # Filter and collect log values
                all_log_values = []
                if len(positive_data) > 0:
                    pos_filtered = positive_data[positive_data > threshold]
                    if len(pos_filtered) > 0:
                        all_log_values.extend(np.log10(pos_filtered))

                if len(negative_data) > 0:
                    neg_filtered = np.abs(negative_data)
                    neg_filtered = neg_filtered[neg_filtered > threshold]
                    if len(neg_filtered) > 0:
                        all_log_values.extend(np.log10(neg_filtered))

                if len(all_log_values) > 0:
                    x_min = np.percentile(all_log_values, 0.5) - 0.5
                    x_max = np.percentile(all_log_values, 99.5) + 0.5

                    actual_min = np.min(all_log_values)
                    actual_max = np.max(all_log_values)
                    print(f"  {name}: log10 range = [{actual_min:.1f}, {actual_max:.1f}], "
                          f"plot range = [{x_min:.1f}, {x_max:.1f}]")
                else:
                    x_min, x_max = -1, 1

                positive_outliers = np.sum(positive_data <= threshold) if len(positive_data) > 0 else 0
                negative_outliers = np.sum(np.abs(negative_data) <= threshold) if len(negative_data) > 0 else 0
                total_outliers = positive_outliers + negative_outliers

                # Subplot 1: Positive values (log scale)
                if len(positive_data) > 0:
                    pos_filtered = positive_data[positive_data > threshold]
                    if len(pos_filtered) > 0:
                        log_positive = np.log10(pos_filtered)
                        axes[0].hist(log_positive, bins=100, range=(x_min, x_max),
                                    density=False, color='green', alpha=0.7, edgecolor='black')
                        title = f'Positive {name}\n({positive_pct:.1f}% of data, n={len(positive_data)})'
                        if positive_outliers > 0:
                            title += f'\n({positive_outliers} outliers < {threshold:.0e} excluded)'
                        axes[0].set_title(title)
                    else:
                        axes[0].text(0.5, 0.5, f'All values < {threshold:.0e}',
                                   ha='center', va='center', transform=axes[0].transAxes)
                        axes[0].set_title(f'Positive {name}\n({positive_pct:.1f}% of data, all outliers)')
                else:
                    axes[0].text(0.5, 0.5, 'No positive values',
                               ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title(f'Positive {name}\n(0.0% of data, n=0)')

                axes[0].set_xlabel(f'log10({name})')
                axes[0].set_ylabel('Count')
                axes[0].set_xlim(x_min, x_max)
                axes[0].grid(axis='y', alpha=0.75)

                # Subplot 2: Negative values (log scale)
                if len(negative_data) > 0:
                    neg_filtered = np.abs(negative_data)
                    neg_filtered = neg_filtered[neg_filtered > threshold]
                    if len(neg_filtered) > 0:
                        log_negative = np.log10(neg_filtered)
                        axes[1].hist(log_negative, bins=100, range=(x_min, x_max),
                                    density=False, color='red', alpha=0.7, edgecolor='black')
                        title = f'Negative {name} (abs)\n({negative_pct:.1f}% of data, n={len(negative_data)})'
                        if negative_outliers > 0:
                            title += f'\n({negative_outliers} outliers < {threshold:.0e} excluded)'
                        axes[1].set_title(title)
                    else:
                        axes[1].text(0.5, 0.5, f'All values < {threshold:.0e}',
                                   ha='center', va='center', transform=axes[1].transAxes)
                        axes[1].set_title(f'Negative {name}\n({negative_pct:.1f}% of data, all outliers)')
                else:
                    axes[1].text(0.5, 0.5, 'No negative values',
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f'Negative {name}\n(0.0% of data, n=0)')

                axes[1].set_xlabel(f'log10(|{name}|)')
                axes[1].set_ylabel('Count')
                axes[1].set_xlim(x_min, x_max)
                axes[1].grid(axis='y', alpha=0.75)

                # Subplot 3: Combined (log scale)
                abs_data = np.abs(finite_data)
                abs_filtered = abs_data[abs_data > threshold]
                if len(abs_filtered) > 0:
                    log_combined = np.log10(abs_filtered)
                    axes[2].hist(log_combined, bins=100, range=(x_min, x_max),
                                density=False, color='blue', alpha=0.7, edgecolor='black')
                    title = f'Combined |{name}|\n(Pos: {positive_pct:.1f}%, Neg: {negative_pct:.1f}%, Zero: {zero_pct:.1f}%, n={len(abs_filtered)})'
                    if total_outliers > 0:
                        title += f'\n({total_outliers} outliers < {threshold:.0e} excluded)'
                    axes[2].set_title(title)
                else:
                    axes[2].text(0.5, 0.5, 'All values are outliers or zero',
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title(f'Combined |{name}| (n=0)')

                axes[2].set_xlabel(f'log10(|{name}|)')
                axes[2].set_ylabel('Count')
                axes[2].set_xlim(x_min, x_max)
                axes[2].grid(axis='y', alpha=0.75)

                plt.tight_layout()
                filename_log = os.path.join(output_dir, f'distribution_{name}_log.png')
                plt.savefig(filename_log, dpi=100)
                plt.close()

                # ==== LINEAR SCALE VERSION ====
                fig, axes = plt.subplots(3, 1, figsize=(10, 15))

                all_values = np.concatenate([positive_data, negative_data]) if len(positive_data) > 0 and len(negative_data) > 0 else \
                            positive_data if len(positive_data) > 0 else negative_data

                if len(all_values) > 0:
                    x_min = np.percentile(all_values, 0.5)
                    x_max = np.percentile(all_values, 99.5)

                    range_width = x_max - x_min
                    x_min -= range_width * 0.1
                    x_max += range_width * 0.1
                else:
                    x_min, x_max = -1, 1

                # Subplot 1: Positive values (linear scale)
                if len(positive_data) > 0:
                    axes[0].hist(positive_data, bins=100, range=(x_min, x_max),
                                density=False, color='green', alpha=0.7, edgecolor='black')
                    axes[0].set_title(f'Positive {name}\n({positive_pct:.1f}% of data, n={len(positive_data)})')
                else:
                    axes[0].text(0.5, 0.5, 'No positive values',
                               ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title(f'Positive {name}\n(0.0% of data, n=0)')

                axes[0].set_xlabel(f'{name}')
                axes[0].set_ylabel('Count')
                axes[0].set_xlim(x_min, x_max)
                axes[0].grid(axis='y', alpha=0.75)

                # Subplot 2: Negative values (linear scale)
                if len(negative_data) > 0:
                    axes[1].hist(negative_data, bins=100, range=(x_min, x_max),
                                density=False, color='red', alpha=0.7, edgecolor='black')
                    axes[1].set_title(f'Negative {name}\n({negative_pct:.1f}% of data, n={len(negative_data)})')
                else:
                    axes[1].text(0.5, 0.5, 'No negative values',
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f'Negative {name}\n(0.0% of data, n=0)')

                axes[1].set_xlabel(f'{name}')
                axes[1].set_ylabel('Count')
                axes[1].set_xlim(x_min, x_max)
                axes[1].grid(axis='y', alpha=0.75)

                # Subplot 3: Combined (linear scale)
                if len(finite_data) > 0:
                    axes[2].hist(finite_data, bins=100, range=(x_min, x_max),
                                density=False, color='blue', alpha=0.7, edgecolor='black')
                    axes[2].set_title(f'Combined {name}\n(Pos: {positive_pct:.1f}%, Neg: {negative_pct:.1f}%, Zero: {zero_pct:.1f}%, n={len(finite_data)})')
                else:
                    axes[2].text(0.5, 0.5, 'All values are zero',
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title(f'Combined {name} (n=0)')

                axes[2].set_xlabel(f'{name}')
                axes[2].set_ylabel('Count')
                axes[2].set_xlim(x_min, x_max)
                axes[2].grid(axis='y', alpha=0.75)

                plt.tight_layout()
                filename_real = os.path.join(output_dir, f'distribution_{name}_real.png')
                plt.savefig(filename_real, dpi=100)
                plt.close()

                print(f"  Saved {name}: Pos={positive_pct:.1f}%, Neg={negative_pct:.1f}%, Zero={zero_pct:.1f}% (log and real scale)")
            else:
                # Linear scale only for 'a' and 'b'
                fig, axes = plt.subplots(3, 1, figsize=(10, 15))

                all_values = np.concatenate([positive_data, negative_data]) if len(positive_data) > 0 and len(negative_data) > 0 else \
                            positive_data if len(positive_data) > 0 else negative_data

                if len(all_values) > 0:
                    x_min = np.percentile(all_values, 0.5)
                    x_max = np.percentile(all_values, 99.5)

                    range_width = x_max - x_min
                    x_min -= range_width * 0.1
                    x_max += range_width * 0.1

                    print(f"  {name}: value range = [{np.min(all_values):.2f}, {np.max(all_values):.2f}], "
                          f"plot range = [{x_min:.2f}, {x_max:.2f}]")
                else:
                    x_min, x_max = -1, 1

                # Subplot 1: Positive values
                if len(positive_data) > 0:
                    axes[0].hist(positive_data, bins=100, range=(x_min, x_max),
                                density=False, color='green', alpha=0.7, edgecolor='black')
                    axes[0].set_title(f'Positive {name}\n({positive_pct:.1f}% of data, n={len(positive_data)})')
                else:
                    axes[0].text(0.5, 0.5, 'No positive values',
                               ha='center', va='center', transform=axes[0].transAxes)
                    axes[0].set_title(f'Positive {name}\n(0.0% of data, n=0)')

                axes[0].set_xlabel(f'{name}')
                axes[0].set_ylabel('Count')
                axes[0].set_xlim(x_min, x_max)
                axes[0].grid(axis='y', alpha=0.75)

                # Subplot 2: Negative values
                if len(negative_data) > 0:
                    axes[1].hist(negative_data, bins=100, range=(x_min, x_max),
                                density=False, color='red', alpha=0.7, edgecolor='black')
                    axes[1].set_title(f'Negative {name}\n({negative_pct:.1f}% of data, n={len(negative_data)})')
                else:
                    axes[1].text(0.5, 0.5, 'No negative values',
                               ha='center', va='center', transform=axes[1].transAxes)
                    axes[1].set_title(f'Negative {name}\n(0.0% of data, n=0)')

                axes[1].set_xlabel(f'{name}')
                axes[1].set_ylabel('Count')
                axes[1].set_xlim(x_min, x_max)
                axes[1].grid(axis='y', alpha=0.75)

                # Subplot 3: Combined
                if len(finite_data) > 0:
                    axes[2].hist(finite_data, bins=100, range=(x_min, x_max),
                                density=False, color='blue', alpha=0.7, edgecolor='black')
                    axes[2].set_title(f'Combined {name}\n(Pos: {positive_pct:.1f}%, Neg: {negative_pct:.1f}%, Zero: {zero_pct:.1f}%, n={len(finite_data)})')
                else:
                    axes[2].text(0.5, 0.5, 'All values are zero',
                               ha='center', va='center', transform=axes[2].transAxes)
                    axes[2].set_title(f'Combined {name} (n=0)')

                axes[2].set_xlabel(f'{name}')
                axes[2].set_ylabel('Count')
                axes[2].set_xlim(x_min, x_max)
                axes[2].grid(axis='y', alpha=0.75)

                plt.tight_layout()
                filename = os.path.join(output_dir, f'distribution_{name}.png')
                plt.savefig(filename, dpi=100)
                plt.close()

                print(f"  Saved {name}: Pos={positive_pct:.1f}%, Neg={negative_pct:.1f}%, Zero={zero_pct:.1f}%")

        else:
            # All-positive data (shouldn't happen for a, b but might for t)
            plt.figure(figsize=(10, 6))

            plt.hist(finite_data, bins=100, density=False, color='blue', alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {name}\n(100.0% positive, n={len(finite_data)})')
            plt.xlabel(f'{name}')
            plt.ylabel('Count')
            plt.grid(axis='y', alpha=0.75)

            filename = os.path.join(output_dir, f'distribution_{name}.png')
            plt.savefig(filename, dpi=100)
            plt.close()

            print(f"  Saved {name}: 100.0% positive, n={len(finite_data)}")

    print("All distribution plots have been saved.")


def main():
    """
    Generate exponential data with the following specifications:
    - x(t) = b * exp(a * t)
    - v(t) = b * a * exp(a * t)
    - a(t) = b * a^2 * exp(a * t)

    Parameter ranges:
    - b: [-1000.000, 1000.000]
    - a: [-10.000, 10.000]
    - t: [1e-3, 10.000]

    Saves to NPZ file with columns: [a, b, t, x_t, v_t, a_t]
    """
    n = 100000  # Target number of samples
    output_filename = 'exponential_test_data.npz'
    output_folder = 'exp_test_data_distributions'
    data = np.zeros((n, 6))

    print(f"Generating {n} valid exponential samples...")
    print(f"Parameter ranges:")
    print(f"  a: [-10.000, 10.000]")
    print(f"  b: [-1000.000, 1000.000]")
    print(f"  t: [1e-3, 10.000]")
    print()

    i = 0
    attempts = 0
    rejected_overflow = 0
    rejected_large = 0
    rejected_small = 0

    while i < n:
        attempts += 1

        # Generate random parameters
        # a: uniform in [-10, 10]
        a = np.random.uniform(-10.0, 10.0)

        # b: uniform in [-1000, 1000]
        b = np.random.uniform(-1000.0, 1000.0)

        # t: uniform in [1e-3, 10] on log scale for better distribution
        # 20% log-uniform, 80% linear uniform
        if np.random.rand() < 0.2:
            t = 10.0 ** np.random.uniform(-3, 1)  # log-uniform from 1e-3 to 10
        else:
            t = np.random.uniform(1e-3, 10.0)  # linear uniform

        is_valid = True
        x_t, v_t, acc_t = np.nan, np.nan, np.nan

        try:
            # Calculate analytical solution
            x_t, v_t, acc_t = analytical_solution_exp(a, b, t)

            # Check for invalid values (NaN or Inf)
            if any(np.isnan([x_t, v_t, acc_t])) or any(np.isinf([x_t, v_t, acc_t])):
                is_valid = False
                rejected_overflow += 1
            # Reject if any value is too large (> 1e3)
            elif any(np.abs([x_t, v_t, acc_t]) > 1e4):
                is_valid = False
                rejected_large += 1
            # Accept only 1% of very small values (< 1e-2)
            elif any(np.abs([x_t, v_t, acc_t]) < 1e-4):
                if np.random.rand() > 0.01:
                    is_valid = False
                    rejected_small += 1

        except Exception:
            is_valid = False
            rejected_overflow += 1

        if is_valid:
            if (i + 1) % 10000 == 0:
                print(f"Progress: {i + 1}/{n} samples generated (attempts: {attempts})")

            # Store data as [a, b, t, x_t, v_t, a_t]
            data[i] = [a, b, t, x_t, v_t, acc_t]
            i += 1

    print(f"\nTotal attempts: {attempts}")
    print(f"  Rejected (overflow/underflow): {rejected_overflow}")
    print(f"  Rejected (too large > 1e3): {rejected_large}")
    print(f"  Rejected (too small < 1e-2): {rejected_small}")
    print(f"  Acceptance rate: {n/attempts*100:.2f}%")

    # Save as npz file
    
    np.savez(output_filename, data=data)
    print(f"\nData generation complete! Saved {n} samples to '{output_filename}'")
    print(f"Data shape: {data.shape}")
    print(f"Columns: [a, b, t, x_t, v_t, a_t]")
    print("\nFirst 5 samples:")
    print(data[:5, :])

    # Print statistics
    print("\n" + "="*60)
    print("Data Statistics:")
    print("="*60)
    for i, name in enumerate(['a', 'b', 't', 'x_t', 'v_t', 'a_t']):
        col_data = data[:, i]
        print(f"{name:5s}: min={np.min(col_data):12.6e}, max={np.max(col_data):12.6e}, "
              f"mean={np.mean(col_data):12.6e}, std={np.std(col_data):12.6e}")
    print("="*60)

    # Generate and save data distribution plots
    function_save_datadistribution(data, output_dir=output_folder)


if __name__ == "__main__":
    main()
