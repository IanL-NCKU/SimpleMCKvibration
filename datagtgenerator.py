import numpy as np
import warnings
import os
import matplotlib.pyplot as plt

def analytical_solution(m, c, k, x0, v0, t):
    Wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    
    if zeta < 1:  # Underdamped case
        Wd = Wn * np.sqrt(1 - zeta**2)
        c1 = x0
        c2 = (v0 + zeta * Wn * x0) / Wd
        x_t = np.exp(-zeta * Wn * t) * (c1 * np.cos(Wd * t) + c2 * np.sin(Wd * t))
        
        v_t = np.exp(-Wn*t*zeta)*(Wd*c2*np.cos(Wd*t) - Wd*c1*np.sin(Wd*t)) \
            - Wn*zeta*np.exp(-Wn*t*zeta)*(c1*np.cos(Wd*t) + c2*np.sin(Wd*t))
        
        a_t = Wn**2*zeta**2*np.exp(-Wn*t*zeta)*(c1*np.cos(Wd*t) + c2*np.sin(Wd*t)) \
            - np.exp(-Wn*t*zeta)*(Wd**2*c1*np.cos(Wd*t) + Wd**2*c2*np.sin(Wd*t)) \
            - 2*Wn*zeta*np.exp(-Wn*t*zeta)*(Wd*c2*np.cos(Wd*t) - Wd*c1*np.sin(Wd*t))
        

    elif zeta == 1:  # Critically damped case
        c1 = x0
        c2 = v0 + Wn * x0
        x_t = np.exp(-Wn * t) * (c1 + c2 * t)
        
        v_t = c2*np.exp(-Wn*t) - Wn*np.exp(-Wn*t)*(c1 + c2*t)
        
        a_t = Wn**2*np.exp(-Wn*t)*(c1 + c2*t) - 2*Wn*c2*np.exp(-Wn*t)

    
    else:  # Overdamped case
        r1 = Wn * (-zeta + np.sqrt(zeta**2 - 1))
        r2 = Wn * (-zeta - np.sqrt(zeta**2 - 1))
        c1 = (v0 + (-r2) * x0) / (r1 - r2)
        c2 = (- v0 + r1 * x0) / (r1 - r2)
        x_t = np.exp(-zeta * Wn * t) * (c1 * np.exp( (r1-r2)/2 * t) + c2 * np.exp( (r2-r1)/2 * t))

        v_t = np.exp(-Wn*t*zeta)*(c1*np.exp(t*(r1/2 - r2/2))*(r1/2 - r2/2) - c2*np.exp(-t*(r1/2 - r2/2))*(r1/2 - r2/2)) \
            - Wn*zeta*np.exp(-Wn*t*zeta)*(c1*np.exp(t*(r1/2 - r2/2)) + c2*np.exp(-t*(r1/2 - r2/2)))
        
        a_t = np.exp(-Wn*t*zeta)*(c1*np.exp(t*(r1/2 - r2/2))*(r1/2 - r2/2)**2 + c2*np.exp(-t*(r1/2 - r2/2))*(r1/2 - r2/2)**2) \
            + Wn**2*zeta**2*np.exp(-Wn*t*zeta)*(c1*np.exp(t*(r1/2 - r2/2)) + c2*np.exp(-t*(r1/2 - r2/2))) \
            - 2*Wn*zeta*np.exp(-Wn*t*zeta)*(c1*np.exp(t*(r1/2 - r2/2))*(r1/2 - r2/2) - c2*np.exp(-t*(r1/2 - r2/2))*(r1/2 - r2/2))

    
    # res = m*a_t + c*v_t + k*x_t
    # print("The residual (should be close to 0): ", res)
    return x_t, v_t, a_t

def unittest_overflow():

    m = 10.0 ** np.random.uniform(-2, 2)
    zeta = 0.5 #10.0 ** np.random.uniform(-3, 1)
    k = 10.0 ** np.random.uniform(-1, 3)
    t = 0#10.0 ** np.random.uniform(-4, 3)
    x0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
    v0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
    c = 2 * zeta * np.sqrt(m * k)
    

    
    # sampling the data make an n by 9 +3 array
    # [m, zeta, k , t, x0, v0, x(t), v(t), a(t)]
    # and store it to a npy file

    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
    print("Zeta value: ", zeta)    
    print("The original x0 and v0: ", x0, v0)
    # print(analytical_solution(m, c, k, x0, v0, t))
    print( "The computed x_t, v_t, a_t: ", x_t, v_t, a_t)


def trycalculate_v_a():
    m = 10.0 ** np.random.uniform(-2, 2)
    zeta = 10.0 ** np.random.uniform(-3, 1)
    k = 10.0 ** np.random.uniform(-1, 3)
    t = 10.0 ** np.random.uniform(-4, 3)
    x0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
    v0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
    c = 2 * zeta * np.sqrt(m * k)
    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
    t_delta = 1e-6
    x_t_plus = analytical_solution(m, c, k, x0, v0, t + t_delta)[0]
    x_t_minus = analytical_solution(m, c, k, x0, v0, t - t_delta)[0]
    v_t_fd = (x_t_plus - x_t_minus) / (2 * t_delta)
    x_t_plus_plus = analytical_solution(m, c, k, x0, v0, t + 2*t_delta)[0]
    x_t_minus_minus = analytical_solution(m, c, k, x0, v0, t - 2*t_delta)[0]

    v_t_plus = (x_t_plus_plus - x_t) / (2*t_delta)
    v_t_minus = (x_t - x_t_minus_minus) / (2*t_delta)
    a_t_fd = (v_t_plus - v_t_minus) / (2*t_delta)
    # a_t_fd = (x_t_plus_plus - 2 * x_t + x_t_minus_minus) / (t_delta **2)


    print(f"Analytical: v_t={v_t}, a_t={a_t}")
    print(f"Finite Diff: v_t={v_t_fd}, a_t={a_t_fd}")
    print("Difference:"
          f" v_t_diff={abs(v_t - v_t_fd)}, a_t_diff={abs(a_t - a_t_fd)}")
    
    return None

def function_save_datadistribution(data, output_dir='data_distributions'):
    """
    Generates and saves histograms for each column of the provided data array.
    Each variable uses its own data-driven axis limits.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    column_names = ['m', 'zeta', 'k', 't', 'x0', 'v0', 'x_t', 'v_t', 'a_t']

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
            # Create 3 subfigures
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))

            # Separate positive, negative, zero
            positive_data = finite_data[finite_data > 0]
            negative_data = finite_data[finite_data < 0]
            zero_data = finite_data[finite_data == 0]

            total_count = len(finite_data)
            positive_pct = (len(positive_data) / total_count) * 100
            negative_pct = (len(negative_data) / total_count) * 100
            zero_pct = (len(zero_data) / total_count) * 100

            # Determine if we should use log scale
            use_log_scale = name in ['x_t', 'v_t', 'a_t']

            if use_log_scale:
                # ==== LOG SCALE with outlier filtering ====
                threshold = 1e-100  # Filter extremely small values
                
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
                    # Use percentiles to avoid extreme outliers
                    x_min = np.percentile(all_log_values, 0.5) - 0.5  # 0.5th percentile
                    x_max = np.percentile(all_log_values, 99.5) + 0.5  # 99.5th percentile
                    
                    # Print diagnostics
                    actual_min = np.min(all_log_values)
                    actual_max = np.max(all_log_values)
                    print(f"  {name}: log10 range = [{actual_min:.1f}, {actual_max:.1f}], "
                          f"plot range = [{x_min:.1f}, {x_max:.1f}]")
                else:
                    x_min, x_max = -1, 1

                # Count filtered outliers
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

            else:
                # ==== LINEAR SCALE for x0, v0 ====
                # Use percentiles to avoid extreme outliers
                all_values = np.concatenate([positive_data, negative_data]) if len(positive_data) > 0 and len(negative_data) > 0 else \
                            positive_data if len(positive_data) > 0 else negative_data

                if len(all_values) > 0:
                    x_min = np.percentile(all_values, 0.5)  # 0.5th percentile
                    x_max = np.percentile(all_values, 99.5)  # 99.5th percentile
                    
                    # Add 10% padding
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
            # All-positive data (log scale for m, zeta, k)
            plt.figure(figsize=(10, 6))
            
            threshold = 1e-100
            filtered_data = finite_data[finite_data > threshold]
            
            if len(filtered_data) > 0:
                log_data = np.log10(filtered_data)
                
                # Use percentiles
                x_min = np.percentile(log_data, 0.5) - 0.5
                x_max = np.percentile(log_data, 99.5) + 0.5
                
                plt.hist(log_data, bins=100, range=(x_min, x_max), 
                        density=False, color='blue', alpha=0.7, edgecolor='black')
                
                outliers = len(finite_data) - len(filtered_data)
                title = f'Distribution of log10({name})\n(100.0% positive, n={len(filtered_data)})'
                if outliers > 0:
                    title += f'\n({outliers} outliers < {threshold:.0e} excluded)'
                plt.title(title)
                
                print(f"  {name}: log10 range = [{np.min(log_data):.1f}, {np.max(log_data):.1f}], "
                      f"plot range = [{x_min:.1f}, {x_max:.1f}]")
            else:
                plt.text(0.5, 0.5, f'All values < {threshold:.0e}', 
                        ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'Distribution of log10({name})\n(All outliers)')
                x_min, x_max = -1, 1
            
            plt.xlabel(f'log10({name})')
            plt.ylabel('Count')
            plt.xlim(x_min, x_max)
            plt.grid(axis='y', alpha=0.75)

            filename = os.path.join(output_dir, f'distribution_{name}.png')
            plt.savefig(filename, dpi=100)
            plt.close()

            print(f"  Saved {name}: 100.0% positive, n={len(finite_data)}")

    print("All distribution plots have been saved.")




def main():
    n = 400000
    
    # np.random.seed(42)
    
    data = np.zeros((n, 9))
    
    print(f"Generating {n} valid samples...")
    error_count = 0
    i = 0
    j = 0
    kk = 0
    attempts = 0
    while i < n:
        attempts += 1
        
        # Generate a single random sample
        m = 10.0 ** np.random.uniform(-3, 3)
        if np.random.rand() < 0.05:
            zeta = 1.0
        else:
            zeta = 10.0 ** np.random.uniform(-3, 2)
        k = 10.0 ** np.random.uniform(-3, 3)
        t_type = np.random.rand()
        if t_type < 0.2:
            t = 10.0 ** np.random.uniform(-3, 3)
        else:
            t = np.random.uniform(1e-3, 1000)
        x0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
        v0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
        
        c = 2 * zeta * np.sqrt(m * k)
        
        is_valid = True
        x_t, v_t, a_t = np.nan, np.nan, np.nan

        # Catch warnings and errors
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")  # Catch all warnings
            warnings.filterwarnings('ignore', category=RuntimeWarning)  # Don't print them

            # Use errstate to catch numerical errors like overflow
            with np.errstate(all='raise'):  # Raise exceptions instead of warnings
                try:
                    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)

                    # Additional check: if results are exactly zero (likely underflow)
                    # and time is large, this indicates underflow that may not trigger warning
                    # has_suspicious_zeros = (x_t == 0 and v_t == 0 and a_t == 0) and t > 100

                    # Debug: print if we catch warnings
                    # if len(w) > 0 and attempts % 1000 == 0:
                    #     print(f"Caught {len(w)} warnings at attempt {attempts}")

                    # Check if any warnings were raised or if results are invalid
                    if len(w) > 0  or \
                        any(np.isnan([x_t, v_t, a_t])) or \
                        any(np.isinf([x_t, v_t, a_t])) or \
                        any(np.abs([x_t, v_t, a_t]) > 1e4):

                        is_valid = False

                    elif any(np.abs([x_t, v_t, a_t]) < 1e-6):
                        # only 1% of the data with very small values
                        j += 1
                        if np.random.rand() > 0.001:
                            is_valid = False

                        # elif any(np.abs([x_t, v_t, a_t]) < 1e-12):
                        #     # the one that smaller will be turn to 0
                        #     x_t = 0.0 if abs(x_t) < 1e-12 else x_t
                        #     v_t = 0.0 if abs(v_t) < 1e-12 else v_t
                        #     a_t = 0.0 if abs(a_t) < 1e-12 else a_t

                        else:
                            kk += 1

                except Exception:
                    is_valid = False

        if is_valid:
            if (i + 1) % 10000 == 0:
                print(f"Progress: {i + 1}/{n} samples generated")
            
            data[i] = [m, zeta, k, t, x0, v0, x_t, v_t, a_t]
            i += 1  # Increment only when a valid sample is generated
            # print(i)
        # If not valid, the loop continues and regenerates a new sample for the same index i

    # check is there any data <1e-9 but not zero
    if np.any((np.abs(data[:, 6:]) < 1e-9) & (data[:, 6:] != 0)):
        # find those rows
        rows = np.where((np.abs(data[:, 6:]) < 1e-9) & (data[:, 6:] != 0))
        # extract those data[rows[0], :] points
        problematic_data = data[rows[0], :]

        print("Warning: There are data points with absolute values < 1e-9 but not zero.")

    print(f"\nTotal attempts to generate {n} valid samples: {attempts}")
    
    # Save as npz file
    np.savez('train_val_vibration_data.npz', data=data)
    print(f"\nData generation complete! Saved {n} samples to 'test_vibration_data.npz'")
    print(f"Data shape: {data.shape}")
    print(f"Columns: [m, zeta, k, t, x0, v0, x(t), v(t), a(t)]")
    print(data[:5, :])  # Print first 5 samples
    # Generate and save data distribution plots
    function_save_datadistribution(data)
    
    for i in range(n):
        if i % 10000 == 0:
            print(f"Progress: {i}/{n} samples finished checking")
        
        m = data[i, 0]
        zeta = data[i, 1]
        k = data[i, 2]
        t = data[i, 3]
        x0 = data[i, 4]
        v0 = data[i, 5]
        
        c = 2 * zeta * np.sqrt(m * k)
       # Catch warnings and errors
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Use errstate to catch numerical errors
            with np.errstate(all='warn'):
                try:
                    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
                    
                    # Check if any warnings were raised
                    if len(w) > 0 or np.isnan(x_t) or np.isnan(v_t) or np.isnan(a_t) or \
                       np.isinf(x_t) or np.isinf(v_t) or np.isinf(a_t):
                        error_count += 1
                        if error_count <= 10:  # Only print first 10 problematic cases
                            print(f"\n{'='*60}")
                            print(f"Problem detected at sample {i}:")
                            print(f"  m = {m:.6e}")
                            print(f"  zeta = {zeta:.6e}")
                            print(f"  k = {k:.6e}")
                            print(f"  t = {t:.6e}")
                            print(f"  x0 = {x0:.6e}")
                            print(f"  v0 = {v0:.6e}")
                            print(f"  c = {c:.6e}")
                            print(f"  Wn = {np.sqrt(k/m):.6e}")
                            print(f"  Results: x_t={x_t:.6e}, v_t={v_t:.6e}, a_t={a_t:.6e}")
                            if len(w) > 0:
                                for warning in w:
                                    print(f"  Warning: {warning.message}")
                            print(f"{'='*60}")
                        
                        # Replace inf/nan with 0 or skip this sample
                        # Option 1: Set to NaN
                        x_t = np.nan if (np.isnan(x_t) or np.isinf(x_t)) else x_t
                        v_t = np.nan if (np.isnan(v_t) or np.isinf(v_t)) else v_t
                        a_t = np.nan if (np.isnan(a_t) or np.isinf(a_t)) else a_t
                        
                except Exception as e:
                    error_count += 1
                    if error_count <= 10:
                        print(f"\n{'='*60}")
                        print(f"Exception at sample {i}: {e}")
                        print(f"  m={m:.6e}, zeta={zeta:.6e}, k={k:.6e}, t={t:.6e}, x0={x0:.6e}, v0={v0:.6e}")
                        print(f"{'='*60}")
                    x_t, v_t, a_t = np.nan, np.nan, np.nan
    print("Total j values below 1e-6: ", j)
    print("Total kk values below 1e-5 accepted: ", kk)


if __name__ == "__main__":
    main()
    # unittest_overflow()
    # trycalculate_v_a()