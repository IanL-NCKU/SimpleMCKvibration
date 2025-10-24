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

    Args:
        data (np.ndarray): The 2D numpy array containing the data.
        output_dir (str): The directory where the plots will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    column_names = ['m', 'zeta', 'k', 't', 'x0', 'v0', 'x_t', 'v_t', 'a_t']
    
    print(f"\nGenerating and saving data distribution plots to '{output_dir}'...")

    for i, name in enumerate(column_names):
        plt.figure(figsize=(10, 6))
        
        # Filter out non-finite values which can't be plotted
        column_data = data[:, i]
        finite_data = column_data[np.isfinite(column_data)]
        
        if finite_data.size == 0:
            print(f"  Skipping '{name}' as it contains no finite data.")
            plt.close()
            continue

        plt.hist(finite_data, bins=50, density=True, color='blue', alpha=0.7)
        plt.title(f'Distribution of {name}')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.grid(axis='y', alpha=0.75)
        
        # Use a log scale for the x-axis if data is all positive
        if np.all(finite_data > 0):
            plt.xscale('log')
            plt.xlabel('Value (log scale)')
        else:
            # For data with negative values, use a symmetric log scale
            # This is useful for visualizing data that spans several orders of magnitude
            # and includes both positive and negative values.
            plt.xscale('symlog')
            plt.xlabel('Value (symlog scale)')


        filename = os.path.join(output_dir, f'distribution_{name}.png')
        plt.savefig(filename)
        plt.close()

    print("All distribution plots have been saved.")

def main():
    n = 100000
    
    np.random.seed(42)
    
    data = np.zeros((n, 9))
    
    print(f"Generating {n} valid samples...")
    error_count = 0
    i = 0
    attempts = 0
    while i < n:
        attempts += 1
        
        # Generate a single random sample
        m = 10.0 ** np.random.uniform(-2, 2)
        if np.random.rand() < 0.05:
            zeta = 1.0
        else:
            zeta = 10.0 ** np.random.uniform(-3, 1)
        k = 10.0 ** np.random.uniform(-1, 3)
        t = 10.0 ** np.random.uniform(-4, 3)
        x0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
        v0 = np.random.choice([-1.0, 1.0]) * np.random.uniform(0, 100)
        
        c = 2 * zeta * np.sqrt(m * k)
        
        is_valid = True
        x_t, v_t, a_t = np.nan, np.nan, np.nan

        # Catch warnings and errors
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Use errstate to catch numerical errors like overflow
            with np.errstate(all='warn'):
                try:
                    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)
                    
                    # Check if any warnings were raised or if results are invalid
                    if len(w) > 0 or any(np.isnan([x_t, v_t, a_t])) or \
                        any(np.isinf([x_t, v_t, a_t])) or \
                        any(np.abs([x_t, v_t, a_t]) > 1e5):
                        is_valid = False
                        
                except Exception:
                    is_valid = False

        if is_valid:
            if (i + 1) % 10000 == 0:
                print(f"Progress: {i + 1}/{n} samples generated")
            
            data[i] = [m, zeta, k, t, x0, v0, x_t, v_t, a_t]
            i += 1  # Increment only when a valid sample is generated
        # If not valid, the loop continues and regenerates a new sample for the same index i

    print(f"\nTotal attempts to generate {n} valid samples: {attempts}")
    
    # Save as npz file
    np.savez('test_vibration_data.npz', data=data)
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



if __name__ == "__main__":
    main()
    # unittest_overflow()
    # trycalculate_v_a()