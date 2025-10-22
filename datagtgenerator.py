import numpy as np
import warnings

def analytical_solution(m, c, k, x0, v0, t):
    Wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    
    if zeta < 1:  # Underdamped case
        Wd = Wn * np.sqrt(1 - zeta**2)
        c1 = x0
        c2 = (v0 + zeta * Wn * x0) / Wd
        x_t = np.exp(-zeta * Wn * t) * (c1 * np.cos(Wd * t) + c2 * np.sin(Wd * t))
        v_t = np.exp(-zeta * Wn * t) * (c1 * Wd * np.sin(Wd * t) + c2 * Wd * np.cos(Wd * t)) \
                + (-zeta * Wn) * x_t
        a_t = np.exp(-zeta * Wn * t) * (c1 * Wd**2 * np.cos(Wd * t) - c2 * Wd**2 * np.sin(Wd * t)) \
                + (-zeta * Wn) * np.exp(-zeta * Wn * t) * (c1 * Wd * np.sin(Wd * t) + c2 * Wd * np.cos(Wd * t)) \
                + (-zeta * Wn) * v_t
                
    elif zeta == 1:  # Critically damped case
        c1 = x0
        c2 = v0 + Wn * x0
        x_t = np.exp(-Wn * t) * (c1 + c2 * t)
        v_t = c2 * np.exp(-Wn * t) + (-Wn) * x_t
        a_t = (-Wn) * v_t + (-Wn) * np.exp(-Wn * t) * c2
    
    else:  # Overdamped case
        r1 = Wn * (-zeta + np.sqrt(zeta**2 - 1))
        r2 = Wn * (-zeta - np.sqrt(zeta**2 - 1))
        c1 = (v0 + (-r2) * x0) / (r1 - r2)
        c2 = (- v0 + r1 * x0) / (r1 - r2)
        x_t = np.exp(-zeta * Wn * t) * (c1 * np.exp( r1* Wn * t) + c2 * np.exp( r2 * Wn * t))
        v_t = np.exp(-zeta * Wn * t) * (c1 * (r1) * np.exp(np.sqrt(zeta**2 - 1) * Wn * t) + c2 * (r2) * np.exp(-np.sqrt(zeta**2 - 1) * Wn * t))
        a_t = np.exp(-zeta * Wn * t) * (c1 * (r1**2) * np.exp(np.sqrt(zeta**2 - 1) * Wn * t) + c2 * (r2**2) * np.exp(-np.sqrt(zeta**2 - 1) * Wn * t))
    
    return x_t, v_t, a_t

def unittest_overflow():

    m = 5.130552e-01
    zeta = 1.2e+00
    k = 1.681688e+01
    t = 2
    x0 = 9.305809e+00
    v0 = -6.974494e+01
    c = 2 * zeta * np.sqrt(m * k)
    print(c)
    # sampling the data make an n by 9 +3 array
    # [m, zeta, k , t, x0, v0, x(t), v(t), a(t)]
    # and store it to a npy file

    x_t, v_t, a_t = analytical_solution(m, c, k, x0, v0, t)

    # print(analytical_solution(m, c, k, x0, v0, t))
    print(x_t, v_t, a_t)



def main():
    n = 100000
    
    np.random.seed(42)
    
    data = np.zeros((n, 9))
    
    print(f"Generating {n} valid samples...")
    
    i = 0
    attempts = 0
    while i < n:
        attempts += 1
        
        # Generate a single random sample
        m = 10.0 ** np.random.uniform(-2, 2)
        zeta = 10.0 ** np.random.uniform(-3, 1)
        k = 10.0 ** np.random.uniform(-1, 3)
        t = 10.0 ** np.random.uniform(-3, 3)
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
    np.savez('vibration_data.npz', data=data)
    print(f"\nData generation complete! Saved {n} samples to 'vibration_data.npz'")
    print(f"Data shape: {data.shape}")
    print(f"Columns: [m, zeta, k, t, x0, v0, x(t), v(t), a(t)]")


if __name__ == "__main__":
    main()
    # unittest_overflow()