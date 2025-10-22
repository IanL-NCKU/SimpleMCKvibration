import numpy as np
# MCK free vibration model 
# underdamped case, critical case, overdamped case
def analytical_solution(m, c, k, x0, v0, t):
    Wn = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(m * k))
    print("zeta:", zeta)
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



def main():
    m = 1 # sampling range 1e-2 to 1e2
    zeta = 0.2 # sampling range 1e-3 to 10
    k = 10 # sampling range 1e-1 to 1e3
    c = 2 * zeta * np.sqrt(m * k) 
    x0 = 6 # sampling range -1e2 to 1e2
    v0 = -5 # sampling range -10 to 10
    t= 1
    # sampling the data make an n by 9 +3 array
    # [m, zeta, k , t, x0, v0, x(t), v(t), a(t)]
    # and store it to a npy file
    # n = 100000
    

    print(analytical_solution(m, c, k, x0, v0, t))


if __name__ == "__main__":
    main()


