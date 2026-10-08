import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

for i in range(-90, 91, 5):
    # print(i)
    ramanShift = [] #cm^-1
    counts = []
    with open(f"Raman/C6D6/CSV/Polarization - {i}.csv", newline = '') as f:
        for line in f:
            if line.strip() == "WL,Power":
                break
        
        reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
        for row in reader:
            ramanShift.append(float(row['WL']))
            counts.append(float(row["Power"]))
    A = dict()
    m = dict()
    gamma = dict()
    a2 = dict()
    delta = dict()
    g2 = dict()
    B = dict()
    C = dict()
    with open(f"Raman/C6D6/CSV/Lorentz Fit Guess - 1552.2 Peak.csv", newline = '') as f:
        reader = csv.DictReader(f)
        for row in reader:
            A[int(row['angle'])] = float(row['amp'])
            m[int(row['angle'])] = float(row['mu'])
            gamma[int(row['angle'])] = float(row['gamma'])
            a2[int(row['angle'])] = float(row['a2'])
            delta[int(row['angle'])] = float(row['delta'])
            g2[int(row['angle'])] = float(row['g2'])
            B[int(row['angle'])] = float(row['B'])
            C[int(row['angle'])] = float(row['C'])

    ramanShift = np.array(ramanShift)
    counts = np.array(counts)
    
    if abs(m[i] - 1552) < abs((m[i] + delta[i] - 1552)) and gamma[i] > 2:
        amp = A[i]
        mu = m[i]
        g = gamma[i]
    else:
        amp = a2[i]
        mu = m[i] + delta[i]
        g = g2[i]  
        
    low, high = mu - 4*g, mu + 4*g

    def lorentz_fit_lin_bg(x, A, m, gamma, B, C):
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)
    p0 = [amp,
          mu,
          g,
          B[i],
          C[i]]
    
    lower_bounds = (
        [0] + 
        [0] + 
        [0] + 
        [-np.inf] + 
        [-np.inf]
    )

    upper_bounds = (
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf]
    )
    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True, bounds = (lower_bounds, upper_bounds), maxfev = 30000)

    N = len(x)
    p = len(popt)

    residuals = y - lorentz_fit_lin_bg(x, *popt)
    chi2 = np.sum((residuals / sigma) ** 2)

    chi2_red = chi2 / (N - p)

    rA, rm, rgamma, rB, rC = popt
    print(f"{rm:.4f}")

    # # --- Plot with Matplotlib ---
    # plt.scatter(x, y, label='C6D6 Spectrum')
    # plt.errorbar(x, y, fmt = 'None')
    # xfit = np.linspace(low, high, 1000)
    # yfit = lorentz_fit_lin_bg(xfit, rA, rm, rgamma, rB, rC)
    # plt.plot(xfit, yfit, 'k--', linewidth=2,
    #          label=f'Lorentz Fit\nReduced χ² = {chi2_red:.2f}')
    # plt.ylabel("Counts")
    # plt.title("C6D6 - 1551.9 cm^-1 Peak")
    # plt.legend()

    # plt.show()

    # plt.figure()
    # plt.scatter(x, residuals)
    # plt.axhline(0, color='k')
    # plt.title("Residuals")
    # plt.show()
