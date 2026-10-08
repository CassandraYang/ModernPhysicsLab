import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DEBUG = False
for i in range(-90, 91, 5):
    if i in [20, 25]:
        continue
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
    
    ramanShift = np.array(ramanShift)
    counts = np.array(counts)
    
    low, high = 555, 596

    def lorentz_fit_lin_bg(x, A, m, gamma, B, C):
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)
    if DEBUG:
        plt.figure()
        plt.scatter(x, y, label='C6H6 Spectrum')
        plt.errorbar(x, y, fmt = 'None')
        plt.show()

    p0 = [max(y),
          573,
          10,
          0,
          5]

    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True, maxfev = 2000)

    N = len(x)
    p = len(popt)

    residuals = y - lorentz_fit_lin_bg(x, *popt)
    chi2 = np.sum((residuals / sigma) ** 2)

    chi2_red = chi2 / (N - p)

    A, m, gamma, B, C = popt
    print(f"{m:.4f}")
    amp = popt[0]
    amp_err = np.sqrt(np.diag(pcov))[0]

    # --- Plot with Matplotlib ---
    plt.scatter(x, y, label='C6H6 Spectrum')
    plt.errorbar(x, y, fmt = 'None')
    xfit = np.linspace(low, high, 1000)
    yfit = lorentz_fit_lin_bg(xfit, A, m, gamma, B, C)
    plt.plot(xfit, yfit, 'k--', linewidth=2,
             label=f'Lorentz Fit\nReduced χ² = {chi2_red:.2f}')
    plt.ylabel("Counts")
    plt.title("C6H6 - 577.6 cm^-1 Peak")
    plt.legend()

    plt.show()

    plt.figure()
    plt.scatter(x, residuals)
    plt.axhline(0, color='k')
    plt.title("Residuals")
    plt.show()
