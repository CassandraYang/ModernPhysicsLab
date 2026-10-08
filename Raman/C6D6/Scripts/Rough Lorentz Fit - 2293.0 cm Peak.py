import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

csv_filename = "Raman/C6D6/CSV/Lorentz Fit Guess - 2293.0 Peak.csv"
with open(csv_filename, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["angle", "amp", "mu", "gamma", "a2", "delta", "g2", "B", "C"])

for i in range(-90, 91, 5):
    print(i)
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
    
    low, high = 2278, 2325

    def lorentz_fit_lin_bg(x, A, m, gamma, a2, delta, g2, B, C):
        m2 = m + delta
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + a2 * (g2 ** 2/((x - m2) ** 2 + g2 ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)

    p0 = [max(y),
          2293,
          10,
          max(y),
          0,
          10,
          0,
          5]
    
    lower_bounds = (
        [0] + 
        [0] + 
        [0] + 
        [0] + 
        [0] + 
        [0] + 
        [-np.inf] + 
        [-np.inf]
    )

    upper_bounds = (
        [max(y)] +
        [np.inf] +
        [np.inf] +
        [max(y)] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf]
    )
    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True, bounds = (lower_bounds, upper_bounds), maxfev = 30000)

    A, m, gamma, a2, delta, g2, B, C = popt
    print(popt)

    with open(csv_filename, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([i, A, m, gamma, a2, delta, g2, B, C])
    
    # --- Plot with Matplotlib ---
    plt.scatter(x, y, label='C6D6 Spectrum')
    plt.errorbar(x, y, fmt = 'None')
    xfit = np.linspace(low, high, 1000)
    yfit = lorentz_fit_lin_bg(xfit, A, m, gamma, a2, delta, g2, B, C)
    plt.plot(xfit, yfit, 'k--', linewidth=2,
             label=f'Lorentz Fit')
    plt.ylabel("Counts")
    plt.title("C6D6 - 2293.0 cm^-1 Peak")
    plt.legend()

    plt.show()