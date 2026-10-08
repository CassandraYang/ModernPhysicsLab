import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ramanShift = [] #cm^-1
counts = []

with open(f"Snapshot.csv", newline = '') as f:
    for line in f:
        if line.strip() == "WL,Power":
            break
        
    reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
    for row in reader:
        ramanShift.append(float(row['WL']))
        counts.append(float(row["Power"]))
    
ramanShift = np.array(ramanShift)
counts = np.array(counts)
    
low, high = 900, 1100

def gauss_fit_const_bg(x, A, m, s, B, C):
    return A * np.exp(-(x - m) ** 2 / (2 * s ** 2)) + B * x + C
    
mask = (ramanShift > low) & (ramanShift < high) & (counts > 0) 
x = ramanShift[mask]
y = counts[mask]
sigma = np.sqrt(y)

p0 = [max(y) + 10,
      991.6,
      15,
      0,
      100]

popt, pcov = curve_fit(gauss_fit_const_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True)

N = len(x)
p = len(popt)

residuals = y - gauss_fit_const_bg(x, *popt)
chi2 = np.sum((residuals / sigma) ** 2)

chi2_red = chi2 / (N - p)

A, m, s, B, C = popt
m_err = np.sqrt(np.diag(pcov))[1]
s_err = np.sqrt(np.diag(pcov))[2]
print(f"{m:.3f} ± {m_err:.3f}")
amp = popt[0]
amp_err = np.sqrt(np.diag(pcov))[0]

# --- Plot with Matplotlib ---
plt.plot(x, y, color='blue', label='C6H6 Spectrum')
xfit = np.linspace(low, high, 1000)
yfit = gauss_fit_const_bg(xfit, A, m, s, B, C)
plt.plot(xfit, yfit, 'k--', linewidth=2,
         label=f'Gaussian Fit\nReduced χ² = {chi2_red:.2f}')

plt.ylabel("Counts")
plt.title("C6H6 - 991.6 cm^-1 Peak")
plt.legend()
plt.savefig("C6H6 - Snapshot.png")
plt.show()

plt.figure()
plt.plot(x, residuals)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()
