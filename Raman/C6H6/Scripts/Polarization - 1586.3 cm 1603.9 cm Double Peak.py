import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def maxAndMinWithErr(L, P, e):
    maxI = 0
    max = -float('inf')
    minI = 0
    min = float('inf')
    for i in range(len(L)):
        if L[i] > max:
            max = L[i]
            maxI = i
        if L[i] < min:
            min = L[i]
            minI = i
    return [max, e[maxI], P[maxI], min, e[minI], P[minI]]

polarization = []
amplitude1 = []
amplitude1_err = []
amplitude2 = []
amplitude2_err = []

for i in range(-90, 91, 5):
    if i in {0, 30, 45, 90}:
        continue
    ramanShift = [] #cm^-1
    counts = []

    with open(f"Raman/C6H6/Polarization - {i}.csv", newline = '') as f:
        for line in f:
            if line.strip() == "WL,Power":
                break
        
        reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
        for row in reader:
            ramanShift.append(float(row['WL']))
            counts.append(float(row["Power"]))
    
    ramanShift = np.array(ramanShift)
    counts = np.array(counts)

    low, high = 1570, 1620

    def lorentz_fit_lin_bg(x, A1, m1, gamma1, A2, m2, gamma2, B, C):
        return A1 * (gamma1 ** 2/((x - m1) ** 2 + gamma1 ** 2)) + A2 * (gamma2 ** 2/((x - m2) ** 2 + gamma2 ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)

    p0 = [max(y),
          1585.5,
          50,
          650,
          1603.3,
          50,
          0,
          200]

    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True)

    amp1 = popt[0]
    amp1_err = np.sqrt(np.diag(pcov))[0]
    amp2 = popt[3]
    amp2_err = np.sqrt(np.diag(pcov))[3]
    
    polarization.append(i)
    amplitude1.append(amp1)
    amplitude1_err.append(amp1_err)
    amplitude2.append(amp2)
    amplitude2_err.append(amp2_err)

# --- Plot with Matplotlib ---

plt.scatter(polarization, amplitude1)
plt.errorbar(polarization, amplitude1, yerr = amplitude1_err, fmt = 'none' )

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("1586.3 cm^-1 Peak")

plt.tight_layout()
plt.savefig("1586.3 cm^-1 Peak.png")
plt.show()

plt.scatter(polarization, amplitude2)
plt.errorbar(polarization, amplitude2, yerr = amplitude2_err, fmt = 'none' )

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("1603.9 cm^-1 Peak")

plt.tight_layout()
plt.savefig("1603.9 cm^-1 Peak.png")
plt.show()

print(maxAndMinWithErr(amplitude1, polarization, amplitude1_err))
print(maxAndMinWithErr(amplitude2, polarization, amplitude2_err))
print(f"{amplitude1[27]:.3f}, {amplitude1_err[27]:.3f}, {polarization[27]:.3f}, {amplitude1[9]:.3f}, {amplitude1_err[9]:.3f}, {polarization[9]:.3f}")
print(f"{amplitude2[27]:.3f}, {amplitude2_err[27]:.3f}, {polarization[27]:.3f}, {amplitude2[9]:.3f}, {amplitude2_err[9]:.3f}, {polarization[9]:.3f}")