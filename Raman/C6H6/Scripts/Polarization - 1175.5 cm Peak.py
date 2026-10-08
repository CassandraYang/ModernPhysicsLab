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
amplitude = []
amplitude_err = []

for i in range(-90, 91, 5):
    if i in {0, 30, 45, 90}:
        continue
    ramanShift = [] #cm^-1
    counts = []

    with open(f"Raman/C6H6/CSV/Polarization - {i}.csv", newline = '') as f:
        for line in f:
            if line.strip() == "WL,Power":
                break
        
        reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
        for row in reader:
            ramanShift.append(float(row['WL']))
            counts.append(float(row["Power"]))
    
    ramanShift = np.array(ramanShift)
    counts = np.array(counts)

    low, high = 1160, 1190

    def lorentz_fit_lin_bg(x, A, m, gamma, B, C):
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)

    p0 = [max(y),
          1174.5,
          25,
          0,
          200]

    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True)

    amp = popt[0]
    amp_err = np.sqrt(np.diag(pcov))[0]

    polarization.append(i)
    amplitude.append(amp)
    amplitude_err.append(amp_err)

# --- Plot with Matplotlib ---

plt.scatter(polarization, amplitude)
plt.errorbar(polarization, amplitude, yerr = amplitude_err, fmt = 'none' )

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("1175.5 cm^-1 Peak")

plt.tight_layout()
plt.savefig("1175.5 cm^-1 Peak.png")
plt.show()

print(maxAndMinWithErr(amplitude, polarization, amplitude_err))
# print(f"{amplitude[27]:.3f}, {amplitude_err[27]:.3f}, {polarization[27]:.3f}, {amplitude[9]:.3f}, {amplitude_err[9]:.3f}, {polarization[9]:.3f}")