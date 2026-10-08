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
    return max, e[maxI], P[maxI], min, e[minI], P[minI]


polarization = []
amplitude = []
amplitude_err = []

for i in range(-90, 91, 5):
    ramanShift = [] #cm^-1
    counts = []
    with open(f"Raman/CCl4/CSV/Polarization - {i}.csv", newline = '') as f:
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
    m2 = dict()
    g2 = dict()
    B = dict()
    C = dict()
    with open(f"Raman/CCl4/CSV/Lorentz Fit Guess - 1532 Peak.csv", newline = '') as f:
        reader = csv.DictReader(f)
        for row in reader:
            A[int(row['angle'])] = float(row['amp'])
            m[int(row['angle'])] = float(row['mu'])
            gamma[int(row['angle'])] = float(row['gamma'])
            a2[int(row['angle'])] = float(row['a2'])
            m2[int(row['angle'])] = float(row['m2'])
            g2[int(row['angle'])] = float(row['g2'])
            B[int(row['angle'])] = float(row['B'])
            C[int(row['angle'])] = float(row['C'])

    ramanShift = np.array(ramanShift)
    counts = np.array(counts)
    
    low, high = m[i] - 4*gamma[i], m[i] + 4*gamma[i]

    def lorentz_fit_lin_bg(x, A, m, gamma, a2, m2, g2, B, C):
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + a2 * (g2 ** 2/((x - m2) ** 2 + g2 ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)
    p0 = [A[i],
          m[i],
          gamma[i],
          a2[i],
          m2[i],
          g2[i],
          B[i],
          C[i]]
    
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
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf] +
        [np.inf]
    )
    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True, bounds = (lower_bounds, upper_bounds), maxfev = 5000)
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
plt.title("1539.4 cm^-1 Peak")

plt.tight_layout()
plt.savefig("Raman/CCl4/Plots/1539.4 cm^-1 Peak.png")
plt.show()

print(maxAndMinWithErr(amplitude, polarization, amplitude_err))
# print(f"{amplitude[27]:.3f}, {amplitude_err[27]:.3f}, {polarization[27]:.3f}, {amplitude[9]:.3f}, {amplitude_err[9]:.3f}, {polarization[9]:.3f}")