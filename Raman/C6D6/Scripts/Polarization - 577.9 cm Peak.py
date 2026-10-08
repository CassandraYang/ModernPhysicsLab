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
    if i in [0]:
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
    A = dict()
    m = dict()
    gamma = dict()
    a2 = dict()
    delta = dict()
    g2 = dict()
    B = dict()
    C = dict()
    with open(f"Raman/C6D6/CSV/Lorentz Fit Guess - 577.6 Peak.csv", newline = '') as f:
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
    
    if g2[i] > gamma[i] and a2[i] > A[i]:
        amp = a2[i]
        mu = m[i] + delta[i]
        g = g2[i]
    else:
        amp = A[i]
        mu = m[i]
        g = gamma[i]
        
    low, high = mu - 6*g, mu + 6*g

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
    amp = popt[0]
    amp_err = np.sqrt(np.diag(pcov))[0]

    polarization.append(i)
    amplitude.append(amp)
    amplitude_err.append(amp_err)
    if i == -55 or i == 45:
        print(amp, amp_err)

# --- Plot with Matplotlib ---

plt.scatter(polarization, amplitude)
plt.errorbar(polarization, amplitude, yerr = amplitude_err, fmt = 'none' )

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("577.9 cm^-1 Peak")

plt.tight_layout()
plt.savefig("577.9 cm^-1 Peak.png")
plt.show()
