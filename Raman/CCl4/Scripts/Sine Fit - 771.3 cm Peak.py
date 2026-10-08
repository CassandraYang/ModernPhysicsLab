import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

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
    with open(f"Raman/CCl4/CSV/Lorentz Fit Guess - 771 Peak.csv", newline = '') as f:
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
    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True, bounds = (lower_bounds, upper_bounds))
    amp = popt[0]
    amp_err = np.sqrt(np.diag(pcov))[0]

    polarization.append(i)
    amplitude.append(amp)
    amplitude_err.append(amp_err)

polarizationPos = np.array(polarization[18:])
amplitudePos = np.array(amplitude[18:])
amplitude_errPos = np.array(amplitude_err[18:])

polarizationNeg = np.array(polarization[:18])
amplitudeNeg = np.array(amplitude[:18])
amplitude_errNeg = np.array(amplitude_err[:18])


def sine_fit_const_bg(x, A, theta0, C):
        return A * np.sin(np.deg2rad(x-theta0))**2 + C

p0Pos = [max(amplitudePos),
      0,
      0]

poptPos, pcovPos = curve_fit(sine_fit_const_bg, polarizationPos, amplitudePos, p0 = p0Pos, sigma = amplitude_errPos, absolute_sigma = True, bounds=([0, -90, 0], [np.inf, 90, np.inf]))
APos, t0Pos, CPos = poptPos
APos_err = np.sqrt(np.diag(pcov))[0]
t0Pos_err= np.sqrt(np.diag(pcov))[1]
CPos_err = np.sqrt(np.diag(pcov))[2]

NPos = len(polarizationPos)
pPos = len(poptPos)

residualsPos = amplitudePos - sine_fit_const_bg(polarizationPos, *poptPos)
chi2Pos = np.sum((residualsPos / amplitude_errPos) ** 2)

chi2_redPos = chi2Pos / (NPos - pPos)

p0Neg = [max(amplitudeNeg),
      t0Pos,
      0]

poptNeg, pcovNeg = curve_fit(sine_fit_const_bg, polarizationNeg, amplitudeNeg, p0 = p0Neg, sigma = amplitude_errNeg, absolute_sigma = True, bounds=([0, -90, 0], [np.inf, 90, np.inf]))
ANeg, t0Neg, CNeg = poptNeg
ANeg_err = np.sqrt(np.diag(pcov))[0]
t0Neg_err = np.sqrt(np.diag(pcov))[1]
CNeg_err = np.sqrt(np.diag(pcov))[2]

NNeg = len(polarizationNeg)
pNeg = len(poptNeg)

residualsNeg = amplitudeNeg - sine_fit_const_bg(polarizationNeg, *poptNeg)
chi2Neg = np.sum((residualsNeg / amplitude_errNeg) ** 2)

chi2_redNeg = chi2Neg / (NNeg - pNeg)



# --- Plot with Matplotlib ---

plt.scatter(polarizationPos, amplitudePos)
plt.errorbar(polarizationPos, amplitudePos, yerr = amplitude_errPos, fmt = 'none' )
xfit = np.linspace(0, 90, 1000)
yfit = sine_fit_const_bg(xfit, APos, t0Pos, CPos)
plt.plot(xfit, yfit, 'k--', linewidth=2,
         label=f'Sine Fit\nReduced χ² = {chi2_redPos:.2f}')

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("771.3 cm^-1 Peak")
plt.legend()

plt.tight_layout()
plt.savefig("771.3 cm^-1 Peak Pos.png")
plt.show()

plt.figure()
plt.scatter(polarizationPos, residualsPos)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()

plt.scatter(polarizationNeg, amplitudeNeg)
plt.errorbar(polarizationNeg, amplitudeNeg, yerr = amplitude_errNeg, fmt = 'none' )
xfit = np.linspace(-90, 0, 1000)
yfit = sine_fit_const_bg(xfit, ANeg, t0Neg, CNeg)
plt.plot(xfit, yfit, 'k--', linewidth=2,
         label=f'Sine Fit\nReduced χ² = {chi2_redNeg:.2f}')

plt.ylabel("Counts")
plt.xlabel("Polarization (theta)")
plt.title("771.3 cm^-1 Peak")
plt.legend()

plt.tight_layout()
plt.savefig("771.3 cm^-1 Peak Neg.png")
plt.show()

plt.figure()
plt.scatter(polarizationNeg, residualsNeg)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()

print(APos + CPos, np.sqrt(APos_err**2 + CPos_err**2), t0Pos, t0Pos_err)
print(CNeg, CNeg_err, t0Neg, t0Neg_err)