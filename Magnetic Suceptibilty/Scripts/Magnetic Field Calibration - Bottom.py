import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

current = [] 
magneticField = []
magError = []

with open(f"Magnetic Suceptibilty/Data/Bottom.csv", newline = '') as f:
    reader = csv.DictReader(f)
    for row in reader:
        current.append(float(row['Electric Current (mV)']))
        magneticField.append(float(row["Average Magnetic Field (G)"]))
        magError.append(float(row["Error"]))
current = np.array(current)
magneticField = np.array(magneticField)
magError = np.array(magError)

def lin_reg(x, B, C):
    return B * x + C

p0 = [66.6,
      20.6,]

popt, pcov = curve_fit(lin_reg, current, magneticField, p0 = p0, sigma = magError, absolute_sigma = True)

N = len(current)
p = len(popt)

residuals = magneticField - lin_reg(current, *popt)
chi2 = np.sum((residuals / magError) ** 2)

chi2_red = chi2 / (N - p)

B, C = popt
B_err = np.sqrt(np.diag(pcov))[0]
C_err = np.sqrt(np.diag(pcov))[1]
print(f"Bottom Calibration Equation: H = ({B:.4f} ± {B_err:.4f})x + ({C:.4f} ± {C_err:.4f}) ")

# --- Plot with Matplotlib ---
plt.scatter(current, magneticField)
plt.errorbar(current, magneticField, magError, fmt = "None")
xfit = np.linspace(min(current), max(current), 1000)
yfit = lin_reg(xfit, B, C)
plt.plot(xfit, yfit, 'k--', linewidth=2,
         label=f'Linear Regression\nReduced χ² = {chi2_red:.2f}')

plt.xlabel("Current (mV)")
plt.ylabel("Magnetic Field (G)")
plt.title("Bottom Calibration")
plt.legend()
plt.savefig("Magnetic Suceptibilty/Plots/Calibration - Bottom.png")
plt.show()

plt.figure()
plt.scatter(current, residuals)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()
