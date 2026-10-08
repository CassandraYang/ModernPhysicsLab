import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

T = np.array([77.15, 293.25])
TErr = np.array([0.1, 0.1])
chi = np.array([-5.7589E-06, -2.3754E-07])
chiErr = np.array([5.8965E-06, 1.1154E-07])

def frac_fit(x, C, T0):
    return C / (x - T0)

p0 = [1,
      .01]

popt, pcov = curve_fit(frac_fit, T, chi, p0 = p0, sigma = chiErr, absolute_sigma = True)

C, T0 = popt
C_err = np.sqrt(np.diag(pcov))[0]
T0_err = np.sqrt(np.diag(pcov))[1]

N = len(T)
p = len(popt)

residuals = chi - frac_fit(T, C, T0)

print(f"x = ({C:.4e} ± {C_err:.4e}) / (T - ({T0:.4e} ± {T0_err:.4e}) ")

# --- Plot with Matplotlib ---
plt.figure(figsize = (7.5, 4.8))
plt.scatter(T, chi)
plt.errorbar(T, chi, yerr = chiErr, xerr = TErr, fmt = "None")
xfit = np.linspace(min(T), max(T), 1000)
yfit = frac_fit(xfit, C, T0)
plt.plot(xfit, yfit, 'k--', linewidth=2)

plt.xlabel(r"T (K)")
plt.ylabel(r"$\chi$ ($cm^3$/g)")
plt.title("Zinc")
# plt.legend()
plt.savefig("Magnetic Suceptibilty/Plots/Zinc.png")
plt.show()

plt.figure()
plt.scatter(T, residuals)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()
