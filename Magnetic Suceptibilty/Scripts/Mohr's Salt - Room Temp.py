import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr

diffSquares = []
diffSquaresErr = []
mass = []
massErr = []

with open(f"Magnetic Suceptibilty/Data/Mohr's Salt - Room Temp.csv", newline = '') as f:
    reader = csv.DictReader(f)
    for row in reader:
        diffSquares.append(float(row["(H_0^2 - H_Top ^2) (G^2)"]))
        diffSquaresErr.append(float(row["(H_0^2 - H_Top ^2) Error (G^2)"]))
        mass.append(float(row["Apparent Mass (g)"]))
        massErr.append(float(row["Apparent Mass Error (g)"]))
diffSquares = np.array(diffSquares)
diffSquaresErr = np.array(diffSquaresErr)
mass = np.array(mass)
massErr = np.array(massErr)

def lin_reg(B, x):
    return B[0] * x + B[1]

model = odr.Model(lin_reg)

data = odr.RealData(diffSquares, mass, sx = diffSquaresErr, sy = massErr)

b0 = [0,
      9.2,]

odr_fit = odr.ODR(data, model, beta0 = b0)
output = odr_fit.run()

B, C = output.beta
B_err, C_err = output.sd_beta 

N = len(diffSquares)
p = len(output.beta)

residuals = mass - lin_reg(output.beta, diffSquares)

chi2_red = output.res_var

print(f"m = ({B:.4e} ± {B_err:.4e})x + ({C:.7e} ± {C_err:.4e}) ")

# --- Plot with Matplotlib ---
plt.scatter(diffSquares, mass)
plt.errorbar(diffSquares, mass, yerr = massErr, xerr = diffSquaresErr, fmt = "None")
xfit = np.linspace(min(diffSquares), max(diffSquares), 1000)
yfit = lin_reg(output.beta, xfit)
plt.plot(xfit, yfit, 'k--', linewidth=2,
         label=f'Linear Regression\nReduced χ² = {chi2_red:.2f}')

plt.xlabel(r"$(H_0^2 - H_{top}^2)$ (G)")
plt.ylabel("Apparent Mass (g)")
plt.title("Mohr's Salt - Room Temp")
plt.legend()
plt.savefig("Magnetic Suceptibilty/Plots/Mohr's Salt - Room Temp.png")
plt.show()

plt.figure()
plt.scatter(diffSquares, residuals)
plt.axhline(0, color='k')
plt.title("Residuals")
plt.show()
