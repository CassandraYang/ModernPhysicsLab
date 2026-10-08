import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

time = []
ratio605 =[]
ratio605Err = []
ratio1175 =[]
ratio1175Err = []
ratio1583 =[]
ratio1583Err = []
ratio3060 =[]
ratio3060Err = []

with open(f"Raman/C6H6/CSV/Raman Spectroscopy - Accquisition Time Plots.csv", newline = '') as f:
    reader = csv.DictReader(f)
    for row in reader:
        time.append(float(row['Integration Time (s)']))
        ratio605.append(float(row["Peak 605"]))
        ratio605Err.append(float(row["Peak 605 Error"]))
        ratio1175.append(float(row["Peak 1175"]))
        ratio1175Err.append(float(row["Peak 1175 Error"]))
        ratio1583.append(float(row["Peak 1583"]))
        ratio1583Err.append(float(row["Peak 1583 Error"]))
        ratio3060.append(float(row["Peak 3060"]))
        ratio3060Err.append(float(row["Peak 3060 Error"]))

time = np.array(time)
ratio605 = np.array(ratio605)
ratio605Err = np.array(ratio605Err)
ratio1175 = np.array(ratio1175)
ratio1175Err = np.array(ratio1175Err)
ratio1583 = np.array(ratio1583)
ratio1583Err = np.array(ratio1583Err)
ratio3060 = np.array(ratio3060)
ratio3060Err = np.array(ratio3060Err)

def lin_reg(x, m, b):
    return m * x + b

bestTime = 0
bestSlope = 100000000
for i in range(2, len(time)):
    t = time[:i]

    p0605 = [0,
             0]

    popt605, pcov605 = curve_fit(lin_reg, t, ratio605[:i], p0 = p0605, sigma = ratio605Err[:i], absolute_sigma = True)

    m605, b605 = popt605
    m605_err = np.sqrt(np.diag(pcov605))[0]
    b605_err = np.sqrt(np.diag(pcov605))[1]

    p01175 = [0,
             0]

    popt1175, pcov1175 = curve_fit(lin_reg, t, ratio1175[:i], p0 = p01175, sigma = ratio1175Err[:i], absolute_sigma = True)

    m1175, b1175 = popt1175
    m1175_err = np.sqrt(np.diag(pcov1175))[0]
    b1175_err = np.sqrt(np.diag(pcov1175))[1]

    p01583 = [0,
             0]

    popt1583, pcov1583 = curve_fit(lin_reg, t, ratio1583[:i], p0 = p01583, sigma = ratio1583Err[:i], absolute_sigma = True)

    m1583, b1583 = popt1583
    m1583_err = np.sqrt(np.diag(pcov1583))[0]
    b1583_err = np.sqrt(np.diag(pcov1583))[1]

    p03060 = [0,
             0]

    popt3060, pcov3060 = curve_fit(lin_reg, t, ratio3060[:i], p0 = p03060, sigma = ratio3060Err[:i], absolute_sigma = True)

    m3060, b3060 = popt3060
    m3060_err = np.sqrt(np.diag(pcov3060))[0]
    b3060_err = np.sqrt(np.diag(pcov3060))[1]

    slope = abs(m605) + abs(m1175) + abs(m1583) + abs(m3060)
    if slope < bestSlope:
        bestSlope = slope
        bestTime = i

t = time[:bestTime]

p0605 = [0,
         0]

popt605, pcov605 = curve_fit(lin_reg, t, ratio605[:bestTime], p0 = p0605, sigma = ratio605Err[:bestTime], absolute_sigma = True)

m605, b605 = popt605
m605_err = np.sqrt(np.diag(pcov605))[0]
b605_err = np.sqrt(np.diag(pcov605))[1]

p01175 = [0,
          0]

popt1175, pcov1175 = curve_fit(lin_reg, t, ratio1175[:bestTime], p0 = p01175, sigma = ratio1175Err[:bestTime], absolute_sigma = True)

m1175, b1175 = popt1175
m1175_err = np.sqrt(np.diag(pcov1175))[0]
b1175_err = np.sqrt(np.diag(pcov1175))[1]

p01583 = [0,
          0]

popt1583, pcov1583 = curve_fit(lin_reg, t, ratio1583[:bestTime], p0 = p01583, sigma = ratio1583Err[:bestTime], absolute_sigma = True)

m1583, b1583 = popt1583
m1583_err = np.sqrt(np.diag(pcov1583))[0]
b1583_err = np.sqrt(np.diag(pcov1583))[1]

p03060 = [0,
          0]

popt3060, pcov3060 = curve_fit(lin_reg, t, ratio3060[:bestTime], p0 = p03060, sigma = ratio3060Err[:bestTime], absolute_sigma = True)

m3060, b3060 = popt3060
m3060_err = np.sqrt(np.diag(pcov3060))[0]
b3060_err = np.sqrt(np.diag(pcov3060))[1]

N = len(time)
p = len(popt605)

residuals605 = ratio605[:bestTime] - lin_reg(t, m605, b605)
chi2_605 = np.sum((residuals605 / ratio605Err[:bestTime]) ** 2)
chi2_605_red = chi2_605 / (N - p)
residuals1175 = ratio1175[:bestTime] - lin_reg(t, m1175, b1175)
chi2_1175 = np.sum((residuals1175 / ratio1175Err[:bestTime]) ** 2)
chi2_1175_red = chi2_1175 / (N - p)
residuals1583 = ratio1583[:bestTime] - lin_reg(t, m1583, b1583)
chi2_1583 = np.sum((residuals1583 / ratio1583Err[:bestTime]) ** 2)
chi2_1583_red = chi2_1583 / (N - p)
residuals3060 = ratio3060[:bestTime] - lin_reg(t, m3060, b3060)
chi2_3060 = np.sum((residuals3060 / ratio3060Err[:bestTime]) ** 2)
chi2_3060_red = chi2_3060 / (N - p)


# --- Plot with Matplotlib ---
fig1, (ax1, ax1_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit605 = np.linspace(min(t), max(t), 1000)
yfit605 = lin_reg(xfit605, m605, b605)

ax1.scatter(time, ratio605)
ax1.errorbar(time, ratio605, yerr = ratio605Err, fmt = "None")
ax1.plot(xfit605, yfit605, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_605_red:.2f}")
ax1.set_ylabel("Ratio")
ax1.set_title(f"Peak at 605 cm^-1")
ax1.legend()

# Residuals
ax1_res.errorbar(t, residuals605, yerr=ratio605Err[:bestTime], fmt='o')
ax1_res.axhline(0, color='k', linestyle='--')
ax1_res.set_ylabel("Residuals")
ax1_res.set_xlabel("Integration Time (s)")

plt.tight_layout()
plt.savefig("Raman/C6H6/Plots/Ratio605.png", dpi=300)
plt.show()

fig2, (ax2, ax2_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit1175 = np.linspace(min(t), max(t), 1000)
yfit1175 = lin_reg(xfit1175, m1175, b1175)

ax2.scatter(time, ratio1175)
ax2.errorbar(time, ratio1175, yerr = ratio1175Err, fmt = "None")
ax2.plot(xfit1175, yfit1175, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_1175_red:.2f}")
ax2.set_ylabel("Ratio")
ax2.set_title(f"Peak at 1175 cm^-1")
ax2.legend()

# Residuals
ax2_res.errorbar(t, residuals1175, yerr=ratio1175Err[:bestTime], fmt='o')
ax2_res.axhline(0, color='k', linestyle='--')
ax2_res.set_ylabel("Residuals")
ax2_res.set_xlabel("Integration Time (s)")

plt.tight_layout()
plt.savefig("Raman/C6H6/Plots/Ratio1175.png", dpi=300)
plt.show()

fig3, (ax3, ax3_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit1583 = np.linspace(min(t), max(t), 1000)
yfit1583 = lin_reg(xfit1583, m1583, b1583)

ax3.scatter(time, ratio1583)
ax3.errorbar(time, ratio1583, yerr = ratio1583Err, fmt = "None")
ax3.plot(xfit1583, yfit1583, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_1583_red:.2f}")
ax3.set_ylabel("Ratio")
ax3.set_title(f"Peak at 1583 cm^-1")
ax3.legend()

# Residuals
ax3_res.errorbar(t, residuals1583, yerr=ratio1583Err[:bestTime], fmt='o')
ax3_res.axhline(0, color='k', linestyle='--')
ax3_res.set_ylabel("Residuals")
ax3_res.set_xlabel("Integration Time (s)")

plt.tight_layout()
plt.savefig("Raman/C6H6/Plots/Ratio1583.png", dpi=300)
plt.show()

fig4, (ax4, ax4_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit3060 = np.linspace(min(t), max(t), 1000)
yfit3060 = lin_reg(xfit3060, m3060, b3060)

ax4.scatter(time, ratio3060)
ax4.errorbar(time, ratio3060, yerr = ratio3060Err, fmt = "None")
ax4.plot(xfit3060, yfit3060, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_3060_red:.2f}")
ax4.set_ylabel("Ratio")
ax4.set_title(f"Peak at 3060 cm^-1")
ax4.legend()

# Residuals
ax4_res.errorbar(t, residuals3060, yerr=ratio3060Err[:bestTime], fmt='o')
ax4_res.axhline(0, color='k', linestyle='--')
ax4_res.set_ylabel("Residuals")
ax4_res.set_xlabel("Integration Time (s)")

plt.tight_layout()
plt.savefig("Raman/C6H6/Plots/Ratio3060.png", dpi=300)
plt.show()
