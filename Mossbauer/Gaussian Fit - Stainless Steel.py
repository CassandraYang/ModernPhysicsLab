import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Calibration constants
a, sigma_a = 1.102, 0.003
b, sigma_b = -0.96, 0.02

def calibrate_velocity(v):
    return a * v + b

v1 = []
counts1 = []
v2 = []
counts2 = []

with open('Mossbauer/Mossbauer - Stainless Steel CSV.csv', newline='') as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        if count < 500:
            v1.append(float(row['velocity']))
            counts1.append(float(row['counts']))
            count += 1
        else:
            v2.append(float(row['velocity']))
            counts2.append(float(row['counts']))

v1 = np.array(v1)
counts1 = np.array(counts1)
v2 = np.array(v2)
counts2 = np.array(counts2)

v1_cal = calibrate_velocity(v1)
v2_cal = calibrate_velocity(v2)

sigma_v1_cal = np.sqrt((a*0)**2 + (v1* sigma_a)**2 + sigma_b**2)  
sigma_v2_cal = np.sqrt((a*0)**2 + (v2* sigma_a)**2 + sigma_b**2)

#rough gaussian fit on areas of interest
left_low, left_high = -20, 20
right_low, right_high = -20, 20

def six_dip_const_bg(x, *params):
    A = params[0]
    m = params[1]
    s = params[2]
    C = params[3]
    
    result = C
    result -= A * np.exp(-(x - m) ** 2 / (2 * s ** 2))
    return result

# Select window
mask_left = (v1_cal > left_low) & (v1_cal < left_high) & (counts1 > 0)
x_left = v1_cal[mask_left]
y_left = counts1[mask_left]
y_err_left = np.sqrt(y_left)

# Initial guesses
p0_left = [max(y_left),
           0, 
           0.5,
           4.2e4]

lower_bounds = (
    [0] +          # A_i ≥ 0 → cannot flip into peaks
    [-20] +        # centers within scan window
    [0] +        # widths must be positive
    [0]              # background positive
)

upper_bounds = (
    [np.inf] +
    [20] +
    [5] +          # widths not absurdly wide
    [np.inf]
)

popt_left, pcov_left = curve_fit(six_dip_const_bg, x_left, y_left, p0=p0_left, bounds = (lower_bounds, upper_bounds), sigma = y_err_left, absolute_sigma=True)

# --- Chi^2 for Soft Iron 1 Gaussian fit ---
yfit_left_data = six_dip_const_bg(x_left, *popt_left)

# Poisson uncertainty (counts)
sigma_y_left = np.sqrt(y_left)

chi2_left = np.sum(((y_left - yfit_left_data) / sigma_y_left)**2)
ndof_left = len(y_left) - len(popt_left)
chi2_red_left = chi2_left / ndof_left

print(f"Stainless Steel 1 Gaussian Reduced Chi^2 = {chi2_red_left:.2f}")

mu_err_left = np.sqrt(np.diag(pcov_left))[1]
sigma_err_left = np.sqrt(np.diag(pcov_left))[2]

A1 = popt_left[0]
m1 = popt_left[1]
s1 = popt_left[2]
C1 = popt_left[3]

print("Stainless Steel 1 Gaussian fit:")
print(f"  Dip 1: mu={m1:.2f} mm/s  sigma={s1:.2f}  amplitude={A1:.2f}")
print(f"  Background = {C1}")

print(f"Dip {1}: mu = {m1:.2f} ± {mu_err_left:.2f} mm/s")
print(sigma_err_left)

mask_right = (v2_cal > right_low) & (v2_cal < right_high) & (counts2 > 0)
x_right = v2_cal[mask_right]
y_right = counts2[mask_right]
y_err_right = np.sqrt(y_right)

p0_right = [max(y_right),
            0, 
            0.5,
            4.2e4]

popt_right, pcov_right = curve_fit(six_dip_const_bg, x_right, y_right, p0=p0_right, bounds = (lower_bounds, upper_bounds), sigma = y_err_right, absolute_sigma=True)

# --- Chi^2 for Soft Iron 2 Gaussian fit ---
yfit_right_data = six_dip_const_bg(x_right, *popt_right)

sigma_y_right = np.sqrt(y_right)

chi2_right = np.sum(((y_right - yfit_right_data) / sigma_y_right)**2)
ndof_right = len(y_right) - len(popt_right)
chi2_red_right = chi2_right / ndof_right

print(f"Stainless Steel 2 Gaussian Reduced Chi^2 = {chi2_red_right:.2f}")

mu_err_right = np.sqrt(np.diag(pcov_right))[1]
sigma_err_right = np.sqrt(np.diag(pcov_right))[2]

A2 = popt_right[0]
m2 = popt_right[1]
s2 = popt_right[2]
C2 = popt_right[3]

print("Stainless Steel 2 Gaussian fit:")
print(f"  Dip 1: mu={m2:.2f} mm/s  sigma={s2:.2f}  amplitude={A2:.2f}")
print(f"  Background = {C2}")

print(f"Dip {1}: mu = {m2:.2f} ± {mu_err_right:.2f} mm/s")
print(sigma_err_right)
# -------- Soft Iron 1 with residuals --------
fig1, (ax1, ax1_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit_left = np.linspace(-20, 20, 1000)
yfit_left = six_dip_const_bg(xfit_left, *popt_left)

ax1.plot(v1_cal, counts1, color='blue', label='Stainless Steel 1')
ax1.plot(xfit_left, yfit_left, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_red_left:.2f}")
ax1.set_yscale('log')
ax1.set_ylabel("Counts")
ax1.set_title(f"Stainless Steel Spectrum 1")
ax1.legend()

# Residuals
residuals_left = (y_left - yfit_left_data)
ax1_res.errorbar(x_left, residuals_left, yerr=sigma_y_left, fmt='o')
ax1_res.axhline(0, color='k', linestyle='--')
ax1_res.set_ylabel("Residuals")
ax1_res.set_xlabel("Velocity (mm/s)")

plt.tight_layout()
plt.savefig("Stainless_Steel_1_with_residuals.png", dpi=300)
plt.show()

# -------- Soft Iron 2 with residuals --------
fig2, (ax2, ax2_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit_right = np.linspace(-20, 20, 1000)
yfit_right = six_dip_const_bg(xfit_right, *popt_right)

ax2.plot(v2_cal, counts2, color='red', label='Stainless Steel 2')
ax2.plot(xfit_right, yfit_right, 'k--', linewidth=2, label=f"Fit: Reduced χ² = {chi2_red_right:.2f}")
ax2.set_yscale('log')
ax2.set_ylabel("Counts")
ax2.set_title(f"Stainless Steel Spectrum 2")
ax2.legend()

# Residuals
residuals_right = (y_right - yfit_right_data)
ax2_res.errorbar(x_right, residuals_right, yerr=sigma_y_right, fmt='o')
ax2_res.axhline(0, color='k', linestyle='--')
ax2_res.set_ylabel("Residuals")
ax2_res.set_xlabel("Velocity (mm/s)")

plt.tight_layout()
plt.savefig("Stainless_Steel_2_with_residuals.png", dpi=300)
plt.show()