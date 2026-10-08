import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Calibration constants
a, sigma_a = 1.102, 0.003
b, sigma_b = -0.96, 0.02

def calibrate_velocity(v):
    return a * v + b

# Physical constants
c = 2.99792458e8           # m/s
mu_N = 5.050783699e-27     # J/T
E0 = 14.4125e3 * 1.60218e-19  # 14.4 keV in joules

K = (c * mu_N / E0) * 1e3   # mm/s per tesla

# g-factors for 57Fe
g_g = 0.18128             
g_e = -0.10313

m_e = np.array([+3/2, +1/2, -1/2, +1/2, -1/2, -3/2])
m_g = np.array([+1/2, +1/2, +1/2, -1/2, -1/2, -1/2])

X = g_e * m_e - g_g * m_g

v1 = []
counts1 = []
v2 = []
counts2 = []

with open('Mossbauer/Mossbauer - Hematite CSV.csv', newline='') as f:
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
    A = params[0:6]
    m = params[6:12]
    s = params[12:18]
    C = params[18]
    
    result = C
    for i in range(6):
        result -= A[i] * np.exp(-(x - m[i]) ** 2 / (2 * s[i] ** 2))
    return result

# Select window
mask_left = (v1_cal > left_low) & (v1_cal < left_high) & (counts1 > 0)
x_left = v1_cal[mask_left]
y_left = counts1[mask_left]
y_err_left = np.sqrt(y_left)

# Initial guesses
p0_left = [max(y_left), max(y_left), max(y_left), max(y_left), max(y_left), max(y_left),
           -18, -11, -4, 2, 11, 17, 
           0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
           4.2e4]

lower_bounds = (
    [0]*6 +          # A_i ≥ 0 → cannot flip into peaks
    [-20]*6 +        # centers within scan window
    [0]*6 +        # widths must be positive
    [0]              # background positive
)

upper_bounds = (
    [np.inf]*6 +
    [20]*6 +
    [5]*6 +          # widths not absurdly wide
    [np.inf]
)

popt_left, pcov_left = curve_fit(six_dip_const_bg, x_left, y_left, p0=p0_left, bounds = (lower_bounds, upper_bounds), sigma=y_err_left, absolute_sigma=True)

# --- Chi^2 for Soft Iron 1 Gaussian fit ---
yfit_left_data = six_dip_const_bg(x_left, *popt_left)

# Poisson uncertainty (counts)
sigma_y_left = np.sqrt(y_left)

chi2_left = np.sum(((y_left - yfit_left_data) / sigma_y_left)**2)
ndof_left = len(y_left) - len(popt_left)
chi2_red_left = chi2_left / ndof_left

print(f"Hematite 1 Gaussian Reduced Chi^2 = {chi2_red_left:.2f}")

mu_err_left = np.sqrt(np.diag(pcov_left))[6:12]

A1 = popt_left[0:6]
m1 = popt_left[6:12]
s1 = popt_left[12:18]
C1 = popt_left[18]

print("Hematite 1 Gaussian fit:")
print(f"  Dip 1: mu={m1[0]:.2f} mm/s  sigma={s1[0]:.2f}  amplitude={A1[0]:.2f}")
print(f"  Dip 2: mu={m1[1]:.2f} mm/s  sigma={s1[1]:.2f}  amplitude={A1[1]:.2f}")
print(f"  Dip 3: mu={m1[2]:.2f} mm/s  sigma={s1[2]:.2f}  amplitude={A1[2]:.2f}")
print(f"  Dip 4: mu={m1[3]:.2f} mm/s  sigma={s1[3]:.2f}  amplitude={A1[3]:.2f}")
print(f"  Dip 5: mu={m1[4]:.2f} mm/s  sigma={s1[4]:.2f}  amplitude={A1[4]:.2f}")
print(f"  Dip 6: mu={m1[5]:.2f} mm/s  sigma={s1[5]:.2f}  amplitude={A1[5]:.2f}")
print(f"  Background = {C1}")

for i in range(6):
    print(f"Dip {i+1}: mu = {m1[i]:.2f} ± {mu_err_left[i]:.2f} mm/s")

mask_right = (v2_cal > right_low) & (v2_cal < right_high) & (counts2 > 0)
x_right = v2_cal[mask_right]
y_right = counts2[mask_right]
y_err_right = np.sqrt(y_right)

p0_right = [max(y_right), max(y_right), max(y_right), max(y_right), max(y_right), max(y_right),
            -18, -11, -4, 2, 11, 17,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 
            4.2e4]

popt_right, pcov_right = curve_fit(six_dip_const_bg, x_right, y_right, p0=p0_right, bounds = (lower_bounds, upper_bounds), sigma = y_err_right, absolute_sigma=True)

# --- Chi^2 for Soft Iron 2 Gaussian fit ---
yfit_right_data = six_dip_const_bg(x_right, *popt_right)

sigma_y_right = np.sqrt(y_right)

chi2_right = np.sum(((y_right - yfit_right_data) / sigma_y_right)**2)
ndof_right = len(y_right) - len(popt_right)
chi2_red_right = chi2_right / ndof_right

print(f"Hematite 2 Gaussian Reduced Chi^2 = {chi2_red_right:.2f}")

mu_err_right = np.sqrt(np.diag(pcov_right))[6:12]

A2 = popt_right[0:6]
m2 = popt_right[6:12]
s2 = popt_right[12:18]
C2 = popt_right[18]

print("Hematite 2 Gaussian fit:")
print(f"  Dip 1: mu={m2[0]:.2f} mm/s  sigma={s2[0]:.2f}  amplitude={A2[0]:.2f}")
print(f"  Dip 2: mu={m2[1]:.2f} mm/s  sigma={s2[1]:.2f}  amplitude={A2[1]:.2f}")
print(f"  Dip 3: mu={m2[2]:.2f} mm/s  sigma={s2[2]:.2f}  amplitude={A2[2]:.2f}")
print(f"  Dip 4: mu={m2[3]:.2f} mm/s  sigma={s2[3]:.2f}  amplitude={A2[3]:.2f}")
print(f"  Dip 5: mu={m2[4]:.2f} mm/s  sigma={s2[4]:.2f}  amplitude={A2[4]:.2f}")
print(f"  Dip 6: mu={m2[5]:.2f} mm/s  sigma={s2[5]:.2f}  amplitude={A2[5]:.2f}")
print(f"  Background = {C2}")

for i in range(6):
    print(f"Dip {i+1}: mu = {m2[i]:.2f} ± {mu_err_right[i]:.2f} mm/s")

X_quad = np.array([-1, -1, -1, +1, +1, +1])  

v_measured = 0.5 * (m1 + m2)
sigma_mu1 = np.sqrt(np.diag(pcov_left))[6:12]
sigma_mu2 = np.sqrt(np.diag(pcov_right))[6:12]

sigma_v_cal = np.sqrt((v_measured * sigma_a)**2 + sigma_b**2)
sigma_v_measured = 0.5 * np.sqrt(sigma_mu1**2 + sigma_mu2**2 + sigma_v_cal**2)
# Design matrix (correct physics)
A = np.column_stack([
    np.ones(len(X)),
    K * X,
    X_quad
])

W = np.diag(1 / sigma_v_measured**2)

ATA = A.T @ W @ A
ATy = A.T @ W @ v_measured

params = np.linalg.solve(ATA, ATy)
cov = np.linalg.inv(ATA)

v_center, B, Delta_EQ = params
sigma_center, sigma_B, sigma_DeltaEQ = np.sqrt(np.diag(cov))

print(f"Isomer shift (center): {v_center:.4f} ± {sigma_center:.4f} mm/s")
print(f"Quadrupole Splitting E_Q : {Delta_EQ:.3f} ± {sigma_DeltaEQ:.3f} V")
print(f"Hyperfine field B:     {B:.3f} ± {sigma_B:.3f} T")

# Correct full model
v_fit = v_center + K * B * X + Delta_EQ * X_quad

chi2 = np.sum(((v_measured - v_fit)**2 / sigma_v_measured))
ndof = len(v_measured) - 3
chi2_red = chi2 / ndof

print(f"Reduced Chi^2 = {chi2_red:.2f}")


# -------- Soft Iron 1 with residuals --------
fig1, (ax1, ax1_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit_left = np.linspace(-20, 20, 1000)
yfit_left = six_dip_const_bg(xfit_left, *popt_left)

ax1.plot(v1_cal, counts1, color='blue', label='Hematite 1')
ax1.plot(xfit_left, yfit_left, 'k--', linewidth=2, label= f"Fit: Reduced χ² = {chi2_red_left:.2f}")
ax1.set_yscale('log')
ax1.set_ylabel("Counts")
ax1.set_title(f"Hematite Spectrum 1")
ax1.legend()

# Residuals
residuals_left = (y_left - yfit_left_data)
ax1_res.errorbar(x_left, residuals_left, yerr=sigma_y_left, fmt='o')
ax1_res.axhline(0, color='k', linestyle='--')
ax1_res.set_ylabel("Residuals")
ax1_res.set_xlabel("Velocity (mm/s)")

plt.tight_layout()
plt.savefig("Hematite_1_with_residuals.png", dpi=300)
plt.show()

# -------- Soft Iron 2 with residuals --------
fig2, (ax2, ax2_res) = plt.subplots(
    2, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

xfit_right = np.linspace(-20, 20, 1000)
yfit_right = six_dip_const_bg(xfit_right, *popt_right)

ax2.plot(v2_cal, counts2, color='red', label='Hematite 2')
ax2.plot(xfit_right, yfit_right, 'k--', linewidth=2, label=f"Fit: Reduced χ² = {chi2_red_right:.2f}")
ax2.set_yscale('log')
ax2.set_ylabel("Counts")
ax2.set_title(f"Hematite Spectrum 2")
ax2.legend()

# Residuals
residuals_right = (y_right - yfit_right_data)
ax2_res.errorbar(x_right, residuals_right, yerr=sigma_y_right, fmt='o')
ax2_res.axhline(0, color='k', linestyle='--')
ax2_res.set_ylabel("Residuals")
ax2_res.set_xlabel("Velocity (mm/s)")

plt.tight_layout()
plt.savefig("Hematite_2_with_residuals.png", dpi=300)
plt.show()

# x-axis: theoretical splitting factor
X_plot = X

# Best-fit model
v_fit = v_center + K * B * X + Delta_EQ *X_quad

fig3, (ax3, ax3_res) = plt.subplots(
    2, 1, figsize=(7, 8), sharex=True,
    gridspec_kw={'height_ratios': [3, 1]}
)

# Data points with error bars
ax3.errorbar(
    X_plot, v_measured,
    yerr=sigma_v_measured,
    fmt='o',
    capsize=4,
    label='Measured lines'
)

# Fit line
ax3.plot(
    X_plot, v_fit,
    'r-',
    linewidth=2,
    label=f"Fit (Reduced χ² = {chi2_red:.2f})"
)

ax3.set_ylabel("Velocity (mm/s)")
ax3.set_title("Mössbauer Zeeman Splitting (Soft Iron)")
ax3.legend()
ax3.grid(alpha=0.3)

# Residuals
residuals_zeeman = v_measured - (v_center + K * B * X_plot)

ax3_res.errorbar(
    X_plot,
    residuals_zeeman,
    yerr=sigma_v_measured,
    fmt='o',
    capsize=4
)

ax3_res.axhline(0, color='k', linestyle='--')
ax3_res.set_xlabel(r"$g_e m_e - g_g m_g$")
ax3_res.set_ylabel("Residuals")

plt.tight_layout()
plt.savefig("Zeeman_Linear_Fit_(Hematite).png", dpi=300)
plt.show()

