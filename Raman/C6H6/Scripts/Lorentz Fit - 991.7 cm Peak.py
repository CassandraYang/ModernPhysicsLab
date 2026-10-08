import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# for i in range(-90, 91, 5):
for i in [45]:
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
    
    low, high = 980, 1000

    def lorentz_fit_lin_bg(x, A, m, gamma, B, C):
        return A * (gamma ** 2/((x - m) ** 2 + gamma ** 2)) + B * x + C
    
    mask = (ramanShift > low) & (ramanShift < high) & (counts > 0)
    x = ramanShift[mask]
    y = counts[mask]
    sigma = np.sqrt(y)

    p0 = [max(y) + 10,
          991.6,
          15,
          0,
          100]

    popt, pcov = curve_fit(lorentz_fit_lin_bg, x, y, p0 = p0, sigma = sigma, absolute_sigma = True)

    N = len(x)
    p = len(popt)

    residuals = y - lorentz_fit_lin_bg(x, *popt)
    chi2 = np.sum((residuals / sigma) ** 2)

    chi2_red = chi2 / (N - p)

    A, m, gamma, B, C = popt
    print(f"{m:.4f}")
    # print(f"{gamma:.4f}")
    amp = popt[0]
    amp_err = np.sqrt(np.diag(pcov))[0]
    b_err = np.sqrt(np.diag(pcov))[3]
    c_err = np.sqrt(np.diag(pcov))[4]

    # --- Plot with Matplotlib ---
    fig2, (ax2, ax2_res) = plt.subplots(
        2, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )
    ax2.scatter(x, y, color='blue', label='C6H6 Spectrum')
    ax2.errorbar(x, y, color='blue', fmt = 'None')
    xfit = np.linspace(low, high, 1000)
    yfit = lorentz_fit_lin_bg(xfit, A, m, gamma, B, C)
    ax2.plot(xfit, yfit, 'k--', linewidth=2,
             label=f'Lorentz Fit\nReduced χ² = {chi2_red:.2f}')
    ax2.set_ylabel("Counts")
    ax2.set_title("C6H6 - 991.7 cm^-1 Peak")
    ax2.legend()

    ax2_res.scatter(x, residuals)
    ax2_res.axhline(0, color='k')
    ax2_res.set_title("Residuals")

    plt.tight_layout()
    plt.savefig("991 Peak.png", dpi = 300)
    plt.show()