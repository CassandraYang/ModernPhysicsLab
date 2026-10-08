import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ramanShift = [] #cm^-1
counts = []
with open(f"Raman/CCl4/CSV/Polarization - -80.csv", newline = '') as f:
    for line in f:
        if line.strip() == "WL,Power":
            break
        
    reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
    for row in reader:
        ramanShift.append(float(row['WL']))
        counts.append(float(row["Power"]))
    
ramanShift = np.array(ramanShift)
counts = np.array(counts)
counts_err = np.sqrt(counts)

 # --- Plot with Matplotlib ---
plt.scatter(ramanShift, counts, color='blue', label='CCl4 Spectrum', s = 5)
plt.errorbar(ramanShift, counts, counts_err, color='blue', fmt = 'None')
plt.xlabel("Raman Shift (cm^-1)")
plt.ylabel("Counts")
plt.title("CCl4 - Spectrum")
plt.legend()

plt.savefig("Raman/CCl4/Plots/CCl4 - Spectrum.png")
plt.show()