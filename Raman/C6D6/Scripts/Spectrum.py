import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ramanShift = [] #cm^-1
counts = []
with open(f"Raman/C6D6/CSV/Polarization - 15.csv", newline = '') as f:
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
plt.scatter(ramanShift, counts, color='blue', label='C6H6 Spectrum', s = 5)
plt.errorbar(ramanShift, counts, counts_err, color='blue', fmt = 'None')
plt.xlabel("Raman Shift (cm^-1)")
plt.ylabel("Counts")
plt.title("C6D6 - Spectrum")
plt.legend()

plt.show()
# plt.savefig("Raman/C6D6/Plots/C6D6 - Spectrum.png")