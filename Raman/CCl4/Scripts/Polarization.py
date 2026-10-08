import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

ramanShift1 = [] #cm^-1
counts1 = []
with open(f"Raman/CCl4/CSV/Polarization - 45.csv", newline = '') as f:
    for line in f:
        if line.strip() == "WL,Power":
            break
        
    reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
    for row in reader:
        ramanShift1.append(float(row['WL']))
        counts1.append(float(row["Power"]))
    
ramanShift1 = np.array(ramanShift1)
counts1 = np.array(counts1)
counts_err1 = np.sqrt(counts1)

ramanShift2 = [] #cm^-1
counts2 = []
with open(f"Raman/CCl4/CSV/Polarization - -55.csv", newline = '') as f:
    for line in f:
        if line.strip() == "WL,Power":
            break
        
    reader = csv.DictReader(f, fieldnames=['WL', 'Power'])
    for row in reader:
        ramanShift2.append(float(row['WL']))
        counts2.append(float(row["Power"]))
    
ramanShift2 = np.array(ramanShift2)
counts2 = np.array(counts2)
counts_err2 = np.sqrt(counts2)

 # --- Plot with Matplotlib ---
plt.scatter(ramanShift1, counts1, color='blue', label='CCl4 Spectrum - Parallel', s = 5)
plt.errorbar(ramanShift1, counts1, counts_err1, color='blue', fmt = 'None')
plt.scatter(ramanShift2, counts2, color='red', label='CCl4 Spectrum - Perpendicular', s = 5)
plt.errorbar(ramanShift2, counts2, counts_err2, color='red', fmt = 'None')
plt.xlabel("Raman Shift (cm^-1)")
plt.ylabel("Counts")
plt.title("CCl4 - Spectrum")
plt.legend()

plt.savefig("Raman/CCl4/Plots/CCl4 - Polarization.png")
plt.show()