import csv
import numpy as np
import matplotlib.pyplot as plt

energy = []
energy_err = []
nothing = []
nothing_err = []
lucite = []
lucite_err = []
copper = []
copper_err = []

with open('Mossbauer/Mossbauer - Rate vs E.csv', newline='') as f:
    reader = csv.DictReader(f)
    count = 0
    for row in reader:
        energy.append(float(row['E (Volts)']))
        energy_err.append(float(row['E Error']))

        nothing.append(float(row['Count Rate - Nothing (events/second)']))
        nothing_err.append(float(row['Nothing Rate Error']))

        lucite.append(float(row['Count Rate - Lucite (events/second)']))
        lucite_err.append(float(row['Lucite Rate Error']))

        copper.append(float(row['Count Rate - Copper (events/second)']))
        copper_err.append(float(row['Copper Rate Error']))
        
plt.figure()
plt.scatter(energy, nothing, label = 'Nothing')
plt.errorbar(energy, nothing, nothing_err, energy_err, fmt = 'None')

plt.scatter(energy, lucite, label = 'Lucite')
plt.errorbar(energy, lucite, lucite_err, energy_err, fmt = 'None', c = 'orange')

plt.scatter(energy, copper, label = 'Copper')
plt.errorbar(energy, copper, copper_err, energy_err, fmt = 'None', c = 'green')

plt.yscale('log')
plt.legend()
plt.title('Rate vs. E')
plt.xlabel('E (V)')
plt.ylabel('Counts')
plt.savefig('RateVsE_all.png')

plt.figure()
plt.scatter(energy, lucite, label = 'Lucite')
plt.errorbar(energy, lucite, lucite_err, energy_err, fmt = 'None')

plt.scatter(energy, copper, label = 'Copper')
plt.errorbar(energy, copper, copper_err, energy_err, fmt = 'None', c = 'orange')

plt.yscale('log')
plt.legend()
plt.title('Rate vs. E')
plt.xlabel('E (V)')
plt.ylabel('Counts')
plt.savefig('RateVsE_LC.png')