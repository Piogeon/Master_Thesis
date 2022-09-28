# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 14:25:35 2022

@author: pio-r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import AutoMinorLocator
from matplotlib.ticker import PercentFormatter

F1_gmm_OS = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/macro_f1_gmm_os.csv', dtype='str', keep_default_na=False)
F1_classic_OS = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/macro_f1_classic_os.csv', dtype='str', keep_default_na=False)
TP_gmm_OS = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/TP_GMM_OS.csv', dtype='str', keep_default_na=False)
TP_classic_OS = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/TP_classic_OS.csv', dtype='str', keep_default_na=False)

F1_gmm_OS["F1-score"] = F1_gmm_OS["F1-score"].astype(float)
F1_classic_OS["F1-score"] = F1_classic_OS["F1-score"].astype(float)
TP_gmm_OS["C"] = TP_gmm_OS["C"].astype(float)
TP_gmm_OS["E"] = TP_gmm_OS["E"].astype(float)
TP_gmm_OS["M"] = TP_gmm_OS["M"].astype(float)
TP_gmm_OS["P"] = TP_gmm_OS["P"].astype(float)
TP_gmm_OS["S"] = TP_gmm_OS["S"].astype(float)
TP_gmm_OS["V"] = TP_gmm_OS["V"].astype(float)

TP_classic_OS["C"] = TP_classic_OS["C"].astype(float)
TP_classic_OS["E"] = TP_classic_OS["E"].astype(float)
TP_classic_OS["M"] = TP_classic_OS["M"].astype(float)
TP_classic_OS["P"] = TP_classic_OS["P"].astype(float)
TP_classic_OS["S"] = TP_classic_OS["S"].astype(float)
TP_classic_OS["V"] = TP_classic_OS["V"].astype(float)

F1_gmm_OS.drop(F1_gmm_OS.columns[F1_gmm_OS.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
F1_classic_OS.drop(F1_classic_OS.columns[F1_classic_OS.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
TP_gmm_OS.drop(TP_gmm_OS.columns[TP_gmm_OS.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
TP_classic_OS.drop(TP_classic_OS.columns[TP_classic_OS.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

# Percentage increase for F1-score

f1_cl_no_OS = 0.08875403806796071
f1_cl_no_OS_std = 0.08718962478100255

f1_ml_no_OS = 0.15601997784284938
f1_ml_no_OS_std = 0.1029099173790253

f1_cl = np.mean(F1_classic_OS["F1-score"])
f1_cl_std = np.std(F1_classic_OS["F1-score"])

f1_ml = np.mean(F1_gmm_OS["F1-score"])
f1_ml_std = np.std(F1_gmm_OS["F1-score"])

f1 = plt.figure()
ax = f1.add_subplot(111)
ax.set_title('MACRO F1-score values', size=30) # Title
ax.label_outer()
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in',width=5, labelsize='15')
ax.tick_params(which='major', direction='in',length=10)
ax.tick_params(which='minor', direction='in',length=7)
ax.yaxis.set_ticks_position('both')
ax.set_xlabel('Classes', fontsize=30)
ax.set_ylabel('Percentage Variation', fontsize=30)
plt.bar("Classic", f1_cl_no_OS, align='center', yerr=f1_cl_no_OS_std, alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("GMM", f1_ml_no_OS, align='center', yerr=f1_ml_no_OS_std, alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("Classic_OS", f1_cl, align='center', yerr=f1_cl_std, alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("GMM_OS", f1_ml, align='center', yerr=f1_ml_std, alpha=1,
       ecolor='black',
       capsize=10)
plt.subplots_adjust(top=0.943,
bottom=0.089,
left=0.058,
right=0.992,
hspace=0.2,
wspace=0.2)
plt.show()

perc_increse = ((f1_ml - f1_cl)/f1_cl)*100 
perc_increase_std = np.sqrt(np.power(f1_ml_std/f1_ml, 2) + np.power(f1_cl_std/f1_cl, 2))*(f1_ml/f1_cl)*100

print('The usage of GMM as feature augmentation increased the Macro F1-score of {} ± {} %'.format(perc_increse, perc_increase_std))

perc_gmm = ((f1_ml - f1_ml_no_OS)/f1_ml_no_OS)*100 
perc_gmm_std = ((f1_ml_std - f1_ml_no_OS_std)/f1_ml_no_OS_std)*100 

perc_class = ((f1_ml_no_OS - f1_cl_no_OS)/f1_cl_no_OS)*100 
perc_tp = pd.DataFrame(columns=["Class", "Classic_OS", "GMM_OS"])

perc_tp["Class"] = ["C", "E", "M", "P", "S", "V"]
perc_tp["Classic_OS"] = [np.mean(TP_classic_OS["C"]), np.mean(TP_classic_OS["E"]), np.mean(TP_classic_OS["M"]),
                         np.mean(TP_classic_OS["P"]), np.mean(TP_classic_OS["S"]), np.mean(TP_classic_OS["V"])]
perc_tp["GMM_OS"] = [np.mean(TP_gmm_OS["C"]), np.mean(TP_gmm_OS["E"]), np.mean(TP_gmm_OS["M"]),
                         np.mean(TP_gmm_OS["P"]), np.mean(TP_gmm_OS["S"]), np.mean(TP_gmm_OS["V"])]

tp_increase = pd.DataFrame(columns=["Class", "Percentage Variation", "Std"])
tp_increase["Class"] = ["C", "E", "M", "P", "S", "V"]
tp_increase["Percentage Variation"] = [(np.mean(TP_gmm_OS["C"])-np.mean(TP_classic_OS["C"]))/np.mean(TP_classic_OS["C"]),
                                       (np.mean(TP_gmm_OS["E"])-np.mean(TP_classic_OS["E"]))/np.mean(TP_classic_OS["E"]),
                                       (np.mean(TP_gmm_OS["M"])-np.mean(TP_classic_OS["M"]))/np.mean(TP_classic_OS["M"]),
                                       (np.mean(TP_gmm_OS["P"])-np.mean(TP_classic_OS["P"]))/np.mean(TP_classic_OS["P"]),
                                       (np.mean(TP_gmm_OS["S"])-np.mean(TP_classic_OS["S"]))/np.mean(TP_classic_OS["S"]),
                                       (np.mean(TP_gmm_OS["V"])-np.mean(TP_classic_OS["V"]))/np.mean(TP_classic_OS["V"])]
tp_increase["Std"] = [np.sqrt(np.power(np.std(TP_gmm_OS["C"])/np.mean(TP_gmm_OS["C"]), 2) + np.power(np.std(TP_classic_OS["C"])/np.mean(TP_classic_OS["C"]), 2))*(np.mean(TP_gmm_OS["C"])/np.mean(TP_classic_OS["C"])),
                                       np.sqrt(np.power(np.std(TP_gmm_OS["E"])/np.mean(TP_gmm_OS["E"]), 2) + np.power(np.std(TP_classic_OS["E"])/np.mean(TP_classic_OS["E"]), 2))*(np.mean(TP_gmm_OS["E"])/np.mean(TP_classic_OS["E"])),
                                       np.sqrt(np.power(np.std(TP_gmm_OS["M"])/np.mean(TP_gmm_OS["M"]), 2) + np.power(np.std(TP_classic_OS["M"])/np.mean(TP_classic_OS["M"]), 2))*(np.mean(TP_gmm_OS["M"])/np.mean(TP_classic_OS["M"])),
                                       np.sqrt(np.power(np.std(TP_gmm_OS["P"])/np.mean(TP_gmm_OS["P"]), 2) + np.power(np.std(TP_classic_OS["P"])/np.mean(TP_classic_OS["P"]), 2))*(np.mean(TP_gmm_OS["P"])/np.mean(TP_classic_OS["P"])),
                                       np.sqrt(np.power(np.std(TP_gmm_OS["S"])/np.mean(TP_gmm_OS["S"]), 2) + np.power(np.std(TP_classic_OS["S"])/np.mean(TP_classic_OS["S"]), 2))*(np.mean(TP_gmm_OS["S"])/np.mean(TP_classic_OS["S"])),
                                       np.sqrt(np.power(np.std(TP_gmm_OS["V"])/np.mean(TP_gmm_OS["V"]), 2) + np.power(np.std(TP_classic_OS["V"])/np.mean(TP_classic_OS["V"]), 2))*(np.mean(TP_gmm_OS["V"])/np.mean(TP_classic_OS["V"]))]

f1 = plt.figure(3)
ax = f1.add_subplot(111)
ax.set_title('Percentage variation of Machine Learning in relation to classical analysis', size=30) # Title
ax.label_outer()
ax.yaxis.set_minor_locator(AutoMinorLocator())
ax.tick_params(which='both', direction='in',width=5, labelsize='15')
ax.tick_params(which='major', direction='in',length=10)
ax.tick_params(which='minor', direction='in',length=7)
ax.yaxis.set_ticks_position('both')
ax.set_xlabel('Classes', fontsize=30)
ax.set_ylabel('Percentage Variation', fontsize=30)
ax.yaxis.set_major_formatter(PercentFormatter(1))
plt.grid()
plt.bar("C", tp_increase["Percentage Variation"][0], align='center', yerr=tp_increase["Std"][0], alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("E", tp_increase["Percentage Variation"][1], align='center', yerr=tp_increase["Std"][1], alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("M", tp_increase["Percentage Variation"][2], align='center', yerr=tp_increase["Std"][2], alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("P", tp_increase["Percentage Variation"][3], align='center', yerr=tp_increase["Std"][3], alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("S", tp_increase["Percentage Variation"][4], align='center', yerr=tp_increase["Std"][4], alpha=1,
       ecolor='black',
       capsize=10)
plt.bar("V", tp_increase["Percentage Variation"][5], align='center', yerr=tp_increase["Std"][5], alpha=1,
       ecolor='black',
       capsize=10)
plt.subplots_adjust(top=0.943,
bottom=0.089,
left=0.058,
right=0.992,
hspace=0.2,
wspace=0.2)
plt.show()