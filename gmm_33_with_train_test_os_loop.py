# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:59:24 2022

@author: pio-r
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN 
from imblearn.over_sampling import SVMSMOTE 

def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
    """Plots a confusion matrix."""
    if classes is not None:
        sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size': 50})
    else:
        sns.heatmap(cm, vmin=0., vmax=1.)
    plt.title(title, fontsize=30)
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)


sr_aeiH = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/tax_SR_ratio.csv', dtype='str', keep_default_na=False)

sr_aeiH["a"] = sr_aeiH["a"].astype(float)
sr_aeiH["e"] = sr_aeiH["e"].astype(float)
sr_aeiH["i"] = sr_aeiH["i"].astype(float)
sr_aeiH["H"] = sr_aeiH["H"].astype(float)
sr_aeiH["nu6"] = sr_aeiH["nu6"].astype(float)
sr_aeiH["5/2"] = sr_aeiH["5/2"].astype(float)
sr_aeiH["2/1"] = sr_aeiH["2/1"].astype(float)
sr_aeiH["HUN"] = sr_aeiH["HUN"].astype(float)
sr_aeiH["3/1"] = sr_aeiH["3/1"].astype(float)
sr_aeiH["PHO"] = sr_aeiH["PHO"].astype(float)
sr_aeiH["JFC"] = sr_aeiH["JFC"].astype(float)

sr_aeiH.drop(sr_aeiH.columns[sr_aeiH.columns.str.contains('unnamed', case = False)], axis = 1, inplace = True)

X = sr_aeiH[["nu6", "5/2", "2/1", "HUN", "3/1", "PHO", "JFC"]]

y = sr_aeiH["Taxon"]

# Percentage of SRs

sr_hist = pd.DataFrame()
sr_hist["SR"] = ["HUN", "NU6", "PHO", "3/1", "2/1", "5/2", "JFC"]
sr_hist["Percentage"] = [np.mean(sr_aeiH["HUN"]), np.mean(sr_aeiH["nu6"]), np.mean(sr_aeiH["PHO"]),
                         np.mean(sr_aeiH["3/1"]), np.mean(sr_aeiH["2/1"]), np.mean(sr_aeiH["5/2"]),
                         np.mean(sr_aeiH["JFC"])]

macro_f1 = []
tp_perc = []
pred = []
true = []

for i in tqdm(range(0, 100)):
    sm = SVMSMOTE(random_state=i)
    # oversample
    x_ros, y_ros = sm.fit_resample(X, y)
    gm = GaussianMixture(n_components=33, covariance_type='diag', 
                     random_state=10)

    gm.fit(x_ros)
    cluster = gm.predict_proba(x_ros)
    
    tr = pd.DataFrame(cluster)
    
    res = pd.DataFrame()
    
    # res["Number"] = sr_aeiH["Number"]
    # res["Name"] = sr_aeiH["Name"]
    # res["Prov.Desig"] = sr_aeiH["Prov.Desig"]
    res["Taxon"] = y_ros
    res["0"] = tr[0]
    res["1"] = tr[1]
    res["2"] = tr[2]
    res["3"] = tr[3]
    res["4"] = tr[4]
    res["5"] = tr[5]
    res["6"] = tr[6]
    res["7"] = tr[7]
    res["8"] = tr[8]
    res["9"] = tr[9]
    res["10"] = tr[10]
    res["11"] = tr[11]
    res["12"] = tr[12]
    res["13"] = tr[13]
    res["14"] = tr[14]
    res["15"] = tr[15]
    res["16"] = tr[16]
    res["17"] = tr[17]
    res["18"] = tr[18]
    res["19"] = tr[19]
    res["20"] = tr[20]
    res["21"] = tr[21]
    res["22"] = tr[22]
    res["23"] = tr[23]
    res["24"] = tr[24]
    res["25"] = tr[25]
    res["26"] = tr[26]
    res["27"] = tr[27]
    res["28"] = tr[28]
    res["29"] = tr[29]
    res["30"] = tr[30]
    res["31"] = tr[31]
    res["32"] = tr[32]
    
    CL_hist = pd.DataFrame()
    
    CL_hist["Cluster"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    CL_hist["Percentage"] = [np.mean(res["0"]), np.mean(res["1"]), np.mean(res["2"]),
                             np.mean(res["3"]), np.mean(res["4"]), np.mean(res["5"]),
                             np.mean(res["6"]), np.mean(res["7"]), np.mean(res["8"]),
                             np.mean(res["9"]), np.mean(res["10"]), np.mean(res["11"]),
                             np.mean(res["12"]), np.mean(res["13"]), np.mean(res["14"]),
                             np.mean(res["15"]), np.mean(res["16"]), np.mean(res["17"]),
                             np.mean(res["18"]), np.mean(res["19"]), np.mean(res["20"]),
                             np.mean(res["21"]), np.mean(res["22"]), np.mean(res["23"]),
                             np.mean(res["24"]), np.mean(res["25"]), np.mean(res["26"]),
                             np.mean(res["27"]), np.mean(res["28"]), np.mean(res["29"]),
                             np.mean(res["30"]), np.mean(res["31"]), np.mean(res["32"])]
    
    res_train, res_test = train_test_split(res, train_size = 0.7, test_size=0.3, random_state=0)
    
    
    tax_E = res_train.loc[res_train["Taxon"] == "E"]
    tax_S = res_train.loc[res_train["Taxon"] == "S"]
    tax_V = res_train.loc[res_train["Taxon"] == "V"]
    tax_C = res_train.loc[res_train["Taxon"] == "C"]
    tax_M = res_train.loc[res_train["Taxon"] == "M"]
    tax_P = res_train.loc[res_train["Taxon"] == "P"]
    
    SR_E = pd.DataFrame()
    SR_E["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_E["Percentage"] = [np.mean(tax_E["0"]), np.mean(tax_E["1"]), np.mean(tax_E["2"]),
            np.mean(tax_E["3"]), np.mean(tax_E["4"]), np.mean(tax_E["5"]), np.mean(tax_E["6"]),
            np.mean(tax_E["7"]), np.mean(tax_E["8"]), np.mean(tax_E["9"]), np.mean(tax_E["10"]),
            np.mean(tax_E["11"]), np.mean(tax_E["12"]), np.mean(tax_E["13"]), np.mean(tax_E["14"]),
            np.mean(tax_E["15"]), np.mean(tax_E["16"]), np.mean(tax_E["17"]), np.mean(tax_E["18"]),
            np.mean(tax_E["19"]), np.mean(tax_E["20"]), np.mean(tax_E["21"]), np.mean(tax_E["22"]),
            np.mean(tax_E["23"]), np.mean(tax_E["24"]), np.mean(tax_E["25"]), np.mean(tax_E["26"]),
            np.mean(tax_E["27"]), np.mean(tax_E["28"]), np.mean(tax_E["29"]), np.mean(tax_E["30"]),
            np.mean(tax_E["31"]), np.mean(tax_E["32"])]
    SR_E["Std"] = [np.std(tax_E["0"])/np.sqrt(len(tax_E)), np.std(tax_E["1"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["2"])/np.sqrt(len(tax_E)), np.std(tax_E["3"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["4"])/np.sqrt(len(tax_E)), np.std(tax_E["5"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["6"])/np.sqrt(len(tax_E)), np.std(tax_E["7"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["8"])/np.sqrt(len(tax_E)), np.std(tax_E["9"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["10"])/np.sqrt(len(tax_E)), np.std(tax_E["11"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["12"])/np.sqrt(len(tax_E)), np.std(tax_E["13"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["14"])/np.sqrt(len(tax_E)), np.std(tax_E["15"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["16"])/np.sqrt(len(tax_E)), np.std(tax_E["17"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["18"])/np.sqrt(len(tax_E)), np.std(tax_E["19"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["20"])/np.sqrt(len(tax_E)), np.std(tax_E["21"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["22"])/np.sqrt(len(tax_E)), np.std(tax_E["23"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["24"])/np.sqrt(len(tax_E)), np.std(tax_E["25"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["26"])/np.sqrt(len(tax_E)), np.std(tax_E["27"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["28"])/np.sqrt(len(tax_E)), np.std(tax_E["29"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["30"])/np.sqrt(len(tax_E)), np.std(tax_E["31"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["32"])/np.sqrt(len(tax_E))]
    
    SR_S = pd.DataFrame()
    SR_S["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_S["Percentage"] = [np.mean(tax_S["0"]), np.mean(tax_S["1"]), np.mean(tax_S["2"]),
            np.mean(tax_S["3"]), np.mean(tax_S["4"]), np.mean(tax_S["5"]), np.mean(tax_S["6"]),
            np.mean(tax_S["7"]), np.mean(tax_S["8"]), np.mean(tax_S["9"]), np.mean(tax_S["10"]),
            np.mean(tax_S["11"]), np.mean(tax_S["12"]), np.mean(tax_S["13"]), np.mean(tax_S["14"]),
            np.mean(tax_S["15"]), np.mean(tax_S["16"]), np.mean(tax_S["17"]), np.mean(tax_S["18"]),
            np.mean(tax_S["19"]), np.mean(tax_S["20"]), np.mean(tax_S["21"]), np.mean(tax_S["22"]),
            np.mean(tax_S["23"]), np.mean(tax_S["24"]), np.mean(tax_S["25"]), np.mean(tax_S["26"]),
            np.mean(tax_S["27"]), np.mean(tax_S["28"]), np.mean(tax_S["29"]), np.mean(tax_S["30"]),
            np.mean(tax_S["31"]), np.mean(tax_S["32"])]
    SR_S["Std"] = [np.std(tax_S["0"])/np.sqrt(len(tax_S)), np.std(tax_S["1"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["2"])/np.sqrt(len(tax_S)), np.std(tax_S["3"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["4"])/np.sqrt(len(tax_S)), np.std(tax_S["5"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["6"])/np.sqrt(len(tax_S)), np.std(tax_S["7"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["8"])/np.sqrt(len(tax_S)), np.std(tax_S["9"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["10"])/np.sqrt(len(tax_S)), np.std(tax_S["11"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["12"])/np.sqrt(len(tax_S)), np.std(tax_S["13"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["14"])/np.sqrt(len(tax_S)), np.std(tax_S["15"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["16"])/np.sqrt(len(tax_S)), np.std(tax_S["17"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["18"])/np.sqrt(len(tax_S)), np.std(tax_S["19"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["20"])/np.sqrt(len(tax_S)), np.std(tax_S["21"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["22"])/np.sqrt(len(tax_S)), np.std(tax_S["23"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["24"])/np.sqrt(len(tax_S)), np.std(tax_S["25"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["26"])/np.sqrt(len(tax_S)), np.std(tax_S["27"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["28"])/np.sqrt(len(tax_S)), np.std(tax_S["29"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["30"])/np.sqrt(len(tax_S)), np.std(tax_S["31"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["32"])/np.sqrt(len(tax_S))]
    
    SR_V = pd.DataFrame()
    SR_V["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_V["Percentage"] = [np.mean(tax_V["0"]), np.mean(tax_V["1"]), np.mean(tax_V["2"]),
            np.mean(tax_V["3"]), np.mean(tax_V["4"]), np.mean(tax_V["5"]), np.mean(tax_V["6"]),
            np.mean(tax_V["7"]), np.mean(tax_V["8"]), np.mean(tax_V["9"]), np.mean(tax_V["10"]),
            np.mean(tax_V["11"]), np.mean(tax_V["12"]), np.mean(tax_V["13"]), np.mean(tax_V["14"]),
            np.mean(tax_V["15"]), np.mean(tax_V["16"]), np.mean(tax_V["17"]), np.mean(tax_V["18"]),
            np.mean(tax_V["19"]), np.mean(tax_V["20"]), np.mean(tax_V["21"]), np.mean(tax_V["22"]),
            np.mean(tax_V["23"]), np.mean(tax_V["24"]), np.mean(tax_V["25"]), np.mean(tax_V["26"]),
            np.mean(tax_V["27"]), np.mean(tax_V["28"]), np.mean(tax_V["29"]), np.mean(tax_V["30"]),
            np.mean(tax_V["31"]), np.mean(tax_V["32"])]
    SR_V["Std"] = [np.std(tax_V["0"])/np.sqrt(len(tax_V)), np.std(tax_V["1"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["2"])/np.sqrt(len(tax_V)), np.std(tax_V["3"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["4"])/np.sqrt(len(tax_V)), np.std(tax_V["5"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["6"])/np.sqrt(len(tax_V)), np.std(tax_V["7"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["8"])/np.sqrt(len(tax_V)), np.std(tax_V["9"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["10"])/np.sqrt(len(tax_V)), np.std(tax_V["11"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["12"])/np.sqrt(len(tax_V)), np.std(tax_V["13"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["14"])/np.sqrt(len(tax_V)), np.std(tax_V["15"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["16"])/np.sqrt(len(tax_V)), np.std(tax_V["17"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["18"])/np.sqrt(len(tax_V)), np.std(tax_V["19"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["20"])/np.sqrt(len(tax_V)), np.std(tax_V["21"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["22"])/np.sqrt(len(tax_V)), np.std(tax_V["23"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["24"])/np.sqrt(len(tax_V)), np.std(tax_V["25"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["26"])/np.sqrt(len(tax_V)), np.std(tax_V["27"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["28"])/np.sqrt(len(tax_V)), np.std(tax_V["29"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["30"])/np.sqrt(len(tax_V)), np.std(tax_V["31"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["32"])/np.sqrt(len(tax_V))]
    
    SR_C = pd.DataFrame()
    SR_C["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_C["Percentage"] = [np.mean(tax_C["0"]), np.mean(tax_C["1"]), np.mean(tax_C["2"]),
            np.mean(tax_C["3"]), np.mean(tax_C["4"]), np.mean(tax_C["5"]), np.mean(tax_C["6"]),
            np.mean(tax_C["7"]), np.mean(tax_C["8"]), np.mean(tax_C["9"]), np.mean(tax_C["10"]),
            np.mean(tax_C["11"]), np.mean(tax_C["12"]), np.mean(tax_C["13"]), np.mean(tax_C["14"]),
            np.mean(tax_C["15"]), np.mean(tax_C["16"]), np.mean(tax_C["17"]), np.mean(tax_C["18"]),
            np.mean(tax_C["19"]), np.mean(tax_C["20"]), np.mean(tax_C["21"]), np.mean(tax_C["22"]),
            np.mean(tax_C["23"]), np.mean(tax_C["24"]), np.mean(tax_C["25"]), np.mean(tax_C["26"]),
            np.mean(tax_C["27"]), np.mean(tax_C["28"]), np.mean(tax_C["29"]), np.mean(tax_C["30"]),
            np.mean(tax_C["31"]), np.mean(tax_C["32"])]
    SR_C["Std"] = [np.std(tax_C["0"])/np.sqrt(len(tax_C)), np.std(tax_C["1"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["2"])/np.sqrt(len(tax_C)), np.std(tax_C["3"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["4"])/np.sqrt(len(tax_C)), np.std(tax_C["5"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["6"])/np.sqrt(len(tax_C)), np.std(tax_C["7"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["8"])/np.sqrt(len(tax_C)), np.std(tax_C["9"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["10"])/np.sqrt(len(tax_C)), np.std(tax_C["11"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["12"])/np.sqrt(len(tax_C)), np.std(tax_C["13"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["14"])/np.sqrt(len(tax_C)), np.std(tax_C["15"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["16"])/np.sqrt(len(tax_C)), np.std(tax_C["17"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["18"])/np.sqrt(len(tax_C)), np.std(tax_C["19"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["20"])/np.sqrt(len(tax_C)), np.std(tax_C["21"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["22"])/np.sqrt(len(tax_C)), np.std(tax_C["23"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["24"])/np.sqrt(len(tax_C)), np.std(tax_C["25"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["26"])/np.sqrt(len(tax_C)), np.std(tax_C["27"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["28"])/np.sqrt(len(tax_C)), np.std(tax_C["29"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["30"])/np.sqrt(len(tax_C)), np.std(tax_C["31"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["32"])/np.sqrt(len(tax_C))]
    
    SR_M = pd.DataFrame()
    SR_M["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_M["Percentage"] = [np.mean(tax_M["0"]), np.mean(tax_M["1"]), np.mean(tax_M["2"]),
            np.mean(tax_M["3"]), np.mean(tax_M["4"]), np.mean(tax_M["5"]), np.mean(tax_M["6"]),
            np.mean(tax_M["7"]), np.mean(tax_M["8"]), np.mean(tax_M["9"]), np.mean(tax_M["10"]),
            np.mean(tax_M["11"]), np.mean(tax_M["12"]), np.mean(tax_M["13"]), np.mean(tax_M["14"]),
            np.mean(tax_M["15"]), np.mean(tax_M["16"]), np.mean(tax_M["17"]), np.mean(tax_M["18"]),
            np.mean(tax_M["19"]), np.mean(tax_M["20"]), np.mean(tax_M["21"]), np.mean(tax_M["22"]),
            np.mean(tax_M["23"]), np.mean(tax_M["24"]), np.mean(tax_M["25"]), np.mean(tax_M["26"]),
            np.mean(tax_M["27"]), np.mean(tax_M["28"]), np.mean(tax_M["29"]), np.mean(tax_M["30"]),
            np.mean(tax_M["31"]), np.mean(tax_M["32"])]
    SR_M["Std"] = [np.std(tax_M["0"])/np.sqrt(len(tax_M)), np.std(tax_M["1"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["2"])/np.sqrt(len(tax_M)), np.std(tax_M["3"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["4"])/np.sqrt(len(tax_M)), np.std(tax_M["5"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["6"])/np.sqrt(len(tax_M)), np.std(tax_M["7"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["8"])/np.sqrt(len(tax_M)), np.std(tax_M["9"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["10"])/np.sqrt(len(tax_M)), np.std(tax_M["11"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["12"])/np.sqrt(len(tax_M)), np.std(tax_M["13"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["14"])/np.sqrt(len(tax_M)), np.std(tax_M["15"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["16"])/np.sqrt(len(tax_M)), np.std(tax_M["17"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["18"])/np.sqrt(len(tax_M)), np.std(tax_M["19"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["20"])/np.sqrt(len(tax_M)), np.std(tax_M["21"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["22"])/np.sqrt(len(tax_M)), np.std(tax_M["23"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["24"])/np.sqrt(len(tax_M)), np.std(tax_M["25"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["26"])/np.sqrt(len(tax_M)), np.std(tax_M["27"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["28"])/np.sqrt(len(tax_M)), np.std(tax_M["29"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["30"])/np.sqrt(len(tax_M)), np.std(tax_M["31"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["32"])/np.sqrt(len(tax_M))]
    SR_P = pd.DataFrame()
    SR_P["SR"] = ["0", "1", "2", "3", "4", "5", "6", "7", "8",
                          "9", "10", "11", "12", "13", "14", "15", "16",
                          "17", "18", "19", "20", "21", "22", "23", "24",
                          "25", "26", "27", "28", "29", "30", "31", "32"]
    SR_P["Percentage"] = [np.mean(tax_P["0"]), np.mean(tax_P["1"]), np.mean(tax_P["2"]),
            np.mean(tax_P["3"]), np.mean(tax_P["4"]), np.mean(tax_P["5"]), np.mean(tax_P["6"]),
            np.mean(tax_P["7"]), np.mean(tax_P["8"]), np.mean(tax_P["9"]), np.mean(tax_P["10"]),
            np.mean(tax_P["11"]), np.mean(tax_P["12"]), np.mean(tax_P["13"]), np.mean(tax_P["14"]),
            np.mean(tax_P["15"]), np.mean(tax_P["16"]), np.mean(tax_P["17"]), np.mean(tax_P["18"]),
            np.mean(tax_P["19"]), np.mean(tax_P["20"]), np.mean(tax_P["21"]), np.mean(tax_P["22"]),
            np.mean(tax_P["23"]), np.mean(tax_P["24"]), np.mean(tax_P["25"]), np.mean(tax_P["26"]),
            np.mean(tax_P["27"]), np.mean(tax_P["28"]), np.mean(tax_P["29"]), np.mean(tax_P["30"]),
            np.mean(tax_P["31"]), np.mean(tax_P["32"])]
    SR_P["Std"] = [np.std(tax_P["0"])/np.sqrt(len(tax_P)), np.std(tax_P["1"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["2"])/np.sqrt(len(tax_P)), np.std(tax_P["3"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["4"])/np.sqrt(len(tax_P)), np.std(tax_P["5"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["6"])/np.sqrt(len(tax_P)), np.std(tax_P["7"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["8"])/np.sqrt(len(tax_P)), np.std(tax_P["9"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["10"])/np.sqrt(len(tax_P)), np.std(tax_P["11"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["12"])/np.sqrt(len(tax_P)), np.std(tax_P["13"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["14"])/np.sqrt(len(tax_P)), np.std(tax_P["15"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["16"])/np.sqrt(len(tax_P)), np.std(tax_P["17"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["18"])/np.sqrt(len(tax_P)), np.std(tax_P["19"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["20"])/np.sqrt(len(tax_P)), np.std(tax_P["21"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["22"])/np.sqrt(len(tax_P)), np.std(tax_P["23"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["24"])/np.sqrt(len(tax_P)), np.std(tax_P["25"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["26"])/np.sqrt(len(tax_P)), np.std(tax_P["27"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["28"])/np.sqrt(len(tax_P)), np.std(tax_P["29"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["30"])/np.sqrt(len(tax_P)), np.std(tax_P["31"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["32"])/np.sqrt(len(tax_P))]
    
    R = pd.DataFrame()
    R["Class"] = ["E", "Std_E", "S", "Std_S", "V", "Std_V", "C", "Std_C", "M", "Std_M", "P", "Std_P"]
    R["0"] = [SR_E["Percentage"][0],
              SR_E["Std"][0],
                SR_S["Percentage"][0],
                SR_S["Std"][0],
                SR_V["Percentage"][0],
                SR_V["Std"][0],
                SR_C["Percentage"][0],
                SR_C["Std"][0],
                SR_M["Percentage"][0],
                SR_M["Std"][0],
                SR_P["Percentage"][0],
                SR_P["Std"][0],
                ]
    R["1"] = [SR_E["Percentage"][1],
              SR_E["Std"][1],
                SR_S["Percentage"][1],
                SR_S["Std"][1],
                SR_V["Percentage"][1],
                SR_V["Std"][1],
                SR_C["Percentage"][1],
                SR_C["Std"][1],
                SR_M["Percentage"][1],
                SR_M["Std"][1],
                SR_P["Percentage"][1],
                SR_P["Std"][1],
                ]
    R["2"] = [SR_E["Percentage"][2],
              SR_E["Std"][2],
                SR_S["Percentage"][2],
                SR_S["Std"][2],
                SR_V["Percentage"][2],
                SR_V["Std"][2],
                SR_C["Percentage"][2],
                SR_C["Std"][2],
                SR_M["Percentage"][2],
                SR_M["Std"][2],
                SR_P["Percentage"][2],
                SR_P["Std"][2],
                ]
    R["3"] = [SR_E["Percentage"][3],
              SR_E["Std"][3],
                SR_S["Percentage"][3],
                SR_S["Std"][3],
                SR_V["Percentage"][3],
                SR_V["Std"][3],
                SR_C["Percentage"][3],
                SR_C["Std"][3],
                SR_M["Percentage"][3],
                SR_M["Std"][3],
                SR_P["Percentage"][3],
                SR_P["Std"][3],
                ]
    R["4"] = [SR_E["Percentage"][4],
              SR_E["Std"][4],
                SR_S["Percentage"][4],
                SR_S["Std"][4],
                SR_V["Percentage"][4],
                SR_V["Std"][4],
                SR_C["Percentage"][4],
                SR_C["Std"][4],
                SR_M["Percentage"][4],
                SR_M["Std"][4],
                SR_P["Percentage"][4],
                SR_P["Std"][4],
                ]
    R["5"] = [SR_E["Percentage"][5],
              SR_E["Std"][5],
                SR_S["Percentage"][5],
                SR_S["Std"][5],
                SR_V["Percentage"][5],
                SR_V["Std"][5],
                SR_C["Percentage"][5],
                SR_C["Std"][5],
                SR_M["Percentage"][5],
                SR_M["Std"][5],
                SR_P["Percentage"][5],
                SR_P["Std"][5],
                ]
    R["6"] = [SR_E["Percentage"][6],
              SR_E["Std"][6],
                SR_S["Percentage"][6],
                SR_S["Std"][6],
                SR_V["Percentage"][6],
                SR_V["Std"][6],
                SR_C["Percentage"][6],
                SR_C["Std"][6],
                SR_M["Percentage"][6],
                SR_M["Std"][6],
                SR_P["Percentage"][6],
                SR_P["Std"][6],
                ]
    R["7"] = [SR_E["Percentage"][7],
              SR_E["Std"][7],
                SR_S["Percentage"][7],
                SR_S["Std"][7],
                SR_V["Percentage"][7],
                SR_V["Std"][7],
                SR_C["Percentage"][7],
                SR_C["Std"][7],
                SR_M["Percentage"][7],
                SR_M["Std"][7],
                SR_P["Percentage"][7],
                SR_P["Std"][7],
                ]
    R["8"] = [SR_E["Percentage"][8],
              SR_E["Std"][8],
                SR_S["Percentage"][8],
                SR_S["Std"][8],
                SR_V["Percentage"][8],
                SR_V["Std"][8],
                SR_C["Percentage"][8],
                SR_C["Std"][8],
                SR_M["Percentage"][8],
                SR_M["Std"][8],
                SR_P["Percentage"][8],
                SR_P["Std"][8],
                ]
    R["9"] = [SR_E["Percentage"][9],
              SR_E["Std"][9],
                SR_S["Percentage"][9],
                SR_S["Std"][9],
                SR_V["Percentage"][9],
                SR_V["Std"][9],
                SR_C["Percentage"][9],
                SR_C["Std"][9],
                SR_M["Percentage"][9],
                SR_M["Std"][9],
                SR_P["Percentage"][9],
                SR_P["Std"][9],
                ]
    R["10"] = [SR_E["Percentage"][10],
              SR_E["Std"][10],
                SR_S["Percentage"][10],
                SR_S["Std"][10],
                SR_V["Percentage"][10],
                SR_V["Std"][10],
                SR_C["Percentage"][10],
                SR_C["Std"][10],
                SR_M["Percentage"][10],
                SR_M["Std"][10],
                SR_P["Percentage"][10],
                SR_P["Std"][10],
                ]
    R["11"] = [SR_E["Percentage"][11],
              SR_E["Std"][11],
                SR_S["Percentage"][11],
                SR_S["Std"][11],
                SR_V["Percentage"][11],
                SR_V["Std"][11],
                SR_C["Percentage"][11],
                SR_C["Std"][11],
                SR_M["Percentage"][11],
                SR_M["Std"][11],
                SR_P["Percentage"][11],
                SR_P["Std"][11],
                ]
    R["12"] = [SR_E["Percentage"][12],
              SR_E["Std"][12],
                SR_S["Percentage"][12],
                SR_S["Std"][12],
                SR_V["Percentage"][12],
                SR_V["Std"][12],
                SR_C["Percentage"][12],
                SR_C["Std"][12],
                SR_M["Percentage"][12],
                SR_M["Std"][12],
                SR_P["Percentage"][12],
                SR_P["Std"][12],
                ]
    R["13"] = [SR_E["Percentage"][13],
              SR_E["Std"][13],
                SR_S["Percentage"][13],
                SR_S["Std"][13],
                SR_V["Percentage"][13],
                SR_V["Std"][13],
                SR_C["Percentage"][13],
                SR_C["Std"][13],
                SR_M["Percentage"][13],
                SR_M["Std"][13],
                SR_P["Percentage"][13],
                SR_P["Std"][13],
                ]
    R["14"] = [SR_E["Percentage"][14],
              SR_E["Std"][14],
                SR_S["Percentage"][14],
                SR_S["Std"][14],
                SR_V["Percentage"][14],
                SR_V["Std"][14],
                SR_C["Percentage"][14],
                SR_C["Std"][14],
                SR_M["Percentage"][14],
                SR_M["Std"][14],
                SR_P["Percentage"][14],
                SR_P["Std"][14],
                ]
    R["15"] = [SR_E["Percentage"][15],
              SR_E["Std"][15],
                SR_S["Percentage"][15],
                SR_S["Std"][15],
                SR_V["Percentage"][15],
                SR_V["Std"][15],
                SR_C["Percentage"][15],
                SR_C["Std"][15],
                SR_M["Percentage"][15],
                SR_M["Std"][15],
                SR_P["Percentage"][15],
                SR_P["Std"][15],
                ]
    R["16"] = [SR_E["Percentage"][16],
              SR_E["Std"][16],
                SR_S["Percentage"][16],
                SR_S["Std"][16],
                SR_V["Percentage"][16],
                SR_V["Std"][16],
                SR_C["Percentage"][16],
                SR_C["Std"][16],
                SR_M["Percentage"][16],
                SR_M["Std"][16],
                SR_P["Percentage"][16],
                SR_P["Std"][16],
                ]
    R["17"] = [SR_E["Percentage"][17],
              SR_E["Std"][17],
                SR_S["Percentage"][17],
                SR_S["Std"][17],
                SR_V["Percentage"][17],
                SR_V["Std"][17],
                SR_C["Percentage"][17],
                SR_C["Std"][17],
                SR_M["Percentage"][17],
                SR_M["Std"][17],
                SR_P["Percentage"][17],
                SR_P["Std"][17],
                ]
    R["18"] = [SR_E["Percentage"][18],
              SR_E["Std"][18],
                SR_S["Percentage"][18],
                SR_S["Std"][18],
                SR_V["Percentage"][18],
                SR_V["Std"][18],
                SR_C["Percentage"][18],
                SR_C["Std"][18],
                SR_M["Percentage"][18],
                SR_M["Std"][18],
                SR_P["Percentage"][18],
                SR_P["Std"][18],
                ]
    R["19"] = [SR_E["Percentage"][19],
              SR_E["Std"][19],
                SR_S["Percentage"][19],
                SR_S["Std"][19],
                SR_V["Percentage"][19],
                SR_V["Std"][19],
                SR_C["Percentage"][19],
                SR_C["Std"][19],
                SR_M["Percentage"][19],
                SR_M["Std"][19],
                SR_P["Percentage"][19],
                SR_P["Std"][19],
                ]
    R["20"] = [SR_E["Percentage"][20],
              SR_E["Std"][20],
                SR_S["Percentage"][20],
                SR_S["Std"][20],
                SR_V["Percentage"][20],
                SR_V["Std"][20],
                SR_C["Percentage"][20],
                SR_C["Std"][20],
                SR_M["Percentage"][20],
                SR_M["Std"][20],
                SR_P["Percentage"][20],
                SR_P["Std"][20],
                ]
    R["21"] = [SR_E["Percentage"][21],
              SR_E["Std"][21],
                SR_S["Percentage"][21],
                SR_S["Std"][21],
                SR_V["Percentage"][21],
                SR_V["Std"][21],
                SR_C["Percentage"][21],
                SR_C["Std"][21],
                SR_M["Percentage"][21],
                SR_M["Std"][21],
                SR_P["Percentage"][21],
                SR_P["Std"][21],
                ]
    R["22"] = [SR_E["Percentage"][22],
              SR_E["Std"][22],
                SR_S["Percentage"][22],
                SR_S["Std"][22],
                SR_V["Percentage"][22],
                SR_V["Std"][22],
                SR_C["Percentage"][22],
                SR_C["Std"][22],
                SR_M["Percentage"][22],
                SR_M["Std"][22],
                SR_P["Percentage"][22],
                SR_P["Std"][22],
                ]
    R["23"] = [SR_E["Percentage"][23],
              SR_E["Std"][23],
                SR_S["Percentage"][23],
                SR_S["Std"][23],
                SR_V["Percentage"][23],
                SR_V["Std"][23],
                SR_C["Percentage"][23],
                SR_C["Std"][23],
                SR_M["Percentage"][23],
                SR_M["Std"][23],
                SR_P["Percentage"][23],
                SR_P["Std"][23],
                ]
    R["24"] = [SR_E["Percentage"][24],
              SR_E["Std"][24],
                SR_S["Percentage"][24],
                SR_S["Std"][24],
                SR_V["Percentage"][24],
                SR_V["Std"][24],
                SR_C["Percentage"][24],
                SR_C["Std"][24],
                SR_M["Percentage"][24],
                SR_M["Std"][24],
                SR_P["Percentage"][24],
                SR_P["Std"][24],
                ]
    R["25"] = [SR_E["Percentage"][25],
              SR_E["Std"][25],
                SR_S["Percentage"][25],
                SR_S["Std"][25],
                SR_V["Percentage"][25],
                SR_V["Std"][25],
                SR_C["Percentage"][25],
                SR_C["Std"][25],
                SR_M["Percentage"][25],
                SR_M["Std"][25],
                SR_P["Percentage"][25],
                SR_P["Std"][25],
                ]
    R["26"] = [SR_E["Percentage"][26],
              SR_E["Std"][26],
                SR_S["Percentage"][26],
                SR_S["Std"][26],
                SR_V["Percentage"][26],
                SR_V["Std"][26],
                SR_C["Percentage"][26],
                SR_C["Std"][26],
                SR_M["Percentage"][26],
                SR_M["Std"][26],
                SR_P["Percentage"][26],
                SR_P["Std"][26],
                ]
    R["27"] = [SR_E["Percentage"][27],
              SR_E["Std"][27],
                SR_S["Percentage"][27],
                SR_S["Std"][27],
                SR_V["Percentage"][27],
                SR_V["Std"][27],
                SR_C["Percentage"][27],
                SR_C["Std"][27],
                SR_M["Percentage"][27],
                SR_M["Std"][27],
                SR_P["Percentage"][27],
                SR_P["Std"][27],
                ]
    R["28"] = [SR_E["Percentage"][28],
              SR_E["Std"][28],
                SR_S["Percentage"][28],
                SR_S["Std"][28],
                SR_V["Percentage"][28],
                SR_V["Std"][28],
                SR_C["Percentage"][28],
                SR_C["Std"][28],
                SR_M["Percentage"][28],
                SR_M["Std"][28],
                SR_P["Percentage"][28],
                SR_P["Std"][28],
                ]
    R["29"] = [SR_E["Percentage"][29],
              SR_E["Std"][29],
                SR_S["Percentage"][29],
                SR_S["Std"][29],
                SR_V["Percentage"][29],
                SR_V["Std"][29],
                SR_C["Percentage"][29],
                SR_C["Std"][29],
                SR_M["Percentage"][29],
                SR_M["Std"][29],
                SR_P["Percentage"][29],
                SR_P["Std"][29],
                ]
    R["30"] = [SR_E["Percentage"][30],
              SR_E["Std"][30],
                SR_S["Percentage"][30],
                SR_S["Std"][30],
                SR_V["Percentage"][30],
                SR_V["Std"][30],
                SR_C["Percentage"][30],
                SR_C["Std"][30],
                SR_M["Percentage"][30],
                SR_M["Std"][30],
                SR_P["Percentage"][30],
                SR_P["Std"][30],
                ]
    R["31"] = [SR_E["Percentage"][31],
              SR_E["Std"][31],
                SR_S["Percentage"][31],
                SR_S["Std"][31],
                SR_V["Percentage"][31],
                SR_V["Std"][31],
                SR_C["Percentage"][31],
                SR_C["Std"][31],
                SR_M["Percentage"][31],
                SR_M["Std"][31],
                SR_P["Percentage"][31],
                SR_P["Std"][31],
                ]
    R["32"] = [SR_E["Percentage"][32],
              SR_E["Std"][32],
                SR_S["Percentage"][32],
                SR_S["Std"][32],
                SR_V["Percentage"][32],
                SR_V["Std"][32],
                SR_C["Percentage"][32],
                SR_C["Std"][32],
                SR_M["Percentage"][32],
                SR_M["Std"][32],
                SR_P["Percentage"][32],
                SR_P["Std"][32],
                ]
    
    R_sr = R.T
    
    new_header = R_sr.iloc[0] #grab the first row for the header
    R_sr = R_sr[1:] #take the data less the header row
    R_sr.columns = new_header #set the header row as the df header
    
    R_sr["E"] = R_sr["E"].astype(float)
    R_sr["S"] = R_sr["S"].astype(float)
    R_sr["C"] = R_sr["C"].astype(float)
    R_sr["V"] = R_sr["V"].astype(float)
    R_sr["M"] = R_sr["M"].astype(float)
    R_sr["P"] = R_sr["P"].astype(float)
    R_sr["Std_E"] = R_sr["Std_E"].astype(float)
    R_sr["Std_S"] = R_sr["Std_S"].astype(float)
    R_sr["Std_V"] = R_sr["Std_V"].astype(float)
    R_sr["Std_C"] = R_sr["Std_C"].astype(float)
    R_sr["Std_M"] = R_sr["Std_M"].astype(float)
    R_sr["Std_P"] = R_sr["Std_P"].astype(float)
    
    R_sr_test = R_sr.copy()
    res_c = res_test.copy()
    res_c = res_c.reset_index(drop=True)
    res_train = res_train.reset_index(drop=True)
    
    prob_res = pd.DataFrame(columns=["True Tax", "E", "Std_E", "S", "Std_S",
                                     "V", "Std_V", "C", "Std_C", "M", "Std_M", "P", "Std_P"])
    
    prob_res["True Tax"] = res_c["Taxon"]
    
    for i in tqdm(range(len(res_c))):
        pr = res_c.loc[i]
        pr = pr[1:].astype(float)
        # pr = pr[1:].astype(float)
        E = []
        std_E = []
        S = []
        std_S = []
        V = []
        std_V = []
        C = []
        std_C = []
        M = []
        std_M = []
        P = []
        std_P = []
        for j in range(len(pr)):
          E.append(pr[j]*R_sr_test["E"].iloc[j])
          std_E.append(pr[j]*R_sr_test["Std_E"].iloc[j])
          S.append(pr[j]*R_sr_test["S"].iloc[j])
          std_S.append(pr[j]*R_sr_test["Std_S"].iloc[j])
          V.append(pr[j]*R_sr_test["V"].iloc[j])
          std_V.append(pr[j]*R_sr_test["Std_V"].iloc[j])
          C.append(pr[j]*R_sr_test["C"].iloc[j])
          std_C.append(pr[j]*R_sr_test["Std_C"].iloc[j])
          M.append(pr[j]*R_sr_test["M"].iloc[j])
          std_M.append(pr[j]*R_sr_test["Std_M"].iloc[j])
          P.append(pr[j]*R_sr_test["P"].iloc[j])
          std_P.append(pr[j]*R_sr_test["Std_P"].iloc[j])
        prob_res["E"].iloc[i] = np.sum(E)
        prob_res["Std_E"].iloc[i] = np.sum(std_E)
        prob_res["S"].iloc[i] = np.sum(S)
        prob_res["Std_S"].iloc[i] = np.sum(std_S)
        prob_res["V"].iloc[i] = np.sum(V)
        prob_res["Std_V"].iloc[i] = np.sum(std_V)
        prob_res["C"].iloc[i] = np.sum(C)
        prob_res["Std_C"].iloc[i] = np.sum(std_C)
        prob_res["M"].iloc[i] = np.sum(M)
        prob_res["Std_M"].iloc[i] = np.sum(std_M)
        prob_res["P"].iloc[i] = np.sum(P)
        prob_res["Std_P"].iloc[i] = np.sum(std_P)
        
    prob_res["E"] = prob_res["E"].astype(float)
    prob_res["S"] = prob_res["S"].astype(float)
    prob_res["C"] = prob_res["C"].astype(float)
    prob_res["V"] = prob_res["V"].astype(float)
    prob_res["M"] = prob_res["M"].astype(float)
    prob_res["P"] = prob_res["P"].astype(float)
    prob_res["Std_E"] = prob_res["Std_E"].astype(float)
    prob_res["Std_S"] = prob_res["Std_S"].astype(float)
    prob_res["Std_V"] = prob_res["Std_V"].astype(float)
    prob_res["Std_C"] = prob_res["Std_C"].astype(float)
    prob_res["Std_M"] = prob_res["Std_M"].astype(float)
    prob_res["Std_P"] = prob_res["Std_P"].astype(float)
    
    y_pred = []
    
    for i in range(len(prob_res)):
        ast = prob_res.loc[i]
        pr = ast[1:].astype(float)
        # pr = ast[1:].astype(float)
        prob = pr[["E", "S", "V", "C", "M", "P"]].tolist()
        sigma = pr[["Std_E", "Std_S", "Std_V", "Std_C","Std_M", "Std_P"]].tolist()
        bet = np.max(prob)
        index = prob.index(bet)
        y_pred.append(index)
        
    for i in range(len(y_pred)):
        if y_pred[i] == 0:
            y_pred[i] = "E"
        elif y_pred[i] == 1:
            y_pred[i] = "S"
        elif y_pred[i] == 2:
            y_pred[i] = "V"
        elif y_pred[i] == 3:
            y_pred[i] = "C"
        elif y_pred[i] == 4:
            y_pred[i] = "M"
        elif y_pred[i] == 5:
            y_pred[i] = "P"
        else:
            continue
    
    cm = confusion_matrix(res_test["Taxon"], y_pred)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    # plt.figure()
    # plot_confusion_matrix(cm_norm, classes=["C", "E", "M", "P", "S", "V"],
    #                       title='Confusion Matrix on the Test Set')
    
    df_cm = pd.DataFrame(cm)
    df_cm.rename(columns = {0:'C', 1:'E', 2:'M', 3:'P', 4:'S', 5:'V'}, inplace = True)
    df_cm["Class"] = ["C", "E", "M", "P", "S", "V"]
    column_to_move = df_cm.pop("Class")
    df_cm.insert(0, "Class", column_to_move)
    
    # Define metrics
    
    prec_C = df_cm["C"].iloc[0]/(df_cm["C"].iloc[0]+np.sum(df_cm["C"][1:])) # TP/(TP+FP)
    rec_C = df_cm["C"].iloc[0]/(df_cm["C"].iloc[0]+np.sum(df_cm.loc[0][2:])) # TP/(TP+FN)
    F1_C = 2*(prec_C*rec_C)/(prec_C+rec_C)
    
    prec_E = df_cm["E"].iloc[1]/(df_cm["E"].iloc[1]+np.sum(df_cm["E"].loc[df_cm["E"] != df_cm["E"].iloc[1]])) # TP/(TP+FP)
    rec_E = df_cm["E"].iloc[1]/(df_cm["E"].iloc[1]+np.sum([n for n in df_cm.loc[1][1:] if n != df_cm["E"].iloc[1]])) # TP/(TP+FN)
    F1_E = 2*(prec_E*rec_E)/(prec_E+rec_E)
    
    prec_M = df_cm["M"].iloc[2]/(df_cm["M"].iloc[2]+np.sum(df_cm["M"].loc[df_cm["M"] != df_cm["M"].iloc[2]])) # TP/(TP+FP)
    rec_M = df_cm["M"].iloc[2]/(df_cm["M"].iloc[2]+np.sum([n for n in df_cm.loc[2][1:] if n != df_cm["M"].iloc[2]])) # TP/(TP+FN)
    F1_M = 2*(prec_M*rec_M)/(prec_M+rec_M)
    
    prec_P = df_cm["P"].iloc[3]/(df_cm["P"].iloc[3]+np.sum(df_cm["P"].loc[df_cm["P"] != df_cm["P"].iloc[3]])) # TP/(TP+FP)
    rec_P = df_cm["P"].iloc[3]/(df_cm["P"].iloc[3]+np.sum([n for n in df_cm.loc[3][1:] if n != df_cm["P"].iloc[3]])) # TP/(TP+FN)
    F1_P = 2*(prec_P*rec_P)/(prec_P+rec_P)
    
    prec_S = df_cm["S"].iloc[4]/(df_cm["S"].iloc[4]+np.sum(df_cm["S"].loc[df_cm["S"] != df_cm["S"].iloc[4]])) # TP/(TP+FP)
    rec_S = df_cm["S"].iloc[4]/(df_cm["S"].iloc[4]+np.sum([n for n in df_cm.loc[4][1:] if n != df_cm["S"].iloc[4]])) # TP/(TP+FN)
    F1_S = 2*(prec_S*rec_S)/(prec_S+rec_S)
    
    prec_V = df_cm["V"].iloc[5]/(df_cm["V"].iloc[5]+np.sum(df_cm["V"].loc[df_cm["V"] != df_cm["V"].iloc[5]])) # TP/(TP+FP)
    rec_V = df_cm["V"].iloc[5]/(df_cm["V"].iloc[5]+np.sum([n for n in df_cm.loc[5][1:] if n != df_cm["V"].iloc[5]])) # TP/(TP+FN)
    F1_V = 2*(prec_V*rec_V)/(prec_V+rec_V)
    
    metric_df = pd.DataFrame(columns=["Class", "Precision", "Recall", "F1-score"])
    metric_df["Class"] = ["C", "E", "M", "P", "S", "V"]
    metric_df["Precision"] = [prec_C, prec_E, prec_M, prec_P, prec_S, prec_V]
    metric_df["Recall"] = [rec_C, rec_E, rec_M, rec_P, rec_S, rec_V]
    metric_df["F1-score"] = [F1_C, F1_E, F1_V, F1_P, F1_S, F1_V]
    
    MACRO_F1 = np.mean(metric_df["F1-score"])
    macro_f1.append(MACRO_F1)
    
    prior_pred = np.unique(y_pred, return_counts=True)
    PR_pred = pd.DataFrame(columns=["Class", "Percentage"])
    PR_pred["Class"] = ["C", "E", "M", "P", "S", "V"]
    PR_pred["Percentage"] = [prior_pred[1][0]/len(y_pred), prior_pred[1][1]/len(y_pred), prior_pred[1][2]/len(y_pred),
                             prior_pred[1][3]/len(y_pred), prior_pred[1][4]/len(y_pred), prior_pred[1][5]/len(y_pred)]
    prior_true = np.unique(prob_res["True Tax"], return_counts=True)
    PR_true = pd.DataFrame(columns=["Class", "Percentage"])
    PR_true["Class"] = ["C", "E", "M", "P", "S", "V"]
    PR_true["Percentage"] = [prior_true[1][0]/len(prob_res["True Tax"]), prior_true[1][1]/len(prob_res["True Tax"]), prior_true[1][2]/len(prob_res["True Tax"]),
                             prior_true[1][3]/len(prob_res["True Tax"]), prior_true[1][4]/len(prob_res["True Tax"]), prior_true[1][5]/len(prob_res["True Tax"])]
    
    pd_cm = pd.DataFrame(cm_norm)
    
    pd_cm_t = pd_cm.copy().T
    pd_cm_t.rename(columns = {0:'C', 1:'E', 2:'M', 3:'P', 4:'S', 5:'V'}, inplace = True)
    for i in range(len(pd_cm_t)):
        pd_cm_t['C'].iloc[i] = (pd_cm_t['C'].iloc[i]*PR_true["Percentage"].iloc[0])/PR_pred["Percentage"].iloc[i]
        pd_cm_t['E'].iloc[i] = (pd_cm_t['E'].iloc[i]*PR_true["Percentage"].iloc[1])/PR_pred["Percentage"].iloc[i]
        pd_cm_t['M'].iloc[i] = (pd_cm_t['M'].iloc[i]*PR_true["Percentage"].iloc[2])/PR_pred["Percentage"].iloc[i]
        pd_cm_t['P'].iloc[i] = (pd_cm_t['P'].iloc[i]*PR_true["Percentage"].iloc[3])/PR_pred["Percentage"].iloc[i]
        pd_cm_t['S'].iloc[i] = (pd_cm_t['S'].iloc[i]*PR_true["Percentage"].iloc[4])/PR_pred["Percentage"].iloc[i]
        pd_cm_t['V'].iloc[i] = (pd_cm_t['V'].iloc[i]*PR_true["Percentage"].iloc[5])/PR_pred["Percentage"].iloc[i]
    pd_cm_t = pd_cm_t.T
    pd_cm_t.rename(columns = {0:'C', 1:'E', 2:'M', 3:'P', 4:'S', 5:'V'}, inplace = True)
    
    # plt.figure()
    # plot_confusion_matrix(pd_cm_t, classes=["C", "E", "M", "P", "S", "V"],
    #                       title='Confusion Matrix on the Test Set')
    
    tp_perc.append(np.diag(pd_cm_t))
    # pred.append(prior_pred)
    # true.append(prior_true)

TP = pd.DataFrame(tp_perc, columns=["C", "E", "M", "P", "S", "V"])  
TP.to_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/TP_GMM_OS.csv')
F1 = pd.DataFrame(columns=["F1-score"])
F1["F1-score"] = macro_f1
F1.to_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/macro_f1_gmm_os.csv')