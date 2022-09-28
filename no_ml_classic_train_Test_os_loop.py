# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 12:25:25 2022

@author: pio-r
"""

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE 
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
from matplotlib.ticker import AutoMinorLocator
import matplotlib.ticker as ticker
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import cross_val_score
import seaborn as sns
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from tqdm import tqdm
from matplotlib.ticker import PercentFormatter
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SVMSMOTE 

# tax_db_aeiH = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/tax_db_aeiH_prob_and_sigma_sr.csv', dtype='str', keep_default_na=False)
tax_db_aeiH = pd.read_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/tax_SR_ratio.csv', dtype='str', keep_default_na=False)

tax_db_aeiH["a"] = tax_db_aeiH["a"].astype(float)
tax_db_aeiH["e"] = tax_db_aeiH["e"].astype(float)
tax_db_aeiH["i"] = tax_db_aeiH["i"].astype(float)
tax_db_aeiH["H"] = tax_db_aeiH["H"].astype(float)
tax_db_aeiH["nu6"] = tax_db_aeiH["nu6"].astype(float)
# tax_db_aeiH["sigma_nu6"] = tax_db_aeiH["sigma_nu6"].astype(float)
tax_db_aeiH["5/2"] = tax_db_aeiH["5/2"].astype(float)
# tax_db_aeiH["sigma_5/2"] = tax_db_aeiH["sigma_5/2"].astype(float)
tax_db_aeiH["2/1"] = tax_db_aeiH["2/1"].astype(float)
# tax_db_aeiH["sigma_2/1"] = tax_db_aeiH["sigma_2/1"].astype(float)
tax_db_aeiH["HUN"] = tax_db_aeiH["HUN"].astype(float)
# tax_db_aeiH["sigma_HUN"] = tax_db_aeiH["sigma_HUN"].astype(float)
tax_db_aeiH["3/1"] = tax_db_aeiH["3/1"].astype(float)
# tax_db_aeiH["sigma_3/1"] = tax_db_aeiH["sigma_3/1"].astype(float)
tax_db_aeiH["PHO"] = tax_db_aeiH["PHO"].astype(float)
# tax_db_aeiH["sigma_PHO"] = tax_db_aeiH["sigma_PHO"].astype(float)
tax_db_aeiH["JFC"] = tax_db_aeiH["JFC"].astype(float)
# tax_db_aeiH["sigma_JFC"] = tax_db_aeiH["sigma_JFC"].astype(float)

tax_db_aeiH.drop(tax_db_aeiH.columns[tax_db_aeiH.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)

SR_hist = pd.DataFrame()

SR = ["nu6", "5/2", "2/1", "HUN", "3/1", "PHO", "JFC"]
PROB = [np.mean(tax_db_aeiH["nu6"]), np.mean(tax_db_aeiH["5/2"]), np.mean(tax_db_aeiH["2/1"]),
        np.mean(tax_db_aeiH["HUN"]), np.mean(tax_db_aeiH["3/1"]), np.mean(tax_db_aeiH["PHO"]),
        np.mean(tax_db_aeiH["JFC"])]
SIGMA = [np.std(tax_db_aeiH["nu6"])/np.sqrt(len(tax_db_aeiH)), np.std(tax_db_aeiH["5/2"])/np.sqrt(len(tax_db_aeiH)),
         np.std(tax_db_aeiH["2/1"])/np.sqrt(len(tax_db_aeiH)), np.std(tax_db_aeiH["HUN"])/np.sqrt(len(tax_db_aeiH)),
         np.std(tax_db_aeiH["3/1"])/np.sqrt(len(tax_db_aeiH)), np.std(tax_db_aeiH["PHO"])/np.sqrt(len(tax_db_aeiH)),
         np.std(tax_db_aeiH["JFC"])/np.sqrt(len(tax_db_aeiH))]

SR_hist["SR"] = SR
SR_hist["Percentage"] = PROB
SR_hist["Sigma"] = SIGMA

F = tax_db_aeiH[["Number", "Name", "Prov.Desig", "Taxon", "nu6", "5/2", "2/1", "HUN", "3/1", "PHO", "JFC"]]

X_t = F[["nu6", "5/2", "2/1", "HUN", "3/1", "PHO", "JFC"]]
y_t = F[["Taxon"]]

macro_f1 = []
tp_perc = []
for i in tqdm(range(0, 100)):
    sm = SVMSMOTE(random_state=i)
    res_x, res_y = sm.fit_resample(X_t, y_t)

    res_x["Taxon"] = res_y
    
    F_train, F_test = train_test_split(res_x, train_size = 0.7, test_size=0.3, random_state=0)
    
    tax_E = F_train.loc[F_train["Taxon"] == "E"]
    tax_S = F_train.loc[F_train["Taxon"] == "S"]
    tax_V = F_train.loc[F_train["Taxon"] == "V"]
    tax_C = F_train.loc[F_train["Taxon"] == "C"]
    tax_M = F_train.loc[F_train["Taxon"] == "M"]
    tax_P = F_train.loc[F_train["Taxon"] == "P"]
    
    SR_E = pd.DataFrame()
    SR_E["SR"] = SR
    SR_E["Percentage"] = [np.mean(tax_E["nu6"]), np.mean(tax_E["5/2"]), np.mean(tax_E["2/1"]),
            np.mean(tax_E["HUN"]), np.mean(tax_E["3/1"]), np.mean(tax_E["PHO"]),
            np.mean(tax_E["JFC"])]
    SR_E["Std"] = [np.std(tax_E["nu6"])/np.sqrt(len(tax_E)), np.std(tax_E["5/2"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["2/1"])/np.sqrt(len(tax_E)), np.std(tax_E["HUN"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["3/1"])/np.sqrt(len(tax_E)), np.std(tax_E["PHO"])/np.sqrt(len(tax_E)),
                   np.std(tax_E["JFC"])/np.sqrt(len(tax_E))]
    SR_S = pd.DataFrame()
    SR_S["SR"] = SR
    SR_S["Percentage"] = [np.mean(tax_S["nu6"]), np.mean(tax_S["5/2"]), np.mean(tax_S["2/1"]),
            np.mean(tax_S["HUN"]), np.mean(tax_S["3/1"]), np.mean(tax_S["PHO"]),
            np.mean(tax_S["JFC"])]
    SR_S["Std"] = [np.std(tax_S["nu6"])/np.sqrt(len(tax_S)), np.std(tax_S["5/2"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["2/1"])/np.sqrt(len(tax_S)), np.std(tax_S["HUN"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["3/1"])/np.sqrt(len(tax_S)), np.std(tax_S["PHO"])/np.sqrt(len(tax_S)),
                   np.std(tax_S["JFC"])/np.sqrt(len(tax_S))]
    
    SR_V = pd.DataFrame()
    SR_V["SR"] = SR
    SR_V["Percentage"] = [np.mean(tax_V["nu6"]), np.mean(tax_V["5/2"]), np.mean(tax_V["2/1"]),
            np.mean(tax_V["HUN"]), np.mean(tax_V["3/1"]), np.mean(tax_V["PHO"]),
            np.mean(tax_V["JFC"])]
    SR_V["Std"] = [np.std(tax_V["nu6"])/np.sqrt(len(tax_V)), np.std(tax_V["5/2"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["2/1"])/np.sqrt(len(tax_V)), np.std(tax_V["HUN"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["3/1"])/np.sqrt(len(tax_V)), np.std(tax_V["PHO"])/np.sqrt(len(tax_V)),
                   np.std(tax_V["JFC"])/np.sqrt(len(tax_V))]
    
    SR_C = pd.DataFrame()
    SR_C["SR"] = SR
    SR_C["Percentage"] = [np.mean(tax_C["nu6"]), np.mean(tax_C["5/2"]), np.mean(tax_C["2/1"]),
            np.mean(tax_C["HUN"]), np.mean(tax_C["3/1"]), np.mean(tax_C["PHO"]),
            np.mean(tax_C["JFC"])]
    SR_C["Std"] = [np.std(tax_C["nu6"])/np.sqrt(len(tax_C)), np.std(tax_C["5/2"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["2/1"])/np.sqrt(len(tax_C)), np.std(tax_C["HUN"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["3/1"])/np.sqrt(len(tax_C)), np.std(tax_C["PHO"])/np.sqrt(len(tax_C)),
                   np.std(tax_C["JFC"])/np.sqrt(len(tax_C))]
    
    SR_M = pd.DataFrame()
    SR_M["SR"] = SR
    SR_M["Percentage"] = [np.mean(tax_M["nu6"]), np.mean(tax_M["5/2"]), np.mean(tax_M["2/1"]),
            np.mean(tax_M["HUN"]), np.mean(tax_M["3/1"]), np.mean(tax_M["PHO"]),
            np.mean(tax_M["JFC"])]
    SR_M["Std"] = [np.std(tax_M["nu6"])/np.sqrt(len(tax_M)), np.std(tax_M["5/2"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["2/1"])/np.sqrt(len(tax_M)), np.std(tax_M["HUN"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["3/1"])/np.sqrt(len(tax_M)), np.std(tax_M["PHO"])/np.sqrt(len(tax_M)),
                   np.std(tax_M["JFC"])/np.sqrt(len(tax_M))]
    
    SR_P = pd.DataFrame()
    SR_P["SR"] = SR
    SR_P["Percentage"] = [np.mean(tax_P["nu6"]), np.mean(tax_P["5/2"]), np.mean(tax_P["2/1"]),
            np.mean(tax_P["HUN"]), np.mean(tax_P["3/1"]), np.mean(tax_P["PHO"]),
            np.mean(tax_P["JFC"])]
    SR_P["Std"] = [np.std(tax_P["nu6"])/np.sqrt(len(tax_P)), np.std(tax_P["5/2"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["2/1"])/np.sqrt(len(tax_P)), np.std(tax_P["HUN"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["3/1"])/np.sqrt(len(tax_P)), np.std(tax_P["PHO"])/np.sqrt(len(tax_P)),
                   np.std(tax_P["JFC"])/np.sqrt(len(tax_P))]
    
    R = pd.DataFrame()
    R["Class"] = ["E", "Std_E", "S", "Std_S", "V", "Std_V", "C", "Std_C", "M", "Std_M", "P", "Std_P"]
    R["nu6"] = [SR_E["Percentage"][0],
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
    R["5/2"] = [SR_E["Percentage"][1],
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
    R["2/1"] = [SR_E["Percentage"][2],
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
    R["HUN"] = [SR_E["Percentage"][3],
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
    R["3/1"] = [SR_E["Percentage"][4],
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
    R["PHO"] = [SR_E["Percentage"][5],
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
    R["JFC"] = [SR_E["Percentage"][6],
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
    
    R_rs_test = R_sr.copy()
    
    from tqdm import tqdm
    
    F_test = F_test.reset_index(drop=True)
    
    prob_res = pd.DataFrame(columns=["True Tax", "E", "Std_E", "S", "Std_S",
                                      "V", "Std_V", "C", "Std_C", "M", "Std_M", "P", "Std_P"])
    
    prob_res["True Tax"] = F_test["Taxon"]
    
    # prob_res = pd.DataFrame(columns=["True Tax", "E", "Std_E", "S", "Std_S",
    #                                  "V", "Std_V", "C", "Std_C", "M", "Std_M", "P", "Std_P"])
    # prob_res["True Tax"] = res_c["Taxon"]
    
    for i in tqdm(range(len(F_test))):
        pr = F_test.loc[i, F_test.columns != 'Taxon']
        pr = pr.astype(float)
        # pr = pr[:33].astype(float)
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
          E.append(pr[j]*R_rs_test["E"].iloc[j])
          # std_E.append(pr[j]*R_sr_test["Std_E"].iloc[j])
          S.append(pr[j]*R_rs_test["S"].iloc[j])
          # std_S.append(pr[j]*R_sr_test["Std_S"].iloc[j])
          V.append(pr[j]*R_rs_test["V"].iloc[j])
          # std_V.append(pr[j]*R_sr_test["Std_V"].iloc[j])
          C.append(pr[j]*R_rs_test["C"].iloc[j])
          # std_C.append(pr[j]*R_sr_test["Std_C"].iloc[j])
          M.append(pr[j]*R_rs_test["M"].iloc[j])
          # std_M.append(pr[j]*R_sr_test["Std_M"].iloc[j])
          P.append(pr[j]*R_rs_test["P"].iloc[j])
          # std_P.append(pr[j]*R_sr_test["Std_P"].iloc[j])
        prob_res["E"].iloc[i] = np.sum(E)
        # prob_res["Std_E"].iloc[i] = np.sum(std_E)
        prob_res["S"].iloc[i] = np.sum(S)
        # prob_res["Std_S"].iloc[i] = np.sum(std_S)
        prob_res["V"].iloc[i] = np.sum(V)
        # prob_res["Std_V"].iloc[i] = np.sum(std_V)
        prob_res["C"].iloc[i] = np.sum(C)
        # prob_res["Std_C"].iloc[i] = np.sum(std_C)
        prob_res["M"].iloc[i] = np.sum(M)
        # prob_res["Std_M"].iloc[i] = np.sum(std_M)
        prob_res["P"].iloc[i] = np.sum(P)
        # prob_res["Std_P"].iloc[i] = np.sum(std_P)
        
    prob_res["E"] = prob_res["E"].astype(float)
    prob_res["S"] = prob_res["S"].astype(float)
    prob_res["C"] = prob_res["C"].astype(float)
    prob_res["V"] = prob_res["V"].astype(float)
    prob_res["M"] = prob_res["M"].astype(float)
    prob_res["P"] = prob_res["P"].astype(float)
    
    y_pred = []
    
    for i in range(len(prob_res)):
        ast = prob_res.loc[i]
        # pr = ast[4:].astype(float)
        pr = ast[1:].astype(float)
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
        
    def plot_confusion_matrix(cm, classes=None, title='Confusion matrix'):
        """Plots a confusion matrix."""
        if classes is not None:
            sns.heatmap(cm, xticklabels=classes, yticklabels=classes, vmin=0., vmax=1., annot=True, annot_kws={'size':50})
        else:
            sns.heatmap(cm, vmin=0., vmax=1.)
        plt.title(title, fontsize=30)
        plt.ylabel('True label', fontsize=30)
        plt.xlabel('Predicted label', fontsize=30)
    
    cm = confusion_matrix(prob_res["True Tax"], y_pred)
    cm_norm = cm/cm.sum(axis=1)[:, np.newaxis]
    plt.figure()
    plot_confusion_matrix(cm_norm, classes=["C", "E", "M", "P", "S", "V"],
                          title='Confusion Matrix on the Test Set')
    
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
    
    plt.figure()
    plot_confusion_matrix(pd_cm_t, classes=["C", "E", "M", "P", "S", "V"],
                          title='Confusion Matrix on the Test Set')
    tp_perc.append(np.diag(pd_cm_t))
    

TP = pd.DataFrame(tp_perc, columns=["C", "E", "M", "P", "S", "V"])
TP.to_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/TP_classic_OS.csv')
F1 = pd.DataFrame(columns=["F1-score"])
F1["F1-score"] = macro_f1
F1.to_csv('C:/Users/pio-r/OneDrive/Desktop/ESA/Internship/Science_Project/Final Analysis/macro_f1_classic_os.csv')