import random
import numpy as np
from sklearn import preprocessing
from keras import backend as K
import pandas as pd
import matplotlib.pyplot as plt


def data_augment(T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB, T_demo, T_meds, T_stage, length):
    counter = 0
    bcd = [None]*9
    for k, ehd_1 in enumerate([T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB, T_demo, T_meds, T_stage]):
        efg = pd.DataFrame()
        for x in T_demo['id']:
            efg = efg.append(ehd_1[ehd_1['id']==x])
        bcd[k] = efg


    for idx in T_demo['id']:
        for i, ehd in enumerate([T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB]):
            if ehd[(ehd['id']==idx)]['time'].count()>3:
                drp = random.sample(range(ehd[(ehd['id']==idx)]['time'].count()-1), int(ehd[(ehd['id']==idx)]['time'].count()/4))
                for j, ehd1 in enumerate([T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB, T_demo, T_meds, T_stage]):
                    data = ehd1[ehd1['id']==idx]
                    if i!=j:
                        add_row = data
                    else:
                        for drp1 in drp:
                            add_row = data.drop(data.loc[data.index==drp1].index, axis=0)
                    add_row['id']= counter+length
                    bcd[j] = bcd[j].append(add_row, ignore_index=True)
                counter = counter+1
    return bcd
    
def data_augment_1(T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB, T_demo, T_meds, T_stage, length):

    bcd = [None]*9
    for k, ehd_1 in enumerate([T_creatinine, T_DBP, T_SBP, T_glucose, T_ldl, T_HGB, T_demo, T_meds, T_stage]):
        efg = pd.DataFrame()
        for x in T_demo['id']:
            efg = efg.append(ehd_1[ehd_1['id']==x])
        bcd[k] = efg

    return bcd

def data_interpolate(bcd):
    
    T_creatinine=bcd[0]
    T_DBP=bcd[1]
    T_SBP=bcd[2]
    T_glucose=bcd[3]
    T_ldl=bcd[4]
    T_HGB=bcd[5]
    T_demo=bcd[6]
    
    ex_num = len(T_demo['id'])
    Creatinine = np.zeros([ex_num, 700,1])
    DBP = np.zeros([ex_num, 700,1])
    Glucose = np.zeros([ex_num, 700,1])
    Ldl = np.zeros([ex_num, 700,1])
    SBP = np.zeros([ex_num, 700,1])
    HGB = np.zeros([ex_num, 700,1])
    Demo = np.zeros([ex_num, 7])
    del_age = max(T_demo['age']) - min(T_demo['age'])
    min_age = min(T_demo['age'])
    for idx1, idx2 in enumerate(T_demo['id']):
        Creatinine[idx1] = np.reshape(np.interp(range(700), T_creatinine[(T_creatinine['id']==idx2)]['time'], T_creatinine[(T_creatinine['id']==idx2)]['value']), (700,1))
        DBP[idx1] = np.reshape(np.interp(range(700), T_DBP[(T_DBP['id']==idx2)]['time'], T_DBP[(T_DBP['id']==idx2)]['value']), (700,1))
        Glucose[idx1] = np.reshape(np.interp(range(700), T_glucose[(T_glucose['id']==idx2)]['time'], T_glucose[(T_glucose['id']==idx2)]['value']), (700,1))
        Ldl[idx1] = np.reshape(np.interp(range(700), T_ldl[(T_ldl['id']==idx2)]['time'], T_ldl[(T_ldl['id']==idx2)]['value']), (700,1))
        SBP[idx1] = np.reshape(np.interp(range(700), T_SBP[(T_SBP['id']==idx2)]['time'], T_SBP[(T_SBP['id']==idx2)]['value']), (700,1))
        HGB[idx1] = np.reshape(np.interp(range(730, 1430), T_HGB[(T_HGB['id']==idx2)]['time'], T_HGB[(T_HGB['id']==idx2)]['value']), (700,1))
        for num, race in enumerate(['Unknown', 'White', 'Black', 'Asian', 'Hispanic']):
            if (T_demo[(T_demo['id']==idx2)]['race']==race).bool():
                Demo[idx1,num]=1
        if (T_demo[(T_demo['id']==idx2)]['gender']=='Male').bool():
            Demo[idx1,5] = 1 
        Demo[idx1,6] = (T_demo[(T_demo['id']==idx2)]['age']-min_age)/del_age
        
    return  Creatinine, DBP, Glucose, Ldl, SBP, HGB, Demo
    


def meds_matrix(Demo):
    ex_num = len(Demo[6]['id'])
    T_meds = Demo[7]
    med_list = T_meds['drug'].unique()
    med_list = med_list.tolist()
    array1 = np.zeros([ex_num, 700, 21])
    for idx1, idx2 in enumerate(Demo[6]['id']):
        med = T_meds[(T_meds['id']==idx2)]
        for drug, dose, start, end in zip(med['drug'], med['daily_dosage'], med['start_day'], med['end_day']):
            for k in range(start, end):
                m = med_list.index(drug)
                array1[idx1,k,m] = dose
    return array1
 
 
def get_final_array(train_list):
    Creatinine, DBP, Glucose, Ldl, SBP, HGB, Demo = data_interpolate(train_list)
    array2 = meds_matrix(train_list)
    for app in [Creatinine, DBP, Glucose, Ldl, SBP, HGB]:
        array2 = np.append(array2, app, axis=2)

    standardized_X = np.zeros([len(train_list[6]['id']), 700, 27])
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    for c in range(27):
        standardized_X[:,:,c] = scaler.fit_transform(array2[:,:,c])
    return standardized_X, Demo
    

def f1_factor(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric."""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric."""
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
    
def round_off(num_array):
    num_return = []
    for num in num_array:
        if num<0.5:
            num_return.append(0)
        else:
            num_return.append(1)
    return num_return
    
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_'+string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_'+string])
    plt.show()