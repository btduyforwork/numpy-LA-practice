import numpy as np
from typing import List, Tuple, Dict
import csv
def load_training_data_from_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        next(reader) 
        data = []
        for row in reader:
            data.append(row[1:])  
        return np.array(data)

def compute_prior_probabilities(train_data):
    
    class_names=["No","Yes"]
    prior_probs=np.zeros(len(class_names))
    total_samples=len(train_data)
    
    for class_idx, name in enumerate(class_names):
        prior_probs[class_idx]=len(train_data[train_data[:,-1]==name])/total_samples
    
    return prior_probs

def compute_conditional_probabilities(train_data):
    class_names=np.array(["No","Yes"]) # 2
    n_features=train_data.shape[1] - 1 #Outlook,Temp,Hum,Wind
    conditional_probs=[] #3D array
    feature_array=[]
    for class_idx,class_val in enumerate(class_names):
        total_class_val=len(train_data[train_data[:,-1]==class_val])
        class_array=[]
        for i in range(n_features):
            feature_names=np.unique(train_data[:,i]) # ['Overcast' 'Rain' 'Sunny']
            feature_val=np.array([])
            if class_idx==0:
                feature_array.append(feature_names)
            for _,f_val in enumerate(feature_names):
                total_f_val=len(train_data[(train_data[:,i]==f_val)&(train_data[:,-1]==class_val)])
                feature_val=np.append(feature_val,total_f_val/total_class_val)
            class_array.append(feature_val)
        conditional_probs.append(class_array)
    return np.array(conditional_probs, dtype=object),np.array(feature_array, dtype=object)
        
def train_navie_bayes(file_path):
    training_data=load_training_data_from_csv(file_path)
    prior_probs=compute_prior_probabilities(training_data)
    conditional_probs,feature_array=compute_conditional_probabilities(training_data)
    return prior_probs,conditional_probs,feature_array

def predict(X,prior_probs,conditional_probs,feature_array):
    class_names=["No","Yes"]
    result=[]
    for class_idx,class_name in enumerate(class_names):
        print("--------", class_name,"--------")
        pro=prior_probs[class_idx]
        for feature_idx,feature_name in enumerate(X):
            idx_fe_arr=np.where(feature_name==feature_array[feature_idx])[0][0]
            pro*=conditional_probs[class_idx][feature_idx][idx_fe_arr]
        print("Probability of P(X/",class_name,"~ ",pro)
        result.append(pro)
    result=np.array(result)
    probability=result[np.argmax(result)]
    prediction=class_names[np.argmax(result)]
    return probability,prediction
        
            
            
    
if __name__ == "__main__":
    prior_probs,conditional_probs,feature_array=train_navie_bayes("Week_2/Data/data.csv")
    probability,prediction=predict(['Sunny','Cool', 'Normal', 'Weak'],prior_probs,conditional_probs,feature_array)
    if prediction=="No":
        print("You should not go!")
    else:
        print("You should go")