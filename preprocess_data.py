import pickle
import numpy as np
import os
import pandas as pd
from tensorflow.keras.utils import load_img,img_to_array,to_categorical
from sklearn.model_selection import train_test_split
from PIL import ImageFile          

from tqdm import tqdm                  

def create_csv(DATA_DIR,filename):
    class_names = os.listdir(DATA_DIR)
    data = list()
    if(os.path.isdir(os.path.join(DATA_DIR,class_names[0]))):
        for class_name in class_names:
            file_names = os.listdir(os.path.join(DATA_DIR,class_name))
            for file in file_names:
                data.append({
                    "Filename":os.path.join(DATA_DIR,class_name,file),
                    "ClassName":class_name
                })
    else:
        class_name = "test"
        file_names = os.listdir(DATA_DIR)
        for file in file_names:
            data.append(({
                "FileName":os.path.join(DATA_DIR,file),
                "ClassName":class_name
            }))
    data = pd.DataFrame(data)
    data.to_csv(os.path.join(os.getcwd(),"csv_files",filename),index=False)


class PreprocessData:
    def __init__(self):

        self.TEST_DIR = os.path.join(os.getcwd(),"data","images","test")
        self.TRAIN_DIR = os.path.join(os.getcwd(),"data","images","train")
        
        create_csv(self.TRAIN_DIR,"train.csv")
        create_csv(self.TEST_DIR,"test.csv")
        print("CSV creation complete")
        data_train = pd.read_csv(os.path.join(os.getcwd(),"csv_files","train.csv"))
        data_test = pd.read_csv(os.path.join(os.getcwd(),"csv_files","test.csv"))
        
        labels_list = list(set(data_train['ClassName'].values.tolist()))
        labels_id = {label_name:id for id,label_name in enumerate(labels_list)}

        data_train['ClassName'].replace(labels_id,inplace=True)
        data_test['ClassName'].replace(labels_id,inplace=True)
        
        with open(os.path.join(os.getcwd(),"pickle_files","labels_list.pkl"),"wb") as handle:
            pickle.dump(labels_id,handle)
        
        labels = to_categorical(data_train['ClassName'])
        xtrain,xtest,ytrain,ytest = train_test_split(data_train.iloc[:,0],labels,
                test_size = 0.2,random_state=42)
        self.train_data = xtrain
        self.train_labels = ytrain

        self.test_data = xtest
        self.test_labels = ytest

        self.train_groups, self.val_groups, self.test_groups = self.initialize()

        self.batch_num = 5

    def initialize(self):
        train_groups = [[],[],[],[],[]]
        for train_data, train_label_ar in zip(self.train_data, self.train_labels):
            train_label = np.argmax(train_label_ar, axis=None, out=None)
            if train_label < 20:
                train_groups[0].append((train_data,train_label))
            elif 20 <= train_label < 40:
                train_groups[1].append((train_data,train_label))
            elif 40 <= train_label < 60:
                train_groups[2].append((train_data,train_label))
            elif 60 <= train_label < 80:
                train_groups[3].append((train_data,train_label))
            elif 80 <= train_label < 100:
                train_groups[4].append((train_data,train_label))
        val_groups = [[],[],[],[],[]]
        for i, train_group in enumerate(train_groups):
            val_groups[i] = train_groups[i][9000:]
            train_groups[i] = train_groups[i][:9000]

        test_groups = [[],[],[],[],[]]
        for test_data, test_label_ar in zip(self.test_data, self.test_labels):
            test_label = np.argmax(test_label_ar, axis=None, out=None)
            if test_label < 20:
                test_groups[0].append((test_data,test_label))
            elif 20 <= test_label < 40:
                test_groups[1].append((test_data,test_label))
            elif 40 <= test_label < 60:
                test_groups[2].append((test_data,test_label))
            elif 60 <= test_label < 80:
                test_groups[3].append((test_data,test_label))
            elif 80 <= test_label < 100:
                test_groups[4].append((test_data,test_label))

        return train_groups, val_groups, test_groups

    def getNextClasses(self, i):
        return self.train_groups[i], self.val_groups[i], self.test_groups[i]

if __name__ == "__main__":
    preprocess_data = PreprocessData()
    print(len(data_preprocess.train_groups[0]))
