import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

class NFOF():
    def __init__(self):
        self.NFOF = pd.read_csv('./data/NFOF_Clustered.csv',index_col=False)
        self.KNN = None
        self.data_result = None
        self.data_after_rm = None
        
    def set_food(self,product_name):
        self.data_result = self.NFOF[self.NFOF['product_name'] == product_name]
        self.data_after_rm = self.NFOF[(self.NFOF['product_name'] != product_name) & (self.NFOF['cluster'] == self.data_result['cluster'].tolist()[0])]
    
    def find(self,product_name):
        ''' Find Product in DataFrame'''
        data = self.NFOF[self.NFOF['product_name'].str.lower().str.contains(product_name.lower(), na = False)]
        data = data.drop_duplicates(subset=['product_name'], keep='first')
        result = []
        for x,y in zip(data.No,data.product_name):
            result = result + [{'No' : x, 'name' : y}]
        return result
    
    def predict(self):
        attrs = ['fat_100g', 'carbohydrates_100g', 'sugars_100g','proteins_100g', 'salt_100g', 'energy_100g','reconstructed_energy', 'g_sum']
        X = self.data_after_rm[attrs]
        y = self.data_after_rm['No']
        KNN = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
        KNN.fit(X,y)
        KNN_raw = KNN.predict(self.data_result[attrs])
        KNN_result = list(set(KNN_raw))
        data = self.NFOF.loc[self.NFOF['No'].isin(KNN_result)]
        result = list(set(data.product_name))
        return result
                    
                    
# x = NFOF()
# print(len(x.find("Rice")))