# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:37:35 2020

@author: liorr
"""
import numpy as np
from baseprep import BasicPreProcessor
import dataReader,pickle
import pandas as pd
import copy
#from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import random
import traceback

class TrainPreProcessor(BasicPreProcessor):
    """
    a class to preprocess the text FOR TRAINING with the following methods:
        1. stop words removal (optional)
        2. contraction expanding
        3. cleaning of [] barckets
        4. bounderies -> sentence level
        5. sciBert for sentence level embeddings
        6. generate bad and penallty
        7. train-dev-test split
        8. padding
        
    """
    def __init__(self,controller,
                 data:dataReader.getData = dataReader.getData
                 ,penalty:str=True):
        """
        Parameters
        ----------
        controller : Controller
            a controller that cordinates between the GUI and the logic.
        data : dataReader.getData, optional
            class to read the data. The default is dataReader.getData.
        penalty : str, optional
            wether to preprocess penalty data or not. The default is True.
        Returns
        -------
        None.

        """
        super(TrainPreProcessor, self).__init__(controller)
        self.data = data
        self.f_data = {}
        
    def __call__(self,penalty:bool,data:str='data/corona'):
        """
        method to manage the preprocessing 

        Parameters
        ----------
        path : str, optional
            path to the data. The default is 'data/corona'.

        Returns
        -------
            pre processed data.

        """
        try:
            self.penalty = penalty
            self.data=self.data()(data)
            self.controller.progress_updater("strating preprocessing")
            for data in ["X1","X2"]:
                self.controller.progress_updater("working on decontractiation and cleaning of {}".format(data))
                self.data[data]=self.data[data].apply(self.decontractiate)
                self.controller.progress_updater("starting sentence level spliting for {}".format(data))
                self.data[data]=self.data[data].apply(self.sentenceSplitStanford)
                self.controller.progress_updater("starting encoding of {}".format(data))
                self.encode(data)
                
            self.generate_bad
            if self.penalty:
                self.generate_penelty
            del self.data 
            for data in ["bad","penelty"] if self.penalty else ["bad"]:
                self.controller.progress_updater("working on pading of {}".format(data))
                for set in ["X1","X2"]:
                    self.data = self.f_data[data]
                    self.pad(set)
                self.split(data)
                self.save(data)

            self.controller.progress_updater("preprocessing ended")
        except Exception as e:
            self.controller.progress_updater('Exiting the program:\n{}\n {}\n'.format(
                     traceback.print_tb(e.__traceback__),e))
        finally:
            return self.f_data

    def split(self,data:str):
        """
        split into train,dev and test sets
        Parameters
        ----------
        data : str
            type of the data to split.

        Returns
        -------
        None.

        """
        def trans(r,c=None):
            return {"X1":np.array(self.f_data[data][r][c].X1.tolist()),
                               "X2":np.array(self.f_data[data][r][c].X2.tolist()),
                               "Y":np.array(self.f_data[data][r][c].Y.tolist()),
                               } if r and c else {"X1":np.array(self.f_data[data][r].X1.tolist()),
                               "X2":np.array(self.f_data[data][r].X2.tolist()),
                               "Y":np.array(self.f_data[data][r].Y.tolist()),
                               }
        
        self.controller.progress_updater("spliting the {} into train,dev and test sets".format(data))
        self.f_data[data] = train_test_split(self.f_data[data],test_size=0.15,
                                             random_state=42)
        self.f_data[data][0] =  train_test_split(self.f_data[data][0], 
                                                 test_size=0.15, random_state=42)
        self.f_data[data] = {"train":trans(0,0),
                             "dev":trans(0,1),"test":trans(1)}
       
             
    def save(self,data:str):
        """
        save the data to disk

        Parameters
        ----------
        data : str
            DESCRIPTION.

        Returns
        -------
        None.

        """
        with open("{}.pkl".format(data),"wb") as f:
             pickle.dump(self.f_data[data],f,protocol=4)
        
    def generate_penelty(self):
        """
        method to genarate damaged summaries for the penelty network

        Returns
        -------
        TYPE
            None.

        """
        def generate_dot5_1(summary):
            l=len(summary)//2
            return np.vstack((summary[:l],summary[:l])) if l>0 else np.concatenate((
                summary[0][:384],summary[0][:384]),axis=None).reshape(1,768)
             
        def generate_dot5_2(summary):
            l=len(summary)//2
            return np.vstack((summary[:l],np.zeros((l,768)))) if l>0 else np.concatenate((
                summary[0][:384],np.zeros(384)),axis=None).reshape(1,768)
            
        def generate_one():    
            temp = copy.deepcopy(self.data)
            temp.X2 = np.random.permutation(temp.X2)
            temp.Y = temp.Y.apply(lambda x: 1)
            return temp
        def generate_dot3(summary):    
            l=len(summary)//3
            return np.vstack((summary[:l],np.zeros((len(summary)-l,768)))) if l>0 else np.concatenate((
                summary[0][:256],np.zeros(512)),axis=None).reshape(1,768)           
        
        self.data.Y = self.data.Y.apply(lambda x:0)
        temp1 = generate_one()
        temp2 = copy.deepcopy(self.data)
        temp2.X2 = temp2.X2.apply(generate_dot5_2)
        temp2.Y = temp2.Y.apply(lambda x: 0.5)
        temp3 = copy.deepcopy(self.data)
        temp3.X2 = temp3.X2.apply(generate_dot5_1)
        temp3.Y = temp3.Y.apply(lambda x: 0.5)
        temp4 = copy.deepcopy(self.data)
        temp4.X2 = temp4.X2.apply(generate_dot5_2)
        temp4.Y = temp4.Y.apply(lambda x:0.7)
        self.data = pd.concat([self.data,temp2,temp3,temp1,temp4],
                                  ignore_index=True)
        self.controller.progress_updater("""done with damaged samples generation, now we have:{} 
                             samples""".format(len(self.data)))
             
    def generate_bad(self):
        """
        method to genarate bad summaries for the penelty network

        Returns
        -------
        TYPE
            None.

        """
        def generate_dot5(summary):
            l=len(summary)//2
            temp = news[news.len >= l].reset_index(drop=True)
            upper_bound = len(temp)
            vec = temp.content[random.randint(0,upper_bound)]
            return np.vstack((summary[:l],vec[:l])) if l>0 else np.concatenate((summary[0][:384],vec[0][:384]),axis=None).reshape(1,768)
        
        def generate_zero():    
            temp = copy.deepcopy(self.data)
            temp.X2 = np.random.permutation(temp.X2)
            temp.Y = temp.Y.apply(lambda x: 0)
            return temp
        
        def read_news_vectors():
            with open("newsenc.pkl","rb") as f:
                news= pickle.load(f)
                upper_bound = len(news)
            return news,upper_bound
                     
        news,upper_bound = read_news_vectors()
        temp2 = copy.deepcopy(self.data)
        temp2.X2 = temp2.X2.apply(generate_dot5)
        temp2.Y = temp2.Y.apply(lambda x: 0.5)
                     
        temp=generate_zero()
        self.data = pd.concat([self.data,temp2,temp],ignore_index=True)
        self.controller.progress_updater("done with bad samples generation, now we have:{} samples"
                         .format(len(self.data)))          
   
  