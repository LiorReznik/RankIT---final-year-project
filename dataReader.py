# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os,json
import numpy as np
import pandas as pd
import spacy
from spacy_langdetect import LanguageDetector
from singelton import Singleton

class getData(metaclass=Singleton):
    """
    a class to read the data and corisponding labels
    this class reads the full text and the abstract, generates bad sumaries 
    and returns pd data frame ([full text],[reviews],[labels])
    """
    
    def __init__(self):
        self.lang_checker = spacy.load('en')
        self.lang_checker.add_pipe(LanguageDetector(), name='language_detector', last=True)
        self.lang_checker.max_length = 999999999999999999999999999999
        
        
    def __call__(self,path='data/corona'):
        self.base = path
        self.x1,self.x2,self.y = [],[],[]
        self.dataReader
        return self.returnData
        
    @property
    def dataReader(self):
        """
        read text and all it's co-responding abstracs
        """
        for path in os.listdir(self.base):
            full_path = os.path.join(self.base, path)
            if os.path.isfile(full_path) and full_path.endswith('.json'):
                with open(full_path,"r",encoding="utf8") as f:
                    file = json.load(f)
                    if len(file.get("abstract",[]))>0 and len(file.get("body_text",[]))>0:

                        x1="\n".join(text.get("text") 
                                     for text in file.get("body_text"))
                        x2="\n".join(text.get("text")
                                             for text in file.get("abstract"))
                        #check if the lang is english
                        doc = self.lang_checker(x2) 

                        if doc._.language['language'] =='en': 
                            self.x1.append(x1)
                            self.x2.append(x2)
                        
    @property
    def returnData(self):
        if len(self.x1)==len(self.x2):
          return pd.DataFrame(zip(self.x1,self.x2,np.ones(len(self.x1))),
                              columns=['X1','X2','Y'])
        else:
            raise ValueError("the articles  must be in JSON format and in english")
      

