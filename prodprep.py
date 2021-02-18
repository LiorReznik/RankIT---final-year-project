 # -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 19:37:35 2020

@author: liorr
"""
import numpy as np
from baseprep import BasicPreProcessor
import pandas as pd
#from multiprocessing import Pool
import traceback
from singelton import Singleton

class ProdPreProcessor(BasicPreProcessor,metaclass=Singleton):
    """
    a class to preprocess the text FOR preparning the raw-data before sending it
    for scoring into the model 
    the following methods are applied:
        1. stop words removal (optional)
        2. contraction expanding
        3. cleaning of [] barckets
        4. bounderies -> sentence level
        5. sciBert for sentence level embeddings
        6. generate bad and penallty
        7. train-dev-test split
        8. padding
        
    """
    def __init__(self,controller):
        """
        Parameters
        ----------
        controller : Controller
            a controller that cordinates between the GUI and the logic.

        Returns
        -------
        None.

        """
        super(ProdPreProcessor, self).__init__(controller)
       
    def __call__(self,data:list):
        """
        method to manage the preprocessing 

        Parameters
        ----------
        data : list
            list of two texts: the article and the summary.

        Returns
        -------
        [Article:numpy array,Summary:numppy array]
            pre-processed data.

        """
        assert len(data) == 2 and type(data[0]) == type(data[-1]) == str,"""
        in prodprep:data should be list that containes two strings"""
        self.data = pd.DataFrame([data],columns=['X1','X2'])
        # it is a partial duplicate of call but I wanted the base class to containe only functions
        try:
            self.controller.progress_updater("strating preprocessing")
            for data in ["X1","X2"]:
                self.controller.progress_updater("working on decontractiation and cleaning of {}".format(data))
                self.data[data]=self.data[data].apply(self.decontractiate)
                self.controller.progress_updater("starting sentence level spliting for {}".format(data))
                self.data[data]=self.data[data].apply(self.sentenceSplitStanford)
                self.controller.progress_updater("starting encoding of {}".format(data))
                self.encode(data)
                self.pad(data)
            self.data = [np.array(self.data.X1.tolist()),np.array(self.data.X2.tolist())]
            assert self.data[0].shape == self.data[-1].shape == (1,650,768)
            self.controller.progress_updater("preprocessing ended")
        except Exception as e:
            self.controller.progress_updater('Exiting the program:\n{}\n {}\n'.format(
                traceback.print_tb(e.__traceback__),e))
        finally:
            return self.data


 
 
   
                    
