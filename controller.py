# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:50:55 2020

@author: liorr
"""
import nn_module,argparse,spacy,json,os,pickle
from prodprep import ProdPreProcessor
from trainprep import TrainPreProcessor
from singelton import Singleton
from logger import Logger
from spacy_langdetect import LanguageDetector


class Controller(metaclass=Singleton):
    def __init__(self,queue=None):
        """
        

        Parameters
        ----------
        queue , optional
            queue to pass massages to the GUI, if the GUI exists.
            The default is None.

        Returns
        -------
        None.

        """
        self.model=nn_module.nn_Model(controller=self)
        self.prod_prep = ProdPreProcessor(controller=self)
        self.train_prep = TrainPreProcessor(controller=self)
        self.queue = queue   
        self.lang_checker = self.logger = None
        
    def progress_updater(self,msg:str):
        """
        Parameters
        ----------
        msg : str
            message to display on the GUI and to save in the logger.

        Returns
        -------
        None.

        """
        if self.logger:
            if msg.startswith("Exiting the program"):
                self.logger.error(msg)
            else:
                self.logger.info(msg)
        else: print(msg)
        if self.queue:
            self.queue.put(msg)
    
    def train_manager(self,path:str,train:bool=True):
        """
                
        method to manage all the (re)training process
        Parameters
        ----------
        path : str
            path to the folder with the articles.
        train : bool, optional
            what pipeline to use, train or retrain. The default is True(train).

        Returns
        -------
        None.

        """
        def parse_config():
            with open(os.path.join(path, 'config.json')) as config:
                return json.load(config)
            
        config = parse_config()        
        logger = "PipeForTrain" if train else "PipeFor(re)Train"            
        with Logger(logger) as self.logger:
            data = self.train_prep(penalty=config.get("penalty"),data=path)
            res_dict = self.model.train_model(data=data,config=config,build=train)  
            if res_dict:
                self.parse(res_dict=res_dict)
           
    def parse(self,res_dict:dict):
        """
        

        Parameters
        ----------
        res_dict : dict
            dictionary that holds the results of the expiriment.
        Returns
        -------
        None.

        """
    
        def save():
            with open("results.pkl","wb") as f:
                pickle.dump(res_dict,f,protocol=4)
        
        def show_on_screen():
            msg ="Finally!,results: train_MSE: {},dev_MSE: {}".format(res_dict["loss"],
                                              res_dict["val_loss"])
            if res_dict.get("pen"):
                msg = "{} on penalty: train_MSE: {},dev_MSE: {}".format(msg,
                                                                        res_dict["pen"]["loss"],
                                                                        res_dict["pen"]["val_loss"])
            self.progress_updater(msg)
                
        show_on_screen()
                
    def score_manager(self,data:list,score_path:str="fm/second_expiriment.h5",
                       penalty_path:str=None
                       ):
        """
        method to manage the scoring process
        the method gets the article and the summary that we want to score
        and manages all the process of the scoring, from preprocessing to scoring

        Parameters
        ----------
        data : list
            data to score .
        score_path : str, optional
            path to the pretrined score model
            for future implementation(the user interface does not supprot this at the moment),
            for not we are defining the (BEST) model oureselfs.
            The default is "fm/second_expiriment.h5".
        penalty_path : str, optional
            path to the pretrined penalty model
            for future implementation(the user interface does not supprot this at the moment),
            for not we are defining the (BEST) model oureselfs.        
            The default is None.

        Raises
        ------
        ValueError
            In case that the article or the summary is not in english.

        Returns
        -------
        None.

        """
        if not self.check_language(data):
            msg = "the article and the summary must be in english"
            self.progress_updater(msg)
            raise ValueError(msg)
        
        with Logger("Scoring") as self.logger:
            data = self.prod_prep(data=data)
            self.progress_updater("Finally!,results: {}".
                                  format(self.model.eval_summary(data,score_path,penalty_path)))
                     
    def check_language(self,data:str)->bool:
        """
        method to check the lang of the input, 
        the method returns True iff the langaue is english
        
        do note that this method checks input for scoring process
        for train and re-train the reader class handales this task

        Parameters
        ----------
        data : str
            list of two strings that represents summary and article.

        Returns
        -------
        bool
            if the language is english.

        """
        if not self.lang_checker :
            self.lang_checker  = spacy.load('en')
            self.lang_checker.add_pipe(LanguageDetector(), name='language_detector', last=True)
            self.lang_checker.max_length = 999999999999999999999999999999
       
        doc = self.lang_checker(data[0]) 
        doc1 = self.lang_checker(data[-1])
        return doc._.language['language'] == doc1._.language['language'] == 'en'
    
    def check_train_input(self,directory:str)->bool:
        """
        method to check if the input for (re)train is valid

        Parameters
        ----------
        directory : str
            path to the data folder.

        Returns
        -------
        bool
            if the input valid or not.

        """        
        dir = os.listdir(directory)
        msg = ""
        flag = True
        if len(dir)==0:
            msg+="the directory is empty!"
            flag = False
        if 'config.json' not in dir:
            msg+=", the directory must contain the config file"
            flag = False
        self.progress_updater(msg)
        return flag
        
    
if __name__ =="__main__":
    def parser_score_manager(data:list)->list:
        """
        function to read and make basic validation of
        the input for score functionality

        Parameters
        ----------
        data : list
            list of two elments: paths to the article and summary.

        Returns
        -------
        list[str]
            list of article and summary.

        """
        def read(ind):
            nonlocal data
            with open(data[ind],"r",encoding="utf8") as f:
                data[ind] = f.read()
        
        if (len(data) !=2 or type(data[0])!=type(data[-1])!=str
            or not data[0].endswith('.txt') or not data[-1].endswith('.txt')):
            parser.error('you must provide two txt files for summary and article')
        else:
            read(-1),read(0)
            return data

    def parser_train_manager(data:list)->bool:
        """
        function to read and make basic validation of
        the input for train and retrain functionality
    
        Parameters
        ----------
        data : list
            list of two elments: paths to the articles,config file and an indecator 
            for the disired functionality.
    
        Returns
        -------
        bool
        true iff the data is valid
    
        """  
        if len(data)!=2:
            parser.error('you must provide path to a directory as well as type of operation')
            return False
        if data[-1] != "t" and data[-1] != "rt":
            parser.error('type of operation must be t for train or rt for retrain')
            return False
        return cont.check_train_input()


    parser = argparse.ArgumentParser(description='RankIT!.your best automatic friend!')
    cont = Controller()
    parser.add_argument('-s','--score', nargs='+',
                   help="""enter path to txt file of the summary
                   and path to txt file for the article""")
    parser.add_argument('-t','--train',type=list,
                        help="""enter path for the data folder and the type of
                                operation t for traning and rt for retraining""")
    from tensorflow.keras.backend import sqrt,mean,square,shape,abs,maximum
    from tensorflow.keras.backend import exp,sum
    args = vars(parser.parse_args())

    if args.get('score'):
        data=parser_score_manager(args['score'])
        cont.score_manager(data)
        
    train = args.get('train')
    if train and parser_train_manager():
        train[-1] = True if train[-1] == "t" else False
        cont.train_manager(train[0],train[-1])
         