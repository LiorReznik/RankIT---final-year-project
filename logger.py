# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 11:29:45 2020

@author: liorr
"""

import logging
from datetime import datetime
class Logger:

    def __init__(self, name:str, level=logging.DEBUG):
        """
        Logger class to save logs into a file.
        each instance of this class opens a dedicated file to log into.
        there ar 4 log levels that are supported:
        debug
        info
        warning
        error

        Parameters
        ----------
        name : str
            the name of the file.
        level, optional
           logging level . The default is logging.DEBUG.

        Returns
        -------
        None.

        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

    def __enter__(self):
        self.fh = logging.FileHandler(datetime.now().strftime('{}_%H_%M_%d_%m_%Y.log'.format(self.name)), 'w')
        self.logger.addHandler(self.fh)
        self.sh = logging.StreamHandler()
        self.logger.addHandler(self.sh)
        return self  
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("closing logger file")
        self.fh.close()
        self.sh.close()
     
    def reformat_msg(self,type:str,msg:str):
        return datetime.now().strftime('{} : %H:%M:%S: {} '.format(type,msg))
    
    def debug(self, msg):
        self.logger.debug(self.reformat_msg("DEBUG",msg))

    def info(self, msg):
        self.logger.info(self.reformat_msg("INFO",msg))

    def warning(self, msg):
        self.logger.warning(self.reformat_msg("WARNING",msg))

    def error(self, msg):
        self.logger.error(self.reformat_msg("ERROR",msg))

        