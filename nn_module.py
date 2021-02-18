# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 01:11:07 2020

@author: liorr
"""
import numpy as np
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle 
from typing import List
import tensorflow as tf

class nn_Model:
    def __init__(self,controller):
        self.controller = controller
        
    def cab_driver_distance(self,vectors):
        return tf.keras.backend.exp(-tf.keras.backend.sum(
            tf.keras.backend.abs(vectors[0]-vectors[-1]), axis=1, keepdims=True))
    
    def output_shape(self,shapes):
        shape1, shape2 = shapes
        return (shape1[0],1)
    
    @property
    def lstm_net_builder(self):
        """
        build the lstm component of the architecture

        Returns
        -------
        lstm network.

        """
        bi, lstm_hidden_units,lstm_dropout_rate = self.comfig.get("bidirectional"),
        self.comfig.get("lstm_hidden_units"), self.comfig.get("lstm_dropout_rate")
        if bi:
            lstm_layers = [tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(h_u, return_sequences=True),
                                            merge_mode='concat') for h_u in lstm_hidden_units]
        if lstm_dropout_rate:
            lstm_layers.append(tf.keras.layers.Dropout(lstm_dropout_rate))
        return tf.keras.Sequential(lstm_layers,name="Siamese-lstm")           
    
    @property
    def cnn_net_builder(self):
        """
        method to build the one channeled cnn component    

        Returns
        -------
        regular cnn net

        """    
        return tf.keras.Sequential([tf.keras.layers.Conv1D(filters=self.config.get("regular_cnn").get("filters") 
                                                ,kernel_size=self.config.get("regular_cnn").get("kernel") ,activation='relu',padding="same"),
                    tf.keras.layers.Dropout(self.config.get("regular_cnn").get("dropout")),
                ],name ="cnn-Siamese")	
        
    @property
    def attention_builder(self):
        """
        method to build the attention mechanisem of the architecture

        Returns
        -------
        attention.

        """
        return tf.keras.Sequential([ 
        tf.keras.layersDense(1, activation='tanh'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Activation('softmax'),
        tf.keras.layers.RepeatVector(self.comfig.get("lstm_hidden_units")[-1]*2),
        tf.keras.layers.Permute([2, 1])
        ],name="Attention")
       
    @property
    def multi_cnn_builder(self)->list:
        """
        method to buld multi channaled cnn

        Returns
        -------
        channels : list
            list of multi-cnn models.

        """
        channels = []
        for i,size in enumerate(self.config.get("multi_cnn").get("kernels")):
            channels.append(tf.keras.Sequential([tf.keras.layers.Conv1D(filters=self.config.get("multi_cnn").get("filters"),
                                          kernel_size=size,
                                          activation='relu'),
                                     tf.keras.layers.Dropout(self.config.get("multi_cnn").get("dropout")),
                                     tf.keras.layers.AveragePooling1D(),
                                     tf.keras.layers.Flatten()],name ="channel_{}".format(i)))
        return channels
    @property
    def load_nn_model(self):
        """
        load pretrained model(s)

        Returns
        -------
        None.

        """
        self.score_model = tf.keras.models.load_model(filepath=self.score_path)
        self.penalty_model =tf.keras.models.load_model(filepath=self.penalty_path) if self.penalty_path else None

    def train_model(self,config:dict,data,build:bool=True)->dict:
        """
        train model

        Parameters
        ----------
        config : dict
            configaration dictionary
        data : numpy array
            the data to train on.
        build : bool
            the disired functionality , train or re train a model 
            true iff train new model.

        Returns
        -------
        results.

        """
        self.config = config
        if build:
            self.shape = data["bad"]["train"]["X1"].shape
            self.score_model = self.build_model
            if self.config.get("penalty"):
                self.penalty_model = self.build_model
        else:
            self.score_path, self.penalty_path = config.get("score_path"),config.get("penalty_path")  
            if not self.score_path:
                self.controller.progress_updater("you must provide path to the score (reward) network") 
                return
            self.load_nn_model
            
        score_hist = self.score_model.fit([data["bad"]["train"]["X1"],["bad"]["train"]["X2"] ],data["bad"]["train"]["Y"]
                                          ,batch_size=self.config.get("batch_size",64),epochs=self.config.get("epochs",25),
                                          validation_data=([data["bad"]["dev"]["X1"],["bad"]["dev"]["X2"] ],data["bad"]["dev"]["Y"]),
                                          callbacks=[SaveBestModel( save_format=None,filepath=self.config.get("model_path","score_net.h5"),monitor="val_loss",
                                                                   patient=self.config.get("patient"))]) 
        score_hist = score_hist.history
                                          
        if self.config.get("penalty"):
            penalty_hist = self.penalty_model.fit([data["penelty"]["train"]["X1"],["penelty"]["train"]["X2"] ],data["penelty"]["train"]["Y"]
                                          ,batch_size=self.config.get("batch_size",64),epochs=self.config.get("epochs",25),
                                          validation_data=([data["penelty"]["dev"]["X1"],["penelty"]["dev"]["X2"] ],data["penelty"]["dev"]["Y"]),
                                          callbacks=[SaveBestModel( save_format=None,filepath=self.config.get("model_path","score_net.h5"),monitor="val_loss",
                                                                   patient=self.config.get("patient"))]) 
            score_hist["pen"] = penalty_hist.history

        return score_hist
        
    @property
    def build_model(self):
        """
        build and compile  model

        Returns
        -------
        compiled model

        """
        use_lstm,use_multi_cnn,use_regular_cnn,use_attention = self.config.get("lstm_hidden_units"),
        self.config.get("multi_cnn"),self.config.get("regular_cnn"),self.config.get("use_attention")
        if  use_lstm and use_multi_cnn:
            self.controller.progress_updater("you cannot have multi cnn and lstm in the same model")
            return
        
        if  use_attention and use_multi_cnn:
            self.controller.progress_updater("you cannot have multi cnn and attention in the same model")
            return
        if not use_lstm and use_regular_cnn:
           self.controller.progress_updater("you cannot have regular cnn without lstm")
           return
       
        left_input , right_input = tf.keras.layers.Input(self.shape), tf.keras.layers.Input(self.shape)
 
        if use_lstm:
            lstm_net = self.lstm_net_builder
        
            if use_regular_cnn:
                cnn_net = self.cnn_net_builder
                combined_l = tf.keras.layers.concatenate([left_input,cnn_net(left_input)],name="combined_left")
                combined_r = tf.keras.layers.concatenate([right_input,cnn_net(right_input)],name="combined_right")   
                res_l,res_r = lstm_net(combined_l), lstm_net(combined_r)
            else:
                res_l,res_r = lstm_net(left_input),lstm_net(right_input)
                
        if use_attention:
            attention = self.attention_builder
            res_r = tf.keras.layers.Lambda(lambda x: tf.keras.layers.sum(x, axis=1))(tf.keras.layers.multiply([attention(res_r), res_r]))
            res_l = tf.keras.layers.Lambda(lambda x: tf.keras.layers.sum(x, axis=1))(tf.keras.layers.multiply([attention(res_l),  res_l]))
            
        if  use_multi_cnn:
            channels = self.multi_cnn_builder
            res_r,res_l =  tf.keras.layers.concatenate([x(right_input) for x in channels]),
            tf.keras.layers.concatenate([x(left_input) for x in channels])
           
        similarity=tf.keras.layers.Lambda(function=self.cab_driver_distance,output_shape=self.output_shape)([res_r, res_l])
        similarity = tf.keras.layers.Dense(1)(similarity)
        model = tf.keras.models.Model([right_input,left_input],similarity)  
        model.compile(loss=self.config.get("optimizer","mse"),optimizer='adam')
        return model


    def eval_summary(self,data:list,
                     score_path:str="models/second_expiriment.h5",
                     penalty_path:str="models/penalty_expiriment.h5")->float:
        """
        method to evaluate a given summary

        Parameters
        ----------
        data : list
            DESCRIPTION.
        score_path : str, optional
            path for score network. The default is "models/score_second_expiriment.h5".
            
        penalty_path : str, optional
              path for penalty network. The default is "models/penalty_expiriment.h5".

        Returns
        -------
        float
            the score.

        """
  
        self.score_path, self.penalty_path = score_path, penalty_path        
        self.load_nn_model
           
        self.score = self.score_model.predict(data).flatten()
        if self.penalty_model:
            self.score -= self.penalty_model.predict(data).flatten()
        return self.score
    
    

class SaveBestModel(tf.keras.callbacks.Callback):
      """
      monitor to save the best model according to val_loss or val_acc
      Parameters
      ----------
      filepath : str
          the disaired model path or name.
      monitor : str, optional
          metric to monitor on it can be "val_loss" or "val_acc". The default is 'val_loss'.
      save_format : str, optional
          the disared format for the saved model. The defalut is 'tf'.
      patient : int, optional
          optional early stopping
      Returns
      -------
      None.
      """
      def __init__(self, filepath:str, monitor:str='val_loss',save_format:str='tf',patient:str=None):
          super(SaveBestModel, self).__init__()
          import warnings
          if monitor not in {'val_loss','val_acc'}:
              warnings.warn("""ModelCheckpoint monitor must be "val_loss" or 
                          "val_acc" but got: {} so monitoring val_loss""".
                          format(monitor),RuntimeWarning)
          self.monitor = monitor
          self.filepath = filepath 
          self.op,self.best = (np.less,np.Inf) if self.monitor =='val_loss' else (np.greater,-np.Inf)
          self.num_epochs = 0 #number since last save
          self.save_format = save_format
          self.patient = patient    
      
      def on_epoch_end(self, epoch, logs={}):
          current = logs.get(self.monitor)
          print(logs.keys())
          if not current:
              warnings.warn('{} is not avilable,skiping'.format(self.monitor), RuntimeWarning)
          elif self.op(current, self.best):
               print('\nEpoch {}:{} improved from {} to {} saving the model'.format(epoch + 1, self.monitor, self.best,current))
               self.best = current
               self.num_epochs = 0
               self.model.save(self.filepath, overwrite=True,save_format=self.save_format )
          else:
               self.num_epochs += 1
               print("\nEpoch {}: {} did not improved from {}, this is the {} epoch without any improvment.".format(epoch + 1, self.monitor, self.best,self.num_epochs))
               if self.patient and self.patient == self.num_epochs:
                   self.model.stop_training = True
                   print('\nstopping the train, did not improved for {}'.format(self.patient))
     
      def on_train_end(self, logs=None):
            print('this is the end my only friend the end')

class Terminate(tf.keras.callbacks.Callback):
    """Callback that terminates training when:
        1.NaN loss is encountered.
        2.val_loss <=0
        3.val_acc  >=0.99
    """
    def on_epoch_end(self, batch, logs=None):
        def halt(msg):
            print(msg)
            self.model.stop_training = True
        logs = logs or {}
        val_acc,loss,val_loss = logs.get('loss'),logs.get('val_acc'),logs.get('val_loss')
        if loss and (np.isnan(loss) or np.isinf(loss)):
            halt('Batch {}: Invalid loss, terminating training'.format(batch))
        if  val_loss and (val_loss<=0 or np.isnan(val_loss)):
            halt('val_loss is at minimum, terminating training')
        if val_acc and val_acc>=0.99:
            halt('val_acc is at maximum, terminating training')


    