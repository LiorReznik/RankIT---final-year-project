# # -*- coding: utf-8 -*-
# """
# Created on Sun Jun  7 04:15:56 2020

# @author: liorr
# """
from tensorflow.keras.backend import sqrt,mean,square,shape,abs,maximum
from tensorflow.keras.backend import exp,sum
from singelton import Singleton
from controller import Controller
import tkinter as tk
import threading
import time
from queue import Queue,Empty
from PIL import ImageTk, Image
from abc import abstractmethod
import os


class App(tk.Tk,metaclass=Singleton):
    def __init__(self, *args, **kwargs):
        super(App,self).__init__( *args, **kwargs)
        def init_container():
            self.container = tk.Frame(self)
            self.container.pack(side="top", fill="both", expand=True)
            self.container.grid_rowconfigure(0, weight=1)
            self.container.grid_columnconfigure(0, weight=1)
      
        def init_menu():
            self.menubar = tk.Menu(self.container)
            self.menuChoise = tk.Menu(self.menubar, tearoff=1)
            self.menuChoise.add_command(label="welcome page",
                                    command=lambda: self.show_page(page=StartPage))
            self.menuChoise.add_separator()
            self.menuChoise.add_command(label="eval",
                                    command=lambda: self.show_page(page=EvalPage))
            self.menuChoise.add_command(label="train",
                                    command=lambda: self.show_page(page=TrainPage))
            self.menuChoise.add_command(label="REtrain",
                                    command=lambda: self.show_page(page=ReTrainPage))
            self.menuChoise.add_separator()
            self.menuChoise.add_command(label="Exit",
                                   command=self.on_closing)
            tk.Tk.config(self, menu=self.menubar)
    
        def init_pages():
            self.pages = {}
            for page in [StartPage,EvalPage,TrainPage,ReTrainPage]:
                frame =page(parent=self)
                self.pages[page] = frame
                frame.grid(row=0, column=0, sticky="nsew")
                
            self.menubar.add_cascade(label="Menu", menu=self.menuChoise)  

        # init graphics
        self.resizable(False, False) # do not allow resizing of the windows
        self.title("RankIT! for your sanity!")
        init_container()
        init_menu()
        init_pages()
        # init the controller and  the queue for comunication
        self.thread_queue = Queue()
        self.controller = Controller(
            queue=self.thread_queue)
        self.show_page(page=StartPage)# show welcome page

    def show_page(self, page):
        frame = self.pages[page]
        frame.tkraise()

    def on_closing(self):
        if tk.messagebox.askokcancel("Quit", "But, why?"):
            self.destroy()
  

class StartPage(tk.Frame,metaclass=Singleton):
    def __init__(self, **kwargs):
        super(StartPage,self).__init__(kwargs.get("parent").container)
        def init_logo():
            self.pack(fill="both", expand=1)
            self.load = Image.open("logo.jpg")
            self.render = ImageTk.PhotoImage(self.load)

            try:
                self.img = tk.Label(self, image=self.render)
            except tk.TclError:pass
            self.img.image = self.render
            self.img.place(x=0, y=0) 
            self.img.configure(relief="sunken")
            
        def init_welcome_label():    
            self.label = tk.Label(self,text="""
                                    Hi there!
                          welcome to your sanity saver!
                          you do not need to read the whole article,
                          you  do not need to relay on bad summaries any more!
                          check the summary with RankIT! your best automatic friend.
                                  SO!why, you are still here?!
                          """, font=("Helvetica", 20))
          #  self.label.configure(relief="sunken")
            self.label.place(x=866 , y=100, anchor="center")
            self.label.configure(background="#ffffff")
        
        self.configure(background="#ffffff")
        init_logo()
        init_welcome_label()
        
        
class FunctionalPage(tk.Frame):
    _is_abstract = True
    def __init__(self,parent):
        def init_design():
            self.configure(background="#d9d9d9")
            self.configure(relief='sunken')
            self.configure(borderwidth="2")
            self.configure(background="#ffffff")
        super(FunctionalPage,self).__init__(parent.container)
    
        if self._is_abstract:
            raise RuntimeError("Abstract class instantiation.")
        init_design()
        self.parent = parent
        self.msg = tk.StringVar()
        self.update_label = tk.Label(self, textvariable=self.msg)
        self.start_button = tk.Button(self,state='normal',
                                      command=self.start_working)
        
        
    def __init_subclass__(self):   
        self._is_abstract = False  

    def listen_for_result(self):
        '''
        Check for messages in the queue, if there is a message diaplay it
        '''
        while True:
            try:
                time.sleep(3)
                self.update()
                self.res = self.parent.thread_queue.get(block=False)
                self.msg.set(self.res)
                self.after(5, self.listen_for_result)
                if (self.res.startswith("Finally!") or self.res.startswith("the article")
                     or self.res.startswith("the number") or self.res.startswith("you")):
                     self.start_button.configure(state="normal")
                #     break                   
            except Empty:pass 

    def start_working(self):
        '''
        Spawn a new thread for running long logic in background
        '''        
        
        if self.get_input:
            self.start_button.configure(state="disabled")
            self.msg.set("starting! be patient, it will take a while")
            self.new_thread = threading.Thread(target=self.target,kwargs=self.kwargs,
                                           daemon=True)
            self.new_thread.start()
        self.after(1, self.listen_for_result)
        
    @abstractmethod     
    def check_input(self): pass
    @abstractmethod     
    def get_input(self):pass   
    
    
class EvalPage(FunctionalPage,metaclass=Singleton):
    
    def __init__(self, parent):
        super(EvalPage,self).__init__(parent)   

        def init_start_button():
            self.start_button.place(relx=0.007, rely=0.904, height=74,
                                    width=157)
            self.start_button.configure(background="#d9d9d9")
            self.start_button.configure(text="RankIT!")

        def init_article_text():
            self.article_text = tk.Text(self)
            self.article_text.place(relx=0.008, rely=0.038, relheight=0.503,
                                    width=1340)
            self.article_text.configure(background="white")
            self.article_text.configure(relief="groove")
            
        def init_article_label():
            self.article_label = tk.Label(self)
            self.article_label.place(relx=0.008, rely=0.013, height=22, width=1340)
            self.article_label.configure(background="#cbe3e7")
            self.article_label.configure(relief="groove")
            self.article_label.configure(text='''Please enter the article''')

        def init_summary_text():
            self.summary_text = tk.Text(self)
            self.summary_text.place(relx=0.008, rely=0.573, height=260,
                                    width=1340)
            self.summary_text.configure(background="white")
            self.summary_text.configure(relief="groove")
         
        def init_summary_label():
            self.summary_label = tk.Label(self)
            self.summary_label.place(relx=0.008, rely=0.548, height=24,
                                     width=1340)
            self.summary_label.configure(background="#cbe3e7")
            self.summary_label.configure(relief="groove")
            self.summary_label.configure(text="Please enter the summary")

        def init_update_label():
            self.update_label.place(relx=0.123, rely=0.905, height=74,
                                    width=1340-157)
            self.update_label.configure(background="#cbe3e7")
            self.update_label.configure(relief="groove")

        
        init_start_button()
        init_article_text()
        init_article_label()
        init_summary_text()
        init_summary_label()
        init_update_label()
        
    @property
    def check_input(self)->bool:
        res = self.summary_text.get('1.0', "end-1c") != '' and self.article_text.get('1.0', "end-1c") != ''       
        if not res:
            #self.parent.controller.progress_updater("you must enter article and summary")
            self.msg.set("you must enter article and summary")
        return res
     
    @property   
    def get_input(self)->bool:
        """
        preapare the input before passing it into the score manager
        """
        if not self.check_input:
            return False
        data = [self.article_text.get("1.0", "end-1c"),
                    self.summary_text.get("1.0", "end-1c")]
        self.article_text.delete('1.0', "end")
        self.summary_text.delete('1.0', "end") 
        self.target , self.kwargs =self.parent.controller.score_manager ,{'data':data}
        return True


class TrainPage(FunctionalPage):
    def __init__(self, parent):
        super(TrainPage,self).__init__(parent)       
        def init_start_button():
            self.start_button.place(x=0,y=2,height=74,
                                    width=157)
            self.start_button.configure(background="#d9d9d9")
            self.start_button.configure(text="TrainIT!",
                                        command=self.start_working)

        def init_info_label():
            self.info_label = tk.Label(self)
            self.info_label.place(x=0,y=0, 
                                     height=22, width=1340)
            self.info_label.configure(background="#cbe3e7")
            self.info_label.configure(relief="groove")
            self.info_label.configure(text="Please browse a Directory")


        def init_update_label():
            self.update_label.place(x=0,y=3,height=74,
                                    width=1340)
            self.update_label.configure(background="#cbe3e7")
            self.update_label.configure(relief="groove")

        
        init_start_button()
        init_info_label()
        init_update_label()
        
    @property
    def check_input(self)->bool:
        return self.parent.controller.check_train_input(self.directory)

     
    @property   
    def get_input(self):
        """
        get and preapare the input before passing it into the train\retrain manager
        """
        self.directory = tk.filedialog.askdirectory()
        if not self.check_input:
            return False
        self.target , self.kwargs =self.parent.controller.train_manager ,{'path':self.directory,"train":True}   
        return True
        
        
class ReTrainPage(TrainPage,metaclass=Singleton):
     def __init__(self, parent):
        super(ReTrainPage,self).__init__(parent)       
        self.start_button.configure(text="reTrainIT!")
   
     @property   
     def get_input(self):
         """
         get and preapare the input before passing it into the train\retrain manager
         """
                
         self.directory = tk.filedialog.askdirectory()
         if not self.check_input:
             return False
         self.target , self.kwargs =self.parent.controller.train_manager ,{'path':self.directory,"train":False}   
         return True

if  __name__  == "__main__":
    app = App(None)
    app.geometry("1385x812+20+32")
    app.mainloop()