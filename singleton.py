# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 02:56:11 2020

@author: liorr
"""
import threading

class Singleton:
	__singleton_lock = threading.Lock()
	__singleton_instance = None

	@classmethod
	def instance(cls):
		if not cls.__singleton_instance:
			with cls.__singleton_lock:
				if not cls.__singleton_instance:
					cls.__singleton_instance = cls()
		return cls.__singleton_instance

