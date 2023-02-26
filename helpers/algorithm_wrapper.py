from __future__ import division
import numpy as np
import torch
import torch.optim as optim
import logging
import torch.nn.functional as F
from helpers.functions import *
from helpers.model import *
from torch.utils.data import DataLoader
from helpers.data_load import *
from helpers.transformer import *
import os
import copy
import math
from numpy.matlib import repmat
from torch.optim.lr_scheduler import MultiStepLR
from algorithms.trainer_proposed import *
from algorithms.trainer_noregeecs import *

			
def algorithmwrapperEECS(args,alg_options,logger):
		
	method			=alg_options['method']
	# Print args
	logger.info(args)
	

	if method=='NOREGEECS':
		test_acc = trainer_noregeecs(args,alg_options,logger)		
	elif method=='VOLMINEECS_LOGDETH' or method=='VOLMINEECS_LOGDETW':
		# Train data, Validation data and Test data	
		if alg_options['loss_function_type']=='cross_entropy':
			logger.info('Using cross entropy....')
			test_acc = trainer_proposed(args,alg_options,logger)							
	else:
		logger.info('Wrong method')
	return test_acc
	


	

