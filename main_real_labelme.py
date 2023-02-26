import torch
import argparse
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from helpers.functions import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import logging
from helpers.data_load import *
from helpers.transformer import *
from helpers.algorithm_wrapper import *
import os
from datetime import datetime
from numpy.random import default_rng
import random
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt

# Set arguments
parser = argparse.ArgumentParser()
parser.add_argument('--M',type=int,help='No of annotators',default=5)
parser.add_argument('--K',type=int,help='No of classes',default=3)
parser.add_argument('--N',type=int,help='No of data samples (synthetic data)',default=10000)
parser.add_argument('--R',type=int,help='Dimension of data samples (synthetic data)',default=5)
parser.add_argument('--annotator_label_pattern',type=str,help='random or correlated or per-sample-budget or per-annotator-budget',default='per-sample-budget')
parser.add_argument('--l',type=int,help='number of annotations per sample or number of samples per annotators',default=1)
parser.add_argument('--p',type=float,help='prob. that an annotator label a sample',default=0.2)
parser.add_argument('--conf_mat_type',type=str,help='separable, random, or diagonally-dominant,'\
'hammer-spammer, classwise-hammer-spammer, pairwise-flipper',default='separable-and-uniform')
parser.add_argument('--gamma',type=float,help='hammer probability in hammer-spammer type',default=0.01)
parser.add_argument('--dataset',type=str,help='synthetic or cifar10 or mnist',default='labelme')
parser.add_argument('--annotator_type',type=str,help='synthetic, machine-classifier, good-bad-annotator-mix or real',default='real')
parser.add_argument('--good_bad_annotator_ratio',type=float,help='ratio of good:bad annotators for good-bad-annotator-mix type ',default=0.1)	
parser.add_argument('--flag_preload_annotations',type=bool,help='True or False (if True, load annotations from file, otherwise generate annotations',\
															default=True)
parser.add_argument('--lam',type=float,help='Volume min regularizer',default=0.2)
parser.add_argument('--seed',type=int,help='Random seed',default=1)
parser.add_argument('--device',type=int,help='GPU device number',default=1)
parser.add_argument('--n_trials',type=int,help='No of trials',default=5)
parser.add_argument('--flag_hyperparameter_tuning',type=bool,help='True or False',default=True)
parser.add_argument('--proposed_init_type',type=str,help='close_to_identity or mle_based or deviation_from_identity',default='close_to_identity')
parser.add_argument('--proposed_projection_type',type=str,help='simplex_projection or softmax or sigmoid_projection',default='simplex_projection')
parser.add_argument('--classifier_NN',type=str,help='resnet9 or resnet18 or resnet34',default='resnet9')


parser.add_argument('--learning_rate',type=float,help='Learning rate',default=0.001)
parser.add_argument('--batch_size',type=int,help='Batch Size',default=100)
parser.add_argument('--n_epoch',type=int,help='Number of Epochs',default=100)
parser.add_argument('--n_epoch_maxmig',type=int,help='Number of Epochs for Maxmig',default=20)
parser.add_argument('--coeff_label_smoothing',type=float,help='label smoothing coefficient',default=0)
parser.add_argument('--log_folder',type=str,help='log folder path',default='results/labelme_real/')
parser.add_argument('--n_epoch_mv',type=int,help='majority_voting_init_epochs',default=20)



# Parser
args=parser.parse_args()

# Setting GPU and cuda settings
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available:
	device = torch.device('cuda:'+str(args.device))
torch.autograd.set_detect_anomaly(True)



# Log file settings
time_now = datetime.now()
time_now.strftime("%b-%d-%Y")
log_file_name = args.log_folder+'log_'+str(time_now.strftime("%b-%d-%Y"))+'_'+args.annotator_type+'_'+args.dataset+'.txt'
result_file=args.log_folder+'result_'+str(time_now.strftime("%b-%d-%Y"))+'_'+args.annotator_type+'_'+args.dataset+'.txt'

if os.path.exists(log_file_name):
	os.remove(log_file_name)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(log_file_name)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
	
	
def main():
	rng = default_rng()

	#Set algorithm flags
	algorithms_list = ['VOLMINEECS_LOGDETH']

	
	
	# Data logging variables
	test_acc_all = np.ones((len(algorithms_list),args.n_trials))*np.nan	
	fileid = open(result_file,"w")
	fileid.write('#########################################################\n')
	fileid.write(str(time_now))
	fileid.write('\n')
	fileid.write('Trial#\t')
	for s in algorithms_list:
		fileid.write(s+str('\t'))
	fileid.write('\n')
	
	annotators_sel = range(args.M)
	alg_options = { 
	'device':device,
	'loss_function_type':'cross_entropy'}
	
	args.n_epoch=100
	args.n_epoch_maxmig=20
	args.K=8
	args.M =59
	if args.flag_hyperparameter_tuning:
		alg_options['lamda_list']=[0.2,0.1]
		alg_options['learning_rate_list']=[0.01,0.001]	
	else:
		alg_options['lamda_list']=[args.lam]	
		alg_options['learning_rate_list']=[args.learning_rate]	

	
	if args.dataset=='cifar10':
		alg_options['flag_lr_scheduler'] = True
		alg_options['milestones'] = [30,60]
	else:
		alg_options['flag_lr_scheduler'] = False
		alg_options['milestones'] = [30,60]		
	
	for t in range(args.n_trials):
		np.random.seed(t+args.seed)
		torch.manual_seed(t+args.seed)
		torch.cuda.manual_seed(t+args.seed)
		random.seed(t+args.seed)
		# Get the train, validation and test dataset 
		train_data 	= labelme_dataset(True, transform=None, target_transform=None,split_per=None, random_seed=t+args.seed,length_train_data = None,args=args,logger=logger)
		val_data 	= labelme_dataset(False, transform=None, target_transform=None,split_per=None, random_seed=t+args.seed,length_train_data = None,args=args,logger=logger)
		test_data 	= labelme_test_dataset(transform=None, target_transform=None)
		args.R = np.prod(np.shape(train_data.train_data)[1:])
		
		alg_options['train_data']=train_data
		alg_options['A_true']=train_data.A_true
		alg_options['data_train']=train_data.train_data
		alg_options['y_train']=train_data.train_labels
		alg_options['data_val']=val_data.val_data
		alg_options['y_val']=val_data.val_labels
		alg_options['data_test']=test_data.test_data
		alg_options['y_test']=test_data.test_labels
		alg_options['annotations']=train_data.annotations
		alg_options['annotations_one_hot']=train_data.annotations_one_hot
		alg_options['annotator_softmax_label_mbem']=train_data.annotator_softmax_label_mbem
		alg_options['annotators_per_sample_mbem']=train_data.annotators_per_sample_mbem
		alg_options['annotations_list_maxmig']=train_data.annotations_list_maxmig
		alg_options['annotators_sel']=annotators_sel
		alg_options['annotator_mask']=train_data.annotator_mask		
		
		# Prepare data for training/validation and testing
		train_loader = DataLoader(dataset=train_data,
								  batch_size=args.batch_size,
								  num_workers=3,
								  shuffle=True,
								  drop_last=True,
								  pin_memory=True)
		val_loader = DataLoader(dataset=val_data,
								  batch_size=args.batch_size,
								  num_workers=3,
								  shuffle=False,
								  drop_last=True,
								  pin_memory=True)
		test_loader = DataLoader(dataset=test_data,
								  batch_size=args.batch_size,
								  num_workers=3,
								  shuffle=False,
								  drop_last=True,
								  pin_memory=True)
								  
								  
		alg_options['train_loader'] = train_loader
		alg_options['val_loader'] = val_loader
		alg_options['test_loader']= test_loader
							  
		#################################Run Algorithms#######################################
		logger.info('Starting trial '+str(t)+'.....................')			
		fileid.write(str(t+1)+'\t')
		for k in range(len(algorithms_list)):
			logger.info('Running '+algorithms_list[k])
			alg_options['method']=algorithms_list[k]
			test_acc=algorithmwrapperEECS(args,alg_options,logger)				
			test_acc_all[k,t]=test_acc*100
			fileid.write("%.4f\t" %(test_acc_all[k,t]))
			#fileid.close()
			#fileid = open(result_file,"a")
		fileid.write('\n')
		
	fileid.write('MEAN\t')
	np.savetxt(fileid,np.transpose(np.nanmean(test_acc_all,axis=1)),fmt='%.4f',delimiter='\t',newline='\t')
	fileid.write('\nMEDIAN\t')
	np.savetxt(fileid,np.transpose(np.nanmedian(test_acc_all,axis=1)),fmt='%.4f',delimiter='\t',newline='\t')
	fileid.write('\nSTD\t')
	np.savetxt(fileid,np.transpose(np.nanstd(test_acc_all,axis=1)),fmt='%.4f',delimiter='\t',newline='\t')
		
	

if __name__ == '__main__':
	main()
	