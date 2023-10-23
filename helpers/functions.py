from __future__ import division
#import mxnet as mx
import numpy as np
import logging,os
import copy
import urllib
import logging,os,sys
from scipy import stats
from random import shuffle
from torch.nn import functional as F
from scipy import special
from scipy.optimize import linear_sum_assignment
import torch
from torch import nn
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader
from helpers.model import *
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import math
from scipy import stats
from sklearn.metrics import confusion_matrix
from PIL import Image
from helpers.transformer import *

def generate_confusion_matrices(M,K,gamma,type):
	A = np.zeros((M,K,K))
	if type == 'random':
		for m in range(M):
			A[m] = np.random.uniform(0,1,(K,K))
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))
	
	elif type == 'separable-and-diagonally-dominant':
		i = np.asscalar(np.random.choice(M,1))
		A[i] = np.identity(K)
		for m in range(0,i):
			A[m] = np.identity(K)+gamma*np.random.uniform(0,1,(K,K)) 
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))		
		for m in range(i+1,M):
			A[m] = np.identity(K)+gamma*np.random.uniform(0,1,(K,K))
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))
			
	elif type == 'diagonally-dominant':
		for m in range(M):
			A[m] = np.identity(K)+gamma*np.random.uniform(0,1,(K,K)) 
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))		
	elif type == 'hammer-spammer':
		A = (1/K)*np.ones((M,K,K))
		for m in range(M):
			if(np.random.uniform(0,1) < gamma):
				A[m] = np.identity(K)
			else:
				A[m]=A[m]+0.0001*np.identity(K)
				A[m]=np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))
	elif type == 'separable-and-uniform-random':
		i = np.asscalar(np.random.choice(M,1))
		A[i] = np.identity(K)+gamma*np.random.uniform(0,1,(K,K)) 
		A[i] = np.matmul(A[i],np.diag(np.divide(1,np.sum(A[i],axis=0))))
		#A = (1/K)*np.ones((M,K,K))
		#i = np.asscalar(np.random.choice(M,1))
		#A[i] = ((1-gamma)/K)*np.ones((K,K))
		#for k in range(K):
		#	A[i,k,k]=gamma
		print(A[i])
		for m in range(0,i):
			A[m] = np.random.uniform(0,1,(K,K)) 
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))	
		print(np.sum(A[m],axis=0))
		for m in range(i+1,M):
			A[m] = np.random.uniform(0,1,(K,K))
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))
	elif type == 'separable-and-uniform':
		A = (1/K)*np.ones((M,K,K))
		i = np.asscalar(np.random.choice(M,1))
		A[i] = np.identity(K)+gamma*np.random.uniform(0,1,(K,K)) 
		A[i] = np.matmul(A[i],np.diag(np.divide(1,np.sum(A[i],axis=0))))
		#A[i] = ((1-gamma)/K)*np.ones((K,K))
		#for k in range(K):
		#	A[i,k,k]=gamma
		print(A[i])
		for m in range(0,i):
			A[m] = A[m]+0.0001*np.identity(K)
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))	
		for m in range(i+1,M):
			A[m] = A[m]+0.0001*np.identity(K)
			A[m] = np.matmul(A[m],np.diag(np.divide(1,np.sum(A[m],axis=0))))
	elif type=='symmetric-flipping':
		A = (gamma/K)*np.ones((M,K,K))
		for m in range(M):
			for k in range(K):
				A[m,k,k]=1-gamma
					
	else:
		print('Incorrect choice for confusion matrix trype')
	return A
		
		
def generate_synthetic_data_and_labels(N,R,K):
			
	# Generate data
	X = np.random.uniform(0,1,(R,N))
	X = X.astype(np.float)
	#Generate labels via a nonlinear function f : Y = f(X)
	W = np.random.normal(0,1,(K,R))
	
	Y = special.softmax(np.tanh(np.matmul(W,X)))
	
	# Generate labels
	y = np.argmax(Y,axis=0)
	X = X.transpose()
	return X, y
	
def generate_annotator_labels(A,pattern,p,l,y):	
	M = np.shape(A)[0]
	N = np.shape(y)[0]
	K = np.shape(A)[1]
	f = np.zeros((N,M,K))
	annotations = -1*np.ones((N,M))
	annotator_label_mask = np.zeros((N,M))
	annotator_label = {}
	for i in range(M):
		annotator_label['softmax' + str(i) + '_label'] = np.zeros((N,K))  
		
	if pattern=='random':
		mask = np.random.binomial(1,p,np.shape(annotations))		
		annotators_per_sample = []
		for n in range(N):
			a = np.argwhere(mask[n,:]==1)
			annotators_per_sample.append(a[0])
			count=0
			for m in annotators_per_sample[n]:
				t =  np.random.multinomial(1,A[m,:,y[n]])
				f[n,m,:] = t
				annotations[n,m]=np.argmax(t)
				annotator_label_mask[n,m]=1
				annotator_label['softmax' + str(count) + '_label'][n] = t
				count = count +1 		
	elif pattern=='per-sample-budget':
		annotators_per_sample = []
		for n in range(N):
			a=np.sort(np.random.choice(M,l,replace=False))
			a = a.reshape(1,-1)
			annotators_per_sample.append(a[:,0])
			count=0
			for m in annotators_per_sample[n]:
				t =  np.random.multinomial(1,A[m,:,y[n]])
				f[n,m,:] = t
				annotations[n,m]=np.argmax(t)
				annotator_label_mask[n,m]=1
				annotator_label['softmax' + str(count) + '_label'][n] = t
				count = count +1 
				
	elif pattern=='correlated':
		a=np.sort(np.random.choice(M,2,replace=False))
		senior_annotator = a[0]
		junior_annotator = a[1]
		mask = np.random.binomial(1,p,np.shape(annotations))		
		annotators_per_sample = []
		for n in range(N):
			a = np.argwhere(mask[n,:]==1)
			annotators_per_sample.append(a[:,0])
			count=0
			for m in annotators_per_sample[n]:
				t =  np.random.multinomial(1,A[m,:,y[n]])
				f[n,m,:] = t
				annotations[n,m]=np.argmax(t)
				annotator_label_mask[n,m]=1
				annotator_label['softmax' + str(count) + '_label'][n] = t
				count = count +1 

		annotations[:,junior_annotator]=annotations[:,senior_annotator]
		f[:,junior_annotator,:]=f[:,senior_annotator,:]
		annotator_label_mask[:,junior_annotator]=annotator_label_mask[:,senior_annotator]
		for n in range(N):
			count=0
			for m in annotators_per_sample[n]:
				annotator_label['softmax' + str(count) + '_label'][n] = f[n,m,:]
				count = count +1 		
		
			
	
	answers_bin_missings = []
	annotations = annotations.astype(int)
	for i in range(N):
		row = []
		for r in range(M):
			if annotations[i, r] == -1:
				row.append(0 * np.ones(K))
			else:
				row.append(one_hot(annotations[i, r], K)[0, :])
		answers_bin_missings.append(row)
		
	answers_bin_missings = np.array(answers_bin_missings)
	

		
		
	return f,annotations,answers_bin_missings,annotator_label,annotators_per_sample,annotator_label_mask
	
def get_real_annotator_labels(annotations,K):
	M = np.shape(annotations)[1]
	N = np.shape(annotations)[0]
	f = np.zeros((N,M,K))
	annotator_label_mask = np.zeros((N,M))
	
	annotator_label = {}
	for i in range(M):
		annotator_label['softmax' + str(i) + '_label'] = np.zeros((N,K))  

	annotators_per_sample = []
	for n in range(N):
		a = np.argwhere(annotations[n,:]!= -1)
		annotators_per_sample.append(a[:,0])
		count=0
		for m in annotators_per_sample[n]:
			f[n,m,annotations[n,m]] = 1
			annotator_label_mask[n,m]=1
			annotator_label['softmax' + str(count) + '_label'][n] = f[n,m,:]
			count = count +1 
			
	answers_bin_missings = []
	#answers_bin_missings = np.zeros((N,M))
	for i in range(N):
		row = []
		for r in range(M):
			if annotations[i, r] == -1:
				row.append(0 * np.ones(K))
			else:
				row.append(one_hot(annotations[i, r], K)[0, :])
		answers_bin_missings.append(row)
	answers_bin_missings = np.array(answers_bin_missings)
	

		
	return f,annotations,answers_bin_missings,annotator_label,annotators_per_sample,annotator_label_mask
	
def generate_machine_classifier_annotations(data_train_1,y_train,data_to_annotate_1,true_labels,args,logger,transform,transform_y):
	# Get paramaters
	M = args.M
	K = args.K
	annotator_label_pattern=args.annotator_label_pattern
	p = args.p
	l = args.l
	flag_preload_annotations=args.flag_preload_annotations
	N = np.shape(data_to_annotate_1)[0]
	
	# Transformation on the data
	data_train = torch.zeros(len(data_train_1),1,28,28)
	for i in range(len(data_train_1)):
		tmp = Image.fromarray(data_train_1[i])
		data_train[i] =transform(tmp)
	data_to_annotate = torch.zeros(len(data_to_annotate_1),1,28,28)
	for i in range(len(data_to_annotate_1)):
		tmp = Image.fromarray(data_to_annotate_1[i])
		data_to_annotate[i] =transform(tmp)
	y_train = transform_y(y_train)
	true_labels = transform_y(true_labels)
	data_train = data_train.numpy()
	data_to_annotate = data_to_annotate.numpy()
	y_train = y_train.numpy()
	true_labels = true_labels.numpy()
	
	# We may choose different sizes of data for training the classifier
	# Batch 1
	#rng = np.random.default_rng()
	#sel_indices = rng.choice(len(data_train),1000,replace=False)
	#data_train = data_train[sel_indices]
	#y_train = y_train[sel_indices]
	
	#machine_classifier_list=['LOGISTIC_REGRESSION','LINEAR_SVM','RBF_SVM','KNN_5','CNN','FCNN']
	if flag_preload_annotations:
		logger.info('Loading the saved annotations.............')
		annotations=np.load('data/machine_classifiers/annotations_mnist_machine_classifiers.npy')
		machine_classifier_list=np.load('data/machine_classifiers/machine_classifier_list.npy')
		acc=np.load('data/machine_classifiers/machine_classifier_acc.npy')
		logger.info('Accuracy of '+str(machine_classifier_list)+' = '+str(acc))
	else:
		machine_classifier_list=['LINEAR_SVM','CNN_5','KNN_3','FCNN_10','LOGISTIC_REGRESSION_N_100',\
								 'RBF_SVM','CNN_10','KNN_5','FCNN_15','LOGISTIC_REGRESSION_N_200',\
								 'POLY_SVM','CNN_15','KNN_7','FCNN_20','LOGISTIC_REGRESSION_N_300']
#								 'LINEAR_SVM','CNN_20','KNN_10','FCNN_25','LOGISTIC_REGRESSION_N_400']
		annotator_training_size_list =100*np.arange(1,5)
		train_args = {'learning_rate': 0.001,
					  'batch_size' : 512,
					  'n_epoch' : 10,
					  'logger' : logger}
		annotations = np.zeros((N,M))
		acc=[]
		for i in range(M):
			logger.info('Training '+machine_classifier_list[i]+'....................')
			if machine_classifier_list[i]=='LOGISTIC_REGRESSION_N_100':	
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=100
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='linear'		
				train_args['n_epoch']=10	
				y_pred=train_eval_NN_classifier(data_train_sel,y_train,data_to_annotate,train_args)			
			elif machine_classifier_list[i]=='LOGISTIC_REGRESSION_N_200':	
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=200
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='linear'		
				train_args['n_epoch']=10	
				y_pred=train_eval_NN_classifier(data_train_sel,y_train,data_to_annotate,train_args)		
			elif machine_classifier_list[i]=='LOGISTIC_REGRESSION_N_300':	
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=300
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='linear'		
				train_args['n_epoch']=10	
				y_pred=train_eval_NN_classifier(data_train_sel,y_train,data_to_annotate,train_args)			
			elif machine_classifier_list[i]=='LOGISTIC_REGRESSION_N_400':	
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=400
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='linear'		
				train_args['n_epoch']=10	
				y_pred=train_eval_NN_classifier(data_train_sel,y_train,data_to_annotate,train_args)					
			elif machine_classifier_list[i]=='LINEAR_SVM':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=9000
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['kernal_type']='linear'
				y_pred=train_eval_svm_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='RBF_SVM':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['kernal_type']='rbf'
				y_pred=train_eval_svm_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='POLY_SVM':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]			
				train_args['kernal_type']='poly'
				y_pred=train_eval_svm_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='KNN_5':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_training_size=9000
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['no_of_neighbors']=5
				y_pred=train_eval_knn_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='KNN_3':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['no_of_neighbors']=3
				y_pred=train_eval_knn_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='KNN_10':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['no_of_neighbors']=10
				y_pred=train_eval_knn_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='KNN_7':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['no_of_neighbors']=7
				y_pred=train_eval_knn_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='CNN_5':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='cnn'				
				train_args['n_epoch']=5				
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='CNN_10':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='cnn'				
				train_args['n_epoch']=10				
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='CNN_15':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='cnn'				
				train_args['n_epoch']=15				
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='CNN_20':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]
				train_args['network_type']='cnn'				
				train_args['n_epoch']=20				
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='FCNN_10':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]			
				train_args['network_type']='fcnn'
				train_args['n_epoch']=10
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='FCNN_15':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]			
				train_args['network_type']='fcnn'
				train_args['n_epoch']=15
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='FCNN_20':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]			
				train_args['network_type']='fcnn'
				train_args['n_epoch']=20
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			elif machine_classifier_list[i]=='FCNN_25':
				rng = np.random.default_rng(i)
				sel_training_size = rng.choice(annotator_training_size_list,1)
				sel_indices = rng.choice(len(data_train),sel_training_size,replace=False)	
				data_train_sel = data_train[sel_indices]
				y_train_sel = y_train[sel_indices]			
				train_args['network_type']='fcnn'
				train_args['n_epoch']=25
				y_pred=train_eval_NN_classifier(data_train_sel,y_train_sel,data_to_annotate,train_args)
			else:
				logger.info('Wrong classifier choice')
			acc.append(np.shape(np.argwhere(y_pred == true_labels))[0]/np.shape(true_labels)[0])
			logger.info('Training done with test accuracy = '+str(acc[i]))
			annotations[:,i]=y_pred
	np.save('data/machine_classifiers/annotations_mnist_machine_classifiers',annotations)
	np.save('data/machine_classifiers/machine_classifier_list',machine_classifier_list)
	np.save('data/machine_classifiers/machine_classifier_acc',acc)
	if annotator_label_pattern=='random':
		mask = np.random.binomial(1,p,np.shape(annotations))
		annotations=annotations+1
		annotations = annotations*mask
		annotations=annotations-1
		#annotations[annotations!=0]=annotations[annotations!=0]-1
		annotations = annotations.astype(int)
		
	else:
		logger.info('Wrong choice')
	f = np.zeros((N,M,K))
	annotator_label_mask = np.zeros((N,M))
	
	annotator_label = {}
	for i in range(M):
		annotator_label['softmax' + str(i) + '_label'] = np.zeros((N,K))  

	annotators_per_sample = []
	for n in range(N):
		a = np.argwhere(annotations[n,:]!= -1)
		annotators_per_sample.append(a[:,0])
		count=0
		for m in annotators_per_sample[n]:
			f[n,m,annotations[n,m]] = 1
			annotator_label_mask[n,m]=1
			annotator_label['softmax' + str(count) + '_label'][n] = f[n,m,:]
			count = count +1 
			
	answers_bin_missings = []
	for i in range(N):
		row = []
		for r in range(M):
			if annotations[i, r] == -1:
				row.append(0 * np.ones(K))
			else:
				row.append(one_hot(annotations[i, r], K)[0, :])
		answers_bin_missings.append(row)
	answers_bin_missings = np.array(answers_bin_missings)
	return f,annotations,answers_bin_missings,annotator_label,annotators_per_sample,annotator_label_mask
				
	
	
def get_estimation_error(A,A_true):
	M = np.shape(A)[0]
	error=0
	for i in range(M):
		row_ind, col_ind  = linear_sum_assignment(-np.dot(np.transpose(A[i]),A_true[i]))
		A[i]  = A[i,:,col_ind]
		error += np.sum(np.abs(A[i]-A_true[i]))/np.sum(np.abs(A_true[i]))
	error = error/M
	return error
	
def train_eval_NN_classifier(data_train,y_train,data_to_annotate,args):
	logger = args['logger']
	network_type = args['network_type']
	if network_type=='linear' or network_type=='fcnn':		
		data_train=data_train.reshape((data_train.shape[0],data_train.shape[2]*data_train.shape[3]))
		data_to_annotate=data_to_annotate.reshape((data_to_annotate.shape[0],data_to_annotate.shape[2]*data_to_annotate.shape[3]))
		input_dim = np.shape(data_train[0])[0]
		K = np.max(y_train)+1
		if network_type=='linear':
			model = LinearClassifier(input_dim,K)
		else:
			model = FCNN(input_dim,K,hidden_units=128,hidden_layers=1)
	elif network_type=='cnn':
		model = CNN()
	else:
		logger.info('Wrong choice for network type')

	train_data = list(zip(data_train,y_train))
	train_loader = DataLoader(dataset=train_data,
							  batch_size=args['batch_size'],
							  num_workers=4,
							  shuffle=True,
							  drop_last=False)	


	optimizer = optim.Adam(model.parameters(), lr=args['learning_rate'], weight_decay=1e-3)
	loss_function = torch.nn.CrossEntropyLoss()
	for epoch in range(args['n_epoch']):
		logger.info('epoch '+str(epoch))
		model.train()
		for batch_x,batch_y in train_loader:
			optimizer.zero_grad()
			y_pred = model.forward(batch_x.float())
			loss = loss_function(y_pred,batch_y)
			loss.backward()
			optimizer.step()
			
	with torch.no_grad():
		model.eval()
		y_pred = model.forward(torch.tensor(data_to_annotate).float())
		y_pred = torch.argmax(y_pred,dim=1)
		y_pred = y_pred.numpy()
	return y_pred
	



def train_eval_svm_classifier(data_train,y_train,data_to_annotate,args):
	data_train=data_train.reshape((data_train.shape[0],data_train.shape[2]*data_train.shape[3]))
	data_to_annotate=data_to_annotate.reshape((data_to_annotate.shape[0],data_to_annotate.shape[2]*data_to_annotate.shape[3]))
	clf = svm.SVC(kernel=args['kernal_type'])
	clf.fit(data_train,y_train)
	y_pred=clf.predict(data_to_annotate)
	return y_pred

	
	
def train_eval_knn_classifier(data_train,y_train,data_to_annotate,args):
	no_of_neighbors=args['no_of_neighbors']
	data_train=data_train.reshape((data_train.shape[0],data_train.shape[2]*data_train.shape[3]))
	data_to_annotate=data_to_annotate.reshape((data_to_annotate.shape[0],data_to_annotate.shape[2]*data_to_annotate.shape[3]))
	clf = KNeighborsClassifier(n_neighbors=no_of_neighbors)
	clf.fit(data_train,y_train)
	y_pred=clf.predict(data_to_annotate)
	return y_pred
		
			
		

#def download_cifar10():
#	fname = ['train.rec', 'train.lst', 'val.rec', 'val.lst']
#	testfile = urllib3.PoolManager()
#	with open(fname[0], 'wb') as out:
#		r = testfile.request('GET', 'http://data.mxnet.io/data/cifar10/cifar10_train.rec', preload_content=False)
#		shutil.copyfileobj(r, out)
#	with open(fname[1], 'wb') as out:
#		r = testfile.request('GET', 'http://data.mxnet.io/data/cifar10/cifar10_train.lst', preload_content=False)
#		shutil.copyfileobj(r, out)
#	with open(fname[2], 'wb') as out:
#		r = testfile.request('GET', 'http://data.mxnet.io/data/cifar10/cifar10_val.rec', preload_content=False)
#		shutil.copyfileobj(r, out)
#	with open(fname[3], 'wb') as out:
#		r = testfile.request('GET', 'http://data.mxnet.io/data/cifar10/cifar10_val.lst', preload_content=False)
#		shutil.copyfileobj(r, out)
#	return fname	 

def calculate_weights_for_cross_entropy(annotations):
	N = np.shape(annotations)[0]
	M = np.shape(annotations)[1]
	weights =[]
	for i in range(N):
		labels=annotations[i,annotations[i,:]!=-1]
		if np.shape(labels)[0]>0:
			mode_labels = stats.mode(labels)
			w1 = mode_labels.count[0]
			w2 = np.shape(labels)[0]
			if w1==0:
				w1=np.finfo(float).eps
		else:
			w1 = np.finfo(float).eps
			w2 = np.finfo(float).eps
		#print('#######################################')
		#print(w1)
		#print(w2)
		#print(w1*w2)
		#print('#######################################')

		weights.append(w1)
	weights = np.array(weights)
	
	return weights
	
def fill_missing_annotations_with_majority_voting(annotations):
	N = np.shape(annotations)[0]
	M = np.shape(annotations)[1]
	weights =[]
	for i in range(N):
		labels=annotations[i,annotations[i,:]!=-1]
		if np.shape(labels)[0]>100:
			mode_labels = stats.mode(labels,axis=None)
			label_sel = mode_labels.mode[0]
			label_count = mode_labels.count[0]
			if(label_count >=3):
				annotations[i,annotations[i,:]==-1]=label_sel


		#print('#######################################')
		#print(w1)
		#print(w2)
		#print(w1*w2)
		#print('#######################################')
	
	return annotations
			
			
def init_confusion_matrices(m):
	M = m.A.size()[0]
	K = m.A.size()[1]
	for i in range(M):
		m.A[i]= torch.eye(K)
		
def init_weights(m):
	if isinstance(m,nn.Linear):
		K = m.weight.data.size()[1]
		m.weight.data = torch.eye(K)

		

	
#def estimate_confusion_matrices_from_groundtruth(annotations,true_labels,K):
#	M = np.shape(annotations)[1]
#	A= np.zeros((M,K,K))
#	for i in range(M):
#		A[i] = confusion_matrix(true_labels, annotations[:,i])
#		#A[i]=np.matmul(A[i],np.diag(np.divide(1,np.sum(A[i],axis=0))))
#	return A
	
def calculate_factor_for_determinant(M,K):
	A= torch.zeros((M,K,K))
	for i in range(M):
		A[i] = 6*torch.eye(K)-5
	A = F.softplus(A)
	A = F.normalize(A,p=1,dim=1)
	W = A.view(M*K,K)
	WW = torch.mm(W.t(),W)
	regularizer_loss = torch.linalg.det(WW)
	factor = round(math.log10(regularizer_loss))
	return factor



	




def map_data(data):
	"""
	Map data to proper indices in case they are not in a continues [0, N) range

	Parameters
	----------
	data : np.int32 arrays

	Returns
	-------
	mapped_data : np.int32 arrays
	n : length of mapped_data

	"""
	uniq = list(set(data))

	id_dict = {old: new for new, old in enumerate(sorted(uniq))}
	data = np.array(list(map(lambda x: id_dict[x], data)))
	n = len(uniq)

	return data, id_dict, n

def one_hot(target, n_classes):
	targets = np.array([target]).reshape(-1)
	one_hot_targets = np.eye(n_classes)[targets]
	return one_hot_targets


def transform_onehot(answers, N_ANNOT, N_CLASSES, empty=-1):
	answers_bin_missings = []
	for i in range(len(answers)):
		row = []
		for r in range(N_ANNOT):
			if answers[i, r] == -1:
				row.append(empty * np.ones(N_CLASSES))
			else:
				row.append(one_hot(answers[i, r], N_CLASSES)[0, :])
		answers_bin_missings.append(row)
	answers_bin_missings = np.array(answers_bin_missings).swapaxes(1, 2)
	return answers_bin_missings
	
def multi_loss(y_true, y_pred, loss_fn=torch.nn.CrossEntropyLoss(reduce='mean').cuda()):
	mask = y_true != -1
	y_pred = torch.transpose(y_pred, 1, 2)
	loss = loss_fn(y_pred[mask], y_true[mask])
	return loss
	
	
def load_data(filename):
	with open(filename, 'rb') as f:
		data = np.load(f)

	f.close()
	return data
	
def confusion_matrix_init_mle_based(ep,M,K):

	#MLE initialization for expert confusion matrix. See apendix B.2 for the detail.

	sum_majority_prob = torch.zeros((M, K))

	expert_tmatrix = torch.zeros((M, K, K))

	ep = torch.tensor(ep)
	for j in range(ep.size()[0]):
		linear_sum_2 = torch.sum(ep[j], dim=0)
		prob_2 = linear_sum_2 / torch.sum(linear_sum_2)

		# prob_2 : all experts' majority voting

		for R in range(M):
			# If missing ....
			if max(ep[j, R]) == 0:
				continue
			_, expert_class = torch.max(ep[j, R], 0)
			expert_tmatrix[R, :, expert_class] += prob_2.float()
			sum_majority_prob[R] += prob_2.float()

	sum_majority_prob = sum_majority_prob + 1 * (sum_majority_prob == 0).float()
	for R in range(M):
		expert_tmatrix[R] = expert_tmatrix[R] / sum_majority_prob[R].unsqueeze(1)

	return expert_tmatrix
	
def kl_div(p, q):
    # p, q is in shape (batch_size, n_classes)
    return (p * p.log2() - p * q.log2()).sum(dim=1)


def symmetric_kl_div(p, q):
    return kl_div(p, q) + kl_div(q, p)


def js_div(p, q):
    # Jensen-Shannon divergence, value is in (0, 1)
    m = 0.5 * (p + q)
    return 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
	
	
"""
Copyright (C) 2014 Dallas Card

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
 to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or
substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.


Description:
Given unreliable observations of patient classes by multiple observers,
determine the most likely true class for each patient, class marginals,
and  individual error rates for each observer, using Expectation Maximization


References:
( Dawid and Skene (1979). Maximum Likelihood Estimation of Observer
Error-Rates Using the EM Algorithm. Journal of the Royal Statistical Society.
Series C (Applied Statistics), Vol. 28, No. 1, pp. 20-28. 
"""

	
def dawid_skene_em(counts, y, logger, tol=0.00001, max_iter=10, init='majority voting'):

    # Get parameters 
    patients = np.shape(counts)[0]
    observers = np.shape(counts)[1]
    classes=np.shape(counts)[2]
    
    # initialize
    iter = 0
    converged = False
    old_class_marginals = None
    old_error_rates = None

    patient_classes = majority_voting_initialization(counts)
    
      
    
    # while not converged do:
    while not converged:     
        iter += 1
        
        # M-step
        (class_marginals, error_rates) = m_step(counts, patient_classes)        
 
        # E-setp
        patient_classes = e_step(counts, class_marginals, error_rates)  
        
        # check likelihood
        log_L = calc_likelihood(counts, class_marginals, error_rates)
        
        # check for convergence
        if old_class_marginals is not None:
            class_marginals_diff = np.sum(np.abs(class_marginals - old_class_marginals))
            error_rates_diff = np.sum(np.abs(error_rates - old_error_rates))
            if (class_marginals_diff < tol and error_rates_diff < tol) or iter > max_iter:
                converged = True
        else:
            None
    
        # update current values
        old_class_marginals = class_marginals
        old_error_rates = error_rates
    
    pred_ds = np.argmax(patient_classes, axis=1)
    u = (pred_ds == y).sum()
    logger.info("Final train accuracy : %f" % (u/len(y)))
    return patient_classes,error_rates
        
    #return (patients, observers, classes, counts, class_marginals, error_rates, patient_classes) 
 


"""
Function: m_step()
    Get estimates for the prior class probabilities (p_j) and the error
    rates (pi_jkl) using MLE with current estimates of true patient classes
    See equations 2.3 and 2.4 in Dawid-Skene (1979)
Input: 
    counts: Array of how many times each response was received
        by each observer from each patient
    patient_classes: Matrix of current assignments of patients to classes
Returns:
    p_j: class marginals [classes]
    pi_kjl: error rates - the probability of observer k receiving
        response l from a patient in class j [observers, classes, classes]
"""
def m_step(counts, patient_classes):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    counts = np.asarray(counts)
    counts = counts.astype(int)
    # compute class marginals
    class_marginals = np.sum(patient_classes,0)/float(nPatients)
    
    # compute error rates 
    error_rates = np.zeros([nObservers, nClasses, nClasses])
    for k in range(nObservers):
        for j in range(nClasses):
            for l in range(nClasses): 
                error_rates[k, j, l] = np.dot(patient_classes[:,j], counts[:,k,l])
            # normalize by summing over all observation classes
            sum_over_responses = np.sum(error_rates[k,j,:])
            if sum_over_responses > 0:
                error_rates[k,j,:] = error_rates[k,j,:]/float(sum_over_responses)  

    return (class_marginals, error_rates)


""" 
Function: e_step()
    Determine the probability of each patient belonging to each class,
    given current ML estimates of the parameters from the M-step
    See equation 2.5 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each observer from each patient
    class_marginals: probability of a random patient belonging to each class
    error_rates: probability of observer k assigning a patient in class j 
        to class l [observers, classes, classes]
Returns:
    patient_classes: Soft assignments of patients to classes
        [patients x classes]
"""      
def e_step(counts, class_marginals, error_rates):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    counts = np.asarray(counts)
    counts = counts.astype(int)    
    patient_classes = np.zeros([nPatients, nClasses])    
    
    for i in range(nPatients):
        for j in range(nClasses):
            estimate = class_marginals[j]
            estimate *= np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))
            
            patient_classes[i,j] = estimate
        # normalize error rates by dividing by the sum over all observation classes
        patient_sum = np.sum(patient_classes[i,:])
        if patient_sum > 0:
            patient_classes[i,:] = patient_classes[i,:]/float(patient_sum)
    
    return patient_classes


"""
Function: calc_likelihood()
    Calculate the likelihood given the current parameter estimates
    This should go up monotonically as EM proceeds
    See equation 2.7 in Dawid-Skene (1979)
Inputs:
    counts: Array of how many times each response was received
        by each observer from each patient
    class_marginals: probability of a random patient belonging to each class
    error_rates: probability of observer k assigning a patient in class j 
        to class l [observers, classes, classes]
Returns:
    Likelihood given current parameter estimates
"""  
def calc_likelihood(counts, class_marginals, error_rates):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    log_L = 0.0
    counts = np.asarray(counts)
    counts = counts.astype(int)     
    for i in range(nPatients):
        patient_likelihood = 0.0
        for j in range(nClasses):
        
            class_prior = class_marginals[j]
            patient_class_likelihood = np.prod(np.power(error_rates[:,j,:], counts[i,:,:]))  
            patient_class_posterior = class_prior * patient_class_likelihood
            patient_likelihood += patient_class_posterior
                              
        temp = log_L + np.log(patient_likelihood)
        
        if np.isnan(temp) or np.isinf(temp):
            sys.exit()

        log_L = temp        
        
    return log_L
	
"""
Function: majority_voting_initialization()
    Alternative initialization # 2
    An alternative way to initialize assignment of patients to classes 
    i.e Get initial estimates for the true patient classes using majority voting
    This is not in the original paper, but could be considered
Input:
    counts: Counts of the number of times each response was received 
        by each observer from each patient: [patients x observers x classes] 
Returns:
    patient_classes: matrix of initial estimates of true patient classes:
        [patients x responses]
"""  
def majority_voting_initialization(counts):
    [nPatients, nObservers, nClasses] = np.shape(counts)
    # sum over observers
    response_sums = np.sum(counts,1)
    
    # create an empty array
    patient_classes = np.zeros([nPatients, nClasses])
    
    # take the most frequent class for each patient 
    for p in range(nPatients):        
        indices = np.argwhere(response_sums[p,:] == np.max(response_sums[p,:]))
        # in the case of ties, take the lowest valued label (could be randomized)        
        patient_classes[p, np.min(indices)] = 1
        
    return patient_classes 	
	
def estimate_confusion_matrices_from_groundtruth(annotations,labels):
	M = np.shape(annotations)[1]
	K = np.max(labels)+1
	A = np.zeros((M,K+1,K+1))
	AA= np.zeros((M,K,K))
	for i in range(M):
		A[i] = confusion_matrix(labels, annotations[:,i])
		AA[i] = A[i,1:K+1,1:K+1]
		A[i][A[i]==0]=1e-12
		A[i]=np.matmul(A[i],np.diag(np.divide(1,np.sum(A[i],axis=0))))
		A[i][np.isnan(A[i])]=0
	return AA	
	
def majority_voting(resp,y,logger):
    # computes majority voting label
    # ties are broken uniformly at random
    resp=np.asarray(resp)
    n = resp.shape[0]
    k = resp.shape[2]
    pred_mv = np.zeros((n), dtype = np.int)
    for i in range(n):
        # finding all labels that have got maximum number of votes
        poss_pred = np.where(np.sum(resp[i],0) == np.max(np.sum(resp[i],0)))[0]
        shuffle(poss_pred)
        # choosing a label randomly among all the labels that have got the highest number of votes
        pred_mv[i] = poss_pred[0]   
    pred_mv_vec = np.zeros((n,k))
    # returning one-hot representation of the majority vote label
    u = (pred_mv == y).sum()
    logger.info("Final train accuracy : %f" % (u/len(y)))
    pred_mv_vec[np.arange(n), pred_mv] = 1
    return pred_mv_vec
		
