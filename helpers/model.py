import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.hub as hub
import torchvision.models as models
from torch.autograd import Variable
import math
from helpers.vgg import *
from torchvision import transforms


class FCNN(nn.Module):
	def __init__(self,R,K,hidden_units,hidden_layers):
		super(FCNN,self).__init__()
		layer_list=[]
		n_in=R
		for i in range(hidden_layers):
			layer_list.append(nn.Linear(n_in,hidden_units))
			layer_list.append(nn.ReLU(inplace=False))
			n_in = hidden_units
		layer_list.append(nn.Linear(hidden_units,K))
		self.layers=nn.Sequential(*layer_list)
		
	def forward(self,x):
		x = self.layers(x)
		p = F.softmax(x,dim=1)
		return p
		
		
		



		

class CrowdNetwork(nn.Module):
	def __init__(self,input_dim,M,K,fnet_type,init_method,A_init):
		super(CrowdNetwork,self).__init__()
		if fnet_type=='fcnn_dropout':
			self.fnet = FCNN_Dropout(input_dim,K)
		elif fnet_type=='lenet':
			self.fnet = Lenet()
		elif fnet_type=='fcnn_dropout_batchnorm':
			self.fnet = FCNN_Dropout_BatchNorm(input_dim,K)
		elif fnet_type=='linear':
			self.fnet = LinearClassifier(input_dim,K)
		elif fnet_type=='resnet9':
			self.fnet = ResNet9(K)
		elif fnet_type=='resnet18':
			self.fnet = ResNet18(K)
		elif fnet_type=='resnet34':
			self.fnet = ResNet34(K)
		else:
			self.fnet = FCNN_Dropout()
		if init_method=='identity':
			self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)
		elif init_method=='mle_based':
			self.P = nn.Parameter(torch.stack([A_init[m] for m in range(M)]), requires_grad=True)
		else:
			self.P = nn.Parameter(torch.stack([torch.eye(K) for _ in range(M)]), requires_grad=True)
		#self.A = None
	def forward(self,x):
		x = self.fnet(x)
		#A = F.softplus(self.P)
		#A = F.normalize(A,p=1,dim=1)
		A = F.softmax(self.P,dim=1)
		y = torch.einsum('ij, bkj -> ibk',x,A)
		#y = F.softmax(y,dim=2)
		return(x,y,A)
		
		

		


class confusion_matrices(nn.Module):
	def __init__(self, device, M, K, init_method='close_to_identity', projection_type='simplex_projection',A_init=[]):
		super(confusion_matrices, self).__init__()
		if init_method=='close_to_identity':		
			P = torch.zeros(M, K, K)
			for i in range(M):
				P[i] = torch.eye(K)#6*torch.eye(K)-5#+ P[i]*C
		elif init_method=='from_file':
			H=torch.tensor(np.loadtxt('A_matrix_init.txt',delimiter=','))
			P =torch.zeros(M,K,K)
			for i in range(M):
				P[i] = H[i*K:(i+1)*K,:]#+ P[i]*C
			print(P.size())
		elif init_method=='mle_based':
			P = torch.log(A_init+0.01)
		elif init_method=='dsem_based':
			P = torch.tensor(A_init).float()
		elif init_method=='deviation_from_identity':
			P = -2*torch.ones(M, K, K)
			C = torch.ones(K, K)
			ind = np.diag_indices(C.shape[0])
			C[ind[0], ind[1]] = torch.zeros(C.shape[0])
			self.C = C.to(device)
			self.identity = torch.eye(K).to(device)		
		self.register_parameter(name='W', param=nn.parameter.Parameter(P))
		self.W.to(device)
		self.M = M
		self.K = K
		self.projection_type=projection_type

	def forward(self):
		if self.projection_type=='simplex_projection':	
			A = F.softplus(self.W)
			A = F.normalize(A.clone(),p=1,dim=1)
		elif self.projection_type=='sigmoid_projection':
			sig = torch.sigmoid(self.W)
			A = torch.zeros((self.M, self.K, self.K))
			for i in range(self.M):
				A[i] = self.identity.detach() + sig[i,:,:]*self.C.detach()
			A = F.normalize(A.clone(), p=1, dim=1)						
		elif self.projection_type=='softmax':
			A = F.softmax(self.W,dim=1)
		else:
			A = []

		return A
		
class confusion_matrices_tracereg(nn.Module):
	def __init__(self, device, M, K, init_method='close_to_identity', A_init=[]):
		super(confusion_matrices_tracereg, self).__init__()
		if init_method=='close_to_identity':		
			P = torch.sigmoid(-10*torch.ones(M, K, K))
			C = torch.ones(K, K)
			ind = np.diag_indices(C.shape[0])
			C[ind[0], ind[1]] = torch.zeros(C.shape[0])
			for i in range(M):
				P[i] = torch.eye(K)#+ P[i]*C
		elif init_method=='from_file':
			H=torch.tensor(np.loadtxt('A_matrix_init.txt',delimiter=','))
			P =torch.zeros(M,K,K)
			for i in range(M):
				P[i] = H[i*K:(i+1)*K,:]#+ P[i]*C
			print(P.size())
		elif init_method=='mle_based':
			P = torch.log(A_init+0.01)
		self.register_parameter(name='W', param=nn.parameter.Parameter(P))
		self.W.to(device)
		self.M = M
		self.K = K

	def forward(self):	
		A = F.relu(self.W)
		A = F.normalize(A.clone(),p=1,dim=1)
		#A = F.softmax(self.W,dim=1)

		return A

class confusion_matrix(nn.Module):
	def __init__(self, device, K, init=2):
		super(confusion_matrix, self).__init__()		
		self.register_parameter(name='w', param=nn.parameter.Parameter(-init*torch.ones(K, K)))
		self.w.to(device)
		co = torch.ones(K, K)
		ind = np.diag_indices(co.shape[0])
		co[ind[0], ind[1]] = torch.zeros(co.shape[0])
		self.co = co.to(device)
		self.identity = torch.eye(K).to(device)
		self.K = K


	def forward(self):
		sig = torch.sigmoid(self.w)
		A = self.identity.detach() + sig*self.co.detach()
		A = F.normalize(A.clone(), p=1, dim=0)
		return A

class Lenet(nn.Module):

	def __init__(self):
		super(Lenet, self).__init__()
		self.conv1 = nn.Conv2d(1,6,5,stride=1,padding=2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1   = nn.Linear(400, 120)
		self.fc2   = nn.Linear(120, 84)
		self.fc3   = nn.Linear(84, 10)


	def forward(self, x):

		out = F.relu(self.conv1(x))
		out = F.max_pool2d(out, 2)
		out = F.relu(self.conv2(out))
		out = F.max_pool2d(out, 2)
		out = out.view(out.size(0), -1)
		out = F.relu(self.fc1(out))
		out = F.relu(self.fc2(out))
		out = self.fc3(out)

		clean = F.softmax(out, 1)


		return clean
		
#class Net(nn.Module):
#	def __init__(self):
#		super(Net, self).__init__()
#		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#		self.conv2_drop = nn.Dropout2d()
#		self.fc1 = nn.Linear(320, 50)
#		self.fc2 = nn.Linear(50, 10)
#
#	def forward(self, x):
#		x = F.relu(F.max_pool2d(self.conv1(x), 2))
#		x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#		x = x.view(-1, 320)
#		x = F.relu(self.fc1(x))
#		x = F.dropout(x, training=self.training)
#		x = self.fc2(x)
#		return F.log_softmax(x)
		


class CNN(nn.Module):
	def __init__(self):
		super(CNN, self).__init__()
		self.conv1 = nn.Conv2d(1, 32, 3, 1)
		self.conv2 = nn.Conv2d(32, 64, 3, 1)
		self.dropout1 = nn.Dropout(0.25)
		self.dropout2 = nn.Dropout(0.5)
		self.fc1 = nn.Linear(9216, 128)
		self.fc2 = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv1(x)
		x = F.relu(x)
		x = self.conv2(x)
		x = F.relu(x)
		x = F.max_pool2d(x, 2)
		x = self.dropout1(x)
		x = torch.flatten(x, 1)
		x = self.fc1(x)
		x = F.relu(x)
		x = self.dropout2(x)
		x = self.fc2(x)
		output = F.log_softmax(x, dim=1)
		return output
		
class LinearClassifier(nn.Module):
	def __init__(self,input_dim,K):
		super(LinearClassifier,self).__init__()
		self.linear = nn.Linear(input_dim,K)
		
	def forward(self,x):
		x=self.linear(x)
		x=F.softmax(x,dim=1)
		return x
		

class FCNN_Dropout(nn.Module):

	def __init__(self,input_dim,K):
		super(FCNN_Dropout, self).__init__()
		layer_list = []
		layer_list.append(nn.Flatten(start_dim=1))
		layer_list.append(nn.Linear(input_dim, 128))
		layer_list.append(nn.ReLU(inplace=False))
		layer_list.append(nn.Dropout(0.5))
		layer_list.append(nn.Linear(128, K))
		self.layers=nn.Sequential(*layer_list)
	
	def forward(self,x):
		out = self.layers(x)
		out = F.softmax(out,dim=1)
		return out
		
class SequentialDataModel(nn.Module):

	def __init__(self,num_words,):
		super(SequentialDataModel, self).__init__()
		layer_list = []
		layer_list.append(nn.Embedding(num_words,300))
		layer_list.append(nn.Conv1d(512,5))
		layer_list.append(nn.ReLU(inplace=False))
		layer_list.append(nn.Dropout(0.5))
		layer_list.append(nn.Linear(128, K))
		self.layers=nn.Sequential(*layer_list)
	
	def forward(self,x):
		out = self.layers(x)
		out = F.softmax(out,dim=1)
		return out
		
def build_base_model():
	base_model = Sequential()
	base_model.add(Embedding(num_words,
						300,
						weights=[embedding_matrix],
						input_length=maxlen,
						trainable=True))
	base_model.add(Conv1D(512, 5, padding="same", activation="relu"))
	base_model.add(Dropout(0.5))
	base_model.add(GRU(50, return_sequences=True))
	base_model.add(TimeDistributed(Dense(N_CLASSES, activation='softmax')))
	base_model.compile(loss='categorical_crossentropy', optimizer='adam')

	return base_model
		
class FCNN_Dropout_BatchNorm(nn.Module):

	def __init__(self,input_dim,K):
		super(FCNN_Dropout_BatchNorm, self).__init__()
		layer_list = []
		layer_list.append(nn.Flatten(start_dim=1))
		layer_list.append(nn.BatchNorm1d(input_dim,affine=False))
		layer_list.append(nn.Linear(input_dim, 128))
		layer_list.append(nn.ReLU(inplace=False))
		layer_list.append(nn.Dropout(0.5))
		layer_list.append(nn.BatchNorm1d(128,affine=False))
		layer_list.append(nn.Linear(128, K))
		self.layers=nn.Sequential(*layer_list)
	
	def forward(self,x):
		out = self.layers(x)
		out = F.softmax(out,dim=1)
		return out
		
#class FCNN_DropoutNosoftmax(nn.Module):
#
#	def __init__(self,input_dim,K):
#		super(FCNN_DropoutNosoftmax, self).__init__()
#		layer_list = []
#		layer_list.append(nn.Flatten(start_dim=1))
#		layer_list.append(nn.Linear(input_dim, 128))
#		layer_list.append(nn.ReLU(inplace=False))
#		layer_list.append(nn.Dropout(0.5))
#		self.layers=nn.Sequential(*layer_list)
#	
#	def forward(self,x):
#		out = self.layers(x)
#		return out
		
		
class FCNN_DropoutNosoftmax(nn.Module):

	def __init__(self,input_dim,K):
		super(FCNN_DropoutNosoftmax, self).__init__()
		layer_list = []
		layer_list.append(nn.Flatten(start_dim=1))
		layer_list.append(nn.Linear(input_dim, 128))
		layer_list.append(nn.ReLU(inplace=False))
		layer_list.append(nn.Dropout(0.5))
		layer_list.append(nn.Linear(128, K))
		self.layers=nn.Sequential(*layer_list)
	
	def forward(self,x):
		out = self.layers(x)
		#out = F.softmax(out,dim=1)
		return out
		
		
class FeatureExtractor(nn.Module):

	def __init__(self, base_model='inception'):
		super(FeatureExtractor, self).__init__()
		if base_model== 'vgg16':
			self.model = models.vgg16(pretrained=True)
		else:
			self.model = hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=True)
			self.model.fc = nn.Identity()

	def forward(self, x):
		feature = self.model(x)
		if self.training:
			feature = feature.logits
		return feature


class Weights(nn.Module):

	def __init__(self, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None):
		super(Weights, self).__init__()
		self.weight_type = weight_type
		if self.weight_type == 'W':
			self.weights = nn.Parameter(torch.ones(n_annotators), requires_grad=True)
		elif self.weight_type == 'I':
			if bottleneck_dim is None:
				self.weights = nn.Linear(feature_dim, n_annotators)
			else:
				self.weights = nn.Sequential(nn.ReLU(), nn.Linear(feature_dim, bottleneck_dim), nn.Linear(bottleneck_dim, n_annotators))
		else:
			raise IndexError("weight type must be 'W' or 'I'.")

	def forward(self, feature):
		if self.weight_type == 'W':
			return self.weights
		else:
			return self.weights(feature).view(-1)


class DoctorNet(nn.Module):

	def __init__(self, n_classes, n_annotators, weight_type='W', feature_dim=2048, bottleneck_dim=None, base_model='inception'):
		super(DoctorNet, self).__init__()
		self.feature_extractor = FeatureExtractor(base_model)
		self.annotators = nn.Parameter(torch.stack([torch.randn(feature_dim, n_classes) for _ in range(n_annotators)]), requires_grad=True)
		self.weights = Weights(n_annotators, weight_type, feature_dim, bottleneck_dim)
		
	def forward(self, x, pred=False, weight=False):
		feature = self.feature_extractor(x)
		decisions = torch.einsum('ik,jkl->ijl', feature, self.annotators)
		weights = self.weights(feature)
		if weight:
			decisions = decisions * weights[None, :, None]
		if pred:
			decisions = torch.sum(decisions, axis=1)
			return decisions
		else:
			return decisions, weights
			
			

				
class ResNet(nn.Module):
	def __init__(self, block, num_blocks, num_classes):
		super(ResNet, self).__init__()
		self.in_planes = 64
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
		self.linear = nn.Linear(512*block.expansion, num_classes)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1]*(num_blocks-1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)

	def forward(self, x, revision=True):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avgpool(out)
		out = out.view(out.size(0), -1)

		out = self.linear(out)

		clean = F.softmax(out, 1)

		return clean
		
def ResNet18(num_classes):
	return ResNet(BasicBlock, [2,2,2,2], num_classes)
	
def ResNet34(num_classes):
	return ResNet(BasicBlock, [3,4,6,3], num_classes)
	
	
class ResNet9(nn.Module):
	def __init__(self, num_classes):
		super().__init__()
		
		# Why does the input size ratchet up to 512 like this??
		self.conv1 = conv_block(3, 64)
		self.conv2 = conv_block(64, 128, pool=True)
		self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
		
		self.conv3 = conv_block(128, 256, pool=True)
		self.conv4 = conv_block(256, 512, pool=True)
		self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
		
		self.classifier = nn.Sequential(nn.MaxPool2d(4), 
										nn.Flatten(), 
										nn.Linear(512, num_classes))
		
	def forward(self, xb):
		out = self.conv1(xb)
		out = self.conv2(out)
		out = self.res1(out) + out
		out = self.conv3(out)
		out = self.conv4(out)
		out = self.res2(out) + out
		out = self.classifier(out)
		out = F.softmax(out, 1)
		return out
		
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)
	
class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)

		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion*planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion*planes)
			)

	def forward(self, x):
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out
