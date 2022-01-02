import argparse 
import os 
import csv 
import torch 
import sys
import librosa
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np 
# from models.models import ResNeXTSpoof, ConvolutionLSTM

class BottleneckBlock(nn.Module): 
	def __init__(self, in_c, kernel_size, bottleneck_depth):
		super().__init__()
		self.conv1 = nn.Conv1d(in_c, bottleneck_depth, kernel_size=1)
		self.conv2 = nn.Conv1d(bottleneck_depth, bottleneck_depth, kernel_size=kernel_size, padding=((kernel_size-1)//2))
		self.conv3 = nn.Conv1d(bottleneck_depth, in_c, kernel_size=1)

		self.bn1 = nn.BatchNorm1d(bottleneck_depth)
		self.bn2 = nn.BatchNorm1d(bottleneck_depth)
		self.bn3 = nn.BatchNorm1d(in_c)

	def forward(self, x): 
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		return x

class ResNeXTBlock(nn.Module):
	def __init__(self, in_c, bottleneck_depth):
		super().__init__()
		self.b_block_1 = BottleneckBlock(in_c, 3, bottleneck_depth) 
		self.b_block_1b = BottleneckBlock(in_c, 5, bottleneck_depth) #Not in original
		self.b_block_2 = BottleneckBlock(in_c, 7, bottleneck_depth)
		self.b_block_2b = BottleneckBlock(in_c, 9, bottleneck_depth) #Not in original
		self.b_block_3 = BottleneckBlock(in_c, 11, bottleneck_depth)
		self.b_block_3b = BottleneckBlock(in_c, 13, bottleneck_depth) #Not in original
		self.b_block_4 = BottleneckBlock(in_c, 15, bottleneck_depth)
		self.b_block_4b = BottleneckBlock(in_c, 17, bottleneck_depth) #Not in original
		self.b_block_5 = BottleneckBlock(in_c, 19, bottleneck_depth)

	def forward(self, x): 
		x1 = self.b_block_1(x)
		x2 = self.b_block_2(x)
		x3 = self.b_block_3(x)
		x4 = self.b_block_4(x)
		x5 = self.b_block_5(x)
		x1b = self.b_block_1b(x)
		x2b = self.b_block_2b(x)
		x3b = self.b_block_3b(x)
		x4b = self.b_block_4b(x)
		combo_x = x1 + x2 + x3 + x4 + x5 + x1b + x2b + x3b + x4b
		return x + combo_x

class ResNeXTSpoof(nn.Module):
	def __init__(self, num_features, dense_dim=256, bottleneck_depth=16):
		super(ResNeXTSpoof, self).__init__() 
		self.layers = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=10, stride=5, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=8, stride=4, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True),

			ResNeXTBlock(dense_dim, bottleneck_depth),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			# ResNeXTBlock(dense_dim, bottleneck_depth),
			# ResNeXTBlock(dense_dim, bottleneck_depth),
			# ResNeXTBlock(dense_dim, bottleneck_depth),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(),
			nn.AdaptiveAvgPool1d(1)
		)
		self.classifier = nn.Linear(dense_dim, dense_dim)
		self.classifier_2 = nn.Linear(dense_dim, 1)

		self.layers.apply(self.init_weights) 
		self.classifier.apply(self.init_weights)
		self.classifier_2.apply(self.init_weights)

	def init_weights(self,m): 
		if type(m) == nn.Conv1d: 
			nn.init.xavier_uniform_(m.weight) 
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch): 
		batch = self.layers(batch) 
		#print("Batch shape: {}".format(batch.shape))
		y_pred = self.classifier_2(F.relu(self.classifier(batch.transpose(1,2))))
		return y_pred

	@staticmethod 
	def get_param_size(model): 
		params = 0 
		for p in model.parameters(): 
			tmp = 1 
			for x in p.size(): 
				tmp *= x 
			params += tmp 
		return params

	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package

	def predict(self, batch): 
		batch = self.layers(batch)
		y_pred = self.classifier_2(F.relu(self.classifier(batch.transpose(1,2))))
		return torch.sigmoid(y_pred)

class ConvolutionLSTM(nn.Module):
	def __init__(self, num_features, dense_dim, hidden_dim, dropout=0.25):
		super(ConvolutionLSTM, self).__init__()

		self.hidden_dim = hidden_dim
		self.dropout = dropout
		self.dense_dim = dense_dim
		self.feature_extractor = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=10, stride=5, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=8, stride=4, bias=False),
			nn.BatchNorm1d(dense_dim), 
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True),
		)

		self.lstm_layer = torch.nn.LSTM(
			input_size=dense_dim, 
			hidden_size=hidden_dim,
			batch_first=True,  
			bidirectional=True
		)

		self.classifier = nn.Linear(hidden_dim, 1)

	def forward(self, batch):
		#Fake initial h_i and initial c_i (both shape: (2, N, hidden_dim))
		h_i = torch.randn((2, batch.shape[0], self.hidden_dim)).cuda()
		c_i = torch.randn((2, batch.shape[0], self.hidden_dim)).cuda()

		#Perform feature extraction. Output shape: (N x C x L)
		features = self.feature_extractor(batch)

		#Transpose into (N x L x C)
		features = torch.transpose(features, 1, 2)

		#Get lstm output (h_i: 2 x N x hidden_dim)
		_, (hidden_state, _) = self.lstm_layer(features, (h_i, c_i))
		hidden_state = hidden_state[0, :, :] + hidden_state[1, :, :]

		y_pred = self.classifier(hidden_state)
		return y_pred

	@staticmethod 
	def get_param_size(model): 
		params = 0
		for p in model.parameters():
			tmp = 1
			for x in p.size():
				tmp *= x 
			params += tmp 
		return params

	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package

	def predict(self, batch): 
		h_i = torch.randn((2, batch.shape[0], self.hidden_dim)).cuda()
		c_i = torch.randn((2, batch.shape[0], self.hidden_dim)).cuda()

		#Perform feature extraction. Output shape: (N x C x L)
		features = self.feature_extractor(batch)

		#Transpose into (N x L x C)
		features = torch.transpose(features, 1, 2)

		#Get lstm output (h_i: 2 x N x hidden_dim)
		_, (hidden_state, _) = self.lstm_layer(features, (h_i, c_i))
		hidden_state = hidden_state[0, :, :] + hidden_state[1, :, :]

		y_pred = self.classifier(hidden_state)

		return torch.sigmoid(y_pred)

class ResNeXTSpoof_Window_Binary(nn.Module):
	def __init__(self, num_features, dense_dim=256, bottleneck_depth=16):
		super(ResNeXTSpoof_Window_Binary, self).__init__() 
		self.layers = nn.Sequential(
			nn.Conv1d(num_features, dense_dim, kernel_size=10, stride=5, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=8, stride=4, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True), 

			nn.Conv1d(dense_dim, dense_dim, kernel_size=4, stride=2, bias=False),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(inplace=True),

			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),
			ResNeXTBlock(dense_dim, bottleneck_depth),
			nn.MaxPool1d(kernel_size=2),
			nn.Dropout(0.25),

			nn.Conv1d(dense_dim, dense_dim, kernel_size=1),
			nn.BatchNorm1d(dense_dim),
			nn.ReLU(),
			nn.Dropout(0.25),	
		)
		self.classifier = nn.Conv1d(dense_dim, 2, kernel_size=1)

		self.layers.apply(self.init_weights) 
		self.classifier.apply(self.init_weights) 

	def init_weights(self, m): 
		if type(m) == nn.Conv1d:
			nn.init.xavier_uniform_(m.weight)
			if m.bias is not None: 
				m.bias.data.fill_(0.01) 

	def forward(self, batch): 
		#Need to figure how to do windowing
		y_pred = self.layers(batch) 
		y_pred = self.classifier(y_pred)
		return y_pred

	@staticmethod
	def get_param_size(model): 
		params = 0
		for p in model.parameters():
			tmp = 1
			for x in p.size():
				tmp *= x 
			params += tmp 
		return params 

	def predict(self, batch): 
		y_pred = self.layers(batch) 
		y_pred = self.classifier(y_pred)
		#return torch.sigmoid(y_pred) 
		return y_pred

	@staticmethod 
	def serialize(model, optimizer=None, epoch=None, iteration=None, loss_results=None, 
		cer_results=None, wer_results=None, avg_loss=None, meta=None):
		package = {
			'state_dict': model.state_dict(), 
		}
		if optimizer is not None: 
			package['optim_dict'] = optimizer.state_dict() 
		if avg_loss is not None: 
			package['avg_loss'] = avg_loss
		if epoch is not None: 
			package['epoch'] = epoch + 1
		if iteration is not None: 
			package['iteration'] = iteration 
		if loss_results is not None: 
			package['loss_results'] = loss_results 
		if meta is not None: 
			package['meta'] = meta
		return package

def to_np(x): 
	"""Convert tensor x to numpy array"""
	return x.cpu().numpy()

# model eg. models\resnext_wide_ft.pth
def predict(wav_path, model_path, device=torch.device("cuda"), model="resnext"):
	"""
	wav_path: path to the wav file for inference
	model_path: path to the pytorch .pth file
	device: defaulted to CUDA, but if using CPU, change to torch.device("cpu")
	model: either "resnext" or "lstm"
	Return: a floating point number between 0 and 1, where 0: real, 1 is fake
	"""
	#Read the file and resample
	raw_audio, _ = librosa.load(wav_path, sr=16000)

	#Now load the model
	package = torch.load(model_path)
	state_dict = package['state_dict']
	if model == "resnext": 
		model_resnext = ResNeXTSpoof(num_features=1, dense_dim=256, bottleneck_depth=16)
	elif model == "lstm": 
		model_resnext = ConvolutionLSTM(num_features=1, dense_dim=256, hidden_dim=256, dropout=0.25)
	else: 
		print("Invalid model type. Currently support 'resnext' and 'lstm'") 
		return -1
	model_resnext.load_state_dict(state_dict) 

	#Now, put the model to device
	model_resnext.to(device)
	#Set to test mode
	model_resnext.eval()

	with torch.no_grad(): 
		#Now turn the audio numpy to something usable by pytorch
		input_tensor = torch.zeros(1, len(raw_audio), 1) #1 sample x t timesteps x 1 channel
		input_tensor[0].narrow(0, 0, len(raw_audio)).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 1)))

		input_tensor = input_tensor.float().to(device) 
		out = model_resnext(input_tensor.transpose(1, 2))

		y_score = to_np(torch.sigmoid(out).view(input_tensor.shape[0]))

	return y_score[0]


def predict_window(wav_path, model_path, device=torch.device("cuda")):
	"""
	wav_path: path to the wav file for inference
	model_path: path to the pytorch .pth file
	device: defaulted to CUDA, but if using CPU, change to torch.device("cpu")
	model: either "resnext" or "lstm"
	Return: a floating point number between 0 and 1, where 0: real, 1 is fake
	"""
	#Read the file and resample
	raw_audio, _ = librosa.load(wav_path, sr=16000)

	#Now load the model
	package = torch.load(model_path)
	state_dict = package['state_dict']
	
	model_resnext = ResNeXTSpoof_Window_Binary(num_features=1, dense_dim=256, bottleneck_depth=16)

	model_resnext.load_state_dict(state_dict) 

	#Now, put the model to device
	model_resnext.to(device)
	#Set to test mode
	model_resnext.eval()

	with torch.no_grad(): 
		#Now turn the audio numpy to something usable by pytorch
		input_tensor = torch.zeros(1, len(raw_audio), 1) #1 sample x t timesteps x 1 channel
		input_tensor[0].narrow(0, 0, len(raw_audio)).copy_(torch.reshape(torch.from_numpy(raw_audio), (-1, 1)))

		input_tensor = input_tensor.float().to(device) 
		out = model_resnext(input_tensor.transpose(1, 2))
		out = to_np(torch.sigmoid(out)[:,1,:])

		print(out.shape)
		#y_score = to_np(torch.sigmoid(out).view(input_tensor.shape[0]))

	return out[0]
