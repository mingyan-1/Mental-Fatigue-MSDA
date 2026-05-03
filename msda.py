import torch
import numpy as np

def euclidean(x1,x2):
	return ((1e-6)+(x1.cuda()-x2.cuda())**2).sum().sqrt()

def k_moment(output_source, output_t, batch_size, k):
	output_s = torch.from_numpy(np.zeros((3, 8, output_source.shape[3])))
	for source_num in range(0, output_source.shape[3]):
		output_s[:, :, source_num] = (output_source[:, :, :, source_num]**k).mean(0)
	output_t = (output_t**k).mean(0)
	moment1 = 0
	for source_num1 in range(0, output_source.shape[3]):
		for source_num2 in range(0, source_num1):
			moment1 = moment1 + euclidean(output_s[:, :, source_num1], output_s[:, :, source_num2])
		moment1 = moment1 + euclidean(output_s[:, :, source_num1], output_t.double())
	return moment1

def msda_regulizer(output_source, output_t, batch_size, belta_moment):
	# print('s1:{}, s2:{}, s3:{}, s4:{}'.format(output_s1.shape, output_s2.shape, output_s3.shape, output_t.shape))
	output_s = torch.from_numpy(np.zeros((batch_size, 3, 8, output_source.shape[3])))
	for source_num in range(0, output_source.shape[3]):
		s_mean = output_source[:, :, :, source_num].squeeze().mean(0)
		output_s[:, :, :, source_num] = output_source[:, :, :, source_num] - s_mean
	t_mean = output_t.mean(0)
	output_t = output_t - t_mean
	moment1 = 0
	for source_num1 in range(0, output_source.shape[3]):
		for source_num2 in range(0, source_num1):
			moment1 = moment1 + euclidean(output_s[:, :, :, source_num1], output_s[:, :, :, source_num2])
		moment1 = moment1 + euclidean(output_s[:, :, :, source_num1], output_t.double())
	reg_info = moment1
	#print(reg_info)
	for i in range(belta_moment-1):
		reg_info += k_moment(output_s, output_t, batch_size, i+2)
	
	return reg_info/(((output_source.shape[3]+1)*output_source.shape[3])/2)
	#return euclidean(output_s1, output_t)

