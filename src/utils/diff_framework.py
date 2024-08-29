"""Collection of utility functions"""
from scipy.fftpack import fft
# from utils.constants import *

import random
import torch
import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import sys
from   torch.autograd import Variable
from   utils.augmentation_operations import *


class DataAugment(Data.Dataset):
    def  __init__(self, x, y, aug_methods, pos_num):
        x_torch = torch.FloatTensor(x)
        y_torch = torch.tensor(y).long()
        self.data  = x_torch
        self.label = y_torch
        self.aug_methods = aug_methods
        self.pos_num     = pos_num

    def __getitem__(self, index):
        data_x = self.data[index]
        for aug_method in self.aug_methods:
            if dispatcher.get(aug_method) is not None:
                data_x = dispatcher[aug_method](data_x, pos_num=self.pos_num)
        data = (data_x, self.label[index])
        return data

    def __len__(self):
        return self.data.shape[0]

# DifDataAugment
class DifDataAugment(Data.Dataset):
    
    def  __init__(self, x, y, pos_num, magnitudes, policies, search_flag=True):
        x_torch    = torch.FloatTensor(x)
        y_torch    = torch.tensor(y).long()
        self.data  = x_torch
        self.label = y_torch
        self.pos_num      = pos_num
        self.magnitudes   = magnitudes
        self.policies     = policies
        self.search_flag  = search_flag

    def __getitem__(self, index):
        
        data_x              = self.data[index]
        if self.search_flag:
            magnitudes          = self.magnitudes.clamp(0, 1)
            probabilities_index = self.probabilities_index
            for i, sub_policy in enumerate(self.policies):
                for j, aug_method in enumerate(sub_policy):
                    if probabilities_index[i][j].item() == 1:
                        if dispatcher.get(aug_method) is not None:
                            data_x = dispatcher[aug_method](torch.FloatTensor(data_x), pos_num=self.pos_num, magnitude=magnitudes[i][j])
        data = (data_x, self.label[index])
        
        return data

    def __len__(self):
        return self.data.shape[0]

class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_data, probability, probability_index, magnitude):
        index    = sum( p_i.item()<<i for i, p_i in enumerate(probability_index))
        com_data = 0
        data     = origin_data
        adds     = 0

        for selection in range(2**len(self.sub_policy)):
            trans_probability = 1
            for i in range(len(self.sub_policy)):
                if selection & (1<<i):
                    trans_probability = trans_probability * probability[i]
                    if selection == index:
                        data = data - magnitude[i]
                        adds = adds + magnitude[i]
                else:
                    trans_probability = trans_probability * ( 1 - probability[i] )
            if selection == index:
                data = data.detach() + adds
                com_data = com_data + trans_probability * data
            else:
                com_data = com_data + trans_probability

        return com_data

class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_polices):
        self._ops = nn.ModuleList()
        self._nums = len(sub_polices)
        for sub_policy in sub_polices:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_data, probabilities, probabilities_index, magnitudes, weights=torch.tensor([1,1])):
        return sum(w * op(origin_data, p, p_i, m)
                   for i, (p, p_i, m, w, op) in
                   enumerate(zip(probabilities, probabilities_index, magnitudes, weights, self._ops)))/2

def _concat(xs):
    return torch.cat([x.contiguous().view(-1) for x in xs])

class Architect(object):

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.augment_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999),
                                          weight_decay=args.arch_weight_decay)
        self.scheduler_arch = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=6, gamma=0.5)

    def _compute_unrolled_model(self, input, target, eta, network_optimizer):
        loss   = self._loss_function(self.model(input, test_flag=True)[0], target) # compute train loss
        theta  = _concat(self.model.param_list).data.detach()
        moment = torch.zeros_like(theta)
        dtheta         = _concat(torch.autograd.grad(loss, self.model.param_list)).data.detach() + self.network_weight_decay * theta
        theta_mid      = theta - eta * (moment + dtheta)              # calculate w'，corrsponding to w' = w − ζ▽wE[Ltrain(w; d)]
        unrolled_model = self._construct_model_from_theta(theta_mid)  # obtain w'-based model，which is used to calculate ▽w'Lval(w')
        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, loss_function, network_optimizer, unrolled): ## eta: learning rate
        self._loss_function = loss_function
        self.optimizer.zero_grad()   ## update DA policy
        if unrolled:
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            self._backward_step(input_valid, target_valid)
        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self._loss_function(self.model(input_valid, test_flag=True)[0], target_valid)
        loss.backward(retain_graph=True)

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        unrolled_model = self._compute_unrolled_model(input_train, target_train, eta, network_optimizer)
        unrolled_model.set_augmenting(False)                             # now d is fixed
        unrolled_loss  = self._loss_function(unrolled_model(input_valid, test_flag=True)[0], target_valid) # calcualte Lval(w')

        unrolled_loss.backward()
        dalpha = []
        vector = [v.grad.data.detach() for v in unrolled_model.parameters()] # calculate ▽w'Lval(w')
        implicit_grads = self._hessian_vector_product(vector, input_train, target_train)
        for ig in implicit_grads:
            if ig is None:
                dalpha += [None]
            else:
                dalpha += [-ig]

        for v, g in zip(self.model.augment_parameters(), dalpha):
            if v.grad is None:
                if not (g is None):
                    v.grad = Variable(g.data)
            else:
                if not (g is None):
                    v.grad.data.copy_(g.data)                   # update the learnalble augmentation parameters d

    def _construct_model_from_theta(self, theta):
        model_new  = self.model.new()
        del model_new._modules['conv_sigma']
        
        model_dict = self.model.state_dict()
        del model_dict['conv_sigma.0.weight']
        del model_dict['conv_sigma.0.bias']
        del model_dict['conv_sigma.1.weight']
        del model_dict['conv_sigma.1.bias']
        del model_dict['conv_sigma.1.running_mean']
        del model_dict['conv_sigma.1.running_var']
        del model_dict['conv_sigma.1.num_batches_tracked']
        

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            if k != 'conv_sigma.0.weight' and k != 'conv_sigma.0.bias' and k != 'conv_sigma.1.weight' and k != 'conv_sigma.1.bias' and k != 'conv_sigma.1.running_mean' and k != 'conv_sigma.1.running_var' and k != 'conv_sigma.1.num_batches_tracked':
                v_length = np.prod(v.size()) # 16*3*3*3，calculate the parameter number of v
                params[k] = theta[offset: offset + v_length].view(v.size())
                offset = offset + v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, input, target, r=1e-2):
        R = r / _concat(vector).data.detach().norm()  ## small scalar in original paper，corresponding to ε=0.01/||▽w'Lval(w')||2
        # for p, v in zip(self.model.parameters(), vector):
        for p, v in zip(self.model.param_list, vector):
            p.data.add_(R, v) # p += R * v            # calculate w+ = w + ε▽w'Lval(w')
            # p.data = p.data + R * v
        loss    = self._loss_function(self.model(input, test_flag=True)[0], target)
        grads_p = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)   # equation 19 left，calculate ▽dE[Ltrain(w+, d)]
        # print(grads_p)

        # for p, v in zip(self.model.parameters(), vector):
        for p, v in zip(self.model.param_list, vector):
            p.data.sub_(2 * R, v) # p -= 2*R * v，calculate w- = w - ε▽w'Lval(w')
            # p.data = p.data - 2*R * v
        loss = self._loss_function(self.model(input, test_flag=True)[0], target)
        grads_n = torch.autograd.grad(loss, self.model.augment_parameters(), retain_graph=True, allow_unused=True)   ## equation 19 right
        # print(grads_n)

        for p, v in zip(self.model.param_list, vector):
            p.data.add_(R, v)                             # restitution

        return [ None if ( x is None ) or ( y is None) else (x - y).div_(2 * R) for x, y in zip(grads_p, grads_n) ]  ## equation 19