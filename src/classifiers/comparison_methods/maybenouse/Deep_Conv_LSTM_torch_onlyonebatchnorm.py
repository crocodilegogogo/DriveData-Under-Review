import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
# import torchvision
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import time
import utils.HMGAN as HMGAN
import utils.FCGAN as FCGAN
from utils.utils import *
from utils.augmentation_methods import *
import os

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)

class Deep_Conv_LSTM(nn.Module):
    def __init__(self, input_2Dfeature_channel, input_channel, kernel_size,
                 feature_channel, hidden_size, drop_rate, num_class, datasetname,
                 args, AUGMENT_METHODS):
        
        super(Deep_Conv_LSTM, self).__init__()
        
        self.input_2Dfeature_channel = input_2Dfeature_channel
        self.input_channel           = input_channel
        self.kernel_size             = kernel_size
        self.feature_channel         = feature_channel
        self.hidden_size             = hidden_size
        self.drop_rate               = drop_rate
        self.num_class               = num_class
        self.datasetname             = datasetname
        self.args                    = args
        self.aug_methods             = AUGMENT_METHODS
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_2Dfeature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            # nn.MaxPool2d(2)
            )
        
        self.lstm = nn.LSTM(
            input_size  = input_channel*feature_channel,
            # e.g. input_size=9*64,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True
            )
        
        self.dropout = nn.Dropout(drop_rate)
        
        self.linear = nn.Linear(hidden_size, num_class)

        # DaDa Initializations
        self.aug_policies    = args.SUBPOLICY_LIST
        if args.INFERENCE_DEVICE == 'TEST_CUDA':
            self.temperature = torch.tensor(args.temperature).cuda()
        else:
            self.temperature = torch.tensor(args.temperature)
        self.mix_augment     = MixedAugment(args.SUBPOLICY_LIST)
        self._initialize_augment_parameters()
        self.augmenting      = True
        
        # DaDa Data Generator Initializations
        self.conv_mu         = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel)
            )
        self.conv_sigma      = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            # nn.BatchNorm2d(feature_channel)
            )

        self.Normal       = torch.distributions.Normal(0, 1)
        self.kl = 0
        
        self.param_list = []
        for n, v in self.named_parameters():
            if n != 'conv_sigma.0.weight' and n != 'conv_sigma.0.bias'\
                and n != 'conv_sigma.1.weight' and n != 'conv_sigma.1.bias'\
                and n != 'conv_sigma.1.running_mean' and n != 'conv_sigma.1.running_var'\
                and n != 'conv_sigma.1.num_batches_tracked': #and n != 'build_prior.prior_mu' and n!='build_prior.prior_logvar':
                self.param_list.append(v)
                # print(n)
    
    def set_augmenting(self, value):
        assert value in [False, True]
        self.augmenting = value
    
    def new(self):
        network_new = Deep_Conv_LSTM(self.input_2Dfeature_channel, self.input_channel, self.kernel_size,
                                     self.feature_channel, self.hidden_size, self.drop_rate, self.num_class,
                                     self.datasetname, self.args, self.aug_methods)
        return network_new
    
    def _initialize_augment_parameters(self):
        num_aug_policies = len(self.aug_policies)
        num_ops = len(self.aug_policies[0])
        if args.INFERENCE_DEVICE == 'TEST_CUDA':
            self.probabilities = Variable(0.5*torch.ones(num_aug_policies, num_ops).cuda(), requires_grad=True) # 2个subpolicy中的两个operation的选择概率
            self.magnitudes    = Variable(0.5*torch.ones(num_aug_policies, num_ops).cuda(), requires_grad=True) # 2个subpolicy中的两个operation的操作强度
            # self.ops_weights   = Variable(1e-3*torch.ones(num_aug_policies).cuda(), requires_grad=True)       # 105个subpolicy的选择概率

        else:
            self.probabilities = Variable(0.5*torch.ones(num_aug_policies, num_ops), requires_grad=True)
            self.magnitudes    = Variable(0.5*torch.ones(num_aug_policies, num_ops), requires_grad=True)
            # self.ops_weights   = Variable(1e-3*torch.ones(num_aug_policies), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.magnitudes,
            # self.ops_weights
        ]
    
    def augment_parameters(self):
        return self._augment_parameters
    
    def sample(self):
        self.probabilities_mid = self.probabilities.clone().clamp(0.0, 1.0)
        probabilities_dist     = torch.distributions.RelaxedBernoulli(self.temperature, self.probabilities_mid) # 见论文中的公式(9)，operation做还是不做
        # print('probabilities:', self.probabilities)
        sample_probabilities      = probabilities_dist.rsample()
        sample_probabilities      = sample_probabilities.clamp(0.0, 1.0)
        self.sample_probabilities_index = sample_probabilities >= 0.5
        self.sample_probabilities = self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities # 这个是干啥？

        # ops_weights_dist   = torch.distributions.RelaxedOneHotCategorical(self.temperature, logits=self.ops_weights) # 见论文中的公式(7)，sunbpolicy选哪个
        # # print('Ops_weights:',self.ops_weights)
        # sample_ops_weights = ops_weights_dist.rsample()
        # sample_ops_weights = sample_ops_weights.clamp(0.0, 1.0)
        # self.sample_ops_weights_index = torch.max(sample_ops_weights, dim=-1, keepdim=True)[1]        # 相当于argmax了，找到概率最大的policy的位置
        # # print('sample_ops_weights_index:',self.sample_ops_weights_index)
        # one_h = torch.zeros_like(sample_ops_weights).scatter_(-1, self.sample_ops_weights_index, 1.0) # 选择哪个policy做成one-hot的形式
        # self.sample_ops_weights = one_h - sample_ops_weights.detach() + sample_ops_weights            # 这个是干啥？

    def data_generator(self, x, test_flag):
        mu          = self.conv_mu(x)
        if test_flag == False:
            sigma   = torch.exp(0.5*self.conv_sigma(x))
            z       = mu + sigma * self.Normal.sample(mu.shape).to(x.device)
            # print(sigma.reshape(sigma.shape[0], -1).sum(1))
            # print(z.reshape(z.shape[0], -1).sum(1))
            # kl_loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
            kl_loss = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).reshape(x.shape[0],-1).mean(dim=1).sum() # .clamp(1.0, 50.0)
            # kl_loss = (sigma.reshape(x.shape[0],-1).mean(dim=1) ** 2 + mu.reshape(x.shape[0],-1).mean(dim=1) ** 2 - torch.log(sigma.reshape(x.shape[0],-1).mean(dim=1)) - 1 / 2).sum()
        else:
            z       = mu
            kl_loss = 0
        return z, kl_loss

    def forward(self, x, test_flag=False, epoch=0):
        
        if self.augmenting and self.aug_methods[0]=='DaDa': # and epoch<15:
            x = self.mix_augment(x, self.sample_probabilities, self.sample_probabilities_index, self.magnitudes)
        # print(self.magnitudes)
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        # hidden = None
        batch_size = x.shape[0]
        feature_channel = x.shape[1]
        input_channel = x.shape[2]
        data_length = x.shape[-1]
        
        # flops
        if len(x.shape) == 3:
            x           = x.unsqueeze(0)
        # flops
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        if self.aug_methods[0]=='DaDa':
            x, kl_loss = self.data_generator(x, test_flag)
        else:
            kl_loss = 0
        # kl_loss = torch.zeros(1).to(x.device)
        
        #############
        x = x.view(batch_size, -1, data_length)
        x = x.permute(0,2,1)
        #############
        # x = x.permute(0,1,3,2)
        # data_length = x.shape[-1]
        # x = x.contiguous().view(batch_size, -1, data_length)
        # x = x.permute(0,2,1)
        #############
        x, _ = self.lstm(x, None)
        x = self.dropout(x)
        x_feature = x
        x = x.view(batch_size, data_length, -1)[:,-1,:]

        output = self.linear(x)

        # return output, x
        return output, kl_loss
    
def train_op(network, aug_methods, EPOCH, BATCH_SIZE, LR, POS_NUM,
             train_x, train_y, val_x, val_y, X_test, y_test,
             output_directory_models, log_training_duration,
             classifier_name, STFT_intervals, args):
    # prepare training_data
    if train_x.shape[0] % BATCH_SIZE == 1:
        drop_last_flag = True
    else:
        drop_last_flag = False

    if aug_methods[0] != 'DaDa':
        if aug_methods is not None:
            torch_dataset_train = DataAugment(train_x, train_y, aug_methods, POS_NUM) ##!
        else:
            torch_dataset_train = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
        train_loader = Data.DataLoader(dataset = torch_dataset_train,
                                       batch_size = BATCH_SIZE,
                                       shuffle = True,
                                       drop_last = drop_last_flag
                                      )
    else:
        torch_dataset_train = DifDataAugment(train_x, train_y, POS_NUM, network.magnitudes, args.SUBPOLICY_LIST, True) ##!
        train_loader        = Data.DataLoader(dataset = torch_dataset_train,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              drop_last = drop_last_flag
                                             )
        torch_dataset_val   = DifDataAugment(val_x, val_y, POS_NUM, network.magnitudes, args.SUBPOLICY_LIST, False) ##!
        val_loader          = Data.DataLoader(dataset = torch_dataset_val,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              drop_last = drop_last_flag
                                             )
        architect = Architect(network, args)

    # init lr&train&test loss&acc log
    lr_results = []
    
    loss_train_results = []
    accuracy_train_results = []
    
    loss_validation_results = []
    accuracy_validation_results = []
    macro_f1_val_results        = []
    
    loss_test_results = []
    accuracy_test_results = []
    macro_f1_test_results       = []
    
    # prepare optimizer&scheduler&loss_function
    parameters = network.parameters()
    optimizer = torch.optim.Adam(parameters, lr = LR, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, 
                                                            patience=5,
                                                            min_lr=LR/10, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
    #                                                         patience=5,
    #                                                         min_lr=LR/10, verbose=True)
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    
    # save init model
    output_directory_init = os.path.join(output_directory_models,'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    
    
    if aug_methods[0] not in ['HMGAN','FCGAN']:
        
        for epoch in range (EPOCH):
            
            if aug_methods[0] != 'DaDa':
                
                for step, (x,y) in enumerate(train_loader):
                    
                    x = x.detach().numpy()
                    y = y.detach().numpy()
                    
                    if 'WDBA' in aug_methods:
                        x = wdba(x, y)
                    elif 'DGW-sD' in aug_methods:
                        x = discriminative_guided_warp(x, y)
                    elif 'RGW-sD' in aug_methods:
                        x = random_guided_warp(x, y)
                    elif 'SFCC' in aug_methods:
                        x, y = sfcc(x, y, np.unique(train_y).shape[0])
                    
                    x = STFT_precessing(x, y, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                    x = torch.FloatTensor(x)
                    y = torch.tensor(y).long()
                    batch_x = x.cuda()
                    batch_y = y.cuda()
                    
                    if 'Mixup' in aug_methods:
                        output_bc,loss  = mixup(batch_x, batch_y, network, loss_function, 3, use_cuda=True)
                    elif 'Cutmix' in aug_methods:
                        output_bc, loss = cutmix(batch_x, batch_y, loss_function, network, 5, True)
                    elif 'Cutmixup' in aug_methods:
                        output_bc, loss = cutmixup(batch_x, batch_y, loss_function, network, 5, True)
                    else:
                        output, _ = network(batch_x)
                        # cal the sum of pre loss per batch 
                        loss      = loss_function(output[0], batch_y)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            else:
                
                network.sample()
                # train_loader.dataset.weights_index       = network.sample_ops_weights_index
                train_loader.dataset.probabilities_index = network.sample_probabilities_index
                
                for step, (x,y) in enumerate(train_loader):
                    
                    network.set_augmenting(True)
                    
                    x = x.detach().numpy()
                    y = y.detach().numpy()
                    x = STFT_precessing(x, y, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                    x = torch.FloatTensor(x)
                    y = torch.tensor(y).long()
                    
                    x = Variable(x, requires_grad=False).cuda()                    # 这个主要是也把它加到graph里面来
                    y = Variable(y, requires_grad=False).cuda(non_blocking=True)
                    
                    # get a minibatch from the search queue with replacement
                    x_search, y_search = next(iter(val_loader))    # 每次迭代的时候从里面取一个，然后就可以删掉这个batch了
                    x_search = x_search.detach().numpy()
                    y_search = y_search.detach().numpy()
                    x_search = STFT_precessing(x_search, y_search, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                    x_search = torch.FloatTensor(x_search)
                    y_search = torch.tensor(y_search).long()
                    
                    x_search = Variable(x_search, requires_grad=False).cuda()
                    y_search = Variable(y_search, requires_grad=False).cuda(non_blocking=True)
                    
                    # if epoch < 15:
                        # architect.step(x, y, x_search, y_search, optimizer.param_groups[0]['lr'], loss_function, optimizer, unrolled=args.unrolled)
                    architect.step(x, y, x_search, y_search, optimizer.param_groups[0]['lr'], loss_function, optimizer, unrolled=args.unrolled)
                    
                    # output_bc, kl_loss = network(x, epoch=epoch)
                    # # cal the sum of pre loss per batch 
                    # c_loss             = loss_function(output_bc, y)
                    # loss               = c_loss + kl_loss * (c_loss.detach()/(kl_loss.detach()+1))
                    
                    mixed_x, y_a, y_b, lam = mixup_data(x, y, 3, use_cuda=True)
                    mixed_x, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
                    output_bc, kl_loss = network(mixed_x, epoch=epoch)
                    c_loss             = mixup_criterion(loss_function, output_bc, y_a, y_b, lam)
                    # loss               = c_loss + kl_loss * (c_loss.detach()/(kl_loss.detach()+1))
                    if kl_loss >= 3 * c_loss:
                        loss               = c_loss + kl_loss * (c_loss.detach()/(kl_loss.detach()+1))
                    else:
                        loss               = c_loss + kl_loss
                    # loss = mixup_criterion(loss_function, output_bc, y_a, y_b, lam) + kl_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
                    optimizer.step()
                    
                    network.sample()
                    # train_loader.dataset.weights_index       = network.sample_ops_weights_index
                    train_loader.dataset.probabilities_index = network.sample_probabilities_index
            
            # test per epoch
            network.eval()
            network.set_augmenting(False)
            # loss_train:loss of training set; accuracy_train:pre acc of training set
            if epoch == 0:
                train_x  = STFT_precessing(train_x, train_y, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                val_x    = STFT_precessing(val_x, val_y, classifier_name, STFT_intervals, POS_NUM, args.test_split)
            loss_train, accuracy_train, _                      = get_test_loss_acc(network, loss_function, train_x, train_y, args.test_split)
            loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, args.test_split)
            loss_test, accuracy_test, macro_f1_test            = get_test_loss_acc(network, loss_function, X_test, y_test, args.test_split)
            network.train()
            
            # update lr
            scheduler.step(accuracy_validation)
            lr = optimizer.param_groups[0]['lr']
            
            # architect.scheduler_arch.step()
            # lr_arch = architect.optimizer.param_groups[0]['lr']
            
            ######################################dropout#####################################
            # loss_train, accuracy_train = get_loss_acc(network.eval(), loss_function, train_x, train_y, args.test_split)
            
            # loss_validation, accuracy_validation = get_loss_acc(network.eval(), loss_function, test_x, test_y, args.test_split)
            
            # network.train()
            ##################################################################################
            
            # log lr&train&validation loss&acc per epoch
            lr_results.append(lr)
            loss_train_results.append(loss_train)
            accuracy_train_results.append(accuracy_train)
            
            loss_validation_results.append(loss_validation)
            accuracy_validation_results.append(accuracy_validation)
            macro_f1_val_results.append(macro_f1_val)
            
            loss_test_results.append(loss_test)    
            accuracy_test_results.append(accuracy_test)
            macro_f1_test_results.append(macro_f1_test)
            
            # print training process
            if (epoch+1) % 1 == 0:
                print('Epoch:', (epoch+1), '|lr:', lr, # '|lr_arch:', lr_arch,
                      '| train_loss:', loss_train, 
                      '| train_acc:', accuracy_train, 
                      '| validation_loss:', loss_validation, 
                      '| validation_acc:', accuracy_validation,
                      '| test_loss:', loss_test, 
                      '| test_acc:', accuracy_test,
                      '| all_loss:', loss,
                      '| kl_loss:', kl_loss * (c_loss.detach()/(kl_loss.detach()+1)))
            
            save_models(network, output_directory_models, 
                        loss_train, loss_train_results, 
                        accuracy_validation, accuracy_validation_results,
                        start_time, training_duration_logs)
    
    elif aug_methods[0] in ['HMGAN']:
        
        args_hmgan  = HMGAN.get_args()
        hmgan       = HMGAN.DASolver_HMGAN(args_hmgan, network, BATCH_SIZE, network.input_channel, POS_NUM, train_x.shape[-1])
        network, EPOCH, lr_results, loss_train_results, accuracy_train_results, loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results\
            = hmgan.train(train_x, train_y, val_x, val_y, X_test, y_test, STFT_intervals, start_time, classifier_name, output_directory_models)
    
    elif aug_methods[0] in ['FCGAN']:
        
        args_fcgan  = FCGAN.get_args()
        fcgan       = FCGAN.DASolver_FCGAN(args_fcgan, network, BATCH_SIZE, network.input_channel, POS_NUM, train_x.shape[-1])
        network, EPOCH, lr_results, loss_train_results, accuracy_train_results, loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results\
            = fcgan.train(train_x, train_y, val_x, val_y, X_test, y_test, STFT_intervals, start_time, classifier_name, output_directory_models)
        
    # log training time 
    per_training_duration = time.time() - start_time
    log_training_duration.append(per_training_duration)
    
    # save last_model
    output_directory_last = os.path.join(output_directory_models,'last_model.pkl')
    torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
    # log history
    history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
                          loss_validation_results, accuracy_validation_results,
                          loss_test_results, accuracy_test_results,
                          output_directory_models)
    
    plot_learning_history(EPOCH, history, output_directory_models)
    
    return(history, per_training_duration, log_training_duration)
    
# def train_op(network, aug_methods, EPOCH, BATCH_SIZE, LR, POS_NUM,
#              train_x, train_y, val_x, val_y, X_test, y_test,
#              output_directory_models, log_training_duration,
#              classifier_name, STFT_intervals, test_split, INFERENCE_DEVICE, sub_policies):
#     # prepare training_data
#     if train_x.shape[0] % BATCH_SIZE == 1:
#         drop_last_flag = True
#     else:
#         drop_last_flag = False

#     if aug_methods is not None:
#         # if aug_method=='MagNoise':
#         #     torch_dataset = Data_RandAug_Transform(train_x, train_y) ##!
#         # elif aug_method=='Mixup':
#         torch_dataset = DataAugment(train_x, train_y, aug_methods, POS_NUM) ##!
#         # torch_dataset = DataAugment(train_x, train_y, aug_methods, classifier_name, STFT_intervals, POS_NUM, test_split) ##!
#     else:
#         torch_dataset = Data.TensorDataset(torch.FloatTensor(train_x), torch.tensor(train_y).long())
#     train_loader = Data.DataLoader(dataset = torch_dataset,
#                                     batch_size = BATCH_SIZE,
#                                     shuffle = True,
#                                     drop_last = drop_last_flag
#                                    )
    
#     # init lr&train&test loss&acc log
#     lr_results = []
    
#     loss_train_results = []
#     accuracy_train_results = []
    
#     loss_validation_results = []
#     accuracy_validation_results = []
#     macro_f1_val_results        = []
    
#     loss_test_results = []
#     accuracy_test_results = []
#     macro_f1_test_results       = []
    
#     # prepare optimizer&scheduler&loss_function
#     parameters = network.parameters()
#     optimizer = torch.optim.Adam(parameters,lr = LR)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, 
#                                                            patience=5,
#                                                            min_lr=LR/10, verbose=True)
#     loss_function = nn.CrossEntropyLoss(reduction='sum')
#     # loss_function = LabelSmoothingCrossEntropy()
    
#     # save init model    
#     output_directory_init = os.path.join(output_directory_models,'init_model.pkl')
#     torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
#     training_duration_logs = []
#     start_time = time.time()
#     for epoch in range (EPOCH):
        
#         for step, (x,y) in enumerate(train_loader):
            
#             # h_state = None      # for initial hidden state
            
#             x = x.detach().numpy()
#             y = y.detach().numpy()
            
#             if 'WDBA' in aug_methods:
#                 x = wdba(x, y)
#             elif 'DGW-sD' in aug_methods:
#                 x = discriminative_guided_warp(x, y)
#             elif 'RGW-sD' in aug_methods:
#                 x = random_guided_warp(x, y)
#             elif 'SFCC' in aug_methods:
#                 x, y = sfcc(x, y, np.unique(train_y).shape[0])
            
#             # x = torch.from_numpy(x)
#             # y = torch.from_numpy(y)
#             x = torch.FloatTensor(x)
#             y = torch.tensor(y).long()
#             batch_x = x.cuda()
#             batch_y = y.cuda()

#             if 'Mixup' in aug_methods:
#                 output_bc,loss = mixup(batch_x, batch_y, network, loss_function, 3, use_cuda=True)
#             elif 'Cutmix' in aug_methods:
#                 output_bc, loss = cutmix(batch_x, batch_y, loss_function, network, 5, True)
#             elif 'Cutmixup' in aug_methods:
#                 output_bc, loss = cutmixup(batch_x, batch_y, loss_function, network, 5, True)
#             else:
#                 output_bc = network(batch_x)[0]                                
#                 # cal the sum of pre loss per batch 
#                 loss = loss_function(output_bc, batch_y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
#         # test per epoch
#         network.eval()
#         # loss_train:loss of training set; accuracy_train:pre acc of training set
#         loss_train, accuracy_train, _ = get_test_loss_acc(network, loss_function, train_x, train_y, test_split)        
#         loss_validation, accuracy_validation, macro_f1_val = get_test_loss_acc(network, loss_function, val_x, val_y, test_split) 
#         loss_test, accuracy_test, macro_f1_test = get_test_loss_acc(network, loss_function, X_test, y_test, test_split)
#         network.train()  
        
#         # update lr
#         scheduler.step(accuracy_validation)
#         lr = optimizer.param_groups[0]['lr']
        
#         ######################################dropout#####################################
#         # loss_train, accuracy_train = get_loss_acc(network.eval(), loss_function, train_x, train_y, test_split)
        
#         # loss_validation, accuracy_validation = get_loss_acc(network.eval(), loss_function, test_x, test_y, test_split)
        
#         # network.train()
#         ##################################################################################
        
#         # log lr&train&validation loss&acc per epoch
#         lr_results.append(lr)
#         loss_train_results.append(loss_train)    
#         accuracy_train_results.append(accuracy_train)
        
#         loss_validation_results.append(loss_validation)    
#         accuracy_validation_results.append(accuracy_validation)
#         macro_f1_val_results.append(macro_f1_val)
        
#         loss_test_results.append(loss_test)    
#         accuracy_test_results.append(accuracy_test)
#         macro_f1_test_results.append(macro_f1_test)
        
#         # print training process
#         if (epoch+1) % 1 == 0:
#             print('Epoch:', (epoch+1), '|lr:', lr,
#                   '| train_loss:', loss_train, 
#                   '| train_acc:', accuracy_train, 
#                   '| validation_loss:', loss_validation, 
#                   '| validation_acc:', accuracy_validation)
        
#         save_models(network, output_directory_models, 
#                     loss_train, loss_train_results, 
#                     accuracy_validation, accuracy_validation_results,
#                     start_time, training_duration_logs)
    
#     # log training time 
#     per_training_duration = time.time() - start_time
#     log_training_duration.append(per_training_duration)
    
#     # save last_model
#     output_directory_last = os.path.join(output_directory_models,'last_model.pkl')
#     torch.save(network.state_dict(), output_directory_last)   # save only the init parameters
    
#     # log history
#     history = log_history(EPOCH, lr_results, loss_train_results, accuracy_train_results, 
#                           loss_validation_results, accuracy_validation_results,
#                           loss_test_results, accuracy_test_results,
#                           output_directory_models)
    
#     # plot_learning_history(EPOCH, history, output_directory_models)
    
#     return(history, per_training_duration, log_training_duration)
