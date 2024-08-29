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
# import utils.HMGAN as HMGAN
# import utils.FCGAN as FCGAN
from utils.utils import *
from utils.augmentation_operations import *
from utils.diff_framework          import *
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
            )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel),
            nn.ReLU(),
            )
        
        self.lstm = nn.LSTM(
            input_size  = input_channel*feature_channel,
            hidden_size = hidden_size,
            num_layers  = 2,
            batch_first = True
            )
        
        self.dropout = nn.Dropout(drop_rate)
        
        self.linear = nn.Linear(hidden_size, num_class)

        # HMC Initializations
        self.aug_policies    = args.SUBPOLICY_LIST
        if args.INFERENCE_DEVICE == 'TEST_CUDA':
            self.temperature = torch.tensor(args.temperature).cuda()
        else:
            self.temperature = torch.tensor(args.temperature)
        self.mix_augment     = MixedAugment(args.SUBPOLICY_LIST)
        self._initialize_augment_parameters()
        self.augmenting      = True
        
        # Data Generator Initializations
        self.conv_mu         = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel)
            )
        self.conv_sigma      = nn.Sequential(
            nn.Conv2d(feature_channel, feature_channel, (1,kernel_size), 1, (0,kernel_size//2)),
            nn.BatchNorm2d(feature_channel)
            )

        self.Normal       = torch.distributions.Normal(0, 1)
        self.kl = 0
        
        self.param_list = []
        for n, v in self.named_parameters():
            if n != 'conv_sigma.0.weight' and n != 'conv_sigma.0.bias'\
                and n != 'conv_sigma.1.weight' and n != 'conv_sigma.1.bias'\
                and n != 'conv_sigma.1.running_mean' and n != 'conv_sigma.1.running_var'\
                and n != 'conv_sigma.1.num_batches_tracked':
                self.param_list.append(v)
    
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
            self.probabilities = Variable(0.5*torch.ones(num_aug_policies, num_ops).cuda(), requires_grad=True) # the initialization of selection probabilities
            self.magnitudes    = Variable(0.5*torch.ones(num_aug_policies, num_ops).cuda(), requires_grad=True) # the initialization of operation magnitudes

        else:
            self.probabilities = Variable(0.5*torch.ones(num_aug_policies, num_ops), requires_grad=True)
            self.magnitudes    = Variable(0.5*torch.ones(num_aug_policies, num_ops), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.magnitudes,
        ]
    
    def augment_parameters(self):
        return self._augment_parameters
    
    def sample(self):
        self.probabilities_mid = self.probabilities.clone().clamp(0.0, 1.0)
        probabilities_dist        = torch.distributions.RelaxedBernoulli(self.temperature, self.probabilities_mid) # RelaxedBernoulli sampling
        sample_probabilities      = probabilities_dist.rsample()
        sample_probabilities      = sample_probabilities.clamp(0.0, 1.0)
        self.sample_probabilities_index = sample_probabilities >= 0.5
        self.sample_probabilities = self.sample_probabilities_index.float() - sample_probabilities.detach() + sample_probabilities # this is for gradient continuity
    
    def data_generator(self, x, test_flag):
        mu          = self.conv_mu(x)
        if test_flag == False:
            sigma         = self.conv_sigma(x)
            sigma_sample  = torch.exp(0.5*sigma)
            sigma_kl      = torch.exp(sigma)
            z             = mu + sigma_sample * self.Normal.sample(mu.shape).to(x.device)
            kl_loss = (sigma_kl + mu ** 2 - sigma - 1).reshape(x.shape[0],-1).mean(dim=1).sum() 
        else:
            z       = mu
            kl_loss = 0
        return z, kl_loss

    def forward(self, x, test_flag=False, epoch=0):
        
        if self.augmenting and self.aug_methods[0]=='DriveData':
            x = self.mix_augment(x, self.sample_probabilities, self.sample_probabilities_index, self.magnitudes)
        
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
        
        if self.aug_methods[0]=='DriveData':
            x, kl_loss = self.data_generator(x, test_flag)
        else:
            kl_loss = 0
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = x.view(batch_size, -1, data_length)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x, None)
        x = self.dropout(x)
        x_feature = x
        x = x.view(batch_size, data_length, -1)[:,-1,:]

        output = self.linear(x)

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

    if aug_methods[0] != 'DriveData':
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
    loss_function = nn.CrossEntropyLoss(reduction='sum')
    
    # save init model
    output_directory_init = os.path.join(output_directory_models,'init_model.pkl')
    torch.save(network.state_dict(), output_directory_init)   # save only the init parameters
    
    training_duration_logs = []
    start_time = time.time()
    
    
    if aug_methods[0] not in ['HMGAN','FCGAN']:
        
        for epoch in range (EPOCH):
            
            if aug_methods[0] != 'DriveData':
                
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
                train_loader.dataset.probabilities_index = network.sample_probabilities_index
                
                for step, (x,y) in enumerate(train_loader):
                    
                    network.set_augmenting(True)
                    
                    x = x.detach().numpy()
                    y = y.detach().numpy()
                    x = STFT_precessing(x, y, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                    x = torch.FloatTensor(x)
                    y = torch.tensor(y).long()
                    
                    x = Variable(x, requires_grad=False).cuda()
                    y = Variable(y, requires_grad=False).cuda(non_blocking=True)
                    
                    # get a minibatch from the search queue with replacement
                    x_search, y_search = next(iter(val_loader))
                    x_search = x_search.detach().numpy()
                    y_search = y_search.detach().numpy()
                    x_search = STFT_precessing(x_search, y_search, classifier_name, STFT_intervals, POS_NUM, args.test_split)
                    x_search = torch.FloatTensor(x_search)
                    y_search = torch.tensor(y_search).long()
                    
                    x_search = Variable(x_search, requires_grad=False).cuda()
                    y_search = Variable(y_search, requires_grad=False).cuda(non_blocking=True)
                    
                    architect.step(x, y, x_search, y_search, optimizer.param_groups[0]['lr'], loss_function, optimizer, unrolled=args.unrolled)
                    
                    mixed_x, y_a, y_b, lam = mixup_data(x, y, 3, use_cuda=True)
                    mixed_x, y_a, y_b = map(Variable, (mixed_x, y_a, y_b))
                    output_bc, kl_loss = network(mixed_x, epoch=epoch)
                    c_loss             = mixup_criterion(loss_function, output_bc, y_a, y_b, lam)
                    if kl_loss >= 2 * c_loss:
                        loss               = c_loss + kl_loss * (c_loss.detach()/(kl_loss.detach()+1))
                    else:
                        loss               = c_loss + kl_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(network.parameters(), args.grad_clip)
                    optimizer.step()
                    
                    network.sample()
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
                      '| kl_loss:', kl_loss) # * (c_loss.detach()/(kl_loss.detach()+1)))
            
            save_models(network, output_directory_models, 
                        loss_train, loss_train_results, 
                        accuracy_validation, accuracy_validation_results,
                        start_time, training_duration_logs)
    
    # elif aug_methods[0] in ['HMGAN']:
        
    #     args_hmgan  = HMGAN.get_args()
    #     hmgan       = HMGAN.DASolver_HMGAN(args_hmgan, network, BATCH_SIZE, network.input_channel, POS_NUM, train_x.shape[-1])
    #     network, EPOCH, lr_results, loss_train_results, accuracy_train_results, loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results\
    #         = hmgan.train(train_x, train_y, val_x, val_y, X_test, y_test, STFT_intervals, start_time, classifier_name, output_directory_models)
    
    # elif aug_methods[0] in ['FCGAN']:
        
    #     args_fcgan  = FCGAN.get_args()
    #     fcgan       = FCGAN.DASolver_FCGAN(args_fcgan, network, BATCH_SIZE, network.input_channel, POS_NUM, train_x.shape[-1])
    #     network, EPOCH, lr_results, loss_train_results, accuracy_train_results, loss_validation_results, accuracy_validation_results, loss_test_results, accuracy_test_results\
    #         = fcgan.train(train_x, train_y, val_x, val_y, X_test, y_test, STFT_intervals, start_time, classifier_name, output_directory_models)
        
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