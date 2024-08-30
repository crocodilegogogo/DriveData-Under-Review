import os
import numpy as np
import yaml
import argparse

def parse_args():
    
    # The training options
      parser = argparse.ArgumentParser(description='TSF for HAR')
      
      parser.add_argument('--PATTERN', type=str, default='TRAIN',
                          help='PATTERN: TRAIN, TEST')
      parser.add_argument('--DATASETS', nargs='+', default=['HAPT'],
      # ['HAPT','HHAR','Opportunity','DSADS','SHO']
                          help='DATASETS: could put multiple datasets into the list')
      parser.add_argument('--CLASSIFIERS', nargs='+', default=['Deep_Conv_Transformer_torch'],
      # ['Deep_Conv_LSTM_torch','DeepSense_torch','Deep_Conv_Transformer_torch']
                          help='CLASSIFIERS: could put multiple classifiers into the list')
      # parser.add_argument('--AUGMENT_METHODS_LIST', nargs='+', default=[['MagNoise'],['DimmagScale'],['MagWarp'],['TimeWarp'],
      #                                                                   ['TimeNoise'],['RandomZoom'],['TimeStepZero'],['TimeShift'],
      #                                                                   ['Scaling'],['CutOut'],['RandomCrop'],['Permutation'],
      #                                                                   ['Resample'],['GenerateHigh'],['GenerateLow'],['AmpNoise'],
      #                                                                   ['PhaseShift'],['AmpPhasePert'],['AmpPhasePertFully'],['RandRotate']],
      # parser.add_argument('--AUGMENT_METHODS_LIST', nargs='+', default=[['MagNoise'],['DimmagScale']],
      #                     help='CLASSIFIERS: could combine multiple augmentation methods into the list')
      parser.add_argument('--AUGMENT_METHODS_LIST', nargs='+', default=[['DriveData']],
      # ['DriveData'],['MagNoise'],['DimmagScale'],['MagWarp'],['TimeWarp'],['TimeNoise'],['RandomZoom'],['TimeStepZero'],['TimeShift'],
      # ['Scaling'],['CutOut'],['RandomCrop'],['Permutation'],['Resample'],['GenerateHigh'],['GenerateLow'],['AmpNoise'],['PhaseShift'],
      # ['AmpPhasePert'],['AmpPhasePertFully'],['RandRotate'],['Mixup'],['Cutmix'],['Cutmixup']
                          help='CLASSIFIERS: could combine multiple augmentation methods into the list')
      parser.add_argument('--test_split', type=int, default=3,
                          help='The testing dataset is seperated into test_split pieces in the inference process.')
      parser.add_argument('--INFERENCE_DEVICE', type=str, default='TEST_CUDA',
                          help='inference device: TEST_CUDA, TEST_CPU')
      parser.add_argument('--SUBPOLICY_LIST', nargs='+', default=[('Magscale','Magwarp','Ampperturb','Cutout'),('Timewarp','Timenoise','Phaseperturb','Timecrop')],
                          help='SUBPOLICY_LIST: each policy has multiply operations')
      parser.add_argument('--temperature',  type=float, default=0.5,
                          help="temperature")
      parser.add_argument('--momentum',     type=float, default=0.9,
                          help='momentum')
      parser.add_argument('--weight_decay', type=float, default=2e-4,
                          help='weight decay')
      parser.add_argument('--arch_learning_rate', type=float, default=0.001,
                          help='learning rate for arch encoding')
      parser.add_argument('--arch_weight_decay',  type=float, default=0,
                          help='weight decay for arch encoding')
      parser.add_argument('--unrolled',     action='store_true', default=True,
                          help='use one-step unrolled validation loss')
      parser.add_argument('--grad_clip',         type=float, default=5,
                          help='gradient clipping')
      args = parser.parse_args()
      
      return args

def get_HAPT_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _) = os.path.split(CUR_DIR)
    DATA_DIR = os.path.join(filepath, 'datasets', 'UCI HAPT', 'HAPT_Dataset')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", 
                  "SITTING", "STANDING", "LAYING",
                  "STAND_TO_SIT", "SIT_TO_STAND", "SIT_TO_LIE",
                  "LIE_TO_SIT", "STAND_TO_LIE", "LIE_TO_STAND"]
    ActID = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    TRAIN_SUBJECTS_ID = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]
    TEST_SUBJECTS_ID  = [2, 4, 9, 10, 12, 13, 18, 20, 24]
    ALL_SUBJECTS_ID   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 
                         25, 26, 27, 28, 29, 30]
    WINDOW_SIZE       = 128 # default: 128
    OVERLAP           = 64  # default: 64
    if separate_gravity_flag == True:
        INPUT_CHANNEL     = 9
    else:
        INPUT_CHANNEL     = 6
    cal_attitude_angle = True
    STFT_intervals    = 16
    POS_NUM           = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, TRAIN_SUBJECTS_ID,\
           TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM,\
           cal_attitude_angle, STFT_intervals

def get_HHAR_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'HHAR', 'Per_subject_npy')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    ACT_LABELS          = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]
    ActID               = [0, 1, 2, 3, 4, 5]
    SUBJECTS            = ["a","b","c","d","e","f","g","h","i"]
    TRAIN_SUBJECTS_ID   = [1, 2, 3, 4, 6, 7, 8]
    TEST_SUBJECTS_ID    = [0, 5]
    WINDOW_SIZE         = 200 # default: 200
    OVERLAP             = 100 # default: 100
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 9
    else:
        INPUT_CHANNEL       = 6
    cal_attitude_angle  = False
    STFT_intervals      = 20
    POS_NUM             = 1
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, ACT_LABELS, ActID, SUBJECTS, \
           TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID, WINDOW_SIZE, OVERLAP, \
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, \
           STFT_intervals

def get_Opportunity_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'Opportunity')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4]
    TRIALS                  = [1,2,3,4,5]
    SELEC_LABEL             = 'MID_LABEL_COL' # 'LOCO_LABEL_COL', 'MID_LABEL_COL', 'HI_LABEL_COL'
    ACT_LABELS              = ['null', 'Open_Door_1', 'Open_Door_2', 'Close_Door_1', 'Close_Door_2', 'Open_Fridge',
                               'Close_Fridge', 'Open_Dishwasher', 'Close_Dishwasher', 'Open Drawer1','Close Drawer1',
                               'Open_Drawer2','Close_Drawer2', 'Open_Drawer3', 'Close_Drawer3', 'Clean_Table',
                               'Drink_Cup', 'Toggle_Switch']
    ACT_ID                  = (np.arange(18)).tolist()
    TRAIN_SUBJECTS_ID       = [1]
    TRAIN_SUBJECTS_TRIAL_ID = [1,2,3,4,5]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 63
    else:
        INPUT_CHANNEL       = 42
    to_NED_flag             = True
    cal_attitude_angle      = False
    STFT_intervals          = 6
    POS_NUM                 = 7
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS, TRIALS, SELEC_LABEL,\
           ACT_LABELS, ACT_ID, TRAIN_SUBJECTS_ID, TRAIN_SUBJECTS_TRIAL_ID,\
           WINDOW_SIZE, OVERLAP, INPUT_CHANNEL, POS_NUM, \
           to_NED_flag, cal_attitude_angle, STFT_intervals

def get_DSADS_dataset_param(CUR_DIR, dataset_name, separate_gravity_flag):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'DSADS')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9','A10',
                               'A11','A12','A13','A14','A15','A16','A17','A18','A19']
    ACT_ID                  = (np.arange(19)+1).tolist()
    WINDOW_SIZE             = 125 # default: 125
    OVERLAP                 = 0   # default: 0
    if separate_gravity_flag == True:
        INPUT_CHANNEL       = 45
    else:
        INPUT_CHANNEL       = 30
    cal_attitude_angle      = False
    STFT_intervals          = 25
    POS_NUM                 = 5
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, ACT_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle,\
           STFT_intervals

def get_SHO_dataset_param(CUR_DIR, dataset_name):
    
    (filepath, _)       = os.path.split(CUR_DIR)
    DATA_DIR            = os.path.join(filepath, 'datasets', 'SHO')
    MODELS_COMP_LOG_DIR = os.path.join(CUR_DIR, 'logs', dataset_name, 'classifiers_comparison')
    SUBJECTS                = [1,2,3,4,5,6,7,8,9,10]
    TRAIN_SUBJECTS_ID       = [1]
    ACT_LABELS              = ['walking',  'standing', 'jogging', 'sitting', 'biking',
                               'upstairs', 'downstairs']
    Act_ID                  = [1,2,3,4,5,6,7]
    WINDOW_SIZE             = 48 # default: 48
    OVERLAP                 = 24 # default: 24
    INPUT_CHANNEL           = 45
    POS_NUM                 = 5
    STFT_intervals          = 6
    cal_attitude_angle      = False
    
    return DATA_DIR, MODELS_COMP_LOG_DIR, SUBJECTS,\
           TRAIN_SUBJECTS_ID, ACT_LABELS, Act_ID, WINDOW_SIZE, OVERLAP,\
           INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals

def create_classifier(dataset_name, classifier_name, input_channel, POS_NUM,
                      data_length, train_size, val_size, test_size, nb_classes, STFT_intervals,
                      BATCH_SIZE, args, AUGMENT_METHODS):
    
    hparam_file     = open(os.path.join('utils','hyperparams.yaml'), mode='r')
    hyperparameters = yaml.load(hparam_file, Loader=yaml.FullLoader)
    conv_chnnl      = hyperparameters[classifier_name]['conv_chnnl'][dataset_name]
    context_chnnl   = hyperparameters[classifier_name]['context_chnnl'][dataset_name]
    
############################## comparison methods #############################

    if classifier_name=='Deep_Conv_LSTM_torch': 
        from classifiers.comparison_methods import Deep_Conv_LSTM_torch 
        # __init__(self, input_2Dfeature_channel, input_channel, kernel_size,
        #      feature_channel, hidden_size, drop_rate, num_class)
        return Deep_Conv_LSTM_torch.Deep_Conv_LSTM(1, input_channel, 5, conv_chnnl, context_chnnl, 0.2, nb_classes, dataset_name, args, AUGMENT_METHODS), Deep_Conv_LSTM_torch

    if classifier_name=='DeepSense_torch':
        from classifiers.comparison_methods import DeepSense_torch
        # __init__(self, input_2Dfeature_channel, input_channel, POS_NUM, kernel_size,
        #      feature_channel, merge_kernel_size1, merge_kernel_size2, merge_kernel_size3,
        #      hidden_size, drop_rate, drop_rate_gru, num_class, datasetname)
        # return DeepSense_torch.DeepSense(1, input_channel, POS_NUM, 3, conv_chnnl, 1, 3, 2, context_chnnl, 0, 0.2, classifier_name, nb_classes, dataset_name, args, STFT_intervals, AUGMENT_METHODS), DeepSense_torch
        return DeepSense_torch.DeepSense(1, input_channel, POS_NUM, 3, conv_chnnl, 1, 3, 2, context_chnnl, 0, 0.2, classifier_name, nb_classes, dataset_name, args, STFT_intervals, AUGMENT_METHODS), DeepSense_torch

    if classifier_name=='Deep_Conv_Transformer_torch': 
        from classifiers.comparison_methods import Deep_Conv_Transformer_torch
        # __init__(self, input_2Dfeature_channel, input_channel, kernel_size, feature_channel,
        #      feature_channel_out, multiheads, drop_rate, data_length, num_class)
        return Deep_Conv_Transformer_torch.Deep_Conv_Transformer(1, input_channel, 7, conv_chnnl, context_chnnl, 8, 0.2, data_length, nb_classes, dataset_name, args, AUGMENT_METHODS), Deep_Conv_Transformer_torch