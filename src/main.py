import os
import sys
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from utils.constants import *
from utils.utils import *
from utils.augmentation_operations import *
from utils.diff_framework          import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
CUR_DIR = os.path.dirname(os.path.abspath(__file__))  # Path to current directory

class Train_Test(object):
    def __init__(self, args):

        # Initial
        self.args  = args

        for dataset_name in args.DATASETS:
            
            sep_flags = []
            BATCH_SIZE, EPOCH, LR = get_hyperparams(os.path.join('utils','hyperparams.yaml'), dataset_name)
            
            for (classifer_id, classifier_name) in enumerate(args.CLASSIFIERS):
                
                ############################### LOAD RAW DATA ################################
                cur_sep_flag, sep_flags = get_sep_flags(classifier_name, dataset_name, sep_flags)
                if classifer_id == 0 or sep_flags[classifer_id] != sep_flags[classifer_id-1]:
                    
                    All_data, All_labels, All_users, ALL_SUBJECTS_ID, X_train, label2act, ACT_LABELS, ActID, \
                       MODELS_COMP_LOG_DIR, INPUT_CHANNEL, POS_NUM, cal_attitude_angle, STFT_intervals = load_all_data(classifer_id,
                                                                    classifier_name, dataset_name, cur_sep_flag, CUR_DIR)
                
                for AUGMENT_METHODS in args.AUGMENT_METHODS_LIST:
                    torch.cuda.empty_cache()
                
                    ########################################## LEAVE ONE SUBJECT OUT ##########################################
                    # set logging settings
                    EXEC_TIME, LOG_DIR, MODEL_DIR, logger, fileHandler = logging_settings(classifier_name, CUR_DIR, dataset_name, AUGMENT_METHODS)
                    time2 = 0
                    
                    for subject_id in ALL_SUBJECTS_ID:
                        # get train and test subjects
                        TEST_SUBJECTS_ID  = [subject_id]
                        TRAIN_SUBJECTS_ID = list(set(ALL_SUBJECTS_ID).difference(set(TEST_SUBJECTS_ID)))
                        
                        ########################## DATA PREPROCESSING #########################
                        X_train, y_train, X_test, y_test, start1, end1 = data_preprocessing(ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID,
                                                                                            TEST_SUBJECTS_ID, All_users, All_data,
                                                                                            All_labels, X_train, classifier_name,
                                                                                            STFT_intervals, POS_NUM, args.test_split)
                        #######################################################################
                        
                        ############### LOG DATASET INFO AND NETWORK PARAMETERS ###############
                        nb_classes = log_dataset_network_info(logger, ALL_SUBJECTS_ID, TRAIN_SUBJECTS_ID, TEST_SUBJECTS_ID,
                                                              X_train, X_test, y_train, y_test, cal_attitude_angle,
                                                              ACT_LABELS, ActID, label2act, BATCH_SIZE, EPOCH, LR)
                        #######################################################################
                        
                        ##### SPLIT TRAINSET TO TRAIN AND VAL DATASETS #####
                        # Initilize the logging variables
                        if subject_id == min(ALL_SUBJECTS_ID):
                            models, scores, log_training_duration = initialize_saving_variables(X_train, X_test, nb_classes)
                        
                        X_tr, X_val, Y_tr, Y_val = train_test_split(X_train, y_train, 
                                                                    test_size=0.1,
                                                                    random_state=6,
                                                                    stratify=y_train)
                        X_test = STFT_precessing(X_test, y_test, classifier_name, STFT_intervals, POS_NUM, args.test_split, AUGMENT_METHODS[0])
                        ######################## Training and Testing Process ############################
                        # Create classifier
                        classifier, classifier_func, classifier_parameter = create_cuda_classifier(dataset_name, classifier_name,
                                                                                                   INPUT_CHANNEL, POS_NUM, X_tr.shape[-1],
                                                                                                   X_tr.shape[0], X_val.shape[0],
                                                                                                   X_test.shape[0], nb_classes, STFT_intervals,
                                                                                                   BATCH_SIZE, args, AUGMENT_METHODS)
                        
                        ###### TRAIN FOR EACH SUBJECT (if have already finished training, print 'Already_done') ######
                        per_training_duration, log_training_duration, output_directory_models = training_process(
                                                                                                   logger, subject_id, X_tr, X_val, X_test, Y_tr, Y_val,
                                                                                                   y_test, nb_classes, classifier_parameter, classifier,
                                                                                                   classifier_func, MODEL_DIR, args.PATTERN, EPOCH, BATCH_SIZE, LR,
                                                                                                   POS_NUM, AUGMENT_METHODS, log_training_duration, classifier_name, 
                                                                                                   STFT_intervals, args)
                        ########## TEST FOR EACH SUBJECT, record inference time ###########
                        X_tr = STFT_precessing(X_tr, Y_tr, classifier_name, STFT_intervals, POS_NUM, args.test_split, AUGMENT_METHODS[0])
                        X_val  = STFT_precessing(X_val, Y_val, classifier_name, STFT_intervals, POS_NUM, args.test_split, AUGMENT_METHODS[0])
                        pred_train, pred_valid, pred_test, scores, time_duration = classifier_func.predict_tr_val_test(dataset_name, classifier, nb_classes, ACT_LABELS,
                                                                                                                       X_tr, X_val, X_test, Y_tr, Y_val, y_test,
                                                                                                                       scores, per_training_duration,
                                                                                                                       subject_id, output_directory_models,
                                                                                                                       args.test_split)
                        
                        time2 = time2 + time_duration
                        ##################################################################################
                    
                    ################ LOG TEST RESULTS, LOG INFERENCE TIME #################
                    log_final_test_results(logger, log_training_duration, scores, label2act, nb_classes, ALL_SUBJECTS_ID,
                                           start1, end1, X_tr, y_train, y_test, args.AUGMENT_METHODS_LIST, AUGMENT_METHODS, 
                                           time2, dataset_name, classifier_name, MODELS_COMP_LOG_DIR,
                                           args.CLASSIFIERS, classifier, classifier_parameter, fileHandler)
                    #######################################################################
                    
def main(args):
    Main = Train_Test(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)