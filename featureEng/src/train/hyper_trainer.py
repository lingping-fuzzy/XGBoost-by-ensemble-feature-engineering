import os
import joblib
import numpy as np
import sys
# Add the root project directory to the sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
# from sklearn.svm import SVC
# from src.train.customeXGB import my_Exgboost
#from . import tuneBaseline, customeXGB, tuneCustomXGB, tuneCustomSVM
# from src.train.customeLgb import my_Exgboost
# from .tuneBaseline import my_tune
# from .tuneCustomXGB import my_tuneExgboost
from .utils import save_results_cc, expanduservars, _build_datasets_UCI
from ..fEngeering import transformData
from src.train import tuneEngineeringXGBv31, tuneEngAssistXGBAv51
import pandas as pd

label_to_name = ['annealing',
 'breast-cancer-wisc-diag',
 'breast-cancer-wisc-prog',
 'congressional-voting',
 'conn-bench-sonar-mines-rocks',
 'conn-bench-vowel-deterding',
 'credit-approval',
 'cylinder-bands',
 'dermatology',
 'flags',
 'heart-cleveland',
 'heart-hungarian',
 'heart-va',
 'hepatitis',
 'horse-colic',
 'ionosphere',
 'libras',
 'molec-biol-promoter',
 'oocytes_merluccius_nucleus_4d',
 'oocytes_merluccius_states_2f',
 'oocytes_trisopterus_nucleus_2f',
 'oocytes_trisopterus_states_5b',
 'parkinsons',
 'planning',
 'primary-tumor',
 'spectf',
 'statlog-australian-credit',
 'statlog-german-credit',
 'statlog-heart',
 'statlog-image',
 'statlog-vehicle',
 'synthetic-control',
 'wine',
 'zoo',
 'pittsburg-bridges-REL-L',
 'plant-texture',
 'pittsburg-bridges-MATERIAL',
 'plant-shape',
 'yeast',
 'breast-tissue',
 'wine-quality-white',
 'glass',
 'pittsburg-bridges-TYPE',
 'plant-margin',
 'pittsburg-bridges-T-OR-D',
 ]
#label_to_name =['abalone', 'acute-inflammation', 'acute-nephritis', 'balance-scale', 'balloons', 'bank', 'blood', 'breast-cancer', 'breast-cancer-wisc', 'car', 'cardiotocography-10clases', 'cardiotocography-3clases', 'chess-krvkp', 'contrac', 'echocardiogram', 'energy-y1', 'energy-y2', 'fertility', 'haberman-survival', 'hayes-roth', 'hill-valley', 'ilpd-indian-liver', 'image-segmentation', 'iris', 'led-display', 'lenses', 'magic', 'mammographic', 'molec-biol-splice', 'monks-1', 'monks-2', 'monks-3', 'mushroom', 'musk-1', 'musk-2', 'optical', 'ozone', 'page-blocks', 'pima', 'pittsburg-bridges-SPAN', 'ringnorm', 'seeds', 'semeion', 'spambase', 'spect', 'statlog-landsat', 'steel-plates', 'teaching', 'thyroid', 'tic-tac-toe', 'titanic', 'twonorm', 'vertebral-column-2clases', 'vertebral-column-3clases', 'wall-following', 'waveform', 'waveform-noise', 'wine-quality-red']


def get_param(args = None):
  confini = dict()
  confini['subsample']	=	args['subsample']
  confini['colsample_bylevel']	=	args['colsample_bylevel']
  confini['colsample_bynode']	=	args['colsample_bynode']
  confini['colsample_bytree']	=	args['colsample_bytree']
  confini['eta']	=	args['eta']
  confini['gamma']	=	args['gamma']
  confini['max_depth'] = args['max_depth']
  confini['n_estimators'] = args['round']
  return confini


def run_feat_save(params: dict, params_file):
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars(('Featdata' + '//' ))
    os.makedirs(output_path, exist_ok=True)

    for gassid in range( len(label_to_name) ): #range(40, 41):len(dataname)
        params['class_label'] = gassid

        print('this is ', label_to_name[gassid], ' -start')
        for seedid in range(4): #( params['folds']):
            params['seed'] = seedid
            train_x, train_y, test_x, test_y, val_x, val_y, space = _build_datasets_UCI(params)

            trs = transformData(train_x, train_y.ravel(), val_x, test_x)
            try:
                x_new, x_test_new, x_val_new = trs._transform_autofeat()
            except:
                x_new, x_test_new, x_val_new = pd.DataFrame(train_x), pd.DataFrame(test_x), pd.DataFrame(val_x)
            save_data(os.path.join(output_path, label_to_name[gassid]), (x_new, x_test_new, x_val_new), splits=('auto'+str(seedid)))
            x_new, x_test_new, x_val_new = trs._transform_hpca()
            save_data(os.path.join(output_path, label_to_name[gassid]), (x_new, x_test_new, x_val_new),
                      splits=('hpca' + str(seedid)))
            x_new, x_test_new, x_val_new = trs._transform_randomProject()
            save_data(os.path.join(output_path, label_to_name[gassid]), (x_new, x_test_new, x_val_new),
                      splits=('randP' + str(seedid)))
            x_new, x_test_new, x_val_new = trs._transform_minmax()
            save_data(os.path.join(output_path, label_to_name[gassid]), (x_new, x_test_new, x_val_new),
                      splits=('minmax' + str(seedid)))
            x_new, x_test_new, x_val_new = trs._transform_robuster()
            save_data(os.path.join(output_path, label_to_name[gassid]), (x_new, x_test_new, x_val_new),
                      splits=('robuster' + str(seedid)))



# for gaussian data and wordnet data
def run_feat_UCI(params: dict, params_file):
    # Create output folder and archive the current code and the parameters there
    output_path = expanduservars((params['output_path'] + '//' + params['method']))
    os.makedirs(output_path, exist_ok=True)

    for gassid in range(len(label_to_name)): #
        params['class_label'] = gassid

        hyp_f1_micro = []
        hyp_f1_macro = []
        hyp_f1_weight = []
        hyp_acc = []
        hyp_recall = []
        hyp_precision = []
        print('this is ', label_to_name[gassid], ' -start')
        for seedid in range(4):#( params['folds']):
            params['seed'] = seedid
            #classifier = tuneEngineeringXGBv31.my_featExgboost(params=params, output_path=output_path,
                                                            #data_name=label_to_name[params['class_label']])

            classifier = tuneEngAssistXGBAv51.my_featExgboost(params=params, output_path=output_path,
                                                            data_name=label_to_name[params['class_label']])


            try:
                best_config = classifier.train()
                acc, f1_macro, f1_micro, f1_weight, precision, recall = classifier.predict(best_config)
            except:
                acc, f1_macro, f1_micro, f1_weight, precision, recall = 0, 0, 0, 0, 0, 0

            print("tree f1 micro: %.4f, f1 macro: %.4f, f1_weight: %.4f. acc %.2f", f1_micro, f1_macro,
                        f1_weight, acc)

            hyp_f1_micro.append(f1_micro)
            hyp_f1_macro.append(f1_macro)
            hyp_f1_weight.append(f1_weight)
            hyp_acc.append(acc)
            hyp_precision.append(precision)
            hyp_recall.append(recall)

        save_results_cc(output_path, params, hyp_f1_micro, hyp_f1_macro, hyp_f1_weight, hyp_acc, hyp_precision, hyp_recall)



def save_data(path, data, splits ='1'):
    os.makedirs(path, exist_ok=True)
    joblib.dump(data, os.path.join(path, splits+'_data.pkl'))


def load_data(path, splits ='1'):
    return joblib.load(os.path.join(path, splits+'_data.pkl'))

def get_score(test_y, test_y_pred):
    f1_micro = f1_score(test_y, test_y_pred, average="micro")
    f1_macro = f1_score(test_y, test_y_pred, average="macro")
    f1_weight = f1_score(test_y, test_y_pred, average="weighted")
    acc = accuracy_score(test_y, test_y_pred)

    precesion = precision_score(test_y, test_y_pred, average="macro", zero_division=0)
    recall = recall_score(test_y, test_y_pred, average="macro", zero_division=0)
    return acc, f1_macro, f1_micro, f1_weight, precesion, recall

