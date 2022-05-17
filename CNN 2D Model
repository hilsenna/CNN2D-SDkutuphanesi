
### Import Libraries
# General system libraries
import os
import gc
#import cv2
import math
import h5py
import locale
import inspect
import argparse
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
from subprocess import call

# Math & Visualization Libs
import math
#import pydot
#import graphviz
import seaborn as sns
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from itertools import cycle
from scipy import stats, interp
from IPython.display import Image

# Multiprocessing
import multiprocessing

# DNA/RNA Analysis Libraries (Biopython, ViennaRNA, pysster) 
# > Biopython Lib
import Bio
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
#from Bio.Alphabet import generic_rna, generic_dna, generic_protein, IUPAC
# > ViennaRNA Lib

import RNA

#RNAssp
os.chdir('/Volumes/hipokrat/CL_RNA_SynthBio-master') #python dilinde cd demek
import src.RNAssp.rna as rnassp

# Import Tensorflow
import tensorflow as tf

# Import Json
import json
import codecs

# Import Keras
from keras import optimizers
from keras import applications
from keras import regularizers
from keras import backend as K
from keras.models import Sequential, load_model
from keras.models import model_from_json, load_model
from keras.layers import Activation, Conv1D, Conv2D, Reshape, BatchNormalization, Dropout, Flatten, Dense, merge, Input, Lambda, InputLayer, Convolution2D, MaxPooling1D, MaxPooling2D, ZeroPadding2D, Bidirectional
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.applications.imagenet_utils import decode_predictions, preprocess_input
from keras.wrappers.scikit_learn import KerasClassifier
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.utils import np_utils
#from keras.utils import multi_gpu_model
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model

#from keras.utils import plot_model
from keras.utils.vis_utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import ModelCheckpoint

#Import Keras-Vis for Saliency
from vis.utils import utils
#from vis.visualization import get_num_filters
#from vis.visualization import visualize_activation, visualize_saliency, visualize_cam, overlay
    ## NOTE: Install in conda enviroment: pip install git+https://github.com/raghakot/keras-vis.git -U

# Import sklearn libs
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics.pairwise import cosine_similarity
    ## NOTE: Activate a new terminal to monitor NVIDIA GPU usage writing
        # watch -n0.5 nvidia-smi
    ## NOTE: If not present, activate GPU persistence mode in terminal with
        # sudo nvidia-smi -pm 1
    ## If you do not see any GPU usage try uncommenting the following line:
        # sess = tf.Session(config=tf.ConfigProto(log_device_placement=True)) #To ensure activation of GPUs in TF Backend

# Progress Bar
from tqdm import tqdm

# Warnings
import warnings
warnings.filterwarnings("ignore")
import pickle 
#Visualization mode
#%matplotlib ipympl

# from numpy import array
# from keras.models import Sequential
# from keras.layers import Dense
# from matplotlib import pyplot
# # prepare sequence
# X = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
# # create model
# model2 = Sequential()
# model2.add(Dense(2, input_dim=1))
# model2.add(Dense(1))
# model2.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae', 'mape', tf.metrics.CosineSimilarity()])
# # train model
# history2 = model2.fit(X, X, epochs=3, batch_size=len(X), verbose=2)
# # plot metrics
# pyplot.plot(history2.history['mse'])
# pyplot.plot(history2.history['mae'])
# pyplot.plot(history2.history['mape'])
# pyplot.plot(history2.history['cosine_similarity'])
# pyplot.show()
# print(history2.history.keys())
###############################################################


## Define helper function to copy full directory for backups
def copy_full_dir(source, target):
    call(['cp', '-a', source, target]) # Unix

#Get available CPUs,
ncpus = multiprocessing.cpu_count()
print('Available CPUs: '+ str(ncpus))

#Get number of available GPUs
def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
ngpus = len(get_available_gpus())
print('Available GPUs: '+ str(ngpus)) 



###############################################################
# Create Data folder if not existent
data_folder = "data/"
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)
    
## Define general path to store all generated models
core_models_path = 'models/'
# Create Data folder if not existent
if not os.path.isdir(core_models_path):
    os.makedirs(core_models_path)

# Define path to load desired Toehold dataset file (.csv)
data_filename = "Toehold_Dataset_Final_2019-10-23.csv"
data_path = '/Users/neslihanyuksel/Desktop/traingirdison.csv'
data = pd.read_csv(data_path)

# Fix random seed for reproducibility
seed = 7
np.random.seed(seed) # Seed can be any number

#nesl biz ekledik alttakini
model_path=('/Volumes/hipokrat/CL_RNA_SynthBio-master/models/cnn_2d_148')
###############################################################

### Datata Sequence ID selection
id_data = data['sequence_id']

### Toehold Switch dataset input/output columns for selection
input_cols = 'seq_SwitchON_GFP'
output_cols = ['ON', 'OFF', 'ON_OFF'] 
qc_levels = [1.1,1]
doTrain = True 
loss_init = 'mae' #'logcosh', #'mse', 'mae', 'r2'
n_foldCV = 10 
verbose_init = True
evaluate  = True
display_init = False 
n_filters_init = 10

### Define data scaler (if any)
scaler_init = False 
scaler = QuantileTransformer(output_distribution='uniform') #This method transforms the features to follow 
#a uniform or a normal distribution

### DEFINE MODEL NAME (e.g. MLP, CNN, LSTM, etc.)
model_name = 'CNN_2D_148'

#Show sample of dataframe structure
data.head()

###############################################################
# Helper function to pass string DNA/RNA sequence to one-hot
def dna2onehot(seq):
    #get sequence into an array
    seq_array = np.array(list(seq))
    
    #integer encode the sequence
    label_encoder = LabelEncoder() #Encode target labels with value between 0 and n_classes-1.
   #This approach is very simple and it involves converting each value in a column to a number. 
    integer_encoded_seq = label_encoder.fit_transform(seq_array)
    
    #one hot the sequence
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    #reshape because that's what OneHotEncoder likes
    integer_encoded_seq = integer_encoded_seq.reshape(len(integer_encoded_seq), 1)
    onehot_encoded_seq = onehot_encoder.fit_transform(integer_encoded_seq)
    #fit_transform() is used on the training data so that we can 
    #scale the training data and also learn the scaling parameters of that data
    
    return onehot_encoded_seq


###############################################################
# Function definition to create complementary matrix of RNA molecule from DNA
def one_hot_special_complementarity_directional_matrix(x, show=False):
    """Produce complementarity matrix for the given RNA molecule.
    by Luis Soenksen 2019-03-15
    Complementary bases (according to Watson-Crick) are assigned the following values:
    G-C are assigned 6 = [0 0 0 0 0 0 1], 
    C-G are assigned 5 = [0 0 0 0 0 1 0], 
    A-U are assigned 4 = [0 0 0 0 1 0 0],
    U-A are assigned 3 = [0 0 0 1 0 0 0],
    G-U are assigned 2 = [0 0 1 0 0 0 0], 
    U-G are assigned 1 = [0 1 0 0 0 0 0], 
    NonWCpairs are   0 = [1 0 0 0 0 0 0],

    Args:
        m: Molecule object (DNA or RNA)
        show (bool): Make a matrix plot of the result.

    Returns:
        p_oh: One-Hot Encoded Categorical Complementarity-directional Matrix
        p: Categorical(integer) Complementarity-directional Matrix
    """
    # Imports
    from tensorflow.keras.utils import to_categorical
    
    if isinstance(x, str):
    # If input is string do this
        # Generate complementary matrix from sequence str & calculated dot-bracket secondary structure
        seq = Bio.Seq.transcribe(x) #Each row in the series is a toehold sequence
        (ss, _ ) = RNA.fold(seq) # Compute corresponding secondary structure (to generate molecule object)
        m = rnassp.Molecule(seq, ss) # Generate molecule object to get complementarity matrix
        l = len(m.seq)
        p = np.zeros((l, l), dtype='int8')
        for i in range(l):
            for j in range(l):
                if m.seq[i] == 'G' and m.seq[j] == 'C' :
                    p[i, j] = 6
                if m.seq[i] == 'A' and m.seq[j] == 'U' :
                    p[i, j] = 4  
                if m.seq[i] == 'G' and m.seq[j] == 'U' :
                    p[i, j] = 2

                # By default... if m.seq[i] == m.seq[j] ; p[i, j] = 0

                if m.seq[i] == 'C' and m.seq[j] == 'G':
                    p[i, j] = 5
                if m.seq[i] == 'U' and m.seq[j] == 'A':
                    p[i, j] = 3  
                if m.seq[i] == 'U' and m.seq[j] == 'G':
                    p[i, j] = 1

        if show:
            fig = plt.figure(dpi=300)
            ax = fig.add_subplot(111)
            cmap = plt.get_cmap('jet', np.max(p)-np.min(p)+1)
            pos = ax.matshow(p, interpolation='nearest', cmap=cmap)
            ax.set_xticks(np.arange(l))
            ax.set_yticks(np.arange(l))
            ax.set_xticklabels([i for i in m.seq], fontsize=2)
            ax.set_yticklabels([i for i in m.seq], fontsize=2)

            # Add colorbar to make it easy to read the energy levels
            cbar = plt.colorbar(pos, ticks=np.arange(np.min(p),np.max(p)+1))
            cbar.ax.set_yticklabels(['N', 'U-G', 'G-U', 'U-A', 'A-U', 'C-G', 'G-C']) 
            plt.show()                    
       
                    
    elif (isinstance(x, np.ndarray) and np.array_equal(x , x.astype(bool))):
    # If input is one-hot encoded
        m = x  
        l = len(m)
        p = np.zeros((l, l), dtype='int8')

        A = np.array([1.,0.,0.,0.])
        C = np.array([0.,1.,0.,0.])
        G = np.array([0.,0.,1.,0.])
        U = np.array([0.,0.,0.,1.]) # "T" or "U"

        for i in range(l):
            for j in range(l):
                if (np.array_equal(m[i], G) and np.array_equal(m[j], C)):
                    p[i, j] = 6
                if (np.array_equal(m[i], A) and np.array_equal(m[j], U)):
                    p[i, j] = 4  
                if (np.array_equal(m[i], G) and np.array_equal(m[j], U)):
                    p[i, j] = 2

                # By default... if m.seq[i] == m.seq[j] ; p[i, j] = 0

                if (np.array_equal(m[i], C) and np.array_equal(m[j], G)):
                    p[i, j] = 5
                if (np.array_equal(m[i], U) and np.array_equal(m[j], A)):
                    p[i, j] = 3  
                if (np.array_equal(m[i], U) and np.array_equal(m[j], G)):
                    p[i, j] = 1 
    
    # elif(isinstance(x,float)):
    #     p = np.empty([len(x),len(x),7])
        #nesli not üstteki iki satır doruğun önerisi
        
        
    
    else:
        p = np.empty([len(x),len(x),7])
    
   # p_oh= np.zeros([l,l+1])
  #  p_oh[np.arange(p.shape[0]), p.T] = 1
   # p_oh 
    p_oh = to_categorical(p[:,:], num_classes = 7) #Convert to one hot     
    p_oh_tensor = tf.convert_to_tensor(p_oh) #Convert to one hot           
    return p_oh_tensor, p_oh, p

###############################################################

# Function to generate single complementary matrix array from linear toehold sequence
def seq2complementaryMap(x):
    _, p_oh , _ = one_hot_special_complementarity_directional_matrix(x, show=False)
    out = p_oh.astype(bool)
    return out
###############################################################

def seq2FlatComplementaryMap(x):
    _, _, p = one_hot_special_complementarity_directional_matrix(x, show=False)
    out = p
    return out

###############################################################

# Function to generate single complementary matrix array from linear toehold sequence
def seq2complementaryMap_output_shape(input_shape):
    # Generate complementary matrix from sequence str & calculated dot-bracket secondary structure
    return (tuple(input_shape[0]),tuple(input_shape[0]),tuple(7))



###############################################################

# INPUT / OUTPUT DEFINITION, PROCESSING & LOADING
def pre_process_data (data, input_cols, output_cols, export_path, qc_level_threshold=1, scaler_init=False, display=True):
    ## OUTPUT / INPUT DEFINITION, PROCESSING & LOADING
    
    #Init process bar
    tqdm.pandas() # Use `progress_apply` when `applying` one hot encoding and complementarity function to dataframe of input sequences

    # Define path to store input / output arrays
    tmp_data_path = 'data/tmp/'
    # Create Data folder if not existent
    if not os.path.isdir(tmp_data_path):
        os.makedirs(tmp_data_path)
    
    # GENERATE & SAVE FULL INPUT VECTORS (only if not exists because it is a large file)
    tmp_data_input_path = (tmp_data_path + 'data_input_file_2d_' + input_cols + '.h5')
    if not os.path.exists(tmp_data_input_path):
        # Data Input selection & Generation if absent (or delete it to re-calculate)
        n_batch_samples = 10000 #We constrain process batches to this number to allow for progressive saving of large files
        n_batches = math.ceil(len(data[input_cols])/n_batch_samples)
        
        if n_batches>1:
            print('Processing in ' + str(n_batches) + ' batches...')
            
        # Process and Append Save:
        with h5py.File(tmp_data_input_path, 'a') as hf:
            is_first=True
            for data_tmp in np.split(data[input_cols], n_batches):
                df_data_input_tmp = data_tmp.progress_apply(seq2complementaryMap)
                data_input_tmp = np.array(list(df_data_input_tmp.values)) 
                if is_first==True:
                    hf.create_dataset("input",  data=data_input_tmp, maxshape=(None, data_input_tmp.shape[1], data_input_tmp.shape[2], data_input_tmp.shape[3]), chunks=(n_batches,data_input_tmp.shape[1], data_input_tmp.shape[2], data_input_tmp.shape[3])) 
                    is_first=False
                else:
                    hf["input"].resize(( hf["input"].shape[0]+data_input_tmp.shape[0],data_input_tmp.shape[1],data_input_tmp.shape[2],data_input_tmp.shape[3]))  
                    hf["input"][-data_input_tmp.shape[0]:,:,:] = data_input_tmp
    
    # GENERATE & SAVE FULL INPUT VECTORS (NPY OPTION)
    tmp_data_input_path = (tmp_data_path + 'data_input_file_2d_' + input_cols + '.npy')
    if not os.path.exists(tmp_data_input_path):
        # Data Input selection & Generation if absent (or delete it to re-calculate)
        df_data_input = data[input_cols].progress_apply(seq2complementaryMap)
        data_input = np.array(list((df_data_input.values)))
    #    #Save:
        np.save(tmp_data_input_path, data_input) #Save npy file
        
    # GENERATE & SAVE FULL OUTPUT VECTORS
    output_ids = "_".join(str(x) for x in output_cols)
    tmp_data_output_path = (tmp_data_path + 'data_output_file_QC_' + str(qc_level_threshold).replace('.','-') + '_' + output_ids + '.h5')
    
    # Data Output selection (QC filtered, OutColumns Only & Drop NaNs)
    df_data_output = data[data.QC_ON_OFF >= qc_level_threshold]
    df_data_output= df_data_output[output_cols] 
    df_data_output = df_data_output.dropna(subset=output_cols)
    data_output = df_data_output.to_numpy().astype('float32')
    #data_output = np.array(list(df_data_output.values), dtype=np.float32)

    #Save:
    with h5py.File(tmp_data_output_path, 'w') as hf:
        hf.create_dataset("output",  data=data_output)
    
    #Load full input array in memory and QC filter
    data_input = np.load(tmp_data_input_path, mmap_mode='r') #Loading with read from hard-disk
    data_input = data_input[df_data_output[:-1].index.values][:][:]
    
    # LOAD FULL OUTPUT ARRAY in memory and QC filter
    with h5py.File(tmp_data_output_path, 'r') as hf:
        data_output = hf['output'][:]
        data_output_orig = data_output
        #Pre-process data (scaler)
        if scaler_init==True:
            data_output = scaler.fit_transform(data_output)
    
    # LOAD FULL LIST OF SEQUENCES after filtering
    data_seqs = data[input_cols][df_data_output.index.values] #input_colsu output cols ilse değiştirdim
    
    
    # Display processed data if desired
    if display==True:
        ### Show example of processed dataset
        ## Display number of retrieved sequences
        print("Number of sequences retrieved: "+str(len(data_input)))
        print()

        #Select ID to show
        toehold_id = 0

        ## Plot Example input complementarity toehold matrix 
        #print('EXAMPLE OF INPUT ONE-HOT COMPLEMENTARITY TOEHOLD MATRIX (COLOR CODED)')
        p_oh_tensor, p_oh, p = one_hot_special_complementarity_directional_matrix(data_seqs.iloc[toehold_id], show=False)
        print()
    
        # Display input size
        print("Input Size: " + str(p_oh_tensor.shape))
        print()

        # Display example of Output vector
        print('EXAMPLE OF OUTPUT VECTOR')
        print(' ' + str(data_output[toehold_id]))
        print()

        # Display Output Values
        for index,item in enumerate(output_cols):
            # Display Output Values
            plt.figure()
            print('Distribution of ' + str(item) + ' Values')
            sns.distplot(data_output[:,index], kde=True, rug=False)
            #Tight plot
            plt.tight_layout()
            # Save figure
            plt.savefig(export_path + "/QC_" + str(qc_level_threshold).replace('.','-') + "_" + str(item) + "_dist.png", bbox_inches='tight', dpi=300)
        
        if scaler_init==True:           
            ## COMPUTE EFFECT OF SCALER
            # Difference between the transformed toehold output values and original toehold output values, 
            # then compute the absolute percentage difference for diplay
            test_metrics = np.zeros((data_output_orig.shape[1],3))
            diff = data_output_orig - data_output
            abstDiff = np.abs(diff)
            # Compute the mean and standard deviation of the absolute difference:
            apd_mean = np.mean(abstDiff, axis=0)
            apd_std = np.std(abstDiff, axis=0)
            apd_r2 = np.zeros_like(apd_mean)
            
            # Plot: Scaled Output values vs. Original values and get R2 value
            for index,item in enumerate(output_cols):
                # R2 (Coefficient of Determination)
                apd_r2[index] = r2(data_output_orig[:,index], data_output[:,index])
                
                # Display Output Values
                x=np.squeeze(data_output_orig[:,index])
                y=np.squeeze(data_output[:,index])
                
                # Display Output Values
                print('')
                print("" + item + " Mean_absolute_error (TRANSFORMATION): " + str(apd_mean[index]) + " (SD: " + str(apd_std[index]) + ")" )
                print('')
                print('TRANSFORMED Values vs. ORIGINAL values (' + item + ')' )
                print('Pearson Correlation: '+ str(stats.pearsonr(x, y)[0]))
                print('Spearman Correlation: '+ str(stats.spearmanr(x, y)[0]))
                print('R2: '+ str(apd_r2[index]))
                print('')
                
                if scaler_init == True:
                    g = sns.jointplot(x, y, kind="reg", color="b", xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
                    
                else:
                    g = sns.jointplot(x, y, kind="reg", color="b", stat_func=r2)
                #g.plot_joint(plt.scatter, c="b", s=1, linewidth=1, marker=".", alpha=0.08)
                #g.plot_joint(sns.kdeplot, zorder=0, color="m", n_levels=6, shade=False)
                g.ax_joint.collections[0].set_alpha(0)
                g.set_axis_labels("$ORIGINAL$", "$TRANSFORMED$");

                # save the figure
                g.savefig(export_path + "/QC_" + str(qc_level_threshold).replace('.','-') + "_" + str(item) + "_data_scaling_" + str(item) + ".png", bbox_inches='tight', dpi=300)
                
                # Store model performance metrics for return   
                test_metrics[index, :] = [apd_mean[index], apd_std[index], apd_r2[index]]
                
            # SAVE METRICS (.npy file)
            np.save(export_path + '/scaling_metrics',test_metrics)
            # SAVE DATA (.npz file)
            np.savez(export_path + '/scaling_data',data_output_orig,data_output) 
            
    return data_input, data_output
###############################################################

## Function to create Keras CNN for regression prediction
def create_2d_cnn(width, height, depth, filters=[32, 64, 128], regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (width, height, depth) #DNA/RNA complementary matrix input sequence (one hot encoded)
    chanDim = -1
    dropout_init = 0.3
    
    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input appropriately
        if i == 0:
            x = inputs
    
        # CONV => RELU => BN => POOL
        #x = Conv2D(f, (5,5), padding="same", kernel_regularizer=regularizers.l1_l2(l1=0.0005, l2=0.0005))(x)
        x = Conv2D(f, (5,5), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=chanDim)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        
    # flatten the volume, then FC => RELU => BN => DROPOUT
    x = Flatten()(x)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Dropout(dropout_init)(x)
    
    # apply another FC layer, this one to potentially match the number of nodes 
    # in a parallel network (i.e MLP with rational parameters)
    x = Dense(16)(x)
    x = Activation("relu")(x)
    
    # check to see if the regression node should be added
    if regress:
        x = Dense(len(output_cols), activation="linear")(x)
    else:
        x = Dense(len(output_cols), activation="sigmoid")(x)

    # Construct the Model
    model = Model(inputs, x) 
    
    # Return the model
    return model

###############################################################
# Helper functions to save/load model and training history
def saveHist(path,history):
    new_hist = {}
    for key in list(history.history.keys()):
        if type(history.history[key]) == np.ndarray:
            new_hist[key] == history.history[key].tolist()
        elif type(history.history[key]) == list:
            if  type(history.history[key][0]) == np.float64:
                new_hist[key] = list(map(float, history.history[key]))

    with codecs.open(path, 'w', encoding='utf-8') as f:
        json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4) 
        
def loadHist(path):
    with codecs.open(path, 'r', encoding='utf-8') as f:
        n = json.loads(f.read())
    return n

def save_model(model, identifier):
    ## MODEL SAVING ON WORKING FOLDER
    # Option 1: Save via Weights + Architecture
    model.save_weights(os.path.abspath(model_path)+'/model_weights_'+str(identifier)+'.h5')
    with open(os.path.abspath(model_path)+'/model_architecture_'+str(identifier)+'.json', 'w') as f:
        f.write(model.to_json())

    # Option 2: Save entire model at once
    model.save(os.path.abspath(model_path)+'/model_'+str(identifier)+'.h5')

###############################################################

#Definition of R2 metric for testing
def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2

#Definition of Custom metric as loss related to Coefficient of Determination (R2) 
#  CoD = 1 - MSE / (variance of inputs), and since this is going to be a loss we want 
#  improvement to point towards zero, so we choose mse/variance of inputs
def custom_r2_loss(y_true, y_pred): 
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return (SS_res/(SS_tot + K.epsilon()))

#Definition of Custom metric as loss related to Weigted Mean Absolute error
#  Improvement points towards zero, but penalizes loss for small values and improves it for larger values
def custom_wmae_loss(y_true, y_pred): 
    from keras import backend as K
    weightedMAE = K.abs((y_true-y_pred)*y_true) #Increase loss for large ON or OFF values -- Skews focus of distribution right
    return weightedMAE

###############################################################

def batch_generator(X, Y, batch_size = 1):
    indices = np.arange(len(X)) 
    batch=[]
    while True:
            for i in indices:
                batch.append(i)
                if len(batch)==batch_size:
                    yield X[batch], Y[batch]
                    batch=[]
###############################################################



### Define our final model architecture (layers & optimizor) and then compile it
def generate_model(model_path, trainX, testX, trainY, testY, verbose_init, evaluate=True):
    ## DEEP-LEARNING TRAINING PARAMETERS(e.g. verbose, patients, epoch size, batch size) to constrain process
    verbose_init = verbose_init #Zero is no keras verbose
    patience_init = 20 # Number of epochs to wait for no model improvement before early stopping a training
    epochs_init = 300 
    batch_size_init = 64*(1+ngpus) # number of samples that will be propagated through the network at every epoch dependent on the number of GPUs
    validation_spit_init = 0.1 # Percentage of testing data to use in internal validation during training
    
    ## Create folder to store model (if not existent)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)
        
    ## Callbacks for training:
    #  Save the model weights to the same file, if and only if the validation accuracy improves.
    checkpoint_file_name = "model_checkpoint_weights.h5"
    model_checkpoint_path = os.path.join(os.path.abspath(model_path), checkpoint_file_name)
    
    if verbose_init==True:
        # Callback to be used for checkpoint generation and early stopping
        callbacks_list = [ModelCheckpoint(model_checkpoint_path, monitor='val_loss', verbose = verbose_init, save_best_only=True, mode='auto'),
                          EarlyStopping(monitor='val_loss', patience=patience_init, verbose = verbose_init),
                          TQDMNotebookCallback()] # Adds Keras integration with TQDM progress bars.
    else:
        # Callback to be used for checkpoint generation and early stopping
        callbacks_list = [ModelCheckpoint(model_checkpoint_path, monitor='val_loss', verbose = False, save_best_only=True, mode='auto'),
                          EarlyStopping(monitor='val_loss', patience=patience_init, verbose = False)]    
    
    ## Create Model (Change for MLP, CNN, ETC)
    # -------------------------------------------------------------------------------------------------------------------------------------
    # Define CNN model input shape
    (width, height, depth) = np.shape(trainX[0])
    
    # Define core model on CPU
    with tf.device("/cpu:0"):  
        model = create_2d_cnn(width, height, depth, filters=[32, 64, 128], regress=True)
        
    ## Initialize the optimizer and Compile model:
    #   Custom metric is used (see above), if we use "Mean absolute percentage error" that
    #   implies that we seek to minimize the absolute percentage difference between 
    #   our *predictions* and *actual* output values. We also calculate other 
    #   valuable metrics for regression evaluation 
    opt = Adam(lr=0.03, epsilon=None, decay=1e-3/200, amsgrad=False) # epsilon=1e-1 for POISSON loss, lr=0.001 is standard but 0.1 leads to more diagonal features
    
    if loss_init=="r2":
        model.compile(loss=custom_r2_loss, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc', custom_r2_loss]) #cosinei değiştirdik nesli not
    elif loss_init =="wmae":
        model.compile(loss=custom_wmae_loss, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc', custom_wmae_loss])
    else:
        model.compile(loss=loss_init, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc']) 
        
    # -------------------------------------------------------------------------------------------------------------------------------------  
    
    
    ## Parallel computing (if multiple GPUs are available)
    # Define model for training (CPU, Single GPU or Multi-GPU depending on availability of resources)
    if ngpus<=1:
        print("[INFO] training with Single GPU or CPU...")
        model_history = model.fit(trainX, trainY, validation_split=validation_spit_init, epochs=epochs_init, batch_size=batch_size_init,  verbose=verbose_init)#nesli
        
    else:
        print("[INFO] training with {} GPUs...".format(ngpus))
        # make the model parallel
        parallel_model = multi_gpu_model(model, gpus=ngpus)
        
        if loss_init=="r2":
            parallel_model.compile(loss=custom_r2_loss, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc', custom_r2_loss])
        elif loss_init =="wmae":
            parallel_model.compile(loss=custom_wmae_loss, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc', custom_wmae_loss])
        else:
            parallel_model.compile(loss=loss_init, optimizer=opt,  metrics=['mse','mae', 'mape', tf.metrics.CosineSimilarity(),'acc']) 
        
        model_history = parallel_model.fit(trainX, trainY, validation_split=validation_spit_init, epochs=epochs_init, batch_size=batch_size_init,  verbose=verbose_init)#nesli
                    

    ## MODEL SAVING
    # Option 1: Save via Weights + Architecture
    model.save_weights(os.path.abspath(model_path)+'/model_weights.h5')
    with open(os.path.abspath(model_path)+'/model_architecture.json', 'w') as f:
        f.write(model.to_json())
    # Option 2: Save entire model at once
    model.save(os.path.abspath(model_path)+'/model.h5')
    # Save model graph to file
    model_graph_path = os.path.abspath(model_path) + '/model_graph.png'
    plot_model(model, to_file=model_graph_path, show_shapes=True, show_layer_names=True)
    # Save training history
    #saveHist(model_path + '/model_history', model_history)
    with open('history.pickle', 'wb') as handle:
        pickle.dump(model_history, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #nesli üst ikiyi biz ekledik
    ## MODEL LOADING (to ensure it works)
    # Option 1: Load latest model via Weights + Architecture
    #with open(os.path.abspath(model_path)+'/model_architecture.json', 'r') as f:
    #    model = model_from_json(f.read())
    #    model.load_weights(os.path.abspath(model_path)+'/model_weights.h5')
    # Option 2: Load latest model via entire model at once
    if loss_init=="r2":
        model = load_model(os.path.abspath(model_path)+'/model.h5', custom_objects={'custom_r2_loss': custom_r2_loss})
    elif loss_init =="wmae":
        model = load_model(os.path.abspath(model_path)+'/model.h5', custom_objects={'custom_wmae_loss': custom_wmae_loss})
    else:
        model = load_model(os.path.abspath(model_path)+'/model.h5')
    
    # Load model training history
    #model_history = loadHist(model_path + '/model_history')
    with open('/Volumes/hipokrat/CL_RNA_SynthBio-master/models/cnn_2d_148/CNN_2D_148-ON-OFF-ON_OFF-QC1-1/base/history.pickle', 'rb') as handle:
        model_history = pickle.load(handle)
    # Init prediction output matrix
    testX_Preds = np.zeros_like(testY) #Empty matrix for full prediction evaluation
    # Init performance metrics matrix
    test_metrics = np.zeros((trainY.shape[1],3)) #Empty matrix for model performance metrics
    
    # GENERATE PREDICTIONS
    if testX.size > 0:
        ## Make predictions on testing data:
        print("Predicting functionality of Test Toeholds ...")
        print("")
        #Predictions in scaled space
        testX_Preds = model.predict(testX)
        
        
        if scaler_init == True:
            testY = scaler.inverse_transform(testY)
            testX_Preds = scaler.inverse_transform(testX_Preds)
        
        ## EVALUATE PERFORMANCE OF MODEL
        if evaluate==True:
            #print(model_history.keys())
            ## Plot training metrics per fold:
            plt.figure ()
            ax1 = plt.subplot(221)
            ax2 = plt.subplot(222)
            ax3 = plt.subplot(223)
            ax4 = plt.subplot(224)
            # Plot MSE metric
            ax1.set_title("Mean squared error")
            ax1.plot(model_history.history['mse'])
            # Plot MAE metric
            ax2.set_title("Mean absolute error")
            ax2.plot(model_history.history['mae'])
            # Plot MAPE metric
            ax3.set_title("Mean absolute percentage error")
            ax3.plot(model_history.history['mape'])
            # Plot CP metric
            ax4.set_title("Cosine Proximity")
            ax4.plot(model_history.history['cosine_similarity'])
            #Tight plot
            plt.tight_layout()
            # Save figure
            plt.savefig(model_path + "/model_training_metrics.png", bbox_inches='tight', dpi=300)

            ## Plot compiled training metrics per fold:
            plt.figure()
            plt.style.use("default")
            N = np.arange(0, len(model_history.history["loss"]))
            # Plot used Loss metric
            plt.plot(N, model_history.history["loss"], label="train_loss")
            plt.plot(N, model_history.history["val_loss"], label="test_loss")
            # Plot used Accuracy metric (applicable only if categorical model)
            plt.plot(N, model_history.history["acc"], label="train_acc")
            plt.plot(N, model_history.history["val_acc"], label="test_acc")
            # Plot MSE metric
            plt.plot(N, model_history.history["mse"], label="train_mse")
            plt.plot(N, model_history.history["val_mse"], label="test_mse")
            # Plot MAE metric
            plt.plot(N, model_history.history["mae"], label="train_mae")
            plt.plot(N, model_history.history["val_mae"], label="test_mae")
            # Plot MAPE metric
            plt.plot(N, model_history.history["mape"], label="train_mape")
            plt.plot(N, model_history.history["val_mape"], label="test_mape")
            # Plot CP metric
            plt.plot(N, model_history.history["cosine_similarity"], label="train_cp")
            plt.plot(N, model_history.history["val_cosine_similarity"], label="test_cp")
            plt.title("CNN Toehold Complementary Rep Data")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) # Place a legend to the right of this smaller subplot.
            # Save figure
            plt.savefig(model_path + "/model_training_compilation.png", bbox_inches='tight', dpi=300)
            
            
            ## COMPUTE PERFORMANCE METRICS
            # Difference between the *predicted* toehold functionality values and *actual* toehold functionality values, 
            # then compute the absolute percentage difference for diplay
            diff = testX_Preds - testY
            abstDiff = np.abs(diff)
            # Compute the mean and standard deviation of the absolute difference:
            apd_mean = np.mean(abstDiff, axis=0)
            apd_std = np.std(abstDiff, axis=0)
            apd_r2 = np.zeros_like(apd_mean)
            # Plot: Predicted values vs. Experimental values and get R2 value
            for index,item in enumerate(output_cols):

                # R2 (Coefficient of Determination)
                apd_r2[index] = r2(testX_Preds[:,index], testY[:,index])
                
                # Display Output Values
                x=np.squeeze(testX_Preds[:,index])
                y=np.squeeze(testY[:,index])
                
                # Display Output Values
                print("" + item + " Mean_absolute_error (TEST): " + str(apd_mean[index]) + " (SD: " + str(apd_std[index]) + ")" )
                print('')
                print('EXPERIMENTAL Values vs. PREDICTED values (' + item + ')' )
                print('Pearson Correlation: '+ str(stats.pearsonr(x, y)[0]))
                print('Spearman Correlation: '+ str(stats.spearmanr(x, y)[0]))
                print('R2: '+ str(apd_r2[index]))
                print('')
                
                if scaler_init == True:
                    g = sns.jointplot(x, y, kind="reg", color="b", xlim=(-0.2, 1.2), ylim=(-0.2, 1.2))
                else:
                    g = sns.jointplot(x, y, kind="reg", color="b")
                
                #g.plot_joint(plt.scatter, c="b", s=1, linewidth=1, marker=".", alpha=0.08)
                #g.plot_joint(sns.kdeplot, zorder=0, color="m", n_levels=6, shade=False)
                g.ax_joint.collections[0].set_alpha(0)
                g.set_axis_labels("$PREDICTED$", "$EXPERIMENTAL$");

                # save the figure
                g.savefig(model_path + "/model_performance_" + str(item) + ".png", bbox_inches='tight', dpi=300)
                
                # Store model performance metrics for return   
                test_metrics[index, :] = [apd_mean[index], apd_std[index], apd_r2[index]]
                
            # SAVE METRICS (.npy file)
            np.save(model_path + '/test_metrics',test_metrics)
            # SAVE DATA (.npz file)
            np.savez(model_path + '/test_data',testX_Preds, testY) 
                
    return model, model_history, testX_Preds, test_metrics

###############################################################
### Define our crossvalidation model generator (layers, optimizor, compilation, training, reporting, etc)
def generate_crossval_model(model_cv_path, X, Y, n_foldCV, verbose_init=True, evaluate=True):
    
    ## CROSSVALIDATION TRAINING
    # Define CV parameters
    n_foldCV = n_foldCV #Number of Crossvalidation bins
    cv_folds = list(StratifiedKFold(n_splits=n_foldCV, shuffle=True, random_state=seed).split(X,Y[:-1].argmax(1))) # Non repeating CV bins
    cv_preds = np.zeros_like(Y) #Empty matrix for full prediction evaluation
    cv_test_metrics = np.zeros((n_foldCV, Y.shape[1], 3))
    deploy_test_metrics = np.zeros((Y.shape[1],3))
    
    # Perform n-fold crossvalidated training and evaluation
    print('')
    print('Performing Crossvalidation...')
    for j, (train_idx, test_idx) in enumerate(cv_folds):
        print('\nFold ',j)
        
        # Define folder for CV fold model
        model_cv_fold_path = model_cv_path + '/Fold' + str(j) 

        ## CrossValidation Strategy: 
        # We use all data for n-crossvalidation this will give us average metrics of performance in future data
        # for this all data will be devided into n bins. In every sequential fold we will use n-1 bins for training 
        # and the remaining bin for testing this split is done in such a way that all data is used for training and
        # testing at some point (sweet!). Testing points will be aggregated to generate an average metric of performance
        # and all the datapoints will be put into a master agreement plot for visualization. 
        # A deploy model will be also trained using all available data without testing
        trainX_cv = X[train_idx]
        trainY_cv = Y[train_idx]
        testX_cv = X[test_idx]
        testY_cv = Y[test_idx]
        # NOTE: Validation set is taken internally from the training set (10% of each fold), this is applied in the the model.fit function
        
        # Create & Train model each fold according to generator function
        model, model_history, testX_Preds, test_metrics = generate_model(model_cv_fold_path, trainX_cv, testX_cv, trainY_cv, testY_cv, verbose_init=verbose_init, evaluate=evaluate)
        
        # Record predicted values of each CV fold training to generate an ensemble reporting
        print("Predicting functionality of CV-Fold Test Toeholds & Model performance metrics ...")
        cv_preds[test_idx,:] = testX_Preds
        cv_test_metrics[j,:,:] = test_metrics
        
        ## MODEL MEMORY RELEASE
        del model_history
        del model
        for i in range(ngpus+1): gc.collect()
        
        ## Free-up keras memmory to prevent leaks
        K.clear_session()
    
    #Transform back data
    if scaler_init == True:
        Y = scaler.inverse_transform(Y)
    
    # SAVE METRICS (.npy file)
    np.save(model_cv_path + '/cv_test_metrics',cv_test_metrics) 
    # SAVE DATA (.npz file)
    np.savez(model_cv_path + '/cv_test_data',cv_preds, Y)
            
    
    ## DEFINE FOLDER FOR DEPLOY MODEL
    model_deploy_path = model_cv_path + '/deploy'
    ## Create folder to store model (if not existent)
    if not os.path.isdir(model_deploy_path):
        os.makedirs(model_deploy_path)
    print('')
    print('Generating deployment model...')
    
    # COMPUTE PERFORMANCE METRICS FOR DEPLOY MODEL
    # Difference between the *predicted* toehold functionality values and *actual* toehold functionality values, 
    # then compute the absolute percentage difference for diplay

    diff = cv_preds - Y
    abstDiff = np.abs(diff)
    # Compute the mean and standard deviation of the absolute difference:
    apd_mean = np.mean(abstDiff, axis=0)
    apd_std = np.std(abstDiff, axis=0)
    apd_r2 = np.zeros_like(apd_mean)

    ## EVALUATE ENSEMBLE CROSSVALIDATION PERFORMANCE OF MODEL
    if evaluate==True:
        for index,item in enumerate(output_cols): 
            # R2 (Coefficient of Determination)
            apd_r2[index] = r2(cv_preds[:,index], Y[:,index])
            # Display Output Values
            x_tot=np.squeeze(cv_preds[:,index])
            y_tot=np.squeeze(Y[:,index])
            print('EXPERIMENTAL Values vs. PREDICTED values (' + item + ')' )
            print('Pearson Correlation: '+ str(stats.pearsonr(x_tot, y_tot)[0]))
            print('Spearman Correlation: '+ str(stats.spearmanr(x_tot, y_tot)[0]))
            print('R2: '+ str(apd_r2[index]))
            print('')
            
            if scaler_init == True:
                g = sns.jointplot(x_tot, y_tot, kind="reg", color="b", xlim=(-0.2, 1.2), ylim=(-0.2, 1.2)) 
            else:
                g = sns.jointplot(x_tot, y_tot, kind="reg", color="b")
            #g.plot_joint(plt.scatter, c="b", s=1, linewidth=1, marker=".", alpha=0.08)
            #g.plot_joint(sns.kdeplot, zorder=0, color="m", n_levels=6, shade=False)
            g.ax_joint.collections[0].set_alpha(0)
            g.set_axis_labels("$PREDICTED$", "$EXPERIMENTAL$");
            
            # save the figure
            g.savefig(model_deploy_path + "/model_ensemble_performance_" + str(item) + ".png", bbox_inches='tight', dpi=300)
            
            # Store model performance metrics for return   
            deploy_test_metrics[index, :] = [apd_mean[index], apd_std[index], apd_r2[index]]
            
    # SAVE METRICS (.npy file)
    np.save(model_deploy_path + '/deploy_test_metrics', deploy_test_metrics) 
    # SAVE DATA (.npz file)
    np.savez(model_deploy_path + '/deploy_test_data',cv_preds, Y)
    
    ## DEPLOYMENT MODEL TRAINING (with full dataset)
    # Partition the data into training (90%), validation (10%), testing (0%) splits 
    (trainX, testX, trainY, testY) = train_test_split(X, Y[:-1], test_size=0.25, random_state=seed)
    # Create model function according to generator function
    model, model_history, _ , _ = generate_model(model_deploy_path, trainX, testX, trainY, testY, verbose_init=True, evaluate=True)
                                                                                                                                               
    ## Return                                     
    return model, model_history, cv_preds, cv_test_metrics, deploy_test_metrics
###############################################################
#Definer function for full model analysis and reporting
def execute_model_analysis(core_models_path, model_name, data, input_cols, output_cols, qc_levels, n_foldCV, verbose_init, evaluate):
      
    #Iterate through all desired Data QC levels
    for j, qc_level in enumerate(qc_levels): 
        
        ### 1) Create all folders per iteration
        ## Define general path to store all generated models
        model_path = core_models_path + model_name.lower()+ '/' + model_name.upper() +'-' + str('-'.join(output_cols) + '-QC' + str(qc_level).replace('.','-') + '/')
        print ("Iteration " + str(j) + ") Building analysis in: " + model_path) 
        # Create Data folder if not existent
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        
        ## SAVE INIT PARAMETER SAVING ON WORKING FOLDER
        init_param_path = model_path + "init_parameters.txt"
        with open(init_param_path, "w+") as variable_file:
            variable_file.write("data_folder = " + str(data_folder)+ '\n' +\
                                "core_models_path = " + str(core_models_path)+ '\n' +\
                                "data_filename = " + str(data_filename)+ '\n' +\
                                "seed = " + str(seed)+ '\n' +\
                                "model_name = " + str(model_name)+ '\n' +\
                                "input_cols = " + str(input_cols)+ '\n' +\
                                "output_cols = " + str(output_cols)+ '\n' +\
                                "qc_level = " + str(qc_level)+ '\n' +\
                                "n_foldCV = " + str(n_foldCV)+ '\n' +\
                                "scaler = " + str(scaler)+ '\n' +\
                                "loss_init = " + str(loss_init)+ '\n' +\
                                "verbose_init = " + str(verbose_init)+ '\n' +\
                                "evaluate  =  " + str(evaluate)+ '\n' +\
                                "display_init =  " + str(display_init)+ '\n')
        
        ## Define path to store base model
        model_base_path = model_path + 'base'
        # Create Data folder if not existent
        if not os.path.isdir(model_base_path):
            os.makedirs(model_base_path)

        ## Define path to store crossvalidation models
        model_cv_path = model_path + 'crossval'
        # Create Data folder if not existent
        if not os.path.isdir(model_cv_path):
            os.makedirs(model_cv_path)

        ## Define path to store all generated model inputs
        model_input_path = model_path + 'input'
        # Create Data folder if not existent
        if not os.path.isdir(model_input_path):
            os.makedirs(model_input_path)
            
        ## Define path to store all generated model outputs
        model_output_path = model_path + 'output'
        # Create Data folder if not existent
        if not os.path.isdir(model_output_path):
            os.makedirs(model_output_path)
    
            
        ## LOAD PREPROCESSED INPUT / OUTPUT
        data_input, data_output = pre_process_data(data, input_cols, output_cols, model_input_path, qc_level_threshold=qc_level, scaler_init=scaler_init, display=display_init)
            
        ### 3) Model Training using Manual Verification Dataset & Evaluation
        # Training with a priori training (75%) & testing (25%) split, with internal training validation from the training set (10% or the 75%)
        # This also does valuation on unseen testing data (25%), and saves base model

        # Create manual working model function according to generator function, train it and display architecture
        # This will be made using 75% of the data for training and 25% for further testing.
        if doTrain==True:
            # Partition the data into training (75%) and testing (25%) splits
            (trainX, testX, trainY, testY) = train_test_split(data_input, data_output[:-1], test_size=0.25, random_state=seed)
            # Generate, Train, Evaluate, Save and Display Model
            print('')
            print('Training Basic Model...')
            model, model_history, testX_Preds, test_metrics = generate_model(model_base_path, trainX, testX, trainY, testY, verbose_init=verbose_init, evaluate=evaluate)
            model.summary()
        
            ## MODEL MEMORY RELEASE
            del model_history
            del model
            for i in range(ngpus+1): gc.collect()
        
            ## Free-up keras memmory to prevent leaks
            K.clear_session()
            print('')

        ### 4) Model Training using k-Fold Cross Validation, Ensemble Evaluation & Full Deployment
        # The gold standard for machine learning model evaluation is k-fold cross validation
        # It provides a robust estimate of the performance of a model on unseen data. 
        # It does this by splitting the training dataset into k subsets and takes turns training models on all subsets except one which is held out, and evaluating model performance on the held out validation dataset. The process is repeated until all subsets are given an opportunity to be the held out validation set. 
        # The performance measure accross all models in the unseen data for each fold
        # The performance is printed for each model and it is stored
        # A final deployment model trained in all data (no testing) is provided for evaluation in future data

        # Create crossvalidated model function according to generator function, train it and display architecture
        # Generate, Train, Evaluate, Save and Display Model
        if ((doTrain==True) & (n_foldCV>0)):
            model, model_history, cv_preds, cv_test_metrics, deploy_test_metrics = generate_crossval_model(model_cv_path, data_input, data_output, n_foldCV=n_foldCV, verbose_init=verbose_init, evaluate=evaluate)
            model.summary()
            
            ## MODEL MEMORY RELEASE
            del model_history
            del model             
            for i in range(ngpus+1): gc.collect()
                
            ## Free-up keras memmory to prevent leaks
            K.clear_session()
    
    
    ## MODEL SAVING ON DATED BACKUP FOLDER
    # Save the entire current model folder to a backup folder
    source_model_path = core_models_path + model_name + '/'
    backup_model_path = 'backup/' + source_model_path +  datetime.now().strftime('%Y%m%d') + '_' + datetime.now().strftime('%H%M')
    
    ## Create folder to store model (if not existent)
    if not os.path.isdir(backup_model_path):
        os.makedirs(backup_model_path)
    # Copy all contents to dated backup
    copy_full_dir(source_model_path, backup_model_path)
###############################################################
## RUN FULL MODEL ANALYSIS AND REPORTING model
execute_model_analysis(core_models_path, model_name, data, input_cols, output_cols, qc_levels, n_foldCV, verbose_init, evaluate)

