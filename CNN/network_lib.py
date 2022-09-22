from tensorflow.keras.utils import plot_model
import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import pickle
import time
import torch
import awkward as ak

#File names and folders
MinBiasdir = "data/MB/" # MB folder name
MinBiastrainfilename = "minbias_train-003.h5" # MB train test and validation name
MinBiastestfilename = "minbias_test.h5"
MinBiasvalidfilename = "minbias_valid.h5"
MinBiasvalidptestfilename = "minbias_valid_p_test.h5" # Valid and test set added together for larger set

LoneMuondir = "data/LM/" # Same for LM
LoneMuontrainfilename = "lone_muon_train_new.h5"
LoneMuontestfilename = "lone_muon_test_new.h5"
LoneMuonvalidfilename = "lone_muon_valid_new.h5"
LoneMuonvalidptestfilename = "lone_muon_valid_p_test_13-10_new.h5"


#Starting parameters
units = 7 #Currenlty loaded in minibatches of 8. This means 7 minibatches of 8 are loaded.
#Notice some mini batches can be smaller.
base = 8 #Mini batch size
bs = units * base #Expected total batch size
nSl = 6 # Number of slices
nL = 8
nusedL = 9
nusedSl= nSl

nbinsphi0 = 216+2*6 #phi-bins
nbinsQOpT = 216+2*2 #pT-bins

size_x = nbinsphi0
size_y = nbinsQOpT

phi0min0 = 0.3 #phi-range (excluding padding)
phi0max0 = 0.5

phi0minstep = (phi0max0-phi0min0)/216 #phi-bin-size

phi0minmin = phi0min0 - 6*phi0minstep #Minimal phi-value including padding
phi0minmid = phi0minmin + 0.5*phi0minstep #Mid of lowest phi bin


QOpTmin0 = -1
QOpTmax0 = 1 #GeV^-1

QOpTminstep = (QOpTmax0-QOpTmin0)/216

QOpTminmin = QOpTmin0 - 2*QOpTminstep
QOpTminmid = QOpTminmin + 0.5*QOpTminstep

req_layers = 6 #Require atleast 6 hit layers

p = 2716/4459232 #Part of bins in minbias+muon that have at least 6 hit layers that actually is a target bin
p6 = 171/3904041
p7 = 958/563841
p8 = 1646/65801

batch_size = 16

HDF5=False
#Getting the files
if HDF5:
    #Train HDF5 file
    MinBiastrainstorage = pd.HDFStore(MinBiasdir + MinBiastrainfilename)
    LoneMuontrainstorage = pd.HDFStore(LoneMuondir + LoneMuontrainfilename)

    #Train key list to file
    Minbiastrainlist = MinBiastrainstorage.keys()
    LoneMuontrainlist = LoneMuontrainstorage.keys()

    #Length of lists
    nMinbiastrainlist = len(Minbiastrainlist)
    nLoneMuontrainlist = len(LoneMuontrainlist)

    ntrainmax = np.min((len(Minbiastrainlist), len(LoneMuontrainlist)))



    MinBiasvalidstorage = pd.HDFStore(MinBiasdir + MinBiasvalidfilename)
    LoneMuonvalidstorage = pd.HDFStore(LoneMuondir + LoneMuonvalidfilename)

    Minbiasvalidlist = MinBiasvalidstorage.keys()
    LoneMuonvalidlist = LoneMuonvalidstorage.keys()

    nMinbiasvalidlist = len(Minbiasvalidlist)
    nLoneMuonvalidlist = len(LoneMuonvalidlist)

    nvalidmax = np.min((len(Minbiasvalidlist), len(LoneMuonvalidlist)))



    MinBiasteststorage = pd.HDFStore(MinBiasdir + MinBiastestfilename)
    LoneMuonteststorage = pd.HDFStore(LoneMuondir + LoneMuontestfilename)

    Minbiastestlist = MinBiasteststorage.keys()
    LoneMuontestlist = LoneMuonteststorage.keys()

    nMinbiastestlist = len(Minbiastestlist)
    nLoneMuontestlist = len(LoneMuontestlist)

    ntestmax = np.min((len(Minbiastestlist), len(LoneMuontestlist)))



    MinBiasvalidpteststorage = pd.HDFStore(MinBiasdir + MinBiasvalidptestfilename)
    LoneMuonvalidpteststorage = pd.HDFStore(LoneMuondir + LoneMuonvalidptestfilename)

    Minbiasvalidptestlist = MinBiasvalidpteststorage.keys()
    LoneMuonvalidptestlist = LoneMuonvalidpteststorage.keys()

    nMinbiasvalidptestlist = len(Minbiasvalidptestlist)
    nLoneMuonvalidptestlist = len(LoneMuonvalidptestlist)

    # nvalidptestmax = np.min((len(Minbiasvalidptestlist), len(LoneMuonvalidptestlist)))
    nvalidptestmax = nLoneMuonvalidptestlist #UPDATED this is better when using the deterministic shuffle, but not the random one.


    nmax = ntrainmax

    def get_saved_truth(df):
        n_events = int(len(df.index)/nSl/nL)

        truthlist = []
        for i in range(n_events):
            for j in range(nSl):
                iqOpT = df['truth_qOpT'].iloc[i*nSl*nL + j*nL]
                if iqOpT != -1:
                    iphi0 = df['truth_phi0'].iloc[i*nSl*nL + j*nL]

                    truthlist += [[i, j, iqOpT, iphi0]]

        return truthlist

    def get_images_allslices(df):
        n_events = int(len(df.index)/nSl/nL)

        images = torch.empty(0,nSl,nL,nbinsQOpT,nbinsphi0, dtype = torch.int)
        for i in range(n_events):
            slices = torch.empty(0,nL,nbinsQOpT,nbinsphi0, dtype = torch.int)
            for j in range(nSl):
                layers = torch.empty(0,nbinsQOpT,nbinsphi0, dtype = torch.int)
                for k in range(nL):
                    layer = torch.zeros((nbinsQOpT,nbinsphi0), dtype = torch.int)
                    layerx = df['x_pos'].iloc[i*nSl*nL + j*nL + k]
                    layery = df['y_pos'].iloc[i*nSl*nL + j*nL + k]
                    value = df['value'].iloc[i*nSl*nL + j*nL + k]
                    layerxnp = ak.to_numpy(layerx).astype(int)
                    layerynp = ak.to_numpy(layery).astype(int)
                    valuenp = ak.to_numpy(value).astype(int)
                    layer[layerynp,layerxnp] = torch.from_numpy(valuenp)

                    layers = torch.cat((layers, layer.unsqueeze(0)),0)

                slices = torch.cat((slices, layers.unsqueeze(0)),0)

            images = torch.cat((images, slices.unsqueeze(0)),0)

        return images.numpy()

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import Model
def network(CPU=False,with_dropout=True):
    if CPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    filters_per_conv_layer = [16,16,24]
    neurons_per_dense_layer = [42,64]

    y = x = x_in = Input(shape=(220,220,1))

    for i,f in enumerate(filters_per_conv_layer):
        print( ('Adding convolutional block {} with N={} filters').format(i,f) )
        x = Conv2D(int(f), kernel_size=(3,3), strides=(1,1), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False,
                   name='conv_{}'.format(i))(x) 
        x = BatchNormalization(name='bn_conv_{}'.format(i))(x)
        x = Activation('relu',name='conv_act_%i'%i)(x)
        x = MaxPooling2D(pool_size = (2,2),name='pool_{}'.format(i) )(x)
        if with_dropout:
            x = Dropout(0.1)(x)
    x = Flatten()(x)
    if with_dropout:
        x = Dropout(0.5)(x)

    for i,n in enumerate(neurons_per_dense_layer):
        print( ('Adding dense block {} with N={} neurons').format(i,n) )
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name='dense_%i'%i, use_bias=False)(x)
        x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
        x = Activation('relu',name='dense_act_%i'%i)(x)
        if with_dropout:
            x = Dropout(0.1)(x)
    x = Dense(2,name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    out = concatenate([x_out,x])
    model = Model(inputs=[x_in], outputs=[out], name='keras_baseline')
    return model

def network2(CPU=False,with_dropout=True):
    if CPU:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    filters_per_conv_layer = [16,16,24]
    neurons_per_dense_layer = [42,64]

    y = x = x_in = Input(shape=(36,36,1))

    for i,f in enumerate(filters_per_conv_layer):
        print( ('Adding convolutional block {} with N={} filters').format(i,f) )
        x = Conv2D(int(f), kernel_size=(3,3), strides=(1,1), kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001), use_bias=False,
                   name='conv_{}'.format(i))(x) 
        x = BatchNormalization(name='bn_conv_{}'.format(i))(x)
        x = Activation('relu',name='conv_act_%i'%i)(x)
        x = MaxPooling2D(pool_size = (2,2),name='pool_{}'.format(i) )(x)
        if with_dropout:
            x = Dropout(0.1)(x)
    x = Flatten()(x)
    if with_dropout:
        x = Dropout(0.5)(x)

    for i,n in enumerate(neurons_per_dense_layer):
        print( ('Adding dense block {} with N={} neurons').format(i,n) )
        x = Dense(n,kernel_initializer='lecun_uniform', kernel_regularizer=l1(0.0001),name='dense_%i'%i, use_bias=False)(x)
        x = BatchNormalization(name='bn_dense_{}'.format(i))(x)
        x = Activation('relu',name='dense_act_%i'%i)(x)
        if with_dropout:
            x = Dropout(0.1)(x)
    x = Dense(2,name='output_dense')(x)
    x_out = Activation('softmax',name='output_softmax')(x)

    model = Model(inputs=[x_in], outputs=[x_out], name='keras_baseline')
    return model


# train_data = False
# if train_data:
#     _arr = np.zeros((220,228))
#     _arry = np.zeros(4)
#     train_set_x = np.array([_arr])
#     train_set_y = np.array([_arry])


#     for i ,f in enumerate(LoneMuontrainlist):
#         LMtrain = LoneMuontrainstorage[f]
#         truth = get_saved_truth(LMtrain)
#         LMimage = get_images_allslices(LMtrain)
#         print(f)
#         #if f=='/lonemuon1346_13':
#          #   break
#         for name in Minbiastrainlist[i:i+6]:
#             try:
#                 MBtrain = MinBiastrainstorage[name]
#                 MBimage = get_images_allslices(MBtrain)
#             except:
#                 continue
#             for t in truth:
#                 train_set_x = np.append(train_set_x,[np.sum(MBimage[0,0],axis=0)+np.sum(LMimage[t[0],t[1]],axis=0)],axis=0)
#                 train_set_y = np.append(train_set_y,[np.array([1,1,t[2],t[3]])],axis=0)

#     for name in Minbiastrainlist:
#         MBtrain = MinBiastrainstorage[name]
#         MBimage = get_images_allslices(MBtrain)
#         img = np.sum(MBimage[0,0],axis=0)
#         argmax = np.argmax(img)
#         train_set_x = np.append(train_set_x,[img],axis=0) # all permutaions of min_bias and lone_muon
#         train_set_y = np.append(train_set_y,[np.array([0,0,argmax // 220, argmax%228])],axis=0)


#     output = open('trainx.pkl', 'wb')
#     pickle.dump(train_set_x, output)
#     output.close()
#     output = open('trainy.pkl', 'wb')
#     pickle.dump(train_set_y, output)
#     output.close()
# else:
#     objectRep = open("trainx.pkl", "rb")
#     train_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("trainy.pkl", "rb")
#     train_set_y = pickle.load(objectRep)
#     objectRep.close()
# train_x = train_set_x[:,:,:-8].reshape(*train_set_x[:,:,:-8].shape,1)
# train_y = train_set_y

# train_data2=False
# if train_data2:
#     train_set_y = np.zeros((len(train_y),2))
#     for i, f in enumerate(train_y):
#         if f[0]:
#             train_set_y[i,0] = 1
#         else:
#             train_set_y[i,1] = 1

#     output = open('train2x.pkl', 'wb')
#     pickle.dump(train_set_x, output)
#     output.close()
#     output = open('train2y.pkl', 'wb')
#     pickle.dump(train_set_y, output)
#     output.close()
# else:
#     objectRep = open("train2x.pkl", "rb")
#     train_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("train2y.pkl", "rb")
#     train_set_y = pickle.load(objectRep)
#     objectRep.close()
# train2_y = train_set_y

# valid_data = False
# if valid_data:
#     _arr = np.zeros((220,228))
#     _arry = np.zeros(4)
#     valid_set_x = np.array([_arr])
#     valid_set_y = np.array([_arry])


#     for f in LoneMuonvalidlist:
#         LMvalid = LoneMuonvalidstorage[f]
#         truth = get_saved_truth(LMvalid)
#         LMimage = get_images_allslices(LMvalid)
#         print(f)
#         #if f=='/lonemuon1346_13':
#          #   break
#         for name in Minbiasvalidlist:
#             try:
#                 MBvalid = MinBiasvalidstorage[name]
#                 MBimage = get_images_allslices(MBvalid)
#             except:
#                 continue
#             for t in truth:
#                 valid_set_x = np.append(valid_set_x,[np.sum(MBimage[0,0],axis=0)+np.sum(LMimage[t[0],t[1]],axis=0)],axis=0)
#                 valid_set_y = np.append(valid_set_y,[np.array([1,1,t[2],t[3]])],axis=0)

#     for name in Minbiasvalidlist:
#         MBvalid = MinBiasvalidstorage[name]
#         MBimage = get_images_allslices(MBvalid)
#         valid_set_x = np.append(valid_set_x,[np.sum(MBimage[0,0],axis=0)],axis=0) # all permutaions of min_bias and lone_muon
#         valid_set_y = np.append(valid_set_y,[np.zeros(4)],axis=0)



#     output = open('validx.pkl', 'wb')
#     pickle.dump(valid_set_x, output)
#     output.close()
#     output = open('validy.pkl', 'wb')
#     pickle.dump(valid_set_y, output)
#     output.close()
# else:
#     objectRep = open("validx.pkl", "rb")
#     valid_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("validy.pkl", "rb")
#     valid_set_y = pickle.load(objectRep)
#     objectRep.close()
# valid_x = valid_set_x[:,:,:-8].reshape(*valid_set_x[:,:,:-8].shape,1)
# valid_y = valid_set_y

# valid_data2=False
# if valid_data2:
#     valid_set_y = np.zeros((len(valid_y),2))
#     for i, f in enumerate(valid_y):
#         if f[0]:
#             valid_set_y[i,0] = 1
#         else:
#             valid_set_y[i,1] = 1

#     output = open('valid2x.pkl', 'wb')
#     pickle.dump(valid_set_x, output)
#     output.close()
#     output = open('valid2y.pkl', 'wb')
#     pickle.dump(valid_set_y, output)
#     output.close()
# else:
#     objectRep = open("valid2x.pkl", "rb")
#     valid_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("valid2y.pkl", "rb")
#     valid_set_y = pickle.load(objectRep)
#     objectRep.close()
# valid2_y = valid_set_y


# test_data = False
# if test_data:
#     _arr = np.zeros((220,228))
#     _arry = np.zeros(4)
#     test_set_x = np.array([_arr])
#     test_set_y = np.array([_arry])


#     for f in LoneMuontestlist:
#         LMtest = LoneMuonteststorage[f]
#         truth = get_saved_truth(LMtest)
#         LMimage = get_images_allslices(LMtest)
#         print(f)
#         #if f=='/lonemuon1346_13':
#          #   break
#         for name in Minbiastestlist:
#             try:
#                 MBtest = MinBiasteststorage[name]
#                 MBimage = get_images_allslices(MBtest)
#             except:
#                 continue
#             for t in truth:
#                 test_set_x = np.append(test_set_x,[np.sum(MBimage[0,0],axis=0)+np.sum(LMimage[t[0],t[1]],axis=0)],axis=0)
#                 test_set_y = np.append(test_set_y,[np.array([1,1,t[2],t[3]])],axis=0)

#     for name in Minbiastestlist:
#         MBtest = MinBiasteststorage[name]
#         MBimage = get_images_allslices(MBtest)
#         test_set_x = np.append(test_set_x,[np.sum(MBimage[0,0],axis=0)],axis=0) # all permutaions of min_bias and lone_muon
#         test_set_y = np.append(test_set_y,[np.zeros(4)],axis=0)




#     output = open('testx.pkl', 'wb')
#     pickle.dump(test_set_x, output)
#     output.close()
#     output = open('testy.pkl', 'wb')
#     pickle.dump(test_set_y, output)
#     output.close()
# else:
#     objectRep = open("testx.pkl", "rb")
#     test_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("testy.pkl", "rb")
#     test_set_y = pickle.load(objectRep)
#     objectRep.close()
# test_x = test_set_x[:,:,:-8].reshape(*test_set_x[:,:,:-8].shape,1)
# test_y = test_set_y

# test_data2=False
# if test_data2:
#     test_set_y = np.zeros((len(test_y),2))
#     for i, f in enumerate(test_y):
#         if f[0]:
#             test_set_y[i,0] = 1
#         else:
#             test_set_y[i,1] = 1

#     output = open('test2x.pkl', 'wb')
#     pickle.dump(test_set_x, output)
#     output.close()
#     output = open('test2y.pkl', 'wb')
#     pickle.dump(test_set_y, output)
#     output.close()
# else:
#     objectRep = open("test2x.pkl", "rb")
#     test_set_x = pickle.load(objectRep)
#     objectRep.close()
#     objectRep = open("test2y.pkl", "rb")
#     test_set_y = pickle.load(objectRep)
#     objectRep.close()
# test2_y = test_set_y
from pathlib import Path
def getReports(indir):
    data_ = {}
    
    report_vsynth = Path('{}/vivado_synth.rpt'.format(indir))
    report_csynth = Path('{}/myproject_prj/solution1/syn/report/myproject_csynth.rpt'.format(indir))
    
    if report_vsynth.is_file() and report_csynth.is_file():
        print('Found valid vsynth and synth in {}! Fetching numbers'.format(indir))
        
        # Get the resources from the logic synthesis report 
        with report_vsynth.open() as report:
            lines = np.array(report.readlines())
            data_['lut']     = int(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[2])
            data_['ff']      = int(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[2])
            data_['bram']    = float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[2])
            data_['dsp']     = int(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[2])
            data_['lut_rel'] = float(lines[np.array(['CLB LUTs*' in line for line in lines])][0].split('|')[5])
            data_['ff_rel']  = float(lines[np.array(['CLB Registers' in line for line in lines])][0].split('|')[5])
            data_['bram_rel']= float(lines[np.array(['Block RAM Tile' in line for line in lines])][0].split('|')[5])
            data_['dsp_rel'] = float(lines[np.array(['DSPs' in line for line in lines])][0].split('|')[5])
        
        with report_csynth.open() as report:
            lines = np.array(report.readlines())
            lat_line = lines[np.argwhere(np.array(['Latency (cycles)' in line for line in lines])).flatten()[0] + 3]
            data_['latency_clks'] = int(lat_line.split('|')[2])
            data_['latency_mus']  = float(lat_line.split('|')[2])*5.0/1000.
            data_['latency_ii']   = int(lat_line.split('|')[6])
    
    return data_