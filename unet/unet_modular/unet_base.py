import sys
import os
# import time
import json
#import cv2
import numpy as np
import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode
import lasagne
from lasagne.layers import batch_norm,ElemwiseSumLayer,NonlinearityLayer
from lasagne.regularization import regularize_network_params, l2, l1
from lasagne.layers.dnn import Conv2DDNNLayer as Conv2DLayer
from lasagne.layers.dnn import MaxPool2DDNNLayer as MaxPool2DLayer
from unet.unet_modular.utilities import *

lasagne.layers.Conv2DLayer = Conv2DLayer
lasagne.layers.MaxPool2DLayer = MaxPool2DLayer

def softmax(x):
     return T.exp(x)/(T.exp(x).sum(1,keepdims=True))

def maxout(x,filters,kernel,maxout):
     x = batch_norm(lasagne.layers.Conv2DLayer(
          x, num_filters=filters*maxout,filter_size=(kernel,kernel),
          nonlinearity=None,pad=1,
          W=initf
     ))
     x = lasagne.layers.FeaturePoolLayer(x, pool_size=maxout)
     return x

def sftmax(x):
     sftmax = x.reshape((x.shape[0],x.shape[1],x.shape[2]*x.shape[3]))
     sftmax = sftmax.dimshuffle((1,0,2))
     sftmax = sftmax.reshape((sftmax.shape[0],sftmax.shape[1]*sftmax.shape[2]))
     sftmax = softmax(T.transpose(sftmax))
     return sftmax

def loss(yp,yt,w):
     return -T.mean(T.log(yp)[T.arange(yp.shape[0]), yt]*w)

def normal(ilayer,fmaps,activation,t='enc',ltype='normal'):
     if t == 'enc':
          x = batch_norm(lasagne.layers.Conv2DLayer(
               ilayer, num_filters=fmaps[0],filter_size=(3,3),
               nonlinearity=None,pad=1,
               W=initf
          ))
     else:
          x = batch_norm(lasagne.layers.Conv2DLayer(
               ilayer, num_filters=fmaps[0],filter_size=(3,3),
               nonlinearity=activation,pad=1,
               W=initf
          ))
     if ltype == 'normal':
          x = batch_norm(lasagne.layers.Conv2DLayer(
               x, num_filters=fmaps[1],filter_size=(3,3),
               nonlinearity=activation,pad=1,
               W=initf
          ))
     elif ltype == 'residual':
          x = batch_norm(lasagne.layers.Conv2DLayer(
               x, num_filters=fmaps[1],filter_size=(3,3),
               nonlinearity=None,pad=1,
               W=initf
          ))
          y = lasagne.layers.Conv2DLayer(
               ilayer, num_filters=fmaps[1],filter_size=(1,1),
               nonlinearity=None,pad='same', W=initf)
          x = ElemwiseSumLayer([x, y])
          x = NonlinearityLayer(x,nonlinearity=activation)
     return x

initf = lasagne.init.GlorotUniform()

def build_network(cfg,input_var):
    encs = cfg['layers']['enc']
    act = cfg['act']
    layertype = cfg['layertype']
    activation = get_activation(act)
    if activation == None:
         return
    inpLayer = encs[0]
    inpShape = tuple(inpLayer['shape'])
    enc_outputs = []
    x = lasagne.layers.InputLayer(
        shape=inpShape,
        input_var=input_var
    )
    x = normal(x,inpLayer['conv'],activation,ltype=layertype)
    enc_outputs.append(x)
    x = lasagne.layers.MaxPool2DLayer(
        x, pool_size=(2,2)
    )

    for enc in encs[1:]:
        x = normal(x,enc['conv'],activation)
        enc_outputs.append(x)
        x = lasagne.layers.MaxPool2DLayer(
            x, pool_size=(2,2)
        )

    x = normal(x,cfg['layers']['bottom']['conv'],activation,t='dec',ltype=layertype)
    x = lasagne.layers.DropoutLayer(
         x,p=0.5
    )
    decs = cfg['layers']['dec']

    for i,dec in enumerate(decs):
        enco = enc_outputs[-(i+1)]
        x = lasagne.layers.Upscale2DLayer(
            x,
            scale_factor=(2,2)
        )
        convf = dec['conv'][0]
        x = batch_norm(lasagne.layers.Conv2DLayer(
            x, num_filters=convf,filter_size=(2,2),
            nonlinearity=None,pad='full',
             W=initf
        ))
        x = lasagne.layers.ConcatLayer(
            [x,enco],
            cropping=['center',None,'center','center']
        )
        x = normal(x,dec['conv'][1:],activation,t='dec',ltype=layertype)

    ox = cfg['layers']['output']
    x = lasagne.layers.Conv2DLayer(
        x, num_filters=ox,filter_size=(1,1),
        nonlinearity=sftmax,
         W=initf
    )

    return x

def get_functions(cfg):
     input_var = T.tensor4('inputs')
     target_var = T.ivector('targets')
     weights_var = T.vector('weights')
     lr=theano.shared(np.float32(0.0000))

     network = build_network(cfg,input_var)
     prediction = lasagne.layers.get_output(network)
     test_prediction = lasagne.layers.get_output(network, deterministic = True)
     output_shape = lasagne.layers.get_output_shape(network)
     l2_penalty = regularize_network_params(network, l2)
     l1_penalty = regularize_network_params(network, l1)
     cost = loss(prediction,target_var,weights_var) + 5*1e-6*(l1_penalty + l2_penalty)
     params = lasagne.layers.get_all_params(network, trainable=True)
     #print (len(params))
     def save_params(path):
          np.savez(path,params)
          return

     def load_params(path):
          data = np.load(path)
          param_values = [ x.get_value() for x in data['arr_0'] ]
        #   print (len(param_values))
          lasagne.layers.set_all_param_values(network, param_values, trainable=True)
          return

     def set_lr(value):
          lr.set_value(value)
          return

     optimiser= cfg['optimiser']
     updates = get_updates(cost,params, optimiser ,lr)

     def acc(yp,yt):
          output = T.argmax(yp,axis=1)
          return T.mean(T.eq(output, target_var))

     accuracy = acc(prediction,target_var)

     train_fn = theano.function([input_var, target_var,weights_var], cost, updates=updates)
     val_fn = theano.function([input_var, target_var], accuracy)
     train_predict_fn = theano.function([input_var], prediction)
     test_predict_fn = theano.function([input_var], test_prediction)

     return train_fn, test_predict_fn, train_predict_fn, save_params, load_params, output_shape, set_lr
