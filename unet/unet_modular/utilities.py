import os
import sys
import lasagne

def gen_config(layers,inchannels,startchannels,outputs,act,ltype,optimiser):
    config = {}
    config['act'] = act
    config['layertype'] = ltype
    config['layers'] = {}
    config['layers']['output'] = outputs

    config['layers']['enc'] = []

    tmp = startchannels
    for i in range(layers):
        if i == 0:
            config['layers']['enc'].append(
            { "conv" : [ tmp, tmp ],
              "shape" : [ None,inchannels,None,None] }
            )
        else:
            config['layers']['enc'].append(
                { "conv" : [ tmp, tmp ] }
            )

        tmp *= 2
    config['layers']['bottom'] = {\
        "conv" : [ tmp, tmp ] }
    tmp = tmp/2
    config['layers']['dec'] = []
    for i in range(layers):
        config['layers']['dec'].append(
            { "conv" : [ tmp, tmp, tmp ] } )
        tmp /= 2
    config['train_params'] = {
        'lr' : 0.0001,
        'mom' : 0.9
        }
    config['optimiser'] = optimiser
    return config

def get_activation(act):
    if act=='relu':
        return lasagne.nonlinearities.rectify
    elif act == 'lrelu':
        return lasagne.nonlinearities.leaky_rectify
    elif act == 'vlrelu':
        return lasagne.nonlinearities.very_leaky_rectify
    elif act == 'elu':
        return lasagne.nonlinearities.elu
    elif act == 'sigmoid':
        return lasagne.nonlinearities.sigmoid
    else:
        return None

def get_updates(cost,params,optimiser, lr):
    if optimiser == 'nesterov_momentum':
        return lasagne.updates.nesterov_momentum(cost, params, learning_rate=lr, momentum=0.9)
    if optimiser == 'adagrad':
        return lasagne.updates.adagrad(cost, params, learning_rate=lr, epsilon=1e-06)
    if optimiser == 'rmsprop':
        return lasagne.updates.rmsprop(cost, params, learning_rate=lr, rho=0.9, epsilon=1e-06)
    if optimiser == 'adadelta':
        return lasagne.updates.adadelta(cost, params, learning_rate=lr, rho=0.95, epsilon=1e-06)
    if optimiser == 'adam':
        return lasagne.updates.adam(cost, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
    if optimiser == 'adamax':
        return lasagne.updates.adamax(cost, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)
