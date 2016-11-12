from __future__ import division
from unet.unet_modular.unet_base import *
from unet.unet_modular.utilities import *
from unet.unet_modular.progbar import *
from data_loader import train_data_loader, valid_data_loader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
import scipy as sp
from sklearn import metrics
import logging
# from datetime import datetime
import time
import csv
from pastalog import Log




seed = 123
rng = np.random.RandomState(seed)

combine_label = True
layers = 3
inputch = 1
filters = 64
if combine_label:
    outputs = 3
else:
    outputs = 4
act = 'relu'
ltype = 'normal'
lr = 1e-4
nb_epoch = 40
nb_samples_per_epoch = 1000
nb_val_samples = 100
patience = 20
optimiser = 'adam'
path = '/data/overalap-chromosomes/models/weights'
train_batch_size = 10
valid_batch_size = 10
# predicted_path = '/data2/processed/luna_segmentation/predicted'
save_path = '/data/overalap-chromosomes/demo/'

def sum_indices(arr, index_list):
    #s = 0
    #for i in index_list:
    #    s += arr[i]
    return sum([arr[i] for i in index_list])

def vis_detections(X_valid,Y_valid,Y_pred,save_image_path):
    fig = plt.figure()
    a=fig.add_subplot(1,3,1)
    plt.imshow(X_valid)
    a.set_title('Original Scan')
    a=fig.add_subplot(1,3,2)
    imgplot = plt.imshow(Y_valid)
    a.set_title('True')
    a=fig.add_subplot(1,3,3)
    plt.imshow(Y_pred)
    a.set_title('Predicted')
    plt.savefig(save_image_path)
    plt.close()


# Function returns the 3 required dice scores given the groundtruths and predictions numpy arrays
def get_dice_score(groundtruths, predictions):
    elements1, counts1 = np.unique(groundtruths, return_counts = True)
    elements2, counts2 = np.unique(predictions, return_counts = True)
    #assert(elements2 == elements1).all()
    unique_counts_groundtruths = np.array([0,0,0,0,0])
    unique_counts_predictions = np.array([0,0,0,0,0])
    for i in xrange(len(elements1)):
        unique_counts_groundtruths[elements1[i]] = counts1[i]
    for i in xrange(len(elements2)):
        unique_counts_predictions[elements2[i]] = counts2[i]

    #intersection_counts1 = [(np.where(groundtruths == i) == np.where(predictions == i)).sum() for i in xrange(5)]
    intersection_counts = np.array([0,0,0,0,0])
    for i in xrange(0, len(groundtruths)):
        if(groundtruths[i] == predictions[i]):
            intersection_counts[groundtruths[i]] += 1

    ct_arr = np.array([1,2,3,4])
    dice_score = 2*sum_indices(intersection_counts, ct_arr)/ (sum_indices(unique_counts_groundtruths, ct_arr)+sum_indices(unique_counts_predictions, ct_arr))
    return dice_score


def main_training(log_tuple, validation_set=0, threshold = 0.5, layers = 3, lr = 1e-2, nb_epoch = 5, nb_samples_per_epoch = 100 ,
nb_val_samples = 20, patience = 20,path = 'models/weights'):
    best_val_loss = np.inf
    not_done_looping = True
    nb_perf_not_improved = 0
    demo_dict = {}
    log_train,log_valid = log_tuple
    for epoch in range(nb_epoch):
        print ("Epoch: {}/{}".format(epoch+1, nb_epoch))
        if not_done_looping:
            progbar = Progbar(target=nb_samples_per_epoch)
            seen = 0
            count_train_samples = 0
            decay = math.pow(0.5, epoch/50)
            lr = lr*decay
            set_lr(lr)
            mean_accuracy = 0
            mean_val_loss = 0
            mean_dice_score = 0
            mean_precision = 0
            mean_recall = 0
            count_valid_samples = 0
            no_of_patches_seen =0
            mean_train_loss= 0
            mean_train_recall =0
            mean_train_precision =0
            mean_train_dice_score =0


            for X_train, Y_train, weights in train_data_loader(train_batch_size, combine_label):
                if count_train_samples == nb_samples_per_epoch:
                    break
                if seen < nb_samples_per_epoch:
                    log_values=[]
                xs = X_train.shape[2]
                ys = Y_train.shape[3]
                Y_train = Y_train.reshape((train_batch_size*xs*ys,))
                weights = weights.reshape((train_batch_size*xs*ys,))
                train_loss = train_fn(X_train.astype('float32'),Y_train.astype('int32'),weights.astype('float32'))
                Y_pred = predict_fn(X_train.astype('float32'))
                Y_pred_class = np.argmax(Y_pred, axis =1)
                dice_score = get_dice_score(Y_train,Y_pred_class)
                mean_train_loss+= train_loss
                mean_train_dice_score+= dice_score
                count_train_samples += X_train.shape[0]
                seen+= X_train.shape[0]
                log_values.append(('train_loss',train_loss))
                if seen < nb_samples_per_epoch:
                    progbar.update(seen,log_values)
            log_values.append(('train_loss',train_loss))
            progbar.update(seen,log_values, force=True)
            mean_train_loss = mean_train_loss/(nb_samples_per_epoch/train_batch_size)
            mean_train_dice_score = mean_train_dice_score/(nb_samples_per_epoch/train_batch_size)
            log_train.post('train_loss', mean_train_loss, epoch)
            log_train.post("mean_train_dice_score",mean_train_dice_score, epoch )


            if epoch % 5 == 0:
                validation_start = time.time()
                count_valid_samples = 0
                for X_valid,Y_valid in valid_data_loader(nb_val_samples, valid_batch_size, combine_label):
                    xs = X_valid.shape[2]
                    ys = Y_valid.shape[3]
                    Y_valid = Y_valid.reshape((valid_batch_size*xs*ys,))
                    Y_pred = test_predict_fn(X_valid.astype('float32'))
                    val_loss = loss(Y_pred.astype('float32'),
                                    Y_valid.astype('int32'),
                                    np.ones((Y_valid.shape[0],)).astype('float32')).eval()
                    Y_pred_class = np.argmax(Y_pred, axis =1)
                    dice_score = get_dice_score(Y_valid,Y_pred_class)
                    Y_pred = Y_pred_class.reshape(valid_batch_size,1,xs,ys)
                    Y_valid = Y_valid.reshape(valid_batch_size,1,xs,ys)
                    save_image_path = os.path.join(save_path, str(epoch), '{}.png'.format(count_valid_samples))
                    if not os.path.exists(os.path.join(save_path, str(epoch))):
                        os.makedirs(os.path.join(save_path, str(epoch)))
                    vis_detections(X_valid[5][0],Y_valid[5][0],Y_pred[5][0],save_image_path)
                    mean_val_loss+= val_loss
                    mean_dice_score+= dice_score
                    count_valid_samples += 1

                mean_val_loss= mean_val_loss/(nb_val_samples/valid_batch_size)
                mean_dice_score = mean_dice_score/(nb_val_samples/valid_batch_size)
                print (mean_val_loss, mean_dice_score)

                log_valid.post("val_loss",mean_val_loss, epoch )
                log_valid.post("mean_val_dice_score",mean_dice_score, epoch )

                print ("mean_val_loss: {} , mean_dice_score: {}".format(mean_val_loss , mean_dice_score))
                validation_end = time.time()
                validation_time = validation_end - validation_start
                print ('validation time : %ds' % validation_time)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_epoch = epoch
                    nb_perf_not_improved = 0
                    dpath = os.path.join(path,"Unet_vald_set_{}_val_loss_{}_epoch_{}".format(validation_set, best_val_loss,best_epoch))
                    save_params(dpath)
                else :
                    nb_perf_not_improved+=1
                    if nb_perf_not_improved > patience:
                        print ("Exiting training as performance  not improving for {} loops".format(patience))
                        not_done_looping = False



    return best_val_loss, best_epoch




cfg = gen_config(layers,inputch,filters,outputs,act,ltype,optimiser)
train_fn, test_predict_fn, predict_fn, save_params, load_params, output_shape, set_lr = get_functions(cfg)
print ("Starting Training")
with open('logs/log_training_2DUnet_lr_{}_optimiser_{}.log'.format(lr,optimiser), 'w') as f:
    sys.stdout = f
    print ("------- Checking for lr = {} ---------- ".format(lr))
    log_train = Log('http://localhost:4152', '2DUnet_train')
    log_valid = Log('http://localhost:4152', '2DUnet_valid')
    log_tuple = (log_train,log_valid)

    best_val_loss, best_epoch = main_training(layers = layers,lr = lr, nb_epoch = nb_epoch, nb_samples_per_epoch = nb_samples_per_epoch ,
                                            nb_val_samples = nb_val_samples, patience = patience,path = path, log_tuple = log_tuple)
    print ("---------------------------------------------------")
sys.stdout = sys.__stdout__

# model_path = os.path.join(path,"Unet_vald_set_{}_val_loss_{}_epoch_{}.npz".format(validation_set,best_val_loss,best_epoch))
# best_model = load_params(model_path)
# folder = 'subset'+str(validation_set)

    # for X,seriesuid in test_data_generator(validation_set =validation_set):
    #     for i in range(X.shape[0]):
    #         X_test = X[i]
    #         X_test = X_test[np.newaxis, np.newaxis,...]
    #         xs = X_test.shape[2]
    #         ys = X_test.shape[3]
    #         Y_pred = test_predict_fn(X_test.astype('float32'))
    #         Y_pred = (Y_pred [:,1]> threshold ).astype('int')
    #         Y_pred = Y_pred.reshape((1,xs,ys))
    #         if i==0 :
    #             Y_pred_final = Y_pred
    #         else:
    #             Y_pred_final = np.append(Y_pred_final, Y_pred, axis = 0)
    #
    #     np.save(os.path.join(predicted_path, folder, 'Y_segmentation_{}.npy'.format(seriesuid)), Y_pred_final)
