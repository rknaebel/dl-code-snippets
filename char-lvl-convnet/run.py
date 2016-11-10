'''
Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python main.py
'''

from __future__ import print_function
from __future__ import division
import json
import datetime
import numpy as np
import data_helpers
np.random.seed(0123)  # for reproducibility

from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D



def getModel(filter_kernels, dense_outputs, maxlen, vocab_size, nb_filter,
          cat_output):
    #Define what the input shape looks like
    inputs = Input(shape=(maxlen, vocab_size), name='input', dtype='float32')

    #All the convolutional layers...
    conv = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[0],
                         border_mode='valid', activation='relu',
                         input_shape=(maxlen, vocab_size))(inputs)
    conv = MaxPooling1D(pool_length=3)(conv)

    conv1 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[1],
                          border_mode='valid', activation='relu')(conv)
    conv1 = MaxPooling1D(pool_length=3)(conv1)

    conv2 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[2],
                          border_mode='valid', activation='relu')(conv1)

    conv3 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[3],
                          border_mode='valid', activation='relu')(conv2)

    conv4 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[4],
                          border_mode='valid', activation='relu')(conv3)

    conv5 = Convolution1D(nb_filter=nb_filter, filter_length=filter_kernels[5],
                          border_mode='valid', activation='relu')(conv4)
    conv5 = MaxPooling1D(pool_length=3)(conv5)
    conv5 = Flatten()(conv5)

    #Two dense layers with dropout of .5
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(conv5))
    z = Dropout(0.5)(Dense(dense_outputs, activation='relu')(z))

    #Output dense layer with softmax activation
    pred = Dense(cat_output, activation='softmax', name='output')(z)
    model = Model(input=inputs, output=pred)
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model

# set parameters:

subset = None

#Whether to save model parameters
save = False
model_name_path = 'params/crepe_model.json'
model_weights_path = 'params/crepe_model_weights.h5'

#Maximum length. Longer gets chopped. Shorter gets padded.
maxlen = 1014

#Model params
#Filters for conv layers
nb_filter = 256
#Number of units in the dense layer
dense_outputs = 1024
#Conv layer kernel size
filter_kernels = [7, 7, 3, 3, 3, 3]
#Number of units in the final output layer. Number of classes.
cat_output = 4

#Compile/fit params
batch_size = 80
nb_epoch = 10

print('Loading data...')
#Expect x to be a list of sentences. Y to be a one-hot encoding of the
#categories.
(xt, yt), (x_test, y_test) = data_helpers.load_ag_data()

print('Creating vocab...')
vocab, reverse_vocab, vocab_size, check = data_helpers.create_vocab_set()

test_data = data_helpers.encode_data(x_test, maxlen, vocab, vocab_size, check)

print('Build model...')

model = getModel(filter_kernels, dense_outputs, maxlen, vocab_size,
                       nb_filter, cat_output)

print('Fit model...')
initial = datetime.datetime.now()
for e in xrange(nb_epoch):
    xi, yi = data_helpers.shuffle_matrix(xt, yt)
    xi_test, yi_test = data_helpers.shuffle_matrix(x_test, y_test)
    if subset:
        batches = data_helpers.mini_batch_generator(xi[:subset], yi[:subset],
                                                    vocab, vocab_size, check,
                                                    maxlen,
                                                    batch_size=batch_size)
    else:
        batches = data_helpers.mini_batch_generator(xi, yi, vocab, vocab_size,
                                                    check, maxlen,
                                                    batch_size=batch_size)

    test_batches = data_helpers.mini_batch_generator(xi_test, yi_test, vocab,
                                                     vocab_size, check, maxlen,
                                                     batch_size=batch_size)

    accuracy = 0.0
    loss = 0.0
    step = 1
    start = datetime.datetime.now()
    print('Epoch: {}'.format(e))
    for x_train, y_train in batches:
        f = model.train_on_batch(x_train, y_train)
        loss += f[0]
        loss_avg = loss / step
        accuracy += f[1]
        accuracy_avg = accuracy / step
        if step % 100 == 0:
            print('  Step: {}'.format(step))
            print('\tLoss: {}. Accuracy: {}'.format(loss_avg, accuracy_avg))
        step += 1

    test_accuracy = 0.0
    test_loss = 0.0
    test_step = 1
    for x_test_batch, y_test_batch in test_batches:
        f_ev = model.test_on_batch(x_test_batch, y_test_batch)
        test_loss += f_ev[0]
        test_loss_avg = loss / step
        test_accuracy += f_ev[1]
        test_accuracy_avg = accuracy / step
        test_step += 1
    stop = datetime.datetime.now()
    e_elap = stop - start
    t_elap = stop - initial
    print('Epoch {}. Loss: {}. Accuracy: {}\nEpoch time: {}. Total time: {}\n'.format(e, test_loss, test_accuracy, e_elap, t_elap))

if save:
    print('Saving model params...')
    json_string = model.to_json()
    with open(model_name_path, 'w') as f:
        json.dump(json_string, f)

    model.save_weights(model_weights_path)
