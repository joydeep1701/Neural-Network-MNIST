import numpy as np

from dataloader import *
from processimage import *
from MLP import *

f = gzip.open('../mnist.pkl.gz')
train_set, valid_set, test_set = pkl.load(f,encoding='iso-8859-1')
f.close()

minibatch_size = 100
print("Creating minibatch of size {}".format(minibatch_size))
print("Training:")
train_data, train_labels = create_minibatches(train_set[0], train_set[1],
                                             minibatch_size,
                                             create_bit_vector=True)
print("Testing:")
valid_data, valid_labels = create_minibatches(valid_set[0], valid_set[1],
                                             minibatch_size,
                                             create_bit_vector=True)

print("Minibatch of size {} created".format(minibatch_size))
print("Length of training data:",len(train_data))


mlp = MLP(layer_config=[784, 100, 100, 10], minibatch_size=minibatch_size)

#mlp.evaluate(train_data, train_labels, valid_data, valid_labels, eval_train=True)

def predict_latest():
    return mlp.predict(latest_image_data())
