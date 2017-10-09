import _pickle as pkl
import gzip
import numpy as np

def label_to_bit_vector(labels, nbits):
    """Returns label in bit vector format"""
    bv = np.zeros((labels.shape[0], nbits))
    for i in range(labels.shape[0]):
        bv[i, labels[i]] = 1.0

    return bv


def create_minibatches(data, labels, batch_size, create_bit_vector=False):
    N = data.shape[0]

    print("Total number of examples: {}".format(N))

    if N % batch_size != 0:
        print("create_minibatches(): batch size {} does not"              "evenly divide number of examples {}".format(batch_size, N))

    chunked_data = []
    chunked_labels = []
    idx = 0

    while idx+batch_size <= N:
        chunked_data.append(data[idx:idx+batch_size, :])
        if not create_bit_vector:
            chunked_labels.append(labels[idx:idx+batch_size])
        else:
            bv = label_to_bit_vector(labels[idx:idx+batch_size], 10)
            chunked_labels.append(bv)

        idx += batch_size

    return chunked_data,chunked_labels
