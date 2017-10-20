import numpy as np
import _pickle as pkl

from NN import *

class MLP:
    def __init__(self, layer_config, minibatch_size=100):
        self.layers = []
        self.num_layers = len(layer_config)
        self.minibatch_size = minibatch_size

        for i in range(self.num_layers-1):
            if i == 0:
                print( "Initializing input layer with size {0}.".format(
                    layer_config[i]
                ))
                # Here, we add an additional unit at the input for the bias
                # weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         minibatch_size,
                                         is_input=True))
            else:
                print( "Initializing hidden layer with size {0}.".format(
                    layer_config[i]
                ))
                # Here we add an additional unit in the hidden layers for the
                # bias weight.
                self.layers.append(Layer([layer_config[i]+1, layer_config[i+1]],
                                         minibatch_size,
                                         activation=f_sigmoid))

        print( "Initializing output layer with size {0}.".format(
            layer_config[-1]
        ))
        self.layers.append(Layer([layer_config[-1], None],
                                 minibatch_size,
                                 is_output=True,
                                 activation=f_softmax))
        print( "Done!")

    def forward_propagate(self, data):
        # We need to be sure to add bias values to the input
        self.layers[0].Z = np.append(data, np.ones((data.shape[0], 1)), axis=1)

        for i in range(self.num_layers-1):
            self.layers[i+1].S = self.layers[i].forward_propagate()
        return self.layers[-1].forward_propagate()

    def backpropagate(self, yhat, labels):
        self.layers[-1].D = (yhat - labels).T
        for i in range(self.num_layers-2, 0, -1):
            # We do not calculate deltas for the bias values
            W_nobias = self.layers[i].W[0:-1, :]

            self.layers[i].D = W_nobias.dot(self.layers[i+1].D) * self.layers[i].Fp

    def update_weights(self, eta):
        for i in range(0, self.num_layers-1):
            W_grad = -eta*(self.layers[i+1].D.dot(self.layers[i].Z)).T
            self.layers[i].W += W_grad

    def evaluate(self, train_data, train_labels, test_data, test_labels,
                 num_epochs=3, eta=0.05, eval_train=False, eval_test=True):

        N_train = len(train_labels)*len(train_labels[0])
        N_test = len(test_labels)*len(test_labels[0])

        print( "Training for {0} epochs...".format(num_epochs))
        for t in range(0, num_epochs):
            out_str = "[{0:4d}] ".format(t)

            for b_data, b_labels in zip(train_data, train_labels):
                output = self.forward_propagate(b_data)
                self.backpropagate(output, b_labels)
                self.update_weights(eta=eta)

            if eval_train:
                errs = 0
                for b_data, b_labels in zip(train_data, train_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Training error: {1:.5f}".format(out_str,
                                                           float(errs)/N_train)

            if eval_test:
                errs = 0
                for b_data, b_labels in zip(test_data, test_labels):
                    output = self.forward_propagate(b_data)
                    yhat = np.argmax(output, axis=1)
                    errs += np.sum(1-b_labels[np.arange(len(b_labels)), yhat])

                out_str = "{0} Test error: {1:.5f}".format(out_str,
                                                       float(errs)/N_test)

            print(out_str)

    def predict(self, data):
        output = self.forward_propagate(np.array([data]))
        yhat = np.argmax(output, axis=1)
        c = 0     
        
        opt = {}
        opt['Predicted:'] = str(yhat[0])
        opt['Probabilities:'] = {}
        for i in output[0]:
            opt['Probabilities:'][str(c)] = "%.2f"%(i*100)
            c += 1
        #print(opt)
        
        return opt
        #return yhat

    def save(self, target='weights.pkl'):
        data = [w.W for w in self.layers]
        op = open(target,'wb')
        pkl.dump(data, op)
        op.close()
        print("Weight matrix saved in {}".format(target))

    def load(self, target='weights.pkl'):
        with open('weights.pkl', 'rb') as inp:
            data = pkl.load(inp)
            inp.close()

            for i,weight in enumerate(data):
                self.layers[i].W = weight
