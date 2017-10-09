import numpy as np

from dataloader import *
from processimage import *
from MLP import *

mlp = MLP(layer_config=[784, 100, 100, 10])

inp = open('weights.pkl', 'rb')
data = pkl.load(inp)
inp.close()

for i,weight in enumerate(data):
    mlp.layers[i].W = weight

print(mlp.predict(latest_image_data()))
