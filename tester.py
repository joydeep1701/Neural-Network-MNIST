import numpy as np

from dataloader import *
from processimage import *
from MLP import *

mlp = MLP(layer_config=[784, 100, 100, 10])

mlp.load()

print(mlp.predict(latest_image_data()))
