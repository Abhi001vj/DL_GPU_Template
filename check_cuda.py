import torch

print("Torch CUDA: ", torch.cuda.is_available())


print("Torch current GPU: ",torch.cuda.current_device())


print("Torch GPU: ",torch.cuda.device(0))


print("Torch GPU count: ",torch.cuda.device_count())


print("Torch GPU name: ",torch.cuda.get_device_name(0))

import tensorflow as tf

if tf.test.is_built_with_cuda():
    print("Tensorflow GPU Device: ", tf.config.list_physical_devices('GPU') )