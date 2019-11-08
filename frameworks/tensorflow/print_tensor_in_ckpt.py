# common usage: python print_tensor_in_ckpt.py > tensors.txt
# ref: https://stackoverflow.com/questions/38218174/how-do-i-find-the-variable-names-and-values-that-are-saved-in-a-checkpoint
import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file


latest_ckp = tf.train.latest_checkpoint('./')
print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')
