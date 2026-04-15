import os
os.environ["KERAS_BACKEND"] = "jax"

import re
import string
import keras
import numpy as np
import tensorflow as tf

strip_chars = string.punctuation
def my_standardize(input_string):
    print(f"Type of input_string: {type(input_string)}")
    print(f"Value: {input_string}")
    input_string = input_string.lower()
    return re.sub(f"[{re.escape(strip_chars)}]", "", input_string)

layer = keras.layers.TextVectorization(standardize=my_standardize)
try:
    print("Adapting with scalar...")
    layer.adapt("Hello, world.")
    print("Adapt successful")
except Exception as e:
    print(f"Caught exception: {e}")

try:
    print("\nAdapting with list of one string...")
    layer.adapt(["Hello, world."])
    print("Adapt successful")
except Exception as e:
    print(f"Caught exception: {e}")
