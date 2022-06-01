import timeit

models = "m1 m2 m3 m4 mt2 mt3 mt4 mt6 mt7".split()
summary = dict()
for model in models:
    setup = f"""
import tensorflow as tf
from PIL import Image
import numpy as np

ipath = "./data/Apple___Apple_scab/image (1).JPG"
m = "m1"
i = tf.lite.Interpreter(model_path="./models/lite/{model}.tflite")
i.allocate_tensors()

id = i.get_input_details()
io = i.get_output_details()

img = Image.open(ipath)

img = np.array(img, dtype=np.float32)
img = img[None, :, :]
"""
    test = """
i.set_tensor(id[0]["index"], img)
i.invoke()
i.get_tensor(io[0]["index"])
"""

    summary[model] = timeit.repeat(setup=setup, stmt=test, repeat=4, number=10)

print(summary)
