import timeit

models = "m1 m2 m3 m4 mt2 mt3 mt4 mt6 mt7".split()
summary = dict()
for model in models:
    setup = f"""
import tensorflow as tf

import numpy as np
from PIL import Image

model = tf.keras.models.load_model("./models/thesis_{model}.h5")

img = Image.open("./data/Apple___Apple_scab/image (1).JPG")
img = np.array(img)
"""
    test = "model.predict(img[None, :, :])"

    summary[model] = timeit.repeat(setup=setup, stmt=test, repeat=4, number=10)

print(summary)
