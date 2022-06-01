import tflite_runtime.interpreter as tf
from PIL import Image
import numpy as np
import time

ms = "m1 m2 m3 m4 mt2 mt3 mt4 mt6 mt7".split()

for m in ms:
    ipath = "./image (1).JPG"
    i = tf.Interpreter(model_path=f"./lite/{m}.tflite")
    i.allocate_tensors()

    idd = i.get_input_details()
    io = i.get_output_details()

    
    img = Image.open(ipath)

    img = np.array(img, dtype=np.float32)
    img = img[None, :, :]
    st = time.time()
    i.set_tensor(idd[0]["index"], img)
    i.invoke()
    od = i.get_tensor(io[0]["index"])
    print(f"{m} in {time.time()-st}")
    del(i)

