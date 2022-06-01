import tensorflow as tf

ms = "m1 m2 m3 m4 mt2 mt3 mt4 mt6 mt7".split()

for m in ms:
    model = tf.keras.models.load_model(f"./models/thesis_{m}.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    open(f"./models/lite/{m}.tflite", "wb").write(tflite_model)
    print(f"Model {m} is done")
