import tensorflow as tf
import numpy as np
from PIL import Image
import itertools
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from datetime import datetime
import matplotlib.pyplot as plt


class ModelBuilderPipeline:
    def __init__(
        self, name, model, input_shape, batch_size=16, epoch=30, optimizer="adam"
    ) -> None:
        self.name = name
        self.model = model
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epoch = epoch
        self.optimizer = optimizer

        self._model_compile()

    def _model_compile(self):
        print("👉 Building and compiling...")
        self.model.build(input_shape=self.input_shape)
        self.model.compile(
            optimizer=self.optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=["accuracy"],
        )
        print(f"Total Parameter {self.model.count_params()}")

    def _fit(self, tds, vds):
        print("\n\n👉 Fitting the model...")
        log_dir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1
        )
        self.history = self.model.fit(
            tds,
            validation_data=vds,
            epochs=self.epoch,
            batch_size=self.batch_size,
            callbacks=[tensorboard_callback],
        )

    def _evaluate(self, ttds):
        print("\n\n👉 Evaluating the model...")
        self.score = self.model.evaluate(ttds)

    def _plot(self):
        print("\n\n👉 Plotting the model...")
        h = self.history
        acc = h.history["accuracy"]
        val_acc = h.history["val_accuracy"]

        loss = h.history["loss"]
        val_loss = h.history["val_loss"]

        epochs_range = range(self.epoch)

        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label="Training Accuracy")
        plt.plot(epochs_range, val_acc, label="Validation Accuracy")
        plt.legend(loc="lower right")
        plt.title("Training and Validation Accuracy")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label="Training Loss")
        plt.plot(epochs_range, val_loss, label="Validation Loss")
        plt.legend(loc="upper right")
        plt.title("Training and Validation Loss")
        plt.show()

    def _save(self):
        print("\n\n👉 Saving the model...")
        self.model.save(f"./models/thesis_{self.name}.h5")

    def _performance(self):
        perf = dict(zip(self.model.metrics_names, self.score))
        perf["name"] = self.name
        return perf

    def run(self, tds, vds, ttds):
        self._fit(tds=tds, vds=vds)
        self._evaluate(ttds=ttds)
        print(self._performance())
        self._save()
        self._plot()


class ModelTestPipeline:
    def __init__(self, labels, name, test_ds=None, auto=True) -> None:
        self.ds = test_ds
        self.labels = labels
        self.name = name
        self.auto = auto
        self.true_label = None
        self.predict_label = None
        self._load()

    def _load(self):
        if self.auto:
            self.model = tf.keras.models.load_model(f"./models/thesis_{self.name}.h5")
        else:
            self.model = tf.keras.models.load_model(f"./models/{self.name}.h5")

    def single_image_test(self):
        for batch_image, batch_label in self.ds.take(1):
            first_image = batch_image[0].numpy().astype("uint8")
            first_label = self.labels[batch_label[0]]

            plt.imshow(first_image)

            batch_prediction = self.model.predict(batch_image)
            print("First Image of batch to predict :")
            print("Actual label : ", first_label)
            print("Predicted label : ", self.labels[np.argmax(batch_prediction[0])])
            plt.axis("off")

    def random_imageset_test(self):
        plt.figure(figsize=(16, 16))
        for batch_image, batch_label in self.ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                image = batch_image[i].numpy().astype("uint8")
                label = self.labels[batch_label[i]]

                plt.imshow(image)

                batch_prediction = self.model.predict(batch_image)
                predicted_class = self.labels[np.argmax(batch_prediction[i])]
                confidence = round(np.max(batch_prediction[i]) * 100, 2)

                plt.title(
                    f"Actual : {label},\n Prediction : {predicted_class},\n Confidence : {confidence}%"
                )

                plt.axis("off")

    @staticmethod
    def conf_plotter(
        cm, classes, normalize=True, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            cm = np.around(cm, decimals=2)
            cm[np.isnan(cm)] = 0.0
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
        # plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    def _preprocess(self):
        tl = []
        pl = []
        for image, target in self.ds:
            tl += [t.numpy().astype("uint8") for t in target]
            pl += [np.argmax(p) for p in self.model.predict(image)]
        self.true_label = tl
        self.predict_label = pl

    def plot_cf(self):
        self._preprocess()
        target_names = self.labels
        # Confusion Matrix
        cm = confusion_matrix(self.true_label, self.predict_label)
        ModelTestPipeline.conf_plotter(
            cm, target_names, normalize=False, title="Confusion Matrix"
        )

    def print_cr(self):
        print("Classification Report")
        print(
            classification_report(
                self.true_label, self.predict_label, target_names=self.labels
            )
        )

    def predict(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)
        return self.labels[np.argmax(self.model.predict(img[None, :, :])[0])]

    def raw_predict(self, image_path):
        img = Image.open(image_path)
        img = np.array(img)
        return self.model.predict(img[None, :, :])[0]
