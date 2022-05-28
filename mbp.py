import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt


class ModelBuilderPipeline:
    def __init__(
        self, name, model, input_shape, batch_size=16, epoch=10, optimizer="adam"
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
