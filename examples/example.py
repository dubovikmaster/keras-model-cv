from keras_model_cv import KerasCV
from sklearn.model_selection import KFold
import tensorflow as tf

tf.get_logger().setLevel("INFO")

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


def build_model(hidden_units, dropout):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(hidden_units, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(10),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    return model


PARAMS = {'hidden_units': 16, 'dropout': .3}

if __name__ == '__main__':
    cv = KerasCV(
        build_model,
        KFold(n_splits=3, random_state=1234, shuffle=True),
        PARAMS,
        preprocessor=tf.keras.layers.Normalization(),
        supervised_preprocessor=False,
        save_history=True,
        directory='my_awesome_project',
        name='my_cv',
        overwrite=True,
        disable_pr_bar=True
    )
    cv.fit(x_train,
           y_train,
           verbose=1,
           epochs=7,
           callbacks=tf.keras.callbacks.ModelCheckpoint('model_chekpoint',
                                                        save_best_only=True,
                                                        monitor='val_accuracy',
                                                        mode='max',
                                                        verbose=0
                                                        ),
           validation_split=.2)

    print(cv.get_cv_score(agg_func={'loss': min, 'accuracy': max}))
    print(cv.get_split_scores())
