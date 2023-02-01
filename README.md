# Keras Cross-validation
`keras-model-cv` allows you to cross-validate  `keras` model. 
## Installation
```python
pip install keras-cv
```
or
```python
pip install git+https://github.com/dubovikmaster/keras-cv.git
```

## Quickstart

```python
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
        save_history=True,
        directory='my_awesome_project',
        name='my_cv',
    )
    cv.fit(x_train, y_train, verbose=0, epochs=3)
    print(cv.get_cv_score())
```
```python
Out: 
                loss  accuracy
        mean  0.283194  0.919783
        std   0.004215  0.002887 
```
You can add another aggregate function (for more info see: [pandas.DataFrame.agg](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.agg.html)):
```python
print(cv.get_cv_score(agg_func={'loss': min, 'accuracy': max}))
```
```python
Out:
        loss        0.27959
        accuracy    0.92010
```
Also, you can get all train history for each splits as `pandas` dataframe:
```python
cv.get_cv_history()
```
```python
Out:
             loss    accuracy    split  epochs
        0  0.957261  0.679375      0       1
        1  0.595646  0.809850      0       2
        2  0.541124  0.824850      0       3
        3  0.835493  0.722475      1       1
        4  0.574581  0.810925      1       2
        5  0.526098  0.829200      1       3
        6  0.813172  0.736200      2       1
        7  0.556871  0.816875      2       2
        8  0.512916  0.829550      2       3
```


What about metrics per splits?
```python
cv.get_split_scores()
```
```python
Out:
            accuracy   loss     split
        0    0.9201  0.282442      0
        1    0.9198  0.290500      1
        2    0.9173  0.279590      2
```
If `save_history=True` train history,  validation metrics and info about split will be saved to the specified directory.
In our example:
```python
my_awesome_project/
 |--my_cv/
      |--split_0/
           |--history.yml
           |--validation_metric.yml
           |--split_info.yml
           
      |--split_1/
      |--split_2/
```
## Licence
 MIT license
