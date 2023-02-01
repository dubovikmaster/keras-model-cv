import time
from typing import (
    Callable,
    Optional,
    Union,
    List,
    Iterable,
    Any

)
from pathlib import Path
from os import PathLike
import shutil
import random

import yaml
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import BaseCrossValidator
import pandas as pd

NAMES = ['mysterious', 'incredible', 'beautiful', 'graceful']
STATUS = {'RUNNING': 0, 'OK': 1}


class KerasCV:

    def __init__(
            self,
            model_builder: Callable,
            cv: BaseCrossValidator,
            params: dict,
            preprocessor: Optional[Union[Callable, List[Callable]]] = None,
            save_history: bool = False,
            directory: Optional[Union[str, PathLike]] = None,
            name: Optional[str] = None,
            custom_evaluate: Optional[Callable] = None,
            overwrite: bool = False,

    ):
        self.model_builder = model_builder
        self.cv = cv
        self.params = params
        self.preprocessor = preprocessor
        self.save_history = save_history
        self.directory = directory
        self.name = name
        self.history = None
        self.overwrite = overwrite
        self.cv_results = None
        self.splits_info = None
        self._multiple_input = None
        self._custom_eval = custom_evaluate

        if self.save_history:
            if self.name is None:
                self.name = random.choice(NAMES) + '_project'
            if self.directory is None:
                raise ValueError('directory must be specified if save_history is "True".')
            self.project_path = Path(self.directory).joinpath(self.name)
            self.project_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def preprocess_data(x: Iterable, preprocessor: Callable, fit_transform: bool):
        if preprocessor is None:
            return x
        if fit_transform:
            if isinstance(preprocessor, keras.layers.Layer):
                preprocessor.adapt(x)
                return preprocessor(x)
            preprocessor.fit(x)
            return preprocessor.transform(x)
        if isinstance(preprocessor, keras.layers.Layer):
            return preprocessor(x)
        return preprocessor.transform(x)

    @staticmethod
    def _save_yaml(obj: dict, filename: PathLike):
        yaml.dump(obj, stream=open(filename, 'w'), default_flow_style=False)

    @staticmethod
    def _load_yaml(filename: PathLike):
        yml = yaml.load(stream=open(filename, 'r'), Loader=yaml.FullLoader)
        return yml

    def _get_split_path(self, split_number: int):
        split_path = self.project_path.joinpath(f'split_{split_number}')
        if not split_path.exists():
            split_path.mkdir(exist_ok=True, parents=True)
        return split_path

    def _del_split_folders(self):
        if self.overwrite:
            for path in self.project_path.iterdir():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)

    def _check_split_status(self, split_number: int):
        status = 0
        split_path = self._get_split_path(split_number)
        info_yml = split_path.joinpath('split_info.yml')
        if info_yml.exists():
            obj = self._load_yaml(info_yml)
            try:
                status = STATUS[obj['status']]
            except KeyError:
                ...
        return status

    def _load_history(self, split_number: int):
        split_path = self._get_split_path(split_number)
        history_yml = split_path.joinpath('history.yml')
        if not history_yml.exists():
            ValueError(f"Can't find history for split {split_number}")
        self.history.append(self._load_yaml(history_yml))
        metrics_yml = split_path.joinpath('validation_metric.yml')
        if not metrics_yml.exists():
            ValueError(f"Can't find validate metrics  for split {split_number}")
        self.cv_results.append(self._load_yaml(metrics_yml))

    def _prepare_data(self, x, idx: List[int], fit_transform: bool):
        if self._multiple_input:
            x_new = [i[idx] for i in x]
            if self.preprocessor is not None:
                x_new = [self.preprocess_data(x_new[i], self.preprocessor[i], fit_transform) for i in
                         range(len(self.preprocessor))]
        else:
            x_new = x[idx]
            if self.preprocessor is not None:
                x_new = self.preprocess_data(x_new, self.preprocessor, fit_transform)
        return x_new

    def get_model(self):
        return self.model_builder(**self.params)

    def fit(self, x, y, **kwargs):
        self.history = []
        self.cv_results = []
        self.splits_info = []
        if self.save_history:
            self._del_split_folders()
        if isinstance(x, list):
            self._multiple_input = True
            n_sample = len(x[0])
        else:
            self._multiple_input = False
            n_sample = len(x)
        for split, (train_index, test_index) in enumerate(
                self.cv.split(range(n_sample), y)):
            if self.save_history and not self.overwrite:
                split_status = self._check_split_status(split)
                if split_status:
                    self._load_history(split)
                    continue
            start = time.perf_counter()
            split_info = {'start_time': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                          'cv_iterator': repr(self.cv),
                          'status': 'RUNNING'
                          }
            if self.save_history:
                split_path = self._get_split_path(split_number=split)
                self._save_yaml(split_info, split_path.joinpath('split_info.yml'))
            keras.backend.clear_session()
            if kwargs.get('verbose', 0):
                tf.get_logger().info(
                    "\n" + "-" * 31 + "\n"
                                      f"Cross-Validation {split + 1}/{self.cv.get_n_splits()}"
                    + "\n"
                    + "-" * 31
                    + "\n"
                )
            # prepare train data
            x_train = self._prepare_data(x, train_index, fit_transform=True)
            y_train = y[train_index]

            # prepare test data
            x_test = self._prepare_data(x, test_index, fit_transform=False)
            y_test = y[test_index]
            # get model
            model = self.get_model()

            # train model
            history = model.fit(x_train, y_train, **kwargs)
            self.history.append(history.history)

            tf.get_logger().info(
                "\n" + "-" * 31 + "\n"
                                  "Evaluate validation performance"
                + "\n"
                + "-" * 31
                + "\n"
            )
            if self._custom_eval:
                y_pred = model.predict(
                    x_test,
                    batch_size=kwargs.get('batch_size', 32),
                    verbose=kwargs.get('verbose', 0),
                )
                val_res = self._custom_eval(y_test, y_pred)
                if isinstance(val_res, (float, int)):
                    val_res = {'custom_metric': val_res}
                self.cv_results.append(val_res)
            else:
                val_res = model.evaluate(
                    x_test,
                    y_test,
                    batch_size=kwargs.get('batch_size', 32),
                    return_dict=True,
                    verbose=kwargs.get('verbose', 0),
                )
                self.cv_results.append(val_res)
            val_res['split'] = split
            end = round(time.perf_counter() - start, 1)
            tf.get_logger().info(
                "\n" + "-" * 31 + "\n"
                                  f"Split {split + 1}/{self.cv.get_n_splits()} time took: {end} s."
                + "\n"
            )

            split_info['status'] = 'OK'
            split_info['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            split_info['elapsed_time'] = end
            self.splits_info.append(split_info)
            if self.save_history:
                split_path = self._get_split_path(split_number=split)
                self._save_yaml(history.history, split_path.joinpath('history.yml'))
                self._save_yaml(val_res, split_path.joinpath('validation_metric.yml'))
                # split_info
                self._save_yaml(split_info, split_path.joinpath('split_info.yml'))

    def get_cv_history(self):
        history_list = list()
        for i, h in enumerate(self.history):
            df = pd.DataFrame(h)
            df['split'] = i
            df['epochs'] = range(1, len(df) + 1)
            history_list.append(df)
        if history_list:
            result = pd.concat(history_list, ignore_index=True)
        else:
            raise ValueError("You must run fit(x,y, **kwargs) before")
        return result

    def get_split_scores(self):
        df = pd.DataFrame(self.cv_results)
        if df.empty:
            raise ValueError("You must run fit(x,y, **kwargs) before")
        return df

    def get_cv_score(self, agg_func: Optional[Any] = None):
        if agg_func is None:
            agg_func = ['mean', 'std']
        df = self.get_split_scores()
        return df.drop(['split'], axis=1).agg(agg_func)
