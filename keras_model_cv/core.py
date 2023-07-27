import contextlib
import random
import shutil
import time
from os import PathLike
from pathlib import Path
from typing import (
    Callable,
    Optional,
    Union,
    List,
    Tuple,
    Iterable,
    Any

)
import copy

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import yaml
from sklearn.model_selection import BaseCrossValidator
from tensorflow import keras

from tqdm.auto import tqdm

NAMES = ['mysterious', 'incredible', 'beautiful', 'graceful']
STATUS = {'RUNNING': 0, 'OK': 1}

logger = tf.get_logger()
logger.propagate = False


class KerasCV:

    def __init__(
            self,
            model_builder: Callable,
            cv: BaseCrossValidator,
            params: dict,
            preprocessor: Optional[Union[Callable, List[Callable]]] = None,
            supervised_preprocessor: Optional[Union[bool, List[bool]]] = None,
            save_history: bool = False,
            directory: Optional[Union[str, PathLike]] = None,
            name: Optional[str] = None,
            custom_evaluate: Optional[Callable] = None,
            needed_test_indexes_for_custom_metric: bool = False,
            overwrite: bool = False,
            distribution_strategy: Optional[tf.distribute.Strategy] = None,
            disable_pr_bar: bool = True,
            random_seed: int = 0


    ):
        self.model_builder = model_builder
        self.cv = cv
        self.params = params
        self.preprocessor = preprocessor
        self.supervised_preprocessor = supervised_preprocessor
        self.save_history = save_history
        self.directory = directory
        self.name = name
        self.history = None
        self.overwrite = overwrite
        self.needed_test_indexes = needed_test_indexes_for_custom_metric
        self.cv_results = None
        self.distribution_strategy = distribution_strategy
        self.splits_info = None
        self.disable_pr_bar = disable_pr_bar
        self.random_seed = random_seed
        self._multiple_input = None
        self._custom_eval = custom_evaluate
        self._model_checkpoint_deepcopy = None
        self._model_checkpoint_new_filepath = None

        if self.save_history:
            if self.name is None:
                self.name = random.choice(NAMES) + '_project'
            if self.directory is None:
                raise ValueError('directory must be specified if save_history is "True".')
            self.project_path = Path(self.directory).joinpath(self.name)
            self.project_path.mkdir(exist_ok=True, parents=True)

    @staticmethod
    def preprocess_data(x: Iterable, y: Iterable, preprocessor: Callable, supervised: bool,  fit_transform: bool):
        if preprocessor is None:
            return x
        if fit_transform:
            if isinstance(preprocessor, keras.layers.Layer):
                preprocessor.adapt(x)
                return preprocessor(x)
            if supervised:
                new_x = preprocessor.fit_transform(x, y)
            else:
                new_x = preprocessor.fit_transform(x)
            return new_x
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

    def _load_best_model(self, model):
        if self._model_checkpoint_deepcopy.save_best_only:
            model = tf.keras.models.load_model(self._model_checkpoint_new_filepath)
        else:
            model.load_weights(self._model_checkpoint_new_filepath)
        return model

    def _find_model_checkpoint(self, kwargs):
        if 'callbacks' in kwargs:
            if isinstance(kwargs['callbacks'], list):
                callbacks = list(kwargs['callbacks'])
                for callback in callbacks:
                    if isinstance(callback, tf.keras.callbacks.ModelCheckpoint):
                        self._model_checkpoint_deepcopy = copy.deepcopy(callback)
                        kwargs['callbacks'].remove(callback)
                        break
            else:
                if isinstance(kwargs['callbacks'], tf.keras.callbacks.ModelCheckpoint):
                    self._model_checkpoint_deepcopy = copy.deepcopy(kwargs['callbacks'])
                    kwargs['callbacks'] = []

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

    def _prepare_data(self, x: Iterable, y: Iterable, idx: List[int], fit_transform: bool):
        y_new = y[idx]
        if self._multiple_input:
            x_new = [i[idx] for i in x]
            if self.preprocessor is not None:
                x_new = [self.preprocess_data(x_new[i], y_new, self.preprocessor[i], self.supervised_preprocessor[i],
                                              fit_transform) for i in
                         range(len(self.preprocessor))]
        else:
            x_new = x[idx]
            if self.preprocessor is not None:
                x_new = self.preprocess_data(x_new, y_new, self.preprocessor, self.supervised_preprocessor,
                                             fit_transform)
        return x_new

    def get_model(self):
        with maybe_distribute(self.distribution_strategy):
            tf.keras.utils.set_random_seed(self.random_seed)
            model = self.model_builder(**self.params)
            return model

    def fit(self, x, y, **kwargs):
        self._model_checkpoint_deepcopy = None
        self._model_checkpoint_new_filepath = None
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
        self._find_model_checkpoint(kwargs)
        for split, (train_index, test_index) in enumerate(tqdm(
                self.cv.split(range(n_sample), y), total=self.cv.n_splits, disable=self.disable_pr_bar)):
            if self._model_checkpoint_deepcopy:
                callback = copy.deepcopy(self._model_checkpoint_deepcopy)
                callback.filepath = self._model_checkpoint_deepcopy.filepath + f'/split_{split}'
                self._model_checkpoint_new_filepath = callback.filepath
                kwargs['callbacks'].append(callback)
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
            if kwargs.get('verbose', 0):
                logger.info(
                    "\n" + "-" * 31 + "\n"
                                      f"Cross-Validation {split + 1}/{self.cv.get_n_splits()}"
                    + "\n"
                    + "-" * 31
                    + "\n"
                )
            # prepare train data
            x_train = self._prepare_data(x, y, train_index, fit_transform=True)
            y_train = y[train_index]

            # prepare test data
            x_test = self._prepare_data(x, y,  test_index, fit_transform=False)
            y_test = y[test_index]
            # get model
            model = self.get_model()
            # train model
            history = model.fit(x_train, y_train, **kwargs)
            self.history.append(history.history)
            if kwargs.get('verbose', 0):
                logger.info(
                    "\n" + "-" * 31 + "\n"
                                      "Evaluate validation performance"
                    + "\n"
                    + "-" * 31
                    + "\n"
                )
            if self._model_checkpoint_deepcopy:
                model = self._load_best_model(model)
            if self._custom_eval:
                y_pred = model.predict(
                    x_test,
                    batch_size=kwargs.get('batch_size', 32),
                    verbose=kwargs.get('verbose', 0),
                )
                if self.needed_test_indexes:
                    val_res = self._custom_eval(y_test, y_pred, test_index)
                else:
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
            if kwargs.get('verbose', 0):
                logger.info(
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

    def get_train_history(self):
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

    def show_train_history(self, metrics: Optional[Union[List[str], str]] = None, fig_size: Tuple[int, int] = (10, 8),
                           save_fig: bool = False, splits: Optional[List[int]] = None, **kwargs):
        if isinstance(metrics, str):
            metrics = [metrics]
        df = self.get_train_history()
        if splits is not None:
            df = df[df.split.isin(splits)]
        if metrics is not None:
            df = df[['split', 'epochs'] + metrics]
        group = df.groupby('split')
        group_cnt = group.ngroups
        rows, extra = divmod(group_cnt, 2)
        if extra:
            rows += 1
        fig, ax = plt.subplots(rows, 2, figsize=fig_size)
        for i, (name, gr) in enumerate(group):
            s = gr.drop(['split'], axis=1).plot(x='epochs', ax=ax.flat[i], title='split_' + str(name), **kwargs)
        for ax in ax.flat[group_cnt:]:
            ax.remove()
        fig.suptitle('Train history for each split', fontsize=15)
        fig.tight_layout()
        if save_fig:
            plt.savefig(self.project_path.joinpath('train_history_plot.png'))
            plt.close()
        else:
            plt.show()


@contextlib.contextmanager
def maybe_distribute(distribution_strategy):
    """Distributes if distribution_strategy is set."""
    if distribution_strategy is None:
        yield
    else:
        with distribution_strategy.scope():
            yield
