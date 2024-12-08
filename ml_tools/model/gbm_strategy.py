from __future__ import annotations
from typing import List, Dict, Optional
import os
import h5py
import lightgbm as lgb
import numpy as np
import pylab as plt

from ml_tools.model.state import StateSeries
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import FeatureProcessor


class GBMStrategy(PredictionStrategy):
    """ A concrete class for a Gradient-Boosting-based prediction strategy

    This prediction strategy is only intended for use with static State-Points, meaning
    non-temporal series, or said another way, State Series with series lengths of one.

    Attributes
    ----------
    boosting_type : str
        The boosting method to be used
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#boosting)
    objective : str
        The loss function to be used
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#objective)
    metric : str
        The metric to use when calculating the loss
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#metric)
    num_leaves : int
        The maximum number of leaves in one tree
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#num_leaves)
    learning_rate: float
        The learning / shrinkage rate
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#learning_rate)
    n_estimators : int
        Number of boosting iterations
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#num_iterations)
    max_depth : int
        The limit on the max depth for the tree model
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#max_depth)
    min_child_samples : int
        Minimum number of data in one leaf required to create a new leaf
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#min_data_in_leaf)
    subsample : float
        The fraction of the training data that is randomly sampled for each iteration / boosting round
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#bagging_fraction)
    colsample_bytree : float
        The fraction of features (columns) that are randomly selected and used for training each tree in the model
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#feature_fraction)
    reg_alpha : float
        The L1 regularization
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#lambda_l1)
    reg_lambda : float
        The L2 regularization
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#lambda_l2)
    verbose : int
        The level of LightGBM’s verbosity
        (see: https://lightgbm.readthedocs.io/en/stable/Parameters.html#verbosity)
    num_boost_round : int
        Number of boosting iterations
        (see: https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.train.html)
    stopping_rounds : int
        The number of rounds the validation score must improve in for training to continue
        (see: https://lightgbm.readthedocs.io/en/stable/Python-Intro.html#early-stopping)
    """

    @property
    def boosting_type(self) -> str:
        return self._boosting_type

    @boosting_type.setter
    def boosting_type(self, boosting_type: str) -> None:
        self._boosting_type = boosting_type

    @property
    def objective(self) -> str:
        return self._objective

    @objective.setter
    def objective(self, objective: str) -> None:
        self._objective = objective

    @property
    def metric(self) -> str:
        return self._objective

    @metric.setter
    def metric(self, metric: str) -> None:
        self._metric = metric

    @property
    def num_leaves(self) -> int:
        return self._num_leaves

    @num_leaves.setter
    def num_leaves(self, num_leaves: int) -> None:
        self._num_leaves = num_leaves

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate: float) -> None:
        self._learning_rate = learning_rate

    @property
    def n_estimators(self) -> int:
        return self._n_estimators

    @n_estimators.setter
    def n_estimators(self, n_estimators: int) -> None:
        self._n_estimators = n_estimators

    @property
    def max_depth(self) -> int:
        return self._max_depth

    @max_depth.setter
    def max_depth(self, max_depth: int) -> None:
        self._max_depth = max_depth

    @property
    def min_child_samples(self) -> int:
        return self._min_child_samples

    @min_child_samples.setter
    def min_child_samples(self, min_child_samples: int) -> None:
        self._min_child_samples = min_child_samples

    @property
    def subsample(self) -> float:
        return self._subsample

    @subsample.setter
    def subsample(self, subsample: float) -> None:
        self._subsample = subsample

    @property
    def colsample_bytree(self) -> float:
        return self._colsample_bytree

    @colsample_bytree.setter
    def colsample_bytree(self, colsample_bytree: float) -> None:
        self._colsample_bytree = colsample_bytree

    @property
    def reg_alpha(self) -> float:
        return self._reg_alpha

    @reg_alpha.setter
    def reg_alpha(self, reg_alpha: float) -> None:
        self._reg_alpha = reg_alpha

    @property
    def reg_lambda(self) -> float:
        return self._reg_lambda

    @reg_lambda.setter
    def reg_lambda(self, reg_lambda: float) -> None:
        self._reg_lambda = reg_lambda

    @property
    def verbose(self) -> int:
        return self._verbose

    @verbose.setter
    def verbose(self, verbose: int) -> None:
        self._verbose = verbose

    @property
    def num_boost_round(self) -> int:
        return self._num_boost_round

    @num_boost_round.setter
    def num_boost_round(self, num_boost_round: int) -> None:
        self._num_boost_round = num_boost_round

    @property
    def stopping_rounds(self) -> int:
        return self._stopping_rounds

    @stopping_rounds.setter
    def stopping_rounds(self, stopping_rounds: int) -> None:
        self._stopping_rounds = stopping_rounds

    @property
    def isTrained(self) -> bool:
        return self._gbm is not None


    def __init__(self,
                 input_features    : Dict[str, FeatureProcessor],
                 predicted_feature : str,
                 boosting_type     : str = "gbdt",
                 objective         : str = "regression",
                 metric            : str = "rmse",
                 num_leaves        : int = 64,
                 learning_rate     : float = 0.07,
                 n_estimators      : int = 1000,
                 max_depth         : int = 4,
                 min_child_samples : int = 20,
                 subsample         : float = 0.8,
                 colsample_bytree  : float = 0.8,
                 reg_alpha         : float = 0.0,
                 reg_lambda        : float = 0.0,
                 verbose           : int = -1,
                 num_boost_round   : int = 20,
                 stopping_rounds   : int = 5) -> None:

        super().__init__()

        self._predicted_feature = predicted_feature
        self._input_features    = input_features

        self.input_features     = input_features
        self.predicted_feature  = predicted_feature
        self.boosting_type      = boosting_type
        self.objective          = objective
        self.metric             = metric
        self.num_leaves         = num_leaves
        self.learning_rate      = learning_rate
        self.n_estimators       = n_estimators
        self.max_depth          = max_depth
        self.min_child_samples  = min_child_samples
        self.subsample          = subsample
        self.colsample_bytree   = colsample_bytree
        self.reg_alpha          = reg_alpha
        self.reg_lambda         = reg_lambda
        self.verbose            = verbose
        self.num_boost_round    = num_boost_round
        self.stopping_rounds    = stopping_rounds

        self._gbm               = None


    def train(self, train_data: List[StateSeries], test_data: Optional[List[StateSeries]] = None, num_procs: int = 1) -> None:

        assert all(len(series) == 1 for series in train_data), \
            "All State Series must be static statepoints (i.e. len(series) == 1)"

        X_train   = self.preprocess_inputs(train_data, num_procs)[:,0,:]
        y_train   = self._get_targets(train_data)[:,0]
        lgb_train = lgb.Dataset(X_train, y_train)

        lgb_eval  = None
        test_data = [] if test_data is None else test_data
        if len(test_data) > 0:
            assert all(len(series) == 1 for series in test_data), \
                "All State Series must be static statepoints (i.e. len(series) == 1"

            X_test   = self.preprocess_inputs(test_data, num_procs)[:,0,:]
            y_test   = self._get_targets(test_data)[:,0]
            lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        params = {"boosting_type"    : self.boosting_type,
                  "objective"        : self.objective,
                  "metric"           : self.metric,
                  "learning_rate"    : self.learning_rate,
                  "n_estimators"     : self.n_estimators,
                  "num_leaves"       : self.num_leaves,
                  "max_depth"        : self.max_depth,
                  "min_child_samples": self.min_child_samples,
                  "subsample"        : self.subsample,
                  "colsample_bytree" : self.colsample_bytree,
                  "reg_alpha"        : self.reg_alpha,
                  "reg_lambda"       : self.reg_lambda,
                  "verbose"          : self.verbose}

        self._gbm = lgb.train(params          = params,
                              train_set       = lgb_train,
                              num_boost_round = self.num_boost_round,
                              valid_sets      = lgb_eval,
                              callbacks       = [lgb.early_stopping(stopping_rounds=self.stopping_rounds)])

    def plot_importances(self) -> None:
        """ A method for plotting the importance of each input feature for a given state series

        This currently only supports plotting 20 state input features.  This should be more than
        sufficient for most use cases
        """

        features            = list(self.input_features)
        feature_importances = self._gbm.feature_importance().astype(float)
        feature_importances *= 100. / np.max(feature_importances)

        idx = np.argsort(feature_importances)[::-1]
        assert len(features) <= 20, "Only 20 features effectively fit on a single plot"

        plt.barh([features[i] for i in idx[:]][::-1], feature_importances[idx[:]][::-1])
        plt.xlabel('Relative Feature Importance [%]')
        plt.show()


    def _predict_one(self, state_series: StateSeries) -> float:

        return self._predict_all([state_series])[0]


    def _predict_all(self, state_series: List[StateSeries]) -> List[float]:

        assert self.isTrained
        assert all(len(series) == 1 for series in state_series), \
            "All State Series must be static statepoints (i.e. len(series) == 1)"

        X = self.preprocess_inputs(state_series)[:,0,:]
        return self._gbm.predict(X, num_iteration=self._gbm.best_iteration)


    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        lgbm_name = file_name.removesuffix(".h5") + ".lgbm" if file_name.endswith(".h5") else file_name + ".lgbm"
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        self._gbm.save_model(lgbm_name)
        with open(lgbm_name, 'rb') as file:
            file_data = file.read()

        with h5py.File(file_name, 'a') as h5_file:
            self.base_save_model(h5_file)
            h5_file.create_dataset('serialized_lgbm_file', data=file_data)


    def load_model(self, file_name: str) -> None:
        """ A method for loading a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to load the model from
        """
        lgbm_name = file_name.removesuffix(".h5") + ".lgbm" if file_name.endswith(".h5") else file_name + ".lgbm"
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        assert os.path.exists(file_name), f"file name = {file_name}"
        read_lgbm_h5 = not os.path.exists(lgbm_name)
        with h5py.File(file_name, 'r') as h5_file:
            self.base_load_model(h5_file)
            if read_lgbm_h5:
                file_data = h5_file['serialized_lgbm_file'][()]
                with open(lgbm_name, 'wb') as file:
                    file.write(file_data)

        self._gbm = lgb.Booster(model_file=lgbm_name)


    @classmethod
    def read_from_hdf5(cls: GBMStrategy, file_name: str) -> GBMStrategy:
        """ A basic factory method for building a GBM Strategy from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read the model

        Returns
        -------
        GBMStrategy:
            The model from the hdf5 file
        """
        assert os.path.exists(file_name), f"file name = {file_name}"

        new_gbm = cls({}, None)
        new_gbm.load_model(file_name)

        return new_gbm
