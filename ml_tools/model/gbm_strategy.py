from typing import List, Type, TypeVar
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from ml_tools.model.state import State
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_processor import NoProcessing

T = TypeVar('T', bound='GBMStrategy')
class GBMStrategy(PredictionStrategy):
    @property
    def isTrained(self) -> bool:
        return self._gbm is not None

    """ A concrete class for a Gradient Boosing prediction strategy

    Attributes
    ----------
    params : dict
        Configuration parameters for LightGBM
    test_fraction : float
        Fraction of the training data that will be used for training: default = 0.2
    """
    def __init__(self, input_features, predicted_feature, params: dict = {}, test_fraction: float=0.2) -> None:

        super().__init__()
        self._predicted_feature = predicted_feature
        self._test_fraction = test_fraction

        self._params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            'metric': 'rmse',           # Use Root Mean Squared Error (RMSE) as the evaluation metric
            'num_leaves': 64,           # Control the complexity of the tree model
            'learning_rate': 0.07,      # Set the learning rate
            'n_estimators': 1000,       # Number of boosting rounds or trees
            'max_depth': 4,             # No limit on the depth of the trees
            'min_child_samples': 20,    # Minimum number of samples required to create a new leaf node
            'subsample': 0.8,           # Fraction of samples to be used for training each tree
            'colsample_bytree': 0.8,    # Fraction of features to be used for training each tree
            'reg_alpha': 0.0,           # L1 regularization term
            'reg_lambda': 0.0,          # L2 regularization term
            "verbose": -1,
        }
        #any changes in params will override the default
        self._params.update(params)

        self._input_features = input_features

        self._gbm = None

    def train(self, states: List[State], num_procs: int = 1) -> None:
        """ The method that approximates the output corresponding to a given input state

        Parameters
        ----------
        states : List[State]
            A list of input states at which to train
        """
        X = self.preprocess_inputs(states, num_procs)
        y = self._get_targets(states)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(self._test_fraction))

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

        self._gbm  = lgb.train(
                        self._params, lgb_train, num_boost_round=20, valid_sets=lgb_eval, callbacks=[lgb.early_stopping(stopping_rounds=5)]
                     )

    def plot_importances(self, state: State):
        import pylab as plt
        names = [feature for feature in self.input_features]
        feature_importances = self._gbm.feature_importance().astype(float)
        feature_importances *= 100. / np.max(feature_importances)
        idx = np.argsort(feature_importances)[::-1]
        plt.barh([names[i] for i in idx[:20]][::-1], feature_importances[idx[:20]][::-1])
        plt.xlabel('Relative Feature Importance [%]')
        plt.show()

    def _predict_one(self, state: State) -> float:
        """ The method that approximates the output corresponding to a given input state

        Parameters
        ----------
        state : State
            The input state at which to predict the output

        Returns
        -------
        float
            The output
        """
        return self._predict_all([state])[0]

    def _predict_all(self, states: List[State]) -> List[float]:
        """ The method that approximates the output corresponding to a given input state

        Parameters
        ----------
        states : List[State]
            The input states at which to predict the output

        Returns
        -------
        List[float]
            The output for each state
        """
        assert(self.isTrained)

        X = self.preprocess_inputs(states)
        return self._gbm.predict(X, num_iteration=self._gbm.best_iteration)

    def save_model(self, file_name: str) -> None:
        """ A method for saving a trained model

        Parameters
        ----------
        file_name : str
            The name of the file to export the model to
        """
        lgbm_name = file_name + ".lgbm"
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
        lgbm_name = file_name + ".lgbm"
        file_name = file_name if file_name.endswith(".h5") else file_name + ".h5"

        assert(os.path.exists(file_name))
        read_lgbm_h5 = not os.path.exists(lgbm_name)
        with h5py.File(file_name, 'r') as h5_file:
            self.base_load_model(h5_file)
            if read_lgbm_h5:
                file_data = h5_file['serialized_lgbm_file'][()]
                with open(lgbm_name, 'wb') as file:
                    file.write(file_data)

        self._gbm = lgb.Booster(model_file=lgbm_name)

    @classmethod
    def read_from_hdf5(cls: Type[T], file_name: str) -> Type[T]:
        """ A basic factory method for building States from an HDF5 file

        Parameters
        ----------
        file_name : str
            The name of the file from which to read the model

        Returns
        -------
        GBMStrategy:
            The model from the hdf5 file
        """
        assert(os.path.exists(file_name))

        new_gbm = cls(None)
        new_gbm.load_model(file_name)

        return new_gbm