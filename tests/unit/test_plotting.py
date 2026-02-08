import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from ml_tools import SeriesCollection, State, StateSeries
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.model.feature_perturbator import NonPerturbator
from ml_tools.utils.plotting import (plot_ref_vs_pred,
                                     plot_hist,
                                     plot_sensitivities,
                                     print_metrics,
                                     plot_corr_matrix,
                                     plot_ice_pdp,
                                     plot_shap)


class DummyStrategy(PredictionStrategy):
    def __init__(self, multiplier: float = 2.0):
        super().__init__()
        self.input_features           = {"x": NoProcessing()}
        self.predicted_features       = {"y": NoProcessing()}
        self._predicted_feature_sizes = {"y": 1}
        self._trained                 = True
        self._multiplier              = multiplier

    @property
    def isTrained(self) -> bool:
        return self._trained

    def train(self, train_data, test_data=None, num_procs: int = 1) -> None:
        self._trained                 = True
        self._predicted_feature_sizes = {"y": 1}

    def to_dict(self) -> dict:
        return {"multiplier": self._multiplier}

    @classmethod
    def read_from_file(cls, file_name: str):
        raise NotImplementedError

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return state_series[:, :1] * self._multiplier


@pytest.fixture
def series_collection():
    series_list = []
    for i in range(3):
        x = np.array([float(i + 1)])
        y = np.array([float((i + 1) * 2)])
        series_list.append(StateSeries([State({"x": x, "y": y})]))
    return SeriesCollection(series_list)


@pytest.fixture
def models():
    return {"dummy": DummyStrategy()}


@pytest.fixture
def perturbators():
    return {"x": NonPerturbator()}


def test_plot_ref_vs_pred(models, series_collection, tmp_path):
    fig_name = tmp_path / "ref_vs_pred"
    plot_ref_vs_pred(models, series_collection, fig_name=str(fig_name))
    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_plot_hist(models, series_collection, tmp_path):
    fig_name = tmp_path / "hist"
    plot_hist(models, series_collection, fig_name=str(fig_name))
    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_plot_sensitivities(models, series_collection, perturbators, tmp_path):
    fig_prefix = tmp_path / "sens"
    plot_sensitivities(models,
                       series_collection,
                       perturbators,
                       number_of_perturbations=1,
                       fig_name_prefix=str(fig_prefix),
                       num_procs=1)
    output = tmp_path / "sens_dummy.png"
    if output.exists():
        output.unlink()


def test_print_metrics(models, series_collection, tmp_path):
    output_file = tmp_path / "metrics.txt"
    print_metrics(models, series_collection, output_file=str(output_file))
    if output_file.exists():
        output_file.unlink()


def test_plot_corr_matrix(series_collection, tmp_path):
    fig_name = tmp_path / "corr_matrix"
    plot_corr_matrix(["x"], series_collection, fig_name=str(fig_name))
    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_plot_ice_pdp(models, series_collection, tmp_path):
    fig_prefix = tmp_path / "ice_pdp"
    plot_ice_pdp(models,
                 series_collection,
                 input_feature="x",
                 fig_name_prefix=str(fig_prefix),
                 num_points=3,
                 num_procs=1,
                 silent=True)
    output = tmp_path / "ice_pdp_dummy_x.png"
    if output.exists():
        output.unlink()


def test_plot_shap(models, series_collection, tmp_path):
    fig_prefix = tmp_path / "shap"
    plot_shap(models,
              series_collection,
              feature_plots={"inputs": ["x"]},
              fig_name_prefix=str(fig_prefix),
              num_samples=3,
              num_procs=1,
              silent=True)
    output = tmp_path / "shap_inputs_dummy.png"
    if output.exists():
        output.unlink()
