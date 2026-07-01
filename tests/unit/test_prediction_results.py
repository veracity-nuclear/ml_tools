import matplotlib

matplotlib.use("Agg")

import numpy as np

from ml_tools import SeriesCollection, State, StateSeries
from ml_tools.model.feature_processor import NoProcessing
from ml_tools.model.feature_perturbator import NonPerturbator
from ml_tools.model.prediction_strategy import PredictionStrategy
from ml_tools.utils.prediction_results import PredictionResults


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

    def _params_to_dict(self) -> dict:
        return {"multiplier": self._multiplier}

    @classmethod
    def read_from_file(cls, file_name: str):
        raise NotImplementedError

    def _predict_one(self, state_series: np.ndarray) -> np.ndarray:
        return state_series[:, :1] * self._multiplier


def test_prediction_results_extracts_reference_and_model_values():
    first_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    second_collection = SeriesCollection([
        StateSeries([State({"x": np.array([4.0]), "y": np.array([8.0])})]),
        StateSeries([State({"x": np.array([5.0]), "y": np.array([10.0])})]),
        StateSeries([State({"x": np.array([6.0]), "y": np.array([12.0])})]),
    ])
    model = DummyStrategy()

    results = PredictionResults([
        PredictionResults.Spec(label="First",
                               model=model,
                               series_collection=first_collection,
                               predicted_feature="y"),
        PredictionResults.Spec(label="Second Scaled",
                               model=model,
                               series_collection=second_collection,
                               predicted_feature="y",
                               value_transform=lambda values: values / 2.0),
    ])

    np.testing.assert_allclose(results.reference_values,
                               np.asarray([[2.0, 4.0],
                                           [4.0, 5.0],
                                           [6.0, 6.0]]))
    np.testing.assert_allclose(results.predicted_values,
                               np.asarray([[2.0, 4.0],
                                           [4.0, 5.0],
                                           [6.0, 6.0]]))

    df = results.to_dataframe()
    assert list(df.columns) == [
        "series_index",
        "reference:First",
        "predicted:First",
        "reference:Second Scaled",
        "predicted:Second Scaled",
    ]
    np.testing.assert_allclose(df["reference:First"].to_numpy(), np.asarray([2.0, 4.0, 6.0]))


def test_prediction_results_plot_ref_vs_pred(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_name = tmp_path / "ref_vs_pred"
    results.plot_ref_vs_pred(fig_name=str(fig_name), title=False)

    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_prediction_results_plot_hist(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_name = tmp_path / "hist"
    results.plot_hist(fig_name=str(fig_name))

    output = fig_name.with_suffix(".png")
    if output.exists():
        output.unlink()


def test_prediction_results_print_metrics(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    output_file = tmp_path / "metrics.txt"
    results.print_metrics(output_file=str(output_file))

    output = output_file.read_text(encoding="utf-8")
    assert "Avg" in output
    assert "Std" in output
    assert "RMS" in output
    assert "Max" in output
    assert "Model :" in output


def test_prediction_results_plot_sensitivities(tmp_path):
    series_collection = SeriesCollection([
        StateSeries([State({"x": np.array([1.0]), "y": np.array([2.0])})]),
        StateSeries([State({"x": np.array([2.0]), "y": np.array([4.0])})]),
        StateSeries([State({"x": np.array([3.0]), "y": np.array([6.0])})]),
    ])
    model = DummyStrategy()
    results = PredictionResults([
        PredictionResults.Spec(label="Model",
                               model=model,
                               series_collection=series_collection,
                               predicted_feature="y"),
    ])

    fig_prefix = tmp_path / "sens"
    results.plot_sensitivities({"x": NonPerturbator()},
                               number_of_perturbations=1,
                               fig_name_prefix=str(fig_prefix),
                               num_procs=1)

    output = tmp_path / "sens_Model.png"
    if output.exists():
        output.unlink()
