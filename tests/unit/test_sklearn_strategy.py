"""Unit tests for SklearnStrategy."""

import unittest
import tempfile
import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from ml_tools.model.sklearn_strategy import SklearnStrategy
from ml_tools.model.state import State, StateSeries, SeriesCollection


class TestSklearnStrategy(unittest.TestCase):
    """Test cases for SklearnStrategy."""

    def setUp(self):
        """Set up test data."""
        # Create simple synthetic data
        self.train_data = self._create_data(n_series=50)
        self.test_data = self._create_data(n_series=10)

    def _create_data(self, n_series: int) -> SeriesCollection:
        """Create synthetic data for testing."""
        series_list = []
        for _ in range(n_series):
            x1 = np.random.randn()
            x2 = np.random.randn()
            y = 2 * x1 + 3 * x2 + np.random.randn() * 0.1

            # Ensure y is a numpy array (not just a scalar)
            state = State({'x1': x1, 'x2': x2, 'y': np.array([y])})
            series_list.append(StateSeries([state]))

        return SeriesCollection(series_list)

    def test_initialization(self):
        """Test strategy initialization."""
        strategy = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=LinearRegression
        )

        self.assertIsNotNone(strategy.estimator)
        self.assertFalse(strategy.isTrained)
        self.assertEqual(len(strategy.input_features), 2)
        self.assertEqual(len(strategy.predicted_features), 1)

    def test_train_and_predict(self):
        """Test training and prediction."""
        strategy = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=LinearRegression
        )

        # Train
        strategy.train(self.train_data)
        self.assertTrue(strategy.isTrained)

        # Predict
        predictions = strategy.predict(self.test_data)
        self.assertEqual(len(predictions), len(self.test_data))

        # Check predictions are reasonable
        for series in predictions:
            for state in series:
                self.assertIn('y', state.features)
                # Check that y is either a scalar or numpy array
                y_val = state['y']
                if isinstance(y_val, np.ndarray):
                    self.assertTrue(y_val.size > 0)
                else:
                    self.assertIsInstance(y_val, (int, float, np.number))

    def test_save_and_load(self):
        """Test saving and loading a trained model."""
        strategy = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=RandomForestRegressor,
            estimator_args={'n_estimators': 10, 'random_state': 42}
        )

        # Train
        strategy.train(self.train_data)
        predictions_before = strategy.predict(self.test_data)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            strategy.save_model(tmp_path)

            # Load
            loaded_strategy = SklearnStrategy.read_from_file(tmp_path)
            self.assertTrue(loaded_strategy.isTrained)

            # Predict with loaded model
            predictions_after = loaded_strategy.predict(self.test_data)

            # Compare predictions
            for series_before, series_after in zip(predictions_before, predictions_after):
                for state_before, state_after in zip(series_before, series_after):
                    np.testing.assert_almost_equal(
                        state_before['y'],
                        state_after['y'],
                        decimal=6
                    )
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_multiple_estimators(self):
        """Test that different sklearn estimators work."""
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.linear_model import Ridge

        estimators = [
            (LinearRegression, {}),
            (Ridge, {'alpha': 1.0}),
            (DecisionTreeRegressor, {'max_depth': 3, 'random_state': 42}),
            (RandomForestRegressor, {'n_estimators': 5, 'max_depth': 3, 'random_state': 42}),
        ]

        for estimator_class, estimator_args in estimators:
            with self.subTest(estimator=estimator_class.__name__):
                strategy = SklearnStrategy(
                    input_features=['x1', 'x2'],
                    predicted_features=['y'],
                    estimator=estimator_class,
                    estimator_args=estimator_args
                )

                strategy.train(self.train_data)
                self.assertTrue(strategy.isTrained)

                predictions = strategy.predict(self.test_data)
                self.assertEqual(len(predictions), len(self.test_data))

    def test_predict_before_train_error(self):
        """Test that predicting before training raises an error."""
        strategy = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=LinearRegression
        )

        with self.assertRaises(AssertionError):
            strategy.predict(self.test_data)

    def test_equality(self):
        """Test equality comparison."""
        from sklearn.linear_model import Ridge
        
        strategy1 = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=LinearRegression
        )

        strategy2 = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=LinearRegression
        )

        # Should be equal (same configuration)
        self.assertEqual(strategy1, strategy2)

        # Different estimator type
        strategy3 = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=Ridge
        )
        self.assertNotEqual(strategy1, strategy3)
        
        # Same class but different args
        strategy4 = SklearnStrategy(
            input_features=['x1', 'x2'],
            predicted_features=['y'],
            estimator=Ridge,
            estimator_args={'alpha': 2.0}
        )
        self.assertNotEqual(strategy3, strategy4)


if __name__ == '__main__':
    unittest.main()
