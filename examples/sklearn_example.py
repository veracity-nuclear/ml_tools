"""Example demonstrating the SklearnStrategy with various scikit-learn models."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ml_tools.model.sklearn_strategy import SklearnStrategy
from ml_tools.model.state import State, StateSeries, SeriesCollection


def create_sample_data(n_series: int = 100, n_timesteps: int = 1) -> SeriesCollection:
    """Create sample data for testing.
    
    Parameters
    ----------
    n_series : int
        Number of series to create
    n_timesteps : int
        Number of timesteps per series (use 1 for static data)
        
    Returns
    -------
    SeriesCollection
        A collection of state series
    """
    series_list = []
    for _ in range(n_series):
        states = []
        for _ in range(n_timesteps):
            # Create random input features
            x1 = np.random.randn()
            x2 = np.random.randn()
            
            # Create target with known relationship: y = 2*x1 + 3*x2 + noise
            y = 2 * x1 + 3 * x2 + np.random.randn() * 0.1
            
            state = State({
                'x1': x1,
                'x2': x2,
                'y': np.array([y])  # Ensure y is a numpy array
            })
            states.append(state)
        series_list.append(StateSeries(states))
    
    return SeriesCollection(series_list)


def test_sklearn_strategy(estimator_class, estimator_args: dict, name: str):
    """Test a scikit-learn estimator with SklearnStrategy.
    
    Parameters
    ----------
    estimator_class : type
        A scikit-learn estimator class
    estimator_args : dict
        Arguments to pass to the estimator
    name : str
        Name of the estimator for display
    """
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print('='*60)
    
    # Create sample data
    train_data = create_sample_data(n_series=100, n_timesteps=1)
    test_data = create_sample_data(n_series=20, n_timesteps=1)
    
    # Create strategy
    strategy = SklearnStrategy(
        input_features=['x1', 'x2'],
        predicted_features=['y'],
        estimator=estimator_class,
        estimator_args=estimator_args
    )
    
    # Train
    print("Training...")
    strategy.train(train_data, test_data)
    print(f"Is trained: {strategy.isTrained}")
    
    # Predict
    print("Predicting...")
    predictions = strategy.predict(test_data)
    
    # Calculate error
    actual_values = []
    predicted_values = []
    for series, pred_series in zip(test_data, predictions):
        for state, pred_state in zip(series, pred_series):
            actual_values.append(state['y'][0] if hasattr(state['y'], '__len__') else state['y'])
            predicted_values.append(pred_state['y'][0] if hasattr(pred_state['y'], '__len__') else pred_state['y'])
    
    actual_values = np.array(actual_values)
    predicted_values = np.array(predicted_values)
    
    mse = np.mean((actual_values - predicted_values) ** 2)
    mae = np.mean(np.abs(actual_values - predicted_values))
    
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Mean Absolute Error: {mae:.6f}")
    
    # Test save/load
    print("\nTesting save/load...")
    strategy.save_model(f"test_sklearn_{name.replace(' ', '_')}.h5")
    loaded_strategy = SklearnStrategy.read_from_file(f"test_sklearn_{name.replace(' ', '_')}.h5")
    print(f"Loaded model is trained: {loaded_strategy.isTrained}")
    
    # Verify loaded model makes same predictions
    loaded_predictions = loaded_strategy.predict(test_data)
    pred_diff = 0
    for pred1, pred2 in zip(predictions, loaded_predictions):
        for state1, state2 in zip(pred1, pred2):
            val1 = state1['y'][0] if hasattr(state1['y'], '__len__') else state1['y']
            val2 = state2['y'][0] if hasattr(state2['y'], '__len__') else state2['y']
            pred_diff += abs(val1 - val2)
    
    print(f"Prediction difference after load: {pred_diff:.10f}")
    print("✓ Test passed!" if pred_diff < 1e-6 else "✗ Test failed!")


if __name__ == "__main__":
    print("SklearnStrategy Examples")
    print("="*60)
    
    # Test various scikit-learn models
    models = [
        (LinearRegression, {}, "Linear Regression"),
        (Ridge, {'alpha': 1.0}, "Ridge Regression"),
        (Lasso, {'alpha': 0.1}, "Lasso Regression"),
        (DecisionTreeRegressor, {'max_depth': 5, 'random_state': 42}, "Decision Tree"),
        (RandomForestRegressor, {'n_estimators': 50, 'max_depth': 5, 'random_state': 42}, "Random Forest"),
        (GradientBoostingRegressor, {'n_estimators': 50, 'max_depth': 3, 'random_state': 42}, "Gradient Boosting"),
        (SVR, {'kernel': 'rbf', 'C': 1.0}, "Support Vector Regression"),
    ]
    
    for estimator_class, estimator_args, name in models:
        try:
            test_sklearn_strategy(estimator_class, estimator_args, name)
        except Exception as e:
            print(f"\n✗ Error testing {name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)
