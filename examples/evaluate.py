from copy import deepcopy
from typing import Dict
import time
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress Tensorflow Warning Messages

import tensorflow as tf
tf.get_logger().setLevel("ERROR")

from ml_tools import MinMaxNormalize, SeriesCollection, PredictionStrategy, GBMStrategy, \
                     NormalPerturbator, RelativeNormalPerturbator
from ml_tools.model.residual_correction_strategy import ResidualCorrectionStrategy
from ml_tools.utils.plotting import plot_ref_vs_pred, plot_hist, plot_sensitivities, print_metrics, plot_corr_matrix, \
                                    plot_ice_pdp, plot_shap
from ml_tools.examples.optimizer import build_dnn_optimizer, build_cnn_optimizer

from data_reader import DataReader


input_features = {'average_exposure'         : MinMaxNormalize(0., 60.),
                  'assembly_enrichment'      : MinMaxNormalize(0., 5.),
                  'measured_fixed_detector'  : MinMaxNormalize(0., 1.3),
                  'boron_concentration'      : MinMaxNormalize(0., 1500.)}


assembly_features = ["average_exposure", "assembly_enrichment"]

detector_features = ['measured_fixed_detector']

scalar_features = [feature for feature in input_features if feature not in assembly_features and
                                                            feature not in detector_features]

predicted_feature = 'cycle_exposure'


def main() -> None:
    """ Example code for doing a model optimization followed by comparison to other models
    """

    series_collection = DataReader.read_data(file_name = "sample.h5", num_procs = 20)

    models = {}
    models['GBM'] = GBMStrategy(input_features, predicted_feature)

    dnn_opt = build_dnn_optimizer(input_features, predicted_feature)
    models['DNN'] = dnn_opt.optimize(series_collection = series_collection.random_sample(10000),
                                     num_trials        = 50,
                                     number_of_folds   = 5,
                                     output_file       = "dnn_optimizer.out",
                                     num_procs         = 20)

    with open("dnn.pkl", 'wb') as file:
        pickle.dump(models['DNN'], file)

    with open("dnn.pkl", 'rb') as file:
        models['DNN'] = pickle.load(file)



    cnn_opt = build_cnn_optimizer(input_features, predicted_feature)
    models['CNN'] = cnn_opt.optimize(series_collection = series_collection.random_sample(10000),
                                     num_trials        = 50,
                                     number_of_folds   = 5,
                                     output_file       = "cnn_optimizer.out",
                                     num_procs         = 20)

    with open("cnn.pkl", 'wb') as file:
        pickle.dump(models['CNN'], file)

    with open("cnn.pkl", 'rb') as file:
        models['CNN'] = pickle.load(file)

    train_collection, valid_collection = series_collection.train_test_split(test_size=0.2)
    train(models, train_collection)

    models['GBM - GBM_Informed'] = ResidualCorrectionStrategy(reference_model=models['GBM'],
                                                              residual_model=deepcopy(models['GBM']),
                                                              reference_model_frozen=True)
    models['DNN - GBM_Informed'] = ResidualCorrectionStrategy(reference_model=models['GBM'],
                                                              residual_model=deepcopy(models['DNN']),
                                                              reference_model_frozen=True)
    models['CNN - GBM_Informed'] = ResidualCorrectionStrategy(reference_model=models['GBM'],
                                                              residual_model=deepcopy(models['CNN']),
                                                              reference_model_frozen=True)

    train({name: model for name, model in models.items() if 'GBM_Informed' in name}, train_collection)

    evaluate(models, valid_collection)



def train(models: Dict[str, PredictionStrategy], train_collection: SeriesCollection) -> None:
    """ Basic function for training all models

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be trained
    train_collection : SeriesCollection
        State series collection data to use for training
    """

    print('Training models...')
    for name, model in models.items():
        print(f'Training {name:s}...')
        start = time.time()
        if name in ['GBM', 'GBM - GBM_Informed']:
            train_data, test_data = train_collection.train_test_split(test_size=0.2)
            model.train(train_data, test_data)
        else:
            model.train(train_collection)
        print(f'  in {time.time()-start:.2f} seconds')


def evaluate(models: Dict[str, PredictionStrategy], valid_collection: SeriesCollection) -> None:
    """Basic function for plotting evaluation results

    Parameters
    ----------
    models : Dict[str, PredictionStrategy]
        Models to be evaluated
    valid_collection : SeriesCollection
        State series collection data to use for validation
    """

    print('Printing Results...')

    base_models            = {name: model for name, model in models.items() if not 'Informed' in name}
    gbm_informed_models    = {name: model for name, model in models.items() if 'GBM_Informed' in name}

    print('Plotting Correlation Matrix...')
    plot_corr_matrix(scalar_features, valid_collection, fig_name = "corr_matrix")

    print('Plotting Reference vs. Predicted Plots...')
    plot_ref_vs_pred(           base_models, valid_collection, title=False, fig_name = "ref_vs_pred_base")
    plot_ref_vs_pred(   gbm_informed_models, valid_collection, title=False, fig_name = "ref_vs_pred_gbm_informed")


    print('Plotting Histograms Plots...')
    plot_hist(           base_models, valid_collection, fig_name = "hist_base")
    plot_hist(   gbm_informed_models, valid_collection, fig_name = "hist_gbm_informed")

    print('Plotting ICE/PDP Plots...')
    plot_ice_pdp(gbm_informed_models, valid_collection, "boron_concentration", num_points=1000, fig_name_prefix = "ice_pdp_boron"    , num_procs=20)

    print('Plotting SHAP Plots...')
    feature_plots = {"scalars"         : scalar_features,
                     "avg_exp"         : ["average_exposure"],
                     "assem_enr"       : ["assembly_enrichment"],
                     "incore_det"      : ["measured_fixed_detector"]}
    plot_shap(gbm_informed_models, valid_collection, feature_plots, num_samples=1000, num_procs=20)

    print_metrics(models, valid_collection)

    perturbators = {'measured_fixed_detector' : RelativeNormalPerturbator(0.026),
                    'assembly_enrichment'     : NormalPerturbator(0.00005),
                    'average_exposure'        : RelativeNormalPerturbator(0.02),
                    'boron_concentration'     : NormalPerturbator(5.0)}

    pert_collection = valid_collection.random_sample(100)

    print('Plotting Sensitivities Plots...')
    plot_sensitivities(models                  = base_models,
                       series_collection       = pert_collection,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_base")

    plot_sensitivities(models                  = gbm_informed_models,
                       series_collection       = pert_collection,
                       perturbators            = perturbators,
                       number_of_perturbations = 100,
                       num_procs               = 30,
                       fig_name_prefix         = "box_plot_informed")


if __name__ == "__main__":
    main()
