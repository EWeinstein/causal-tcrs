"""
Hyperparameter optimization for CAIRE.

This script implements Bayesian hyperparameter optimization for the CAIRE model, using the Ax platform.
"""
from aim import Text
from ax.service.ax_client import AxClient, ObjectiveProperties
import argparse
import configparser
from datetime import datetime
import json
import numpy as np
import os
import pandas as pd
import pickle
import subprocess

from CausalReceptors.manager import create_run


def parse_config(config):
    """Convert config to parameters format for Ax, taking care of type casting."""
    parameters = []
    for elem in config.sections():
        parameter = {}
        for key in config[elem]:
            if key in ['bounds', 'values', 'value']:
                if config[elem][key] in ['True', 'False']:
                    config[elem][key] = config[elem][key].lower()
                parameter[key] = json.loads(config[elem][key])
            elif key in ['log_scale', 'is_ordered']:
                parameter[key] = config[elem].getboolean(key)
            else:
                parameter[key] = config[elem][key]
        parameters.append(parameter)

    return parameters


def parse_params(params, seq_per_validation):
    """Convert from Ax parameter format to model format."""
    unit_batch = int(2 ** params['unit-batch-log2'])
    subunit_batch = int(2 ** params['subunit-batch-log2'])
    params['validate-iter'] = max(int(seq_per_validation / (unit_batch * subunit_batch)), 1)
    sub_args = []
    for elem in params:
        if params[elem] is True:
            sub_args.append('--{}'.format(elem))
        elif params[elem] is False or 'log2' in elem or elem == 'conv-kernels':
            continue
        else:
            sub_args.append('--{}={}'.format(elem, params[elem]))
    if 'unit-batch-log2' in params:
        sub_args.append('--unit-batch={}'.format(unit_batch))
        sub_args.append('--unit-batch-eval={}'.format(unit_batch))
    if 'subunit-batch-log2' in params:
        sub_args.append('--subunit-batch={}'.format(subunit_batch))
        sub_args.append('--subunit-batch-eval={}'.format(subunit_batch))
    if 'conv-kernels' in params:
        # Lock all convolution kernels to the same size.
        sub_args.append('--conv-kernel={}'.format(params['conv-kernels']))
        sub_args.append('--selection-conv-kernel={}'.format(params['conv-kernels']))
        sub_args.append('--encoder-conv-kernel={}'.format(params['conv-kernels']))

    return sub_args


def evaluate(params, args):
    """Train model and return validation score."""
    sub_args = parse_params(params, args.seq_per_validation)
    # Note: we evaluate the model to help post-hoc understanding, but hyperparameter optimization is done only with validation results.
    if not args.eval_all:
        sub_args.append('--no-test-set-evaluation')
    # Run model.
    cmd = ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'python', args.modelfile, args.datafile] + sub_args
    print(' '.join(cmd))
    sub_results = subprocess.run(' '.join(cmd), capture_output=True, shell=True)

    # Check for failure.
    if sub_results.returncode == 1:
        stderr = sub_results.stderr.decode('utf-8')
        if 'CUDA error: an illegal memory access' in stderr or 'CUDA out of memory' in stderr or 'Input contains NaN' in stderr:
            failed_trial = True
            return None, None, failed_trial
        else:
            print(stderr)
            assert False
    else:
        failed_trial = False

    # Extract results.
    test_eval_file = None
    for elem in sub_results.stdout.decode('utf-8').split('\n'):
        if 'ValidationResults:' in elem:
            validation_scores_file, best_model = elem.split(' ')[1:]
        elif 'TestResults:' in elem:
            test_eval_file = elem.split(' ')[1]
            break
    validation_scores = np.load(validation_scores_file)

    return {"score": validation_scores[0], "select_accuracy_mean": np.nan_to_num(validation_scores[1]),
            "propensity_pearson_mean": np.nan_to_num(validation_scores[2]),
            "outcome_score": np.nan_to_num(validation_scores[3]),
            "elbo_average": validation_scores[4], "convergence_diagnostic": np.nan_to_num(validation_scores[5])
            }, best_model, test_eval_file, failed_trial


def save(path, client, trial_evals, eval_all=False):
    """Save hyperparameter optimization results."""
    ax_result_file = os.path.join(path, 'ax_client.csv')
    ax_df = client.get_trials_data_frame()
    if eval_all:
        reformat_eval = []
        for trial_index, inner_dict in trial_evals.items():
            inner_dict['trial_index'] = trial_index
            reformat_eval.append(inner_dict)
        eval_df = pd.DataFrame(reformat_eval)
        ax_df = pd.merge(ax_df, eval_df, how='left', on='trial_index')
    ax_df.to_csv(ax_result_file)

    return ax_result_file


if __name__ == "__main__":
    # Load config file.
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for CAIRE.")
    parser.add_argument('modelfile', help='Model path (path to select_outcome_model.py)')
    parser.add_argument('datafile', help='Input data')
    parser.add_argument('config', help='Configuration file')
    parser.add_argument('trials', type=int, help='Number of trials')
    parser.add_argument('--max-batch-log2', default=20, type=int,
                        help='Maximum total batch size (unit * subunit) log 2.')
    parser.add_argument('--seq-per-validation', default=240000000, type=int,
                        help='Number of sequences before calculating validation set scores (used to compute the validate-iter argument of select_outcome_model.py).')
    parser.add_argument('--eval-all', default=False, action="store_true",
                        help='Run full evaluation of every model, not just best (just for understanding; validation scores are still used for optimization).')
    args = parser.parse_args()

    # Set up aim.
    aim_run = create_run('cir-hyo', args)

    # Parse config file.
    config = configparser.ConfigParser()
    config.read(args.config)
    for section in config.sections():
        aim_run['config_' + section] = dict(config[section])

    # Create Ax client.
    ax_client = AxClient()

    # Create Ax experiment.
    parameter_constraints = []
    if 'unit-batch-log2' in config and 'subunit-batch-log2' in config:
        if config['unit-batch-log2']['type'] == 'range' and config['unit-batch-log2']['type'] == 'range':
            parameter_constraints.append("unit-batch-log2 + subunit-batch-log2 <= {}".format(args.max_batch_log2))
            print('Added batch constraints: max batch 2^{} seqs'.format(args.max_batch_log2))
    ax_experiment = ax_client.create_experiment(
        name="tune_cir",
        parameters=parse_config(config),
        objectives={"score": ObjectiveProperties(minimize=False)},
        tracking_metric_names=['select_accuracy_mean', 'propensity_pearson_mean', 'outcome_score',
                               'elbo_average', 'convergence_diagnostic'],
        parameter_constraints=parameter_constraints
    )

    # Run Ax experiments.
    per_trial_models = dict()
    per_trial_evals = dict()
    per_trial_eval_files = dict()
    for i in range(args.trials):
        # Generate next set of parameters try.
        parameters, trial_index = ax_client.get_next_trial()
        print(trial_index)

        # Log on aim.
        aim_run.track({'trial': trial_index}, step=i)
        for key in parameters:
            if (key not in config) or (config[key]['type'] != 'fixed'):
                if type(parameters[key]) is str:
                    aim_run.track({key: Text(parameters[key])}, step=i)
                else:
                    aim_run.track({key: parameters[key]}, step=i)

        # Run trial (and handle exceptions, e.g. from CUDA out-of-memory).
        t0 = datetime.now()
        trial_result_data, best_model, test_eval_file, failed_trial = evaluate(parameters, args)
        print(best_model)
        print(test_eval_file)
        trial_time = (datetime.now() - t0).seconds / 60

        # Log failed trial.
        if failed_trial or np.isnan(trial_result_data['score']):
            ax_client.log_trial_failure(trial_index=trial_index)
            # Log on aim
            aim_run.track({'failed': 1}, step=i)
            continue
        else:
            aim_run.track({'failed': 0}, step=i)

        # Log completed trial on Ax.
        per_trial_models[trial_index] = best_model
        per_trial_eval_files[trial_index] = test_eval_file
        if test_eval_file is not None:
            with open(test_eval_file, 'rb') as f:
                test_eval = pickle.load(f)
            per_trial_evals[trial_index] = test_eval
            aim_run.track(test_eval, step=i)
        else:
            per_trial_evals[trial_index] = None
        ax_client.complete_trial(trial_index=trial_index, raw_data=trial_result_data,
                                 metadata=per_trial_evals[trial_index])

        # Log evaluations on aim, including trial time.
        aim_run.track({key: trial_result_data[key][0] for key in trial_result_data}, step=i)
        aim_run.track({'time': trial_time}, step=i)

        # Save intermediate results in case of interruption.
        save(aim_run['local_dir'], ax_client, per_trial_evals, args.eval_all)

    # Get the best model.
    best_trial, best_params = ax_client.get_best_trial(use_model_predictions=False)[:2]
    print('best trial', best_trial)

    # Evaluate best model on the test set.
    if args.eval_all:
        eval_results_file = per_trial_eval_files[best_trial]
        eval_results = per_trial_evals[best_trial]
    else:
        sub_args = parse_params(best_params, args.seq_per_validation)
        sub_args.append('--pretrained-model-params=' + per_trial_models[best_trial])
        # Run model.
        cmd = ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'python', args.modelfile, args.datafile] + sub_args
        print(' '.join(cmd))
        sub_results = subprocess.run(' '.join(cmd), capture_output=True, shell=True)
        # Check for failure.
        if sub_results.returncode == 1:
            print(sub_results.stderr.decode('utf-8'))
            assert False
        # Extract results.
        for elem in sub_results.stdout.decode('utf-8').split('\n'):
            if 'TestResults:' in elem:
                eval_results_file = elem.split(' ')[1]
                break
        # Load results.
        with open(eval_results_file, 'rb') as f:
            eval_results = pickle.load(f)
    # Log evaluation results on aim.
    aim_run.track(eval_results, step=args.trials)
    for key in best_params:
        if type(best_params[key]) is str:
            aim_run.track({'best_' + key: Text(best_params[key])}, step=args.trials)
        else:
            aim_run.track({'best_' + key: best_params[key]}, step=args.trials)
    aim_run.track({'best_trial': best_trial}, step=args.trials)
    aim_run.track({'best_model': Text(per_trial_models[best_trial])}, step=args.trials)

    # Save ax client.
    ax_result_file = save(aim_run['local_dir'], ax_client, per_trial_evals, args.eval_all)

    # Print results.
    print(f'OptimizationResults: {eval_results_file} {ax_result_file}')

