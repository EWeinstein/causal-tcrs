"""
Semisynthetic data experiments for CAIRE.

This script generates semisynthetic datasets under different conditions, and then,
for each dataset, hyperparameter optimizes and evaluates CAIRE, as well as any comparison models.
"""
import argparse
from ax.service.ax_client import AxClient
import configparser
from datetime import datetime
import json
import os
import pandas as pd
import pickle
import subprocess

from CausalReceptors.manager import create_run


def parse_config(experimentConfig, modelConfig):
    # Extract global parameters.
    n_scenarios = int(experimentConfig['global']['n_scenarios'])
    n_models = int(experimentConfig['global']['n_models'])

    # Construct list of data generating scenario parameters.
    data_generation_scenarios = []
    # Load fixed settings.
    fixed_settings = experimentConfig['fixed']
    # Load variable settings, creating unique data generation scenario for each.
    variable_settings = experimentConfig['variable']
    for i in range(n_scenarios):
        scenario = dict(fixed_settings)
        for setting in variable_settings:
            # Take ith entry of each variable setting.
            scenario[setting] = str(json.loads(variable_settings[setting])[i])
        data_generation_scenarios.append(scenario)

    # Construct list of model class parameters. These take the form of hyperparameter optimization settings.
    model_classes = []
    # Load fixed settings.
    fixed_settings = modelConfig
    # Load variable settings, creating a unique model class for each.
    variable_settings = experimentConfig['models']
    for i in range(n_models):
        model_class = {ke: dict(fixed_settings[ke]) for ke in fixed_settings.sections()}
        for setting in variable_settings:
            # Take ith entry of each variable setting.
            value, value_type = json.loads(variable_settings[setting])[i]
            model_class[setting] = {'name': setting,
                                    'type': 'fixed',
                                    'value': str(value),
                                    'value_type': str(value_type)}
        model_classes.append(model_class)

    return data_generation_scenarios, model_classes


def update_seed(seed):
    return str(int(seed) + 1)


def parse_params(params):
    cmd = []
    for ke in params:
        val = params[ke]
        if val.lower() == 'true':
            cmd.append(f'--{ke}')
        elif val.lower() == 'false':
            continue
        else:
            cmd.append(f'--{ke}={val}')

    return cmd


def generate_data(datascript, experimentConfig, scenario):
    # Assemble inputs to semisynthetic.py
    cmd = ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'python', datascript, experimentConfig['global']['basedata']
           ] + parse_params(scenario)
    print(' '.join(cmd))
    subp_results = subprocess.run(' '.join(cmd), capture_output=True, shell=True)

    # Check for failure.
    if subp_results.returncode == 1:
        stderr = subp_results.stderr.decode('utf-8')
        print(stderr)
        assert False

    # Extract results.
    for elem in subp_results.stdout.decode('utf-8').split('\n'):
        if 'GeneratedData:' in elem:
            datafile = elem.split(' ')[1]
            break

    return datafile


def train_model(datafile, modelscript, hyoscript, model_class, indx, hyperparam_trials,
                aim_run, smoke=False, eval_all=False):
    # Write config.
    configWrite = configparser.ConfigParser()
    configWrite.read_dict(model_class)
    modelClassFile = '{}/model_{}.ini'.format(aim_run['local_dir'], indx)
    with open(modelClassFile, 'w') as f:
        configWrite.write(f)

    # Assemble input
    cmd = ['OMP_NUM_THREADS=1', 'MKL_NUM_THREADS=1', 'python', hyoscript, modelscript, datafile,
           modelClassFile, str(hyperparam_trials)]
    if smoke:
        cmd.append('--seq-per-validation=1')
    if eval_all:
        cmd.append('--eval-all')
    print(' '.join(cmd))
    subp_results = subprocess.run(' '.join(cmd), capture_output=True, shell=True)

    # Check for failure.
    if subp_results.returncode == 1:
        stderr = subp_results.stderr.decode('utf-8')
        print(stderr)
        assert False

    # Extract results.
    for elem in subp_results.stdout.decode('utf-8').split('\n'):
        if 'OptimizationResults:' in elem:
            eval_results_file, ax_result_file = elem.split(' ')[1:]
            break
    print(eval_results_file, ax_result_file)

    # Load results.
    with open(eval_results_file, 'rb') as f:
        eval_results = pickle.load(f)
    ax_df = pd.read_csv(ax_result_file)

    return eval_results, ax_df


def main(args):
    # Load config information.
    experimentConfig = configparser.ConfigParser()
    experimentConfig.read(args.experimentconfig)
    modelConfig = configparser.ConfigParser()
    modelConfig.read(args.modelconfig)
    data_generation_scenarios, model_classes = parse_config(experimentConfig, modelConfig)
    smoke = experimentConfig['fixed'].getboolean('smoke')
    n_repetitions = int(experimentConfig['global']['n_repetitions'])
    hyperparam_trials = int(experimentConfig['global']['hyperparameter_trials'])
    eval_all = experimentConfig['global'].getboolean('eval-all')

    # Save configs.
    for section in experimentConfig.sections():
        aim_run['experiment_config_' + section] = dict(experimentConfig[section])
    for section in modelConfig.sections():
        aim_run['model_config_' + section] = dict(modelConfig[section])

    # Compute time estimate.
    generation_time = 2
    model_time = 2 + float(modelConfig['max-time']['value'])
    total_time = len(data_generation_scenarios) * n_repetitions * (generation_time + len(model_classes) * hyperparam_trials * model_time)
    print('Estimated time: ', total_time)
    t0 = datetime.now()

    # Iterate over data generation scenarios.
    mi = 0
    storage = []
    sweep_results_file = os.path.join(aim_run['local_dir'], 'sweep_results.pkl')
    print('Sweep results:', sweep_results_file)
    for i, data_generation_scenario in enumerate(data_generation_scenarios):
        storage.append([])
        # Iterate over independent seeds.
        for j in range(n_repetitions):
            storage[-1].append([])
            # Update data generation and model seeds.
            data_generation_scenario['seed'] = update_seed(data_generation_scenario['seed'])
            data_generation_scenario['data-seed'] = update_seed(data_generation_scenario['data-seed'])

            # Generate data.
            datafile = generate_data(args.datascript, experimentConfig, data_generation_scenario)

            # Iterate over model classes.
            for k, model_class in enumerate(model_classes):
                storage[-1][-1].append([])
                # Update model seeds.
                model_class['seed']['value'] = update_seed(model_class['seed']['value'])
                model_class['data-seed']['value'] = update_seed(model_class['data-seed']['value'])

                # Train models (including hyperparameter optimization).
                print('sweep stage:', i, j, k)
                eval_results, ax_df = train_model(datafile, args.modelscript, args.hyoscript, model_class, mi,
                                                  hyperparam_trials, aim_run, smoke=smoke, eval_all=eval_all)

                # Store results.
                storage[i][j][k] = {'eval': eval_results, 'ax': ax_df}
                aim_run.track(eval_results, step=mi)
                aim_run.track({'data_scenario': i, 'repeat': j, 'model class': k, 'iterate': mi}, step=mi)
                mi += 1

                # Save results.
                with open(sweep_results_file, 'wb') as f:
                    pickle.dump(storage, f)

    print(f'SweepResults: {sweep_results_file}')

    print('Total time:', datetime.now() - t0, '|| Estimate was', total_time)


if __name__ == "__main__":
    # Load config file.
    parser = argparse.ArgumentParser(description='Evaluate models under a range of data generating scenarios.')
    parser.add_argument('datascript', help='Data generation path (semisynthetic.py)')
    parser.add_argument('modelscript', help='Model path (select_outcome_model.py)')
    parser.add_argument('hyoscript', help='Hyperparameter optimization path (hyper_optimization.py)')
    parser.add_argument('experimentconfig', help='Configuration file for the data generating scenarios and models to evaluate')
    parser.add_argument('modelconfig', help='Base configuration file for models (following format of config files for hyper_optimization.py).')
    args0 = parser.parse_args()

    # Set up aim.
    aim_run = create_run('cir-sysw', args0)

    main(args0)

