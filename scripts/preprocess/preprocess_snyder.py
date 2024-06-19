"""
Preprocessing code specifically for the Snyder et al. (2020) dataset.

This code assigns a disease severity score to the patients, and extracts metadata,
for all those patient samples for which the data is available.
"""

import json
import numpy as np
import os
import pandas as pd


def make_covariates_and_paths(file, repertoires_folder, processed, processed_metadata_tsv):
    # Load data with pandas
    data = pd.read_csv(file, delimiter='\t')

    # Initialize metadata of processes dataset.
    metadata = processed.create_group('metadata')

    # Create hospitalization score.
    # We first create a composite hospitalization score.
    # It is True if hospitalized = True, days_in_hospital > 0 OR covid_unit_admit = True
    # If all these values are NaN it is NaN.
    # Otherwise, it is False.
    hosp_true = (data['hospitalized'] == True) | (data['covid_unit_admit'] == True) | (data['days_in_hospital'] > 0)
    hosp_nan = data['hospitalized'].isnull() & data['covid_unit_admit'].isnull() & data['days_in_hospital'].isnull()
    hosp_true[hosp_nan] = pd.NA
    data['composite_hospitalized'] = hosp_true
    print(data['composite_hospitalized'].iloc[-10:])
    print('Initial hospitalized score has ',
          np.sum(data['hospitalized'] == True), ' true values,',
          np.sum(data['hospitalized'] == False), ' false values,',
          np.sum(~data['hospitalized'].isnull()), ' non-nan values,',
          np.sum(data['hospitalized'].isnull()), ' nan values')
    print('Created composite_hospitalized score, with ',
          np.sum(data['composite_hospitalized'] == True), ' true values,',
          np.sum(data['composite_hospitalized'] == False), ' false values,',
          np.sum(~data['composite_hospitalized'].isnull()), ' non-nan values,',
          np.sum(data['composite_hospitalized'].isnull()), ' nan values')

    # We now create a composite severity score for each patient.
    # 0 - no hospitalization
    # 1 - hospitalized
    # 2 - icu or death
    # nan - no data.
    severity = data['composite_hospitalized'] * np.nan
    severity[data['composite_hospitalized'] == False] = 0
    severity[data['composite_hospitalized'] == True] = 1
    severity[data['icu_admit'] == True] = 2
    severity[data['death'] == True] = 2
    print('Created severity score, with ',
          np.sum(severity == 0), ' no hospitalization cases,',
          np.sum(severity == 1), ' hospitalization cases,',
          np.sum(severity == 2), ' icu or death cases,',
          np.sum(~severity.isnull()), ' non-nan values total,',
          np.sum(severity.isnull()), ' nan values total')
    data['severity'] = severity

    # Get rows without missing data.
    # Note: we could use the missing data to help learn the selection model.
    # Here we do not, due to the computational cost of preprocessing an extra ~900 repertoires
    data_nomiss = data[~severity.isnull()]
    severity_nomiss = severity[~severity.isnull()]

    # Construct paths to each repertoire sample.
    samples_files = [elem + '.tsv' for elem in data_nomiss['sample_name']]
    samples_paths = [os.path.join(repertoires_folder, elem) for elem in samples_files]
    nsamples = len(samples_paths)
    metadata.attrs['samples_files'] = ','.join(samples_files)
    metadata.attrs['samples_paths'] = ','.join(samples_paths)
    metadata.attrs['nsamples'] = nsamples

    # Save outcomes.
    outcome_data = processed.require_dataset('outcomes', (nsamples,), dtype='i')
    for row in range(len(data_nomiss)):
        outcome_data[row] = severity_nomiss.iloc[row]

    # Construct table of covariates (plausible confounders).
    # Data is stored as an int (to make hdf5 happy), with negative values corresponding to missing data.
    covariate_cols = ['Age', 'Biological Sex', 'Racial Group']
    processed.attrs['covariate_cols'] = covariate_cols
    covariate_data = processed.require_dataset(
        'covariates', (nsamples, len(covariate_cols)), dtype='i')
    biological_sex_vals = {'Male': 0, 'Female': 1}
    racial_group_vals = {'Caucasian': 0, 'Hispanic': 1, 'Asian or Pacific Islander': 2, 'Black or African American': 3,
                         'Native American or Alaska Native': 4,
                         'Mixed racial group': 5, 'Unknown racial group': -1}
    metadata.attrs['attribute_dict'] = json.dumps({'Biological Sex': biological_sex_vals,
                                                   'Racial Group': racial_group_vals})
    for row in range(len(data_nomiss)):
        categ = 'Age'
        col = covariate_cols.index(categ)
        if data_nomiss[categ].isnull().iloc[row]:
            val = -1
        else:
            val = int(data_nomiss[categ].iloc[row].split(' ')[0])
        covariate_data[row, col] = val

        categ = 'Biological Sex'
        col = covariate_cols.index(categ)
        entry = data_nomiss[categ].iloc[row]
        if type(entry) is str:
            val = biological_sex_vals[entry]
        else:
            val = -1
        covariate_data[row, col] = val

        categ = 'Racial Group'
        col = covariate_cols.index(categ)
        entry = data_nomiss[categ].iloc[row]
        if type(entry) is str:
            val = racial_group_vals[entry]
        else:
            val = -1
        if data_nomiss['Ethnic Group'].iloc[row] == 'Hispanic or Latino':
            val = racial_group_vals['Hispanic']
        covariate_data[row, col] = val

    # Save csv with full metadata.
    data_nomiss.to_csv(processed_metadata_tsv, sep='\t', index=False)