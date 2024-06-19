"""
Preprocessing code specifically for the Emerson et al. (2017) dataset.
"""

import json
import numpy as np
import os
import pandas as pd


class AttributeDict:
    """Dictionary for storing patient covariate metadata.
    Keys: attributes (e.g. ethnicity, etc.).
    Values: dictionary of possible values and assigned indices
    (e.g. {hispanic: 0, ...})).
    """
    def __init__(self, categories):

        self.a = {el: dict() for el in categories}
        self.tot = {el: dict() for el in categories}

    def add(self, category, value):
        if value.split(' ')[0] == 'Unknown':
            return -1
        if value not in self.a[category]:
            self.a[category][value] = len(self.a[category])
            self.tot[category][value] = 0
        self.tot[category][value] += 1
        return self.a[category][value]

    def length(self):
        return len(self.a)


def make_covariate_table(hla_dict, metadata, nsamples, processed, samples_paths):
    # Covariate column labels.
    covariate_cols = (['Age', 'Biological Sex', 'Ethnic Group', 'Racial Group']
                      + list(hla_dict.a['HLA MHC class I']))
    processed.attrs['covariate_cols'] = covariate_cols
    attribute_dict = AttributeDict(covariate_cols[1:])
    covariate_data = processed.create_dataset(
        'covariates', (nsamples, len(covariate_cols)), dtype='i')
    outcome_data = processed.create_dataset(
        'outcomes', (nsamples,), dtype='i')
    # Store covariates and outcomes.
    for row, file in enumerate(samples_paths):
        # Extract patient level data.
        data = pd.read_csv(file, delimiter='\t', nrows=2)
        properties = {el.split(':')[0]: el.split(':')[1] for el in
                      data['sample_catalog_tags'].iloc[0].split(',')}

        # Store data (all as integers).
        categ = 'Age'
        col = covariate_cols.index(categ)
        if categ in properties:
            val = int(properties[categ].split(' ')[0])
        else:
            val = -1
        covariate_data[row, col] = val

        for categ in ['Biological Sex', 'Ethnic Group', 'Racial Group']:
            col = covariate_cols.index(categ)
            if categ in properties:
                val = attribute_dict.add(categ, properties[categ])
            else:
                val = -1
            covariate_data[row, col] = val

        # HLA
        for el in data['sample_catalog_tags'].iloc[0].split(','):
            categ, val = el.split(':')
            if categ == 'HLA MHC class I' and val != 'HLA MHC Class I':
                col = covariate_cols.index(val)
                covariate_data[row, col] = 1

        categ = 'Virus Diseases'
        if categ in properties:
            if properties['Virus Diseases'] == 'Cytomegalovirus -':
                val = 0
            elif properties['Virus Diseases'] == 'Cytomegalovirus +':
                val = 1
            else:
                assert False, 'Unanticipated outcome in {}: {}'.format(
                    file, properties['Virus Diseases'])
        else:
            val = -1
        outcome_data[row] = val
    metadata.attrs['attribute_dict'] = json.dumps(attribute_dict.a)
    metadata.attrs['attribute_dict'] = json.dumps(attribute_dict.tot)


def make_samples_paths(directory, fileList, metadata):
    # Load list of patient files.
    samples_files = open(fileList, 'r').read().strip('\n').split('\n')
    samples_paths = [os.path.join(directory, el) for el in samples_files]
    nsamples = len(samples_paths)
    metadata.attrs['samples_files'] = ','.join(samples_files)
    metadata.attrs['samples_paths'] = ','.join(samples_paths)
    metadata.attrs['nsamples'] = nsamples
    return nsamples, samples_paths


def extract_hla(metadata, samples_paths):
    # Extract HLA types.
    hla_names = ['HLA MHC class I', 'HLA MHC class II']
    hla_dict = AttributeDict(hla_names)
    for row, file in enumerate(samples_paths):
        # Extract patient level data.
        data = pd.read_csv(file, delimiter='\t', nrows=2)
        # Extract HLA types.
        for el in data['sample_catalog_tags'].iloc[0].split(','):
            categ, val = el.split(':')
            if categ in hla_names and val != 'HLA MHC Class I':
                hla_dict.add(categ, val)
    metadata.attrs['hla_dict'] = json.dumps(hla_dict.a)
    metadata.attrs['hla_tot'] = json.dumps(hla_dict.tot)

    return hla_dict