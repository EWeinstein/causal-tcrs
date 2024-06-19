"""
Patch the output of preprocess_igor.py for the Snyder et al. dataset,
to remove redundant samples taken from the same patient.
"""


import argparse
import h5py
import os
import pandas as pd
import numpy as np

from CausalReceptors.manager import create_run


def better_sample(sample0, sample1):
    """
    Choose from among two samples taken from the same patient.
    Logic: take sample where patient is assigned larger severity;
    otherwise, take sample from earlier visit;
    otherwise, take sample with larger dataset.
    """
    samples = [sample0, sample1]
    if sample1['severity'] != sample0['severity']:
        return samples[np.argmax([sample0['severity'], sample1['severity']])]
    elif (not np.isnan(sample0['visit'])) and (not np.isnan(sample1['visit'])) and (
            sample1['visit'] != sample0['visit']):
        return samples[np.argmin([sample0['visit'], sample1['visit']])]
    else:
        # Note this will always resolve, since sequence numbers are all unique in this dataset.
        return samples[np.argmax([sample0['seqs'], sample1['seqs']])]


def mask_list(a, mask):
    # Return values of list a for which mask is True.
    return [elem for elem, inc in zip(a, mask) if inc]


def main(args):
    # Load metadata.
    metadata_tsv = os.path.join(args.processed_folder, 'preprocessed_metadata.tsv')
    metadata = pd.read_csv(metadata_tsv, delimiter='\t')

    # Load preprocessed data.
    pfile = os.path.join(args.processed_folder, 'preprocessed_igor.hdf5')
    processed_in = h5py.File(pfile, 'r')
    pmetadata_in = processed_in['metadata']
    tot_seqs_per_patient = pmetadata_in.attrs['seq_nums'][:]

    # Get list of unique subjects, and initialize storage for the best sample of each
    subjects = list(set([metadata['subject_id'].iloc[j] for j in range(len(metadata))]))
    best_sample = {elem: None for elem in subjects}

    # Iterate over samples
    for j in range(len(metadata)):
        # Get subject id.
        subject_id = metadata['subject_id'].iloc[j]
        # Extract severity, visit and data size.
        sample_stats = {'index': j,
                        'severity': metadata['severity'].iloc[j],
                        'visit': metadata['visit'].iloc[j],
                        'seqs': tot_seqs_per_patient[j]}

        # Update best sample for subject.
        if best_sample[subject_id] is None:
            best_sample[subject_id] = sample_stats
        else:
            best_sample[subject_id] = better_sample(best_sample[subject_id], sample_stats)

    # Get list of best samples
    best_samples = np.array([elem['index'] for elem in best_sample.values()])
    assert len(best_samples) == len(set(best_samples))

    # Create mask.
    mask = np.zeros(len(metadata), dtype=np.int32)
    mask[best_samples] = 1
    mask = mask.astype(np.bool_)
    metadata_out = metadata[mask]

    # Create output folder.
    os.makedirs(args.out, exist_ok=True)

    # Save new metadata file.
    metadata_out_tsv = os.path.join(args.out, 'preprocessed_metadata.tsv')
    metadata_out.to_csv(metadata_out_tsv)

    # Create new hdf5 file.
    out_h5f = os.path.join(args.out, 'preprocessed_igor.hdf5')
    run['out_h5f'] = out_h5f
    processed = h5py.File(out_h5f, 'w')
    pmetadata = processed.create_group('metadata')

    # -- New subsetted outcome and covariate data. --
    # Paths metadata.
    samples_files = mask_list(pmetadata_in.attrs['samples_files'].split(','), mask)
    samples_paths = mask_list(pmetadata_in.attrs['samples_paths'].split(','), mask)
    nsamples = sum(mask)
    pmetadata.attrs['samples_files'] = ','.join(samples_files)
    pmetadata.attrs['samples_paths'] = ','.join(samples_paths)
    pmetadata.attrs['nsamples'] = nsamples

    # Outcome data.
    outcome_data_in = processed_in['outcomes'][:]
    processed.create_dataset('outcomes', data=outcome_data_in[mask])

    # Covariate data.
    processed.attrs['covariate_cols'] = processed_in.attrs['covariate_cols']
    covariate_data_in = processed_in['covariates'][:]
    processed.create_dataset('covariates', data=covariate_data_in[mask])
    pmetadata.attrs['attribute_dict'] = pmetadata_in.attrs['attribute_dict']

    # -- New subsetted naive and mature repertoires. --
    max_len = pmetadata_in.attrs['max_len']
    seq_nums_in = pmetadata_in.attrs['seq_nums']
    seq_nums = mask_list(seq_nums_in, mask)
    naive_nums_in = pmetadata_in.attrs['naive_nums'][:]
    naive_nums = naive_nums_in[mask]
    naive_paths = mask_list(pmetadata_in.attrs['naive_paths'].split(','), mask)
    pmetadata.attrs['max_len'] = max_len
    pmetadata.attrs['seq_nums'] = seq_nums
    pmetadata.attrs['nonprod_nums'] = mask_list(pmetadata_in.attrs['nonprod_nums'], mask)
    pmetadata.attrs['naive_nums'] = naive_nums
    pmetadata.attrs['max_len_naive'] = pmetadata_in.attrs['max_len_naive']
    pmetadata.attrs['naive_paths'] = ','.join(naive_paths)
    pmetadata.attrs['aa_alphabet'] = pmetadata_in.attrs['aa_alphabet']

    productive_aa_in = processed_in['productive_aa'][:]
    productive_freq_in = processed_in['productive_freq'][:]
    naive_aa_in = processed_in['naive_aa'][:]
    productive_aa, productive_freq, naive_aa = [], [], []
    productive_pos = 0
    naive_pos = 0
    for j in range(len(seq_nums_in)):
        next_productive_pos = productive_pos + seq_nums_in[j]
        next_naive_pos = naive_pos + naive_nums_in[j]
        if mask[j]:
            p_slice = slice(productive_pos, next_productive_pos)
            n_slice = slice(naive_pos, next_naive_pos)
            productive_aa.append(productive_aa_in[p_slice])
            productive_freq.append(productive_freq_in[p_slice])
            naive_aa.append(naive_aa_in[n_slice])
        productive_pos = next_productive_pos
        naive_pos = next_naive_pos
    productive_aa = np.concatenate(productive_aa, axis=0)
    productive_freq = np.concatenate(productive_freq, axis=0)
    naive_aa = np.concatenate(naive_aa, axis=0)
    assert productive_aa.shape[0] == sum(seq_nums)
    assert productive_freq.shape[0] == sum(seq_nums)
    assert naive_aa.shape[0] == sum(naive_nums)
    processed.create_dataset('productive_aa', data=productive_aa)
    processed.create_dataset('productive_freq', data=productive_freq)
    processed.create_dataset('naive_aa', data=naive_aa)
    processed.close()

    with open(os.path.join(args.out, 'README.txt'), 'w') as rf:
        rf.write('This data is the output of aim run preprocess_patch {}.'.format(run['local_dir']))


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
            description='Patch the preprocessed Snyder et al. dataset, to remove redundant samples.'
    )
    parser.add_argument('processed_folder', help='Output folder from initial preprocessing.')
    parser.add_argument('out', help='Output folder for results of this patch.')
    args = parser.parse_args()

    # Logging.
    run = create_run('preprocess_patch', args)

    main(args)

