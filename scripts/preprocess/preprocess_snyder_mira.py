"""
Preprocess MIRA assay SARS-CoV-2 binding data from the Snyder et al. (2020) study.
"""


import argparse
import chardet
import csv
import pandas as pd
import numpy as np
import os
import h5py
import sys

from CausalReceptors.manager import create_run
from preprocess_igor import get_cdr3s_size
from preprocess_snyder_patch import better_sample


def create_dataset(hf, name, shape, dtype=None, data=None):
    """Create hdf5 sub-dataset."""
    if name in hf:
        del hf[name]
    return hf.create_dataset(name, shape, dtype=dtype, data=data)


def main(args):
    # Open processed data.
    print('Setting up...')
    processed = h5py.File(os.path.join(args.processed_folder, 'preprocessed_igor.hdf5'), 'r+')
    processed.require_group('mira')

    # Open list of MIRA experiments.
    mira_subjects_f = os.path.join(args.mira_folder, 'subject-metadata.csv')
    with open(mira_subjects_f, 'rb') as f:
        rawdata = f.read()
        encoding = chardet.detect(rawdata)['encoding']
    mira_subjects = pd.read_csv(mira_subjects_f, encoding=encoding)

    # Extract list of MIRA experiments done on COVID patients and cI or cII peptides (not minigenes).
    mira_experiments = [[], []]  # class I, class II; same order as mira_files below
    mira_experiment_to_subject = dict()
    mira_retain_subjects = []
    for j in range(len(mira_subjects)):
        # Get COVID patients and non-minigene MIRA experiments.
        if mira_subjects['Cohort'].iloc[j] != 'Healthy (No known exposure)':
            target_type = mira_subjects['Target Type'].iloc[j]
            experiment = mira_subjects['Experiment'].iloc[j]
            subject = mira_subjects['Subject'].iloc[j]
            if target_type == 'C19_cI':
                mira_experiments[0].append(experiment)
                mira_experiment_to_subject[experiment] = subject
                mira_retain_subjects.append(subject)
            elif target_type == 'C19_cII':
                mira_experiments[1].append(experiment)
                mira_experiment_to_subject[experiment] = subject
                mira_retain_subjects.append(subject)
    mira_experiments_all = mira_experiments[0] + mira_experiments[1]
    processed['mira'].attrs['experiments_list'] = mira_experiments_all

    # Get paths to repertoires.
    repertoires_metadata = pd.read_csv(args.repertoires_file, delimiter='\t')
    best_sample = dict()
    for j in range(len(repertoires_metadata)):
        if repertoires_metadata['Dataset'].iloc[j] == 'COVID-19-Adaptive-MIRAMatched':

            # Get subject id.
            subject_id = int(repertoires_metadata['subject_id'].iloc[j])
            if subject_id in mira_retain_subjects:
                # Get sample name and path.
                sample_name = repertoires_metadata['sample_name'].iloc[j]
                sample_path = os.path.join(args.repertoires_folder, sample_name + '.tsv')
                with open(sample_path, 'r') as sr:
                    nseqs = len(sr.readlines()) - 1
                # Extract severity, visit and data size.
                sample_stats = {'index': j,
                                'severity': 0,
                                'visit': repertoires_metadata['visit'].iloc[j],
                                'seqs': nseqs,
                                'sample_path': sample_path,
                                'subject': subject_id}

                # Update best sample for subject.
                if subject_id not in best_sample:
                    best_sample[subject_id] = sample_stats
                else:
                    best_sample[subject_id] = better_sample(best_sample[subject_id], sample_stats)
    mira_matched_files = [elem['sample_path'] for elem in best_sample.values()]
    mira_matched_subjects = [elem['subject'] for elem in best_sample.values()]
    mira_nsamples = len(mira_matched_files)
    processed['mira'].attrs['repertoire_paths'] = ','.join(mira_matched_files)
    processed['mira'].attrs['nrepertoires'] = mira_nsamples
    processed['mira'].attrs['patients'] = ','.join(map(str, mira_matched_subjects))

    # List of MIRA files from Snyder et al., 2020, along with a final entry for repertoire sequences (non-hits).
    mira_files = ['peptide-ci', 'peptide-cii', 'nonhit']
    processed['mira'].attrs['mira_files'] = mira_files
    # MIRA data paths.
    mira_detail_f = {'peptide-ci': os.path.join(args.mira_folder, 'peptide-detail-ci.csv'),
                     'peptide-cii': os.path.join(args.mira_folder, 'peptide-detail-cii.csv')}

    # Get MIRA data and data sizes.
    mira_num_seqs = np.zeros(len(mira_files), dtype=np.int64)
    mira_data = dict()
    unproductive = 0
    for k, ke in enumerate(mira_files[:-1]):
        mira_detail = pd.read_csv(mira_detail_f[ke])
        include = np.ones(len(mira_detail), dtype=np.bool_)
        for j in range(len(mira_detail)):
            tcr_experiment = mira_detail['Experiment'].iloc[j]
            if tcr_experiment not in mira_experiments[k]:
                include[j] = False
                continue
            tcr_subject = mira_experiment_to_subject[tcr_experiment]
            if tcr_subject not in mira_matched_subjects:
                include[j] = False
                continue
            cdr3_raw = mira_detail['TCR BioIdentity'].iloc[j].split('+')[0]
            if cdr3_raw == 'unproductive':
                unproductive += 1
                include[j] = False

        mira_data[ke] = mira_detail[include]
        mira_save_f = os.path.join(args.processed_folder, ke + '.csv')
        mira_data[ke].to_csv(mira_save_f)
        mira_num_seqs[k] = len(mira_data[ke])
        print('Hits from ', ke, len(mira_detail), '->', len(mira_data[ke]))
    print('Hits with unproductive CDR3:', unproductive)

    # Number of a.a. to cut from start and end of (productive) CDR3 region.
    ncut = (args.ncut_start, args.ncut_end)

    # Get repertoire data sizes.
    print('Getting repertoire sizes..')
    _, _, repertoire_seq_nums = get_cdr3s_size(processed['mira'], ncut, mira_matched_files, run)
    mira_num_seqs[-1] = np.sum(repertoire_seq_nums)

    # Initialize storage of CDR3s.
    create_dataset(processed['mira'], 'num_seqs', mira_num_seqs.shape, dtype='i', data=mira_num_seqs)
    hits = np.zeros(sum(mira_num_seqs), dtype=np.int64)
    hits[:mira_num_seqs[:-1].sum()] = 1
    create_dataset(processed['mira'], 'hits', hits.shape, dtype='i', data=hits)
    max_len = processed['metadata'].attrs['max_len']
    seqs = create_dataset(processed['mira'], 'seq_aa', (sum(mira_num_seqs), max_len+1), dtype='i')
    patients = create_dataset(processed['mira'], 'patient', (sum(mira_num_seqs),), dtype='i')
    mhc_class = create_dataset(processed['mira'], 'class', (sum(mira_num_seqs),), dtype='i')
    experiments = create_dataset(processed['mira'], 'experiment', (sum(mira_num_seqs),), dtype='i')
    aa_alph_str = processed['metadata'].attrs['aa_alphabet']
    aa_alph = {aa: aai for aai, aa in enumerate(list(aa_alph_str))}

    # Save MIRA data CDR3s.
    print('Saving MIRA hit sequences...')
    seq_ind = 0
    length_violations = 0
    for k, ke in enumerate(mira_files[:-1]):
        run.track(k, name='save_mira_file')
        mira_detail = mira_data[ke]
        for j in range(len(mira_detail)):
            # Extract sequence.
            cdr3_raw = mira_detail['TCR BioIdentity'].iloc[j].split('+')[0]
            # Truncate and add stop symbol to indicate length.
            cdr3 = cdr3_raw[ncut[0]:-ncut[1]] + '*'

            # Check for length violations (since the MIRA data was not included in the original max length calculation).
            if len(cdr3) > max_len + 1:
                cdr3 = cdr3[:(max_len + 1)]
                length_violations += 1

            # Store sequence.
            seq = -1 * np.ones(max_len + 1, dtype=np.int64)
            for si, s in enumerate(list(cdr3)):
                seq[si] = aa_alph[s]
            seqs[seq_ind, :] = seq

            # Record subject (patient) index and MHC class for binding.
            experiment = mira_detail['Experiment'].iloc[j]
            experiment_ind = mira_experiments_all.index(experiment)
            subject = mira_experiment_to_subject[experiment]
            subject_ind = mira_matched_subjects.index(subject)
            experiments[seq_ind] = experiment_ind
            patients[seq_ind] = subject_ind
            mhc_class[seq_ind] = k + 1

            seq_ind += 1

    # Save repertoire CDR3s.
    print('Saving repertoire non-hit sequences...')
    for fi, file in enumerate(mira_matched_files):
        run.track(fi, name='save_repertoire_file')
        with open(file, 'r') as csvfile:
            # Read rows (TCR sequences):
            tr = csv.reader(csvfile, delimiter='\t')
            for il, line in enumerate(tr):
                if il == 0:
                    # Header information.
                    frame_ind = line.index('frame_type')
                    prod_ind = line.index('amino_acid')
                else:
                    if line[frame_ind] == 'In':
                        # Encode variable length CDR3 aa seq as integers.
                        # Trim start (assumed to be C...) and end
                        # (assumed to be ...F)
                        seq = -1 * np.ones(max_len + 1, dtype=np.int64)
                        cdr3 = line[prod_ind][ncut[0]:-ncut[1]] + '*'

                        # Check for length violations (since evaluation data was not included in the original max length calculation).
                        if len(cdr3) > max_len + 1:
                            cdr3 = cdr3[:(max_len + 1)]
                            length_violations += 1

                        # Store sequence.
                        for si, s in enumerate(list(cdr3)):
                            seq[si] = aa_alph[s]
                        seqs[seq_ind, :] = seq

                        # Store subject (patient) index
                        patients[seq_ind] = fi
                        mhc_class[seq_ind] = 0
                        experiments[seq_ind] = -1

                        seq_ind += 1

    # Clean up.
    processed['mira'].attrs['length_violations'] = length_violations
    print('Length violations:', length_violations, length_violations/np.sum(mira_num_seqs))
    print('Finished.')
    processed.close()


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description='Preprocess experimental antigen binding data')
    parser.add_argument('repertoires_file', help='File containing master list of patient repertoires.')
    parser.add_argument('repertoires_folder', default='.', type=str, help='Folder with repertoires.')
    parser.add_argument('processed_folder', help='Output folder from original preprocessing (preprocess_snyder_patch.py)')
    parser.add_argument('mira_folder', help='Folder with MIRA results from Snyder et al., 2020.')
    parser.add_argument('--ncut_start', default=1, type=int,
                        help='Make sure this matches the value for the original preprocessed data.')
    parser.add_argument('--ncut_end', default=1, type=int,
                        help='Make sure this matches the value for the original preprocessed data.')
    args = parser.parse_args()

    # Logging.
    run = create_run('preprocess_mira', args)

    main(args)