"""
Preprocessing repertoire datasets for CAIRE.

This code takes in repertoire sequencing study data, specifically the data from
Emerson et al. 2017 (which is used in the semisynthetic experiments)
Snyder et al. 2020 (which is used in the COVID application).

The code performs the following steps:
1. It extracts the relevant sequence data and metadata from the original data files.
2. It trains IGoR models on the nonproductive sequences, and draws samples from them.
3. It stores the entire dataset as an hdf5 file.
"""



import aim
import argparse
import numpy as np
import subprocess
import os
import h5py
import csv
from Bio.Seq import Seq
import multiprocessing
from datetime import datetime
import json
import sys

from CausalReceptors.manager import create_run

import preprocess_emerson
import preprocess_snyder


def get_cdr3s_size(metadata, ncut, samples_paths, run):
    # Extract number of CDR3s and max lengths.
    max_len = 0
    seq_nums = []
    nonprod_nums = []
    for fi, file in enumerate(samples_paths):
        with open(file) as csvfile:
            # Total productive & nonproductive CDR3s.
            tot_seqs = 0
            tot_nonprod = 0
            # Read rows (TCR sequences).
            tr = csv.reader(csvfile, delimiter='\t')
            for il, line in enumerate(tr):
                if il == 0:
                    # Header information.
                    frame_ind = line.index('frame_type')
                    prod_ind = line.index('amino_acid')
                else:
                    if line[frame_ind] == 'In':
                        tot_seqs += 1
                        max_len = np.maximum(max_len,
                                             len(line[prod_ind]) - sum(ncut))
                    else:
                        tot_nonprod += 1
            # Record total.
            seq_nums.append(tot_seqs)
            nonprod_nums.append(tot_nonprod)
        run.track(fi, name='get_cdr3s_size_file')
    metadata.attrs['max_len'] = max_len
    metadata.attrs['seq_nums'] = seq_nums
    metadata.attrs['nonprod_nums'] = nonprod_nums
    return max_len, nonprod_nums, seq_nums


def run_igor(fileinfo):
    ncut, fi, file, seq_num, nonprod_num = fileinfo
    run.track(fi, name='run_igor_file')
    igor_batch = os.path.split(file)[-1].split('.')[0]
    # -- Read non-productive sequences and save. --
    fasta = file.split('.')[0] + '_nonproductives.txt'
    check = Checkpoint(args.out, 'read_{}_{}'.format(fi, igor_batch))
    if check.status():
        with open(file, 'r') as csvfile:
            with open(fasta, 'w') as fastafile:
                # Read rows (TCR sequences).
                tr = csv.reader(csvfile, delimiter='\t')
                for il, line in enumerate(tr):
                    if il == 0:
                        # Header information.
                        frame_ind = line.index('frame_type')
                        rearrange_ind = line.index('rearrangement')
                    else:
                        if line[frame_ind] != 'In':
                            # Nonproductive sequences - save formatted to train IGoR.
                            # This truncation at 65 matches IGoRs guidelines, which expects ~60nt for TRB. In the
                            # Emerson dataset, they provide this truncation as a separate field, 'rearrangement_trunc',
                            # but they do not in Snyder. So, truncating the 'rearrangement' field works for both.
                            fastafile.write('{}\n'.format(line[rearrange_ind][-65:]))
        check.close()

    # -- Run IGoR. --
    igor_cmd = ['taskset', '--cpu-list', args.cpu_list,
                args.igor,  # Igor executable.
                '-set_wd', args.out,  # Set working directory.
                '-batch', igor_batch]  # Run name (sample file).
    igor_model_dir = os.path.join(os.path.split(os.path.split(args.igor)[0])[0],
                                  'models', 'human', 'tcr_beta')
    # Alignment.
    check = Checkpoint(args.out, 'align_{}_{}'.format(fi, igor_batch))
    # Read sequences command. Option to subsample for scalability.
    read_cmd = igor_cmd.copy()
    if args.subsample < nonprod_num:
        subsamp = int(args.subsample)
        read_cmd += ['-subsample', str(subsamp)]
    else:
        subsamp = nonprod_num
    read_cmd += ['-read_seqs', fasta]
    # Align sequences command.
    align_cmd = igor_cmd.copy() + [
        '-species', 'human', '-chain', 'beta',
        '-set_CDR3_anchors',
        '--V', os.path.join(igor_model_dir, 'ref_genome', 'V_gene_CDR3_anchors.csv'),
        '--J', os.path.join(igor_model_dir, 'ref_genome', 'J_gene_CDR3_anchors.csv'),
        '-align', '--all']
    if check.status():
        # Read sequences.
        print(' '.join(read_cmd))
        subprocess.run(read_cmd, check=True, stdout=check.stdout, stderr=check.stderr)
        # Align sequences.
        print(' '.join(align_cmd))
        subprocess.run(align_cmd, check=True, stdout=check.stdout, stderr=check.stderr)
        check.close()

    # Inference.
    check = Checkpoint(args.out, 'infer_{}_{}'.format(fi, igor_batch))
    if check.status():
        inf_cmd = igor_cmd.copy() + [
                              '-set_CDR3_anchors',
                              '--V', os.path.join(igor_model_dir, 'ref_genome', 'V_gene_CDR3_anchors.csv'),
                              '--J', os.path.join(igor_model_dir, 'ref_genome', 'J_gene_CDR3_anchors.csv'),
                              '-set_custom_model',
                              os.path.join(igor_model_dir, 'models', 'model_parms.txt'),
                              os.path.join(igor_model_dir, 'models', 'model_marginals.txt'),
                              '-infer']
        if args.MLSO:
            inf_cmd += ['--MLSO']
        print(' '.join(inf_cmd))
        restarts = args.restarts
        while restarts > 0:
            try:
                run.track(restarts, name='restarts')
                run.track(subsamp, name='subsample')
                subprocess.run(inf_cmd, timeout=args.timeout, check=True, stdout=check.stdout, stderr=check.stderr)
                restarts = -1
            except subprocess.TimeoutExpired:
                print('Hung process, retrying.', restarts)
                restarts -= 1
                if restarts % 3 == 0:
                    subsamp -= 10
                    if '-subsample' not in read_cmd:
                        read_cmd = read_cmd[:-2] + ['-subsample', str(subsamp)] + read_cmd[-2:]
                    else:
                        read_cmd[-3] = str(subsamp)
                    print(' '.join(read_cmd))
                    subprocess.run(read_cmd, check=True, stdout=check.stdout, stderr=check.stderr)
                    print(' '.join(align_cmd))
                    subprocess.run(align_cmd, check=True, stdout=check.stdout, stderr=check.stderr)

        if restarts == 0:
            print('Ran out of restarts!')
        else:
            check.close()

    # Generation.
    check = Checkpoint(args.out, 'generate_{}_{}'.format(fi, igor_batch))
    if check.status():
        generate_cmd = igor_cmd.copy() + [
            '-set_CDR3_anchors',
            '--V', os.path.join(igor_model_dir, 'ref_genome', 'V_gene_CDR3_anchors.csv'),
            '--J', os.path.join(igor_model_dir, 'ref_genome', 'J_gene_CDR3_anchors.csv'),
            '-set_custom_model',
            os.path.join(args.out, igor_batch + '_inference', 'final_parms.txt'),
            os.path.join(args.out, igor_batch + '_inference', 'final_marginals.txt'),
            '-generate', str(seq_num + nonprod_num), '--CDR3',
            '--seed', str(args.seed + fi)]
        print(' '.join(generate_cmd))
        subprocess.run(generate_cmd, check=True, stdout=check.stdout, stderr=check.stderr)
        check.close()
    naive_cdr3_file = os.path.join(args.out, igor_batch + '_generated', 'generated_seqs_werr_CDR3_info.csv')
    # Get number of CDR3s and max length.
    check = Checkpoint(args.out, 'cdr3_stats_{}_{}'.format(fi, igor_batch))
    if check.status():
        naive_num = 0
        max_len_naive = 0
        with open(naive_cdr3_file, 'r') as csvfile:
            # Read rows (generated sequences).
            tr = csv.reader(csvfile, delimiter=',')
            for il, line in enumerate(tr):
                if il == 0:
                    # Header information
                    cdr3_ind = line.index('nt_CDR3')
                    frame_ind = line.index('is_inframe')
                    anchor_ind = line.index('anchors_found')
                else:
                    if int(line[frame_ind]) == 0 or int(line[anchor_ind]) == 0:
                        continue
                    cdr3_full = str(Seq(line[cdr3_ind]).translate())
                    if '*' in cdr3_full:
                        continue
                    naive_num += 1
                    max_len_naive = np.maximum(max_len_naive, int(len(line[cdr3_ind]) / 3) - sum(ncut))
        np.save(os.path.join(args.out, 'naive_num_{}_{}.npy'.format(fi, igor_batch)), naive_num)
        np.save(os.path.join(args.out, 'max_len_naive_{}_{}.npy'.format(fi, igor_batch)), max_len_naive)
        check.close()
    else:
        naive_num = np.load(os.path.join(
                        args.out, 'naive_num_{}_{}.npy'.format(fi, igor_batch)))
        max_len_naive = np.load(os.path.join(
                        args.out, 'max_len_naive_{}_{}.npy'.format(fi, igor_batch)))

    return naive_cdr3_file, naive_num, max_len_naive


class Checkpoint:
    """Checkpoint class for saving/loading preprocessing progress."""
    def __init__(self, out_dir, name='checkpoint'):
        self.out_dir = out_dir
        self.checkpoint_file = os.path.join(out_dir, name) + '.txt'
        self.time = datetime.now()
        self.stdout = open(os.path.join(out_dir, name) + "-out.txt", "a")
        sys.stdout = self.stdout
        self.stderr = open(os.path.join(out_dir, name) + "-err.txt", "a")
        sys.stderr = self.stderr

    def status(self):
        """Check that checkpoint file doesn't exist."""
        return not os.path.exists(self.checkpoint_file)

    def close(self):
        """Save checkpoint file"""
        with open(self.checkpoint_file, 'w') as f:
            f.write('{}'.format(datetime.now() - self.time))

    def save(self, data):
        """Save extra data to checkpoint file"""
        with open(self.checkpoint_file, 'w') as f:
            f.write(json.dumps(data))


def main(args):
    """Preprocess data from Emerson et al. 2017 or Snyder et al. 2020."""
    # Load constants/settings.
    # Number of a.a. to cut from start and end of (productive) CDR3 region.
    ncut = (args.ncut_start, args.ncut_end)

    # Create file to store preprocessed dataset, in hdf5 format.
    os.makedirs(args.out, exist_ok=True)
    processed_file = os.path.join(args.out, 'preprocessed_igor.hdf5')
    run['processed_file'] = processed_file

    # Process metadata.
    if args.datasource == 'emerson':
        # Preprocessing for the Emerson et al. (2017) CMV dataset.
        # Initial setup: make the processed file and instantiate with list of repertoire files.
        check = Checkpoint(args.out, 'setup')
        if check.status():
            processed = h5py.File(processed_file, 'w')
            metadata = processed.create_group('metadata')

            # Make samples files paths.
            directory = os.path.split(args.fileList)[0]
            nsamples, samples_paths = preprocess_emerson.make_samples_paths(directory, args.fileList, metadata)

            check.close()
        else:
            processed = h5py.File(processed_file, 'r+')
            metadata = processed['metadata']
            samples_paths = metadata.attrs['samples_paths'].split(',')
            nsamples = len(samples_paths)

        # Extract HLA types
        check = Checkpoint(args.out, 'hla')
        if check.status():
            hla_dict = preprocess_emerson.extract_hla(metadata, samples_paths)
            check.close()
        else:
            hla_dict = json.loads(metadata.attrs['hla_dict'])

        # Collect all the covariates and outcomes for each patient.
        check = Checkpoint(args.out, 'covariates')
        if check.status():
            preprocess_emerson.make_covariate_table(hla_dict, metadata, nsamples, processed, samples_paths)
            check.close()

    elif args.datasource == 'snyder':
        # Preprocessing for the Snyder et al. (2020) COVID dataset.
        check = Checkpoint(args.out, 'covariates')
        if check.status():
            processed = h5py.File(processed_file, 'w')
            processed_metadata_tsv = os.path.join(args.out, 'preprocessed_metadata.tsv')

            preprocess_snyder.make_covariates_and_paths(
                args.fileList, args.repertoires_folder, processed, processed_metadata_tsv)
            check.close()
        else:
            processed = h5py.File(processed_file, 'r+')

        metadata = processed['metadata']
        samples_paths = metadata.attrs['samples_paths'].split(',')
        nsamples = len(samples_paths)

    # Get the CDR3 max length, productive/nonproductive stats.
    check = Checkpoint(args.out, 'cdr3_stats')
    if check.status():
        max_len_prod, nonprod_nums, seq_nums = get_cdr3s_size(metadata, ncut, samples_paths, run)
        check.close()
    else:
        max_len_prod = metadata.attrs['max_len']
        nonprod_nums = metadata.attrs['nonprod_nums']
        seq_nums = metadata.attrs['seq_nums']

    # --- Construct naive repertoire estimate. ---
    check = Checkpoint(args.out, 'igor')
    if check.status():
        igor_args = zip([ncut]*nsamples, range(nsamples), samples_paths,
                        seq_nums, nonprod_nums)

        if args.nproc == 1:
            # Non-parallelized
            naive_paths, naive_nums, max_len_naives = zip(*map(run_igor, igor_args))
        else:
            # Parallelized
            with multiprocessing.Pool(args.nproc) as pool:
                naive_paths, naive_nums, max_len_naives = zip(*pool.map(run_igor, igor_args))
        naive_nums = np.array(naive_nums, dtype=np.int64)
        max_len_naive = np.max(max_len_naives)
        max_len = np.maximum(max_len_prod, max_len_naive)
        metadata.attrs['naive_nums'] = naive_nums
        metadata.attrs['max_len_naive'] = max_len_naive
        metadata.attrs['max_len'] = max_len
        metadata.attrs['naive_paths'] = ','.join(naive_paths)
        check.close()
    else:
        naive_nums = metadata.attrs['naive_nums']
        max_len = metadata.attrs['max_len']
        naive_paths = metadata.attrs['naive_paths'].split(',')

    # Save repertoires.
    aa_alph_str = 'ACDEFGHIKLMNPQRSTVWY*'
    aa_alph = {aa: aai for aai, aa in enumerate(list(aa_alph_str))}
    metadata.attrs['aa_alphabet'] = aa_alph_str
    # Save post-selection repertoire.
    check = Checkpoint(args.out, 'save_prod')
    if check.status():
        seqs = processed.require_dataset(
            'productive_aa', (sum(seq_nums), max_len + 1), dtype='i')
        freqs = processed.require_dataset(
            'productive_freq', (sum(seq_nums),), dtype=float)
        seq_ind = 0
        for fi, file in enumerate(samples_paths):
            # -- Read productive sequences and save. --
            run.track(fi, name='save_mature_file')
            with open(file, 'r') as csvfile:
                # Read rows (TCR sequences):
                tr = csv.reader(csvfile, delimiter='\t')
                for il, line in enumerate(tr):
                    if il == 0:
                        # Header information.
                        frame_ind = line.index('frame_type')
                        prod_ind = line.index('amino_acid')
                        freq_ind = line.index('productive_frequency')
                    else:
                        if line[frame_ind] == 'In':
                            # Encode variable length CDR3 aa seq as integers.
                            # Trim start (assumed to be C...) and end
                            # (assumed to be ...F)
                            seq = -1 * np.ones(max_len + 1, dtype=np.int64)
                            cdr3 = line[prod_ind][ncut[0]:-ncut[1]] + '*'
                            for si, s in enumerate(list(cdr3)):
                                seq[si] = aa_alph[s]
                            seqs[seq_ind, :] = seq
                            freqs[seq_ind] = float(line[freq_ind])
                            seq_ind += 1
        check.close()

    # Save pre-selection repertoire.
    check = Checkpoint(args.out, 'save_naive')
    if check.status():
        naive_seqs = processed.require_dataset(
            'naive_aa', (sum(naive_nums), max_len + 1), dtype='i')
        seq_ind = 0
        for fi, file in enumerate(naive_paths):
            run.track(fi, name='save_naive_file')
            with open(file, 'r') as csvfile:
                # Read rows (generated sequences).
                tr = csv.reader(csvfile, delimiter=',')
                for il, line in enumerate(tr):
                    if il == 0:
                        # Header information
                        cdr3_ind = line.index('nt_CDR3')
                        frame_ind = line.index('is_inframe')
                        anchor_ind = line.index('anchors_found')
                    else:
                        if int(line[frame_ind]) == 0 or int(line[anchor_ind]) == 0:
                            continue
                        cdr3_full = str(Seq(line[cdr3_ind]).translate())
                        if '*' in cdr3_full:
                            continue
                        seq = -1 * np.ones(max_len + 1, dtype=np.int64)
                        cdr3 = cdr3_full[ncut[0]:-ncut[1]] + '*'
                        for si, s in enumerate(list(cdr3)):
                            seq[si] = aa_alph[s]
                        naive_seqs[seq_ind, :] = seq
                        seq_ind += 1
        check.close()
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__
    processed.close()

    return processed_file


if __name__ == '__main__':
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description='Preprocess Emerson et al. or Snyder et al. datasets, using IGoR to infer naive repertoire.')
    parser.add_argument('fileList', help='File containing master list of patient repertoires.')
    parser.add_argument('igor', help='IGoR executable.')
    parser.add_argument('--out', help='Output folder.')
    parser.add_argument('--MLSO', default=False, action="store_true",
                        help='Run IGoR inference with the MLSO flag (Viterbi like algorithm) ')
    parser.add_argument('--timeout', default=600, type=int,
                        help='Time in seconds before IGoR is considered frozen and is restarted.')
    parser.add_argument('--restarts', default=60, type=int,
                        help='Number of IGoR restarts to perform.')
    parser.add_argument('--nproc', default=1, type=int, help='This option is deprecated.')
    parser.add_argument('--subsample', default=np.inf, type=int,
                        help='Train IGoR on only a subsample of nonproductive sequences, for computational scalability.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed used for IGoR.')
    parser.add_argument('--ncut_start', default=1, type=int)
    parser.add_argument('--ncut_end', default=1, type=int)
    parser.add_argument('--datasource', default='emerson', type=str,
                        help='Currently implemented: emerson OR snyder')
    parser.add_argument('--repertoires-folder', default='.', type=str,
                        help='Folder with repertoires (needed only for the Snyder data).')
    parser.add_argument('--cpu-list', default='0-10', type=str,
                        help='Limit IGoR to use these cpus (with the linux command taskset).')
    args = parser.parse_args()

    # Logging.
    run = create_run('preprocess', args)
    if args.out is None:
        args.out = run['local_dir']

    # Run main code.
    main(args)
