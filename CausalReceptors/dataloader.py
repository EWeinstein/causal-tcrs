import pandas as pd
import h5py
from io import StringIO
import numpy as np
import pandas as pd
import torch
import torch.utils.data as tdata

from CausalReceptors.layers import one_hot


def tots_to_inds(tots):
    """Convert list of sequence totals to list of indices for start/end locations."""
    return torch.cat([torch.tensor([0]), torch.cumsum(tots, 0)], 0)


class RepertoiresDataset(tdata.Dataset):

    def __init__(self, file, outcome_type='binary', flip_outcome=False, mean_subtract=True, seq_batch=10000,
                 uniform_sample_seq=False, generator=None,
                 dtype=torch.float32, cuda_data=False, cuda=False, synthetic_test=False,
                 deterministic_batch=False):

        super().__init__()

        # Deprecated: cuda_data = True.
        if cuda_data:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.outcome_type = outcome_type
        self.flip_outcome = flip_outcome

        # Sequence batching.
        self.seq_batch = seq_batch

        # Load h5py file.
        self.h5f = h5py.File(file, 'r')

        # Load metadata.
        self.tot_patients = self.h5f['metadata'].attrs['nsamples']
        self.aa_alphabet = self.h5f['metadata'].attrs['aa_alphabet']
        self.repertoire_length = self.h5f['metadata'].attrs['max_len'] + 1
        self.repertoire_alphabet = self.aa_alphabet
        self.mature_name = 'productive_aa'
        # Non-productive sequences are generated with IGoR and encoded as a.a.
        self.datatype = 'aa'
        self.naive_mature_alphabet = self.aa_alphabet
        self.naive_mature_length = self.repertoire_length
        self.naive_name = 'naive_aa'

        # Repertoire sizes.
        self.tot_seqs_per_patient = torch.tensor(self.h5f['metadata'].attrs['seq_nums'])
        self.seq_ind = tots_to_inds(self.tot_seqs_per_patient)
        self.mature_counts = None
        if 'mature_counts' in self.h5f:
            # Deprecated option.
            self.mature_counts = torch.tensor(self.h5f['mature_counts'][:], dtype=torch.float32, device=self.device)
            self.mature_totals = torch.zeros(self.tot_patients, dtype=torch.float32, device=self.device)
            for i in range(self.tot_patients):
                self.mature_totals[i] = self.mature_counts[self.seq_ind[i]:self.seq_ind[i+1]].sum()
        elif 'productive_freq' in self.h5f:
            # Use clonotype frequencies for weighting.
            self.mature_counts = torch.tensor(self.h5f['productive_freq'][:], dtype=torch.float32, device=self.device)
            # Use number of clonotypes as a rough guess for the number of cells.
            self.mature_totals = self.tot_seqs_per_patient
        else:
            self.mature_totals = self.tot_seqs_per_patient
        self.tot_naive_per_patient = torch.tensor(self.h5f['metadata'].attrs['naive_nums'])
        self.naive_ind = tots_to_inds(self.tot_naive_per_patient)

        # Convert from hdf5 to tensor in RAM or cuda.
        self.mature_seqs = torch.tensor(self.h5f[self.mature_name][:], dtype=torch.int8, device=self.device)
        self.naive_seqs = torch.tensor(self.h5f[self.naive_name][:], dtype=torch.int8, device=self.device)
        if self.outcome_type == 'binary':
            self.outcomes = one_hot(torch.tensor(self.h5f['outcomes'][:], dtype=torch.long, device=self.device),
                                2, torch.float)
            if flip_outcome:
                assert False, 'Flip outcome not yet implemented for binary outcome.'
        elif self.outcome_type == 'continuous':
            self.outcomes = torch.tensor(self.h5f['outcomes'][:], dtype=torch.float, device=self.device)[:, None]
            # Mean subtract
            if mean_subtract:
                self.outcomes = self.outcomes - self.outcomes.mean()
            # Option to flip sign.
            if flip_outcome:
                self.outcomes = -self.outcomes
            # TODO: option to divide by standard deviation?

        # Set up uniform sampling of repertoire, as opposed to clonotype sampling.
        self.uniform_sample_seq = uniform_sample_seq
        if uniform_sample_seq and (self.mature_counts is not None):
            self.uniform_counts = torch.ones_like(self.mature_counts)
            for i in range(self.tot_patients):
                self.mature_counts[self.seq_ind[i]:self.seq_ind[i+1]] = (
                        self.mature_counts[self.seq_ind[i]:self.seq_ind[i+1]]/
                        self.mature_counts[self.seq_ind[i]:self.seq_ind[i+1]].sum())

        # Pin big data tensors to memory for fast cpu->gpu transfer.
        if cuda and not cuda_data:
            self.mature_seqs.pin_memory()
            self.naive_seqs.pin_memory()
            self.outcomes.pin_memory()

        self.dtype = dtype
        self.batch = torch.tensor([seq_batch], device=self.device)
        self.synthetic_test = synthetic_test
        self.deterministic_batch = deterministic_batch
        self.generator = generator

    def __len__(self):

        return self.tot_patients

    def get_batch(self, seqs, start_indx, end_indx, seq_counts=None, patient_i=0):
        """Subsample a batch of sequences from a repertoire."""
        if self.seq_batch > -1:
            # Take random batch.
            if self.uniform_sample_seq and (seq_counts is not None):
                if not self.deterministic_batch:
                    seq_indx = (torch.multinomial(seq_counts[start_indx:end_indx],
                                                  self.seq_batch, replacement=True) + start_indx
                                ).to(device=self.device)
                else:
                    # Draw fixed, reproducible batch for each patient.
                    rng = torch.Generator(device=self.device).manual_seed(patient_i)
                    seq_indx = (torch.multinomial(seq_counts[start_indx:end_indx],
                                                  self.seq_batch, replacement=True, generator=rng) + start_indx
                                ).to(device=self.device)
            else:
                seq_indx = torch.randint(low=start_indx, high=end_indx, size=(self.seq_batch,),
                                         device=self.device)
        else:
            # Take all sequences.
            seq_indx = slice(start_indx, end_indx)
        if seq_counts is not None:
            if self.uniform_sample_seq:
                return seqs[seq_indx], self.uniform_counts[:len(seq_indx)]
            else:
                return seqs[seq_indx], seq_counts[seq_indx]
        else:
            return seqs[seq_indx], torch.nan

    def __getitem__(self, i):
        """Get patient i's data (mature and naive repertoire, outcome)."""

        # Mature sequences.
        matures, mature_counts = self.get_batch(self.mature_seqs, self.seq_ind[i], self.seq_ind[i+1],
                                                self.mature_counts, patient_i=i)
        tot_matures = self.mature_totals[i]

        # Naive sequences
        naives, _ = self.get_batch(self.naive_seqs, self.naive_ind[i], self.naive_ind[i+1])
        tot_naives = self.naive_ind[i+1] - self.naive_ind[i]

        # Outcomes.
        outcomes = self.outcomes[i]

        if self.synthetic_test:
            # Basic synthetic data test for the outcome model. Inject motifs for positive class.
            if torch.allclose(outcomes[1], torch.tensor(1., device=self.device)):
                matures[:100, 2:6] = torch.tensor([1, 2, 3, 4], dtype=torch.int8, device=self.device)

        return matures, naives, outcomes, tot_matures, tot_naives, mature_counts


class BindingDataset(tdata.Dataset):
    def __init__(self, file, bind_name, cuda=False):
        """Load a dataset of binding information.
        file: input hdf5 file path.
        bind_type: type of binding experiment/group of h5f file. Only current option is 'mira'.
        """
        super().__init__()
        self.device = 'cpu'
        self.bind_type = bind_name

        # Load h5py file.
        self.h5f = h5py.File(file, 'r')

        # Load sequences.
        self.seqs = torch.tensor(self.h5f[bind_name]['seq_aa'][:],
                                 dtype=torch.int8, device=self.device)
        # Load hit/nonhit labels.
        self.hits = torch.tensor(self.h5f[bind_name]['hits'][:],
                                 dtype=torch.int8, device=self.device)

        # Load patient labels.
        self.patients = torch.tensor(self.h5f[bind_name]['patient'][:],
                                     dtype=torch.int8, device=self.device)
        self.num_patients = len(self.h5f[bind_name].attrs['patients'].split(','))

        # Load MHC class labels.
        self.mhc_class = torch.tensor(self.h5f[bind_name]['class'][:],
                                      dtype=torch.int8, device=self.device)

        # Get indices for each class (hits and unlabeled, from corresponding patients).
        self.mhc_class_eval = dict()
        for c in [1, 2]:
            mc_ind = self.mhc_class == c
            patients_ind = torch.zeros_like(mc_ind)
            for j in range(self.num_patients):
                # Check that experiments with MHC class c were run on the patient was run on patients.
                patient_j_ind = self.patients == j
                if torch.sum(patient_j_ind[mc_ind]) > 0:
                    patients_ind += patient_j_ind
            self.mhc_class_eval[c] = mc_ind | ((self.mhc_class == 0) & patients_ind)

        # Load metadata.
        self.nseqs = len(self.hits)

        # Pin big data tensors to memory for fast cpu->gpu transfer.
        if cuda:
            self.seqs.pin_memory()
            self.hits.pin_memory()

    def __len__(self):
        return self.nseqs

    def __getitem__(self, i):
        """Get sequence i info (amino acid sequence and hit label)."""
        return self.seqs[i], self.hits[i]


# Deprecated.
class DeepRCDataset(tdata.Dataset):
    def __init__(self, repertoire_file, outcome_file, seq_batch=10000, generator=None, dtype=torch.float32, cuda_data=False, cuda=False,
                 synthetic_test=False):
        super().__init__()

        if cuda_data:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Sequence batching.
        self.seq_batch = seq_batch

        # Load h5py file.
        self.h5f = h5py.File(repertoire_file, 'r')

        # Load csv file.
        self.outcome_dataset = pd.read_csv(outcome_file, sep='\t', header=0, dtype=str)
        self.cmv_status = self.outcome_dataset['Known CMV status']

        # Confirm indices line up.
        pd_subjects = list(self.outcome_dataset['Subject ID'][:])
        h5_subjects = [el.decode('utf-8').split('.')[0] for el in self.h5f['metadata']['sample_keys'][:]]
        for a, b in zip(h5_subjects, pd_subjects):
            assert a == b, 'File indices do not line up'

        # Load repertoire metadata.
        self.tot_patients = self.h5f['metadata']['n_samples'][()]
        self.aa_alphabet = self.h5f['metadata']['aas'][()].decode('utf-8') + '*'
        self.repertoire_alphabet = self.aa_alphabet

        # Mature repertoire.
        seq_start_end = torch.tensor(self.h5f['sampledata']['sample_sequences_start_end'][:])
        self.tot_seqs_per_patient = seq_start_end[:, 1] - seq_start_end[:, 0]
        self.seq_ind = torch.cat([torch.tensor([0]), torch.cumsum(self.tot_seqs_per_patient, 0)], 0)

        # Repertoire amino acid sequences.
        self.mature_seqs = torch.tensor(self.h5f['sampledata']['amino_acid_sequences'][:], dtype=torch.int8,
                                        device=self.device)
        self.mature_seqs = torch.cat([self.mature_seqs, -torch.ones((self.mature_seqs.shape[0], 1),
                                                                           dtype=torch.int8, device=self.device)], dim=1)
        seq_lens = torch.tensor(self.h5f['sampledata']['seq_lens'][:], dtype=torch.long, device=self.device)
        self.mature_seqs[torch.arange(len(seq_lens), dtype=torch.long), seq_lens] = (
                    torch.tensor(len(self.aa_alphabet) - 1, dtype=torch.int8, device=self.device))
        self.repertoire_length = self.mature_seqs.shape[1]

        # Outcomes
        outcomes = -torch.ones(len(self.cmv_status), dtype=torch.long)
        outcomes += torch.tensor(self.cmv_status == '+') * 2 + torch.tensor(self.cmv_status == '-')
        self.outcomes = one_hot(outcomes.to(device=self.device), 2, torch.float)

        # Counts per sequence.
        self.counts_per_seq = torch.tensor(self.h5f['sampledata']['counts_per_sequence'][:],
                                           dtype=torch.int, device=self.device)

        # Pin big data tensors to memory for fast cpu->gpu transfer.
        if cuda and not cuda_data:
            self.mature_seqs.pin_memory()
            self.outcomes.pin_memory()
            self.counts_per_seq.pin_memory()

        self.dtype = dtype
        self.generator = generator

    def __len__(self):

        return self.tot_patients

    def get_batch(self, seqs, counts, start_indx, end_indx):
        """Subsample a batch of sequences from a repertoire."""
        if self.seq_batch > 0:
            # Take random batch.
            seq_indx = torch.randint(low=start_indx, high=end_indx, size=(self.seq_batch,),
                                     device=self.device)
        else:
            # Take all sequences.
            seq_indx = slice(start_indx, end_indx)
        seq_subset = seqs[seq_indx]
        count_subset = counts[seq_indx]
        return seq_subset, count_subset

    def __getitem__(self, i):
        """Get patient i's data (mature and naive repertoire, outcome)."""

        # Mature sequences.
        matures, seq_counts = self.get_batch(self.mature_seqs, self.counts_per_seq,
                                             self.seq_ind[i], self.seq_ind[i + 1])
        tot_matures = self.seq_ind[i + 1] - self.seq_ind[i]

        # Naive sequences
        naives = matures
        tot_naives = tot_matures

        # Outcomes.
        outcomes = self.outcomes[i]

        return matures, naives, outcomes, tot_matures, tot_naives, seq_counts


class RepertoireTensorDataset(tdata.Dataset):
    """Single repertoire dataset, loaded from tensor (rather than hdf5)."""
    def __init__(self, seqs, seq_ind, confounder=None, seq_nums=None, seq_batch=-1, generator=None, cuda=False):
        super().__init__()

        self.tot_patients = len(seq_ind) - 1
        self.seqs = seqs
        self.seq_ind = seq_ind
        self.confounder = confounder
        self.seq_nums = seq_nums
        self.seq_batch = seq_batch
        self.generator = generator
        if cuda:
            self.seqs.pin_memory()
            if confounder is not None:
                self.confounder.pin_memory()
            if seq_nums is not None:
                self.seq_nums.pin_memory()

    def __len__(self):

        return self.tot_patients

    def __getitem__(self, i):
        """Subsample a batch of sequences from a repertoire."""
        start_indx, end_indx = self.seq_ind[i], self.seq_ind[i+1]
        if self.seq_batch <= 0 or start_indx == end_indx:
            # Take all sequences.
            seq_indx = slice(start_indx, end_indx)
        else:
            # Take random batch of sequences.
            seq_indx = torch.randint(low=start_indx, high=end_indx, size=(self.seq_batch,))

        seq_subset = self.seqs[seq_indx]
        if self.confounder is None and self.seq_nums is None:
            return seq_subset
        else:
            out = [seq_subset]

        if self.confounder is not None:
            subset_confounder = self.confounder[i]
            out.append(subset_confounder)

        if self.seq_nums is not None:
            count_subset = self.seq_nums[seq_indx]
            out.append(count_subset)

        return out


# --- Data features (one-hot encoding, position encoding). ---
class DataFeatures:
    def __init__(self, max_seq_len, alphabet, args):
        """Initialize"""
        # TODO: switch to explicit args
        super().__init__()

        # Save arguments.
        self.args = args
        self.max_seq_len = max_seq_len
        self.alphabet_len = len(alphabet)

        if args.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # CUDA constants.
        self.zero = torch.tensor(0., device=self.device, dtype=self.args.low_dtype)
        self.one = torch.tensor(1., device=self.device, dtype=self.args.low_dtype)
        self.ten = torch.tensor(10., device=self.device, dtype=self.args.low_dtype)

        if self.args.pos_encode:
            # Precompute position encoding features.
            self.pos_features = self._compute_pos_encoding_vectors(max_seq_len)

        # BLOSUM encoding
        if self.args.blosum_encode:
            num_feats = self.alphabet_len + 3
            self.aa_encode = torch.zeros((num_feats, num_feats), dtype=self.args.low_dtype, device=self.device)
            self.aa_encode[:self.alphabet_len, :self.alphabet_len] = torch.tensor(
                    [[blosum[a0][a1] / blosum.std().mean() for a1 in list(alphabet)]
                     for a0 in list(alphabet)], dtype=self.args.low_dtype, device=self.device)

    def _compute_pos_encoding_vectors(self, max_seq_len):
        # Precompute for all possible sequence lengths.
        seq_len = torch.arange(max_seq_len, device=self.device, dtype=self.args.low_dtype).unsqueeze(-1)
        half_seq_len = seq_len / 2
        rang = torch.arange(max_seq_len, device=self.device, dtype=self.args.low_dtype)
        # Start of sequence feature.
        feat1 = torch.nn.functional.relu((half_seq_len - rang) / half_seq_len)
        # End of sequence feature.
        feat2 = torch.nn.functional.relu((rang - half_seq_len + 1) / half_seq_len) * (rang < seq_len)
        # Center of sequence feature.
        feat3 = (1 - torch.nn.functional.relu(torch.abs(rang - half_seq_len) / half_seq_len)) * (rang < seq_len)
        # Complete feature.
        pos_features = torch.cat([feat1.unsqueeze(-1), feat2.unsqueeze(-1), feat3.unsqueeze(-1)], -1)
        # Set sequence length zero to zero, not nan.
        pos_features[0] = self.zero
        return pos_features

    def featurize_seqs(self, seqs, seq_counts=None):
        """
        Convert sequence, encoded as integers, to one-hot encoded with position features, on the gpu.
        Follows __compute_features__ in DeepRC.
        """
        # Transfer to cuda.
        seqs = seqs.to(device=self.device, non_blocking=True).to(dtype=torch.int)
        if self.args.drc_seq_counts:
            # Deprecated.
            seq_counts = seq_counts.to(device=self.device, non_blocking=True).to(dtype=self.args.low_dtype)
        # Compute sequence lengths (using stop symbol position).
        seq_len = seqs.argmax(dim=-1)
        # Allocate full one-hot encoded tensor.
        N, M, _ = seqs.shape
        num_feats = self.alphabet_len + 3 * self.args.pos_encode
        # num_feats + 1 handles entries of seqs that are -1 (missing data). The feature will be dropped later.
        feats = torch.zeros((N, M, self.max_seq_len, num_feats+1), dtype=self.args.low_dtype, device=self.device)
        # Reshape: (N . M . max_seq_len) x num_feats+1
        feats = feats.view((-1, num_feats+1))
        # One hot encode. seqs + 1 handles entries of seqs that are -1 (missing).
        rang = torch.arange(feats.shape[0], device=self.device)
        feats[rang, (seqs+1).view((-1,))] = self.one
        # Drop missing data feature, and reshape to N x M x max_seq_len x num_feats
        feats = feats[..., 1:].view((N, M, self.max_seq_len, num_feats))
        # Encode with BLOSUM.
        if self.args.blosum_encode:
            feats = torch.einsum('nmlj,jk->nmlk', feats, self.aa_encode)
        if self.args.drc_seq_counts:
            # Scale features by sequence counts
            feats = feats * seq_counts[:, :, None, None].abs().log1p()
        if self.args.pos_encode:
            # Add positional encoding.
            feats[..., -3:] = self.pos_features[seq_len]
        # Batch standardization
        if self.args.batch_standardize:
            feats = feats / feats.std()
        elif self.args.approx_batch_standardize:
            # Multiply by typical batch standard deviation observed in preliminary experiments.
            feats = self.ten * feats
        return feats


# BLOSUM50
blosum = pd.read_csv(StringIO("""
   A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  5 -2 -1 -2 -1 -1 -1  0 -2 -1 -2 -1 -1 -3 -1  1  0 -3 -2  0 -2 -1 -1 -5
R -2  7 -1 -2 -4  1  0 -3  0 -4 -3  3 -2 -3 -3 -1 -1 -3 -1 -3 -1  0 -1 -5
N -1 -1  7  2 -2  0  0  0  1 -3 -4  0 -2 -4 -2  1  0 -4 -2 -3  4  0 -1 -5
D -2 -2  2  8 -4  0  2 -1 -1 -4 -4 -1 -4 -5 -1  0 -1 -5 -3 -4  5  1 -1 -5
C -1 -4 -2 -4 13 -3 -3 -3 -3 -2 -2 -3 -2 -2 -4 -1 -1 -5 -3 -1 -3 -3 -2 -5
Q -1  1  0  0 -3  7  2 -2  1 -3 -2  2  0 -4 -1  0 -1 -1 -1 -3  0  4 -1 -5
E -1  0  0  2 -3  2  6 -3  0 -4 -3  1 -2 -3 -1 -1 -1 -3 -2 -3  1  5 -1 -5
G  0 -3  0 -1 -3 -2 -3  8 -2 -4 -4 -2 -3 -4 -2  0 -2 -3 -3 -4 -1 -2 -2 -5
H -2  0  1 -1 -3  1  0 -2 10 -4 -3  0 -1 -1 -2 -1 -2 -3  2 -4  0  0 -1 -5
I -1 -4 -3 -4 -2 -3 -4 -4 -4  5  2 -3  2  0 -3 -3 -1 -3 -1  4 -4 -3 -1 -5
L -2 -3 -4 -4 -2 -2 -3 -4 -3  2  5 -3  3  1 -4 -3 -1 -2 -1  1 -4 -3 -1 -5
K -1  3  0 -1 -3  2  1 -2  0 -3 -3  6 -2 -4 -1  0 -1 -3 -2 -3  0  1 -1 -5
M -1 -2 -2 -4 -2  0 -2 -3 -1  2  3 -2  7  0 -3 -2 -1 -1  0  1 -3 -1 -1 -5
F -3 -3 -4 -5 -2 -4 -3 -4 -1  0  1 -4  0  8 -4 -3 -2  1  4 -1 -4 -4 -2 -5
P -1 -3 -2 -1 -4 -1 -1 -2 -2 -3 -4 -1 -3 -4 10 -1 -1 -4 -3 -3 -2 -1 -2 -5
S  1 -1  1  0 -1  0 -1  0 -1 -3 -3  0 -2 -3 -1  5  2 -4 -2 -2  0  0 -1 -5
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  2  5 -3 -2  0  0 -1  0 -5
W -3 -3 -4 -5 -5 -1 -3 -3 -3 -3 -2 -3 -1  1 -4 -4 -3 15  2 -3 -5 -2 -3 -5
Y -2 -1 -2 -3 -3 -1 -2 -3  2 -1 -1 -2  0  4 -3 -2 -2  2  8 -1 -3 -2 -1 -5
V  0 -3 -3 -4 -1 -3 -3 -4 -4  4  1 -3  1 -1 -3 -2  0 -3 -1  5 -4 -3 -1 -5
B -2 -1  4  5 -3  0  1 -1  0 -4 -4  0 -3 -4 -2  0  0 -5 -3 -4  5  2 -1 -5
Z -1  0  0  1 -3  4  5 -2  0 -3 -3  1 -1 -4 -1  0 -1 -2 -2 -3  2  5 -1 -5
X -1 -1 -1 -1 -2 -1 -1 -2 -1 -1 -1 -1 -1 -2 -2 -1  0 -3 -1 -1 -1 -1 -1 -5
* -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5 -5  1
"""), index_col=0, delimiter=' +', header=0, engine='python')

