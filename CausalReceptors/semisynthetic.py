from aim import Figure, Distribution
import pandas as pd
import argparse
import h5py
import numpy as np
import os
import plotly.express as px
import torch
import torch.distributions as dist
from torch import nn
from torch.utils.data import DataLoader, random_split

from CausalReceptors.manager import create_run
from CausalReceptors.layers import KmerEmbed
from CausalReceptors.dataloader import RepertoiresDataset, RepertoireTensorDataset, tots_to_inds, DataFeatures


class SemisyntheticGenerator:
    def __init__(self, real_data, synthetic, generator, smoke=False):
        # Save input.
        self.generator = generator

        # Copy naive repertoire and metadata.
        real = real_data.h5f
        for elem in ['nsamples', 'aa_alphabet', 'max_len']:
            synthetic['metadata'].attrs[elem] = real['metadata'].attrs[elem]
            if smoke and elem == 'nsamples':
                synthetic['metadata'].attrs[elem] = 8
        self.synthetic = synthetic
        self.nsamples = int(synthetic['metadata'].attrs['nsamples'])
        self.repertoire_length = real_data.repertoire_length
        self.alphabet = real_data.repertoire_alphabet

        # Load naive data.
        self.full_naive_seqs = torch.tensor(real['naive_aa'][:], dtype=torch.int8)
        self.full_tot_naive_per_patient = torch.tensor(real['metadata'].attrs['naive_nums'])
        self.full_naive_ind = tots_to_inds(self.full_tot_naive_per_patient)

    def create_naive(self, causal_motif, patient_frac, causal_seq_frac, confound_motifs, confound_seq_frac,
                     position_start, position_end, aim_run):
        """Inject motif into naive repertoire."""
        # Convert motif dtype to match data.
        causal_motif = causal_motif.to(dtype=torch.int8)
        motif_length = len(causal_motif)
        confound_motifs = confound_motifs.to(dtype=torch.int8).reshape([-1, confound_motifs.shape[-1]])
        nmotifs = 1 + confound_motifs.shape[0]

        # Sample injected patients.
        inject_patients = torch.bernoulli(patient_frac * torch.ones(self.nsamples), generator=self.generator)

        # Take sequences above minimum length
        min_length = motif_length + position_end + 1
        naive_length = self.full_naive_seqs.argmax(dim=-1)
        nseqs = len(naive_length)
        select_length = naive_length > min_length
        aim_run.track({'fraction_sequences_above_min_length': select_length.to(torch.float).mean()})

        # Sample sequences and positions for motifs.
        seq_frac = torch.cat(
                [torch.tensor([1. - causal_seq_frac - (nmotifs-1) * confound_seq_frac, causal_seq_frac]),
                        confound_seq_frac * torch.ones(nmotifs-1)], dim=0)
        add_ind = torch.bitwise_and(dist.OneHotCategorical(probs=seq_frac).sample((nseqs,))[:, 1:].to(torch.bool),
                                    select_length[:, None])
        add_pos = torch.randint(position_start, position_end, (nseqs, nmotifs), generator=self.generator)

        # Iterate over injected patients.
        seqs_sets = [[], []]
        tots_sets = [[], []]
        for i in range(self.nsamples):
            for k in range(nmotifs):
                if k == 0:
                    motif = causal_motif
                else:
                    motif = confound_motifs[k-1]
                # For the causal motif, presence is randomized in the naive repertoire;
                # for confounder motifs, changes come just from selection.
                if inject_patients[i] > 0.5 or k > 0:
                    # Get sequences to inject.
                    rep_ind = slice(self.full_naive_ind[i], self.full_naive_ind[i + 1])
                    inject_seqs_rep = self.full_naive_seqs[rep_ind][add_ind[rep_ind, k]]
                    inject_pos_rep = add_pos[rep_ind, k][add_ind[rep_ind, k]]

                    # Add motif.
                    for j in range(motif_length):
                        inject_seqs_rep[torch.arange(inject_seqs_rep.shape[0]), inject_pos_rep + j] = motif[j]

                    # Put back into full data.
                    self.full_naive_seqs[rep_ind][add_ind[rep_ind, k]] = inject_seqs_rep

                    # Record summary stats.
                    aim_run.track({'naive_injected_seqs_frac_{}'.format(k):
                                   inject_seqs_rep.shape[0]/(self.full_naive_ind[i + 1] - self.full_naive_ind[i])},
                                  step=i)

            # Split sequences into sets used for naive, mature and evaluation
            full_indices = torch.arange(self.full_naive_ind[i], self.full_naive_ind[i + 1])
            subsets = random_split(full_indices, [0.5, 0.5], generator=self.generator)
            for j, subset in enumerate(subsets):
                seqs_sets[j].append(self.full_naive_seqs[full_indices[subset.indices]])
                tots_sets[j].append(len(subset))
            aim_run.track({'naive_seqs': len(full_indices)/2}, step=i)

        # Compile data.
        self.naive_seqs, self.QA_seqs = (torch.cat(s) for s in seqs_sets)
        self.tot_naive_per_patient, self.tot_QA_per_patient = (torch.tensor(t) for t in tots_sets)
        self.naive_ind, self.QA_ind = (tots_to_inds(torch.tensor(t)) for t in tots_sets)

        # Store data.
        self.inject_patients = inject_patients
        self.synthetic.create_dataset('inject_patients', data=self.inject_patients.numpy())
        self.synthetic.create_dataset('naive_aa', data=self.naive_seqs.numpy())
        self.synthetic['metadata'].attrs['naive_nums'] = self.tot_naive_per_patient

        return None

    def create_confounder(self, nconfounders, confound_probability):
        # Sample confounder.
        self.confound = torch.bernoulli(confound_probability * torch.ones([self.nsamples, nconfounders]),
                                        generator=self.generator)

        # Store data.
        self.synthetic.create_dataset('confounder', data=self.confound.numpy())

        # Store half of confounders as observed.
        nobserved = int(nconfounders/2)
        self.synthetic.attrs['covariate_cols'] = ['synthetic_covariate_{}'.format(j) for j in range(nobserved)]
        self.synthetic.create_dataset('covariates', data=self.confound[:, :nobserved].numpy())

        return None

    def create_mature(self, confound_motifs, positive_select_strength, negative_select_strength,
                      aim_run, low_dtype=torch.float32, cuda=False):

        # Initialize dataset, dataloader and featurizer
        repertoire_data = RepertoireTensorDataset(self.QA_seqs, self.QA_ind, self.confound, cuda=cuda)
        repertoire_load = DataLoader(repertoire_data, batch_size=1, shuffle=False, num_workers=1, generator=None,
                                     drop_last=False, persistent_workers=False, pin_memory=cuda)
        data_featurizer = DataFeaturizer(self.repertoire_length, self.alphabet, low_dtype=low_dtype, cuda=cuda)

        # Initialize fitness model.
        synthetic_fitness = SyntheticFitness(self.repertoire_length, self.alphabet, confound_motifs,
                                             positive_select_strength, negative_select_strength,
                                             low_dtype=low_dtype, cuda=cuda)
        one = torch.tensor(1., device=synthetic_fitness.device)

        # Iterate over patients.
        matures = []
        mature_counts = []
        tot_mature_per_patient = []
        selection_naive_means = []
        selection_mature_means = []
        QA_weights = []
        for naives_cpu, confounders_cpu in repertoire_load:
            # Transfer to cuda and featurize data.
            naives = data_featurizer.featurize_seqs(naives_cpu)
            if cuda:
                confounders = confounders_cpu.cuda(non_blocking=True)
            else:
                confounders = confounders_cpu

            # Compute relative fitness and approximate QA.
            QA_weight, QA_positive_selection, QA_negative_selection = synthetic_fitness(naives, confounders)
            QA_weight = QA_weight.squeeze(0).cpu()
            QA_positive_selection, QA_negative_selection = QA_positive_selection.squeeze(0), QA_negative_selection.squeeze(0)
            QA_weights.append(QA_weight)

            # Sample repertoire from QA.
            M = naives.shape[1]
            draws = torch.multinomial(QA_weight, M, replacement=True, generator=self.generator)
            draw_counts = torch.bincount(draws, minlength=M)

            # Take sequences with non-zero counts.
            non_zero = draw_counts > 0
            matures.append(naives_cpu[0, non_zero])
            mature_counts.append(draw_counts[non_zero])
            tot_mature_per_patient.append(matures[-1].shape[0])

            # Record summary stats.
            # Sequences under selection (in naive repertoire).
            selection_naive = (QA_positive_selection + QA_negative_selection).clamp(max=one).cpu()
            selection_naive_mean = selection_naive.mean()
            selection_positive_mean = QA_positive_selection.clamp(max=one).mean().cpu()
            selection_negative_mean = QA_negative_selection.clamp(max=one).mean().cpu()
            # Fraction of mature repertoire arising from sequences under selection.
            selection_mature_mean = (selection_naive * draw_counts).sum() / draw_counts.sum()
            selection_naive_means.append(selection_naive_mean)
            selection_mature_means.append(selection_mature_mean)
            # Fraction of unique mature repertoire arising from sequences under selection.
            selection_unique_mature_mean = selection_naive[non_zero].mean()
            aim_run.track({'mature_generated': tot_mature_per_patient[-1],
                           'fraction_naive_under_selection': selection_naive_mean,
                           'fraction_naive_under_positive_selection': selection_positive_mean,
                           'fraction_naive under negative_selection': selection_negative_mean,
                           'fraction_mature_under_selection': selection_mature_mean,
                           'fraction_unique_mature_under_selection': selection_unique_mature_mean})

        # Store sequences and sequence counts.
        self.QA_weights = torch.cat(QA_weights, 0)
        self.matures = torch.cat(matures, 0)
        self.mature_counts = torch.cat(mature_counts, 0)
        self.tot_mature_per_patient = torch.tensor(tot_mature_per_patient)
        self.mature_ind = tots_to_inds(self.tot_mature_per_patient)
        self.synthetic.create_dataset('productive_aa', data=self.matures.numpy())
        self.synthetic['metadata'].attrs['seq_nums'] = self.tot_mature_per_patient
        self.synthetic.create_dataset('mature_counts', data=self.mature_counts.numpy())

        # Plot key summaries.
        # How much does the confounder drive the mature repertoire?
        for j in range(self.confound.shape[1]):
            selection_naive_table = [float(el) for i, el in enumerate(selection_naive_means)
                                     if np.isclose(self.confound[i].sum(), j)]

            selection_mature_table = [float(el) for i, el in enumerate(selection_mature_means)
                                      if np.isclose(self.confound[i].sum(), j)]

            aim_run.track(Distribution(selection_naive_table),
                          name='selection_naive_confounder_sum_{}'.format(j))
            aim_run.track(Distribution(selection_mature_table),
                          name='selection_mature_confounder_sum_{}'.format(j))

    def create_outcome(self, inject_motif, motif_threshold, motif_effect, confounder_effect, base_effect, outcome_type,
                       aim_run, outcome_noise=0.1, low_dtype=torch.float32, cuda=False):

        # Initialize dataset, dataloader and featurizer
        repertoire_data = RepertoireTensorDataset(self.QA_seqs, self.QA_ind, self.confound, seq_nums=self.QA_weights,
                                                  cuda=cuda)
        repertoire_load = DataLoader(repertoire_data, batch_size=1, shuffle=False, num_workers=1, generator=None,
                                     drop_last=False, persistent_workers=False, pin_memory=cuda)
        data_featurizer = DataFeaturizer(self.repertoire_length, self.alphabet, low_dtype=low_dtype, cuda=cuda)

        # Initialize outcome model.
        synthetic_outcome = SyntheticOutcome(self.repertoire_length, self.alphabet, inject_motif, motif_threshold,
                                             motif_effect, confounder_effect, base_effect,
                                             low_dtype=low_dtype, cuda=cuda)

        # Iterate over patients.
        outcomes = []
        outcome_los, motif_contribs, confound_contribs = [], [], []
        for matures, confounders, matures_count in repertoire_load:
            # Transfer to cuda and featurize data.
            matures = data_featurizer.featurize_seqs(matures)
            if cuda:
                confounders = confounders.cuda(non_blocking=True)
                matures_count = matures_count.cuda(non_blocking=True)

            # Compute outcome log odds.
            outcome_lo, motif_contrib, confound_contrib = synthetic_outcome(matures, matures_count, confounders)
            outcome_lo = outcome_lo.cpu()
            motif_contrib = motif_contrib.cpu()
            confound_contrib = confound_contrib.cpu()

            if outcome_type == 'bernoulli':
                # Binary outcome: transform log odds to probability, draw from Bernoulli.
                outcome = torch.bernoulli(outcome_lo.sigmoid()[0], generator=self.generator)
            elif outcome_type == 'continuous':
                # Continuous outcome: draw from normal.
                outcome = torch.normal(outcome_lo, outcome_noise * torch.ones_like(outcome_lo), generator=self.generator)[0]

            # Record.
            outcomes.append(outcome)
            motif_contribs.append(motif_contrib)
            confound_contribs.append(confound_contrib)
            outcome_los.append(outcome_lo)
            aim_run.track({'outcome_generated': outcome,
                           'outcome_log-odds': outcome_lo[0],
                           'outcome_motif contribution': motif_contrib[0],
                           'outcome_motif_frac': motif_contrib[0]/motif_effect,
                           'outcome_confound_contribution': confound_contrib[0],
                           'injected_motif': self.inject_patients[len(outcomes)-1]},
                          step=len(outcomes)-1)

        # Save full data.
        if outcome_type == 'bernoulli':
            self.outcomes = torch.tensor(outcomes).to(dtype=torch.long)
        elif outcome_type == 'continuous':
            self.outcomes = torch.tensor(outcomes).to(dtype=torch.float32)
        self.synthetic.create_dataset('outcomes', data=self.outcomes.numpy())
        # Save information about the contribution of the causal motif and confounder to the outcome.
        # This is used for efficient computation of the effect we are interested in.
        self.motif_contribs = torch.tensor(motif_contribs)
        self.synthetic.create_dataset('motif_contribs', data=self.motif_contribs.numpy())
        self.confound_contribs = torch.tensor(confound_contribs)
        self.synthetic.create_dataset('confound_contribs', data=self.motif_contribs.numpy())

        # Plot key summaries.
        aim_run.track({'outcome_mean': torch.tensor(outcomes).mean().cpu().numpy()})
        # Is the range of outcome log odds reasonable?
        outcome_los_table = [float(el) for el in outcome_los]
        aim_run.track(Distribution(outcome_los_table), name='outcome_log_odds')
        # How much does the causal motif drive the outcome log odds?
        motif_contrib_present = [float(torch.log1p(mc)) for mc, ip in zip(motif_contribs, self.inject_patients)
                                 if ip > 0.5]
        motif_contrib_absent = [float(torch.log1p(mc)) for mc, ip in zip(motif_contribs, self.inject_patients)
                                if ip < 0.5]
        aim_run.track(Distribution(motif_contrib_present), name='motif_contrib_present_log1p')
        aim_run.track(Distribution(motif_contrib_absent), name='motif_contrib_absent_log1p')


class SyntheticFitness(nn.Module):
    """Compute relative fitness of each sequence in the naive repertoire."""
    def __init__(self, repertoire_length, alphabet, confound_motifs, positive_select_strength, negative_select_strength,
                 low_dtype=torch.float32, cuda=False):
        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.low_dtype = low_dtype

        # Kmer extraction.
        n_confound, n_motif_per = confound_motifs.shape[0], confound_motifs.shape[1]
        confound_motifs_flat = confound_motifs.reshape([-1, confound_motifs.shape[2]]
                                                       ).to(dtype=low_dtype, device=self.device)
        n_motifs = confound_motifs_flat.shape[0]
        self.kmer_embed = KmerEmbed(repertoire_length, alphabet, confound_motifs.shape[2],
                                    custom_kmers=confound_motifs_flat, dtype=low_dtype, cuda=cuda)
        self.confound_mask = torch.zeros((n_confound, n_motifs), dtype=low_dtype, device=self.device)
        for j in range(n_confound):
            self.confound_mask[j, (n_motif_per*j):(n_motif_per*(j+1))] = (
                    torch.tensor(1., dtype=low_dtype, device=self.device))

        # Selection strength.
        self.positive_select_strength = positive_select_strength
        self.negative_select_strength = negative_select_strength

        # Constants.
        self.one = torch.tensor(1., dtype=torch.float32, device=self.device)

    def forward(self, repertoires, confounds):
        # Evaluate presence/absence of each kmer in each sequence. Out: N x M x n_kmers
        embed_seq = self.kmer_embed(repertoires).clamp(max=self.one)

        # Compute kmers under selection. Out: N x 1 x n_kmers
        kmer_positive_select = torch.sum(confounds[:, :, None] * self.confound_mask[None, :, :], dim=1, keepdim=True)
        kmer_negative_select = torch.sum((1 - confounds[:, :, None]) * self.confound_mask[None, :, :], dim=1, keepdim=True)

        # Compute relative fitness of each sequence.
        seq_positive_selection = (embed_seq * kmer_positive_select).sum(dim=-1).to(torch.float32)
        seq_negative_selection = (embed_seq * kmer_negative_select).sum(dim=-1).to(torch.float32)
        fitness = (self.positive_select_strength * seq_positive_selection +
                   self.negative_select_strength * seq_negative_selection)

        # Compute growth fraction.
        growth = torch.softmax(fitness, dim=-1)

        return growth, seq_positive_selection, seq_negative_selection


class SyntheticOutcome(nn.Module):
    """Generate outcome from mature repertoire."""
    def __init__(self, repertoire_length, alphabet, inject_motif, motif_threshold,
                 motif_effect, confounder_effect, base_effect,
                 low_dtype=torch.float32, cuda=False):
        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.low_dtype = low_dtype

        # Kmer extraction.
        inject_motif_flat = inject_motif[None, :].to(dtype=low_dtype, device=self.device)
        self.kmer_embed = KmerEmbed(repertoire_length, alphabet, len(inject_motif), custom_kmers=inject_motif_flat,
                                    dtype=low_dtype, cuda=cuda)
        self.motif_threshold = torch.tensor(motif_threshold, dtype=torch.float32, device=self.device)

        # Coefficients.
        self.motif_effect = torch.tensor(motif_effect, dtype=torch.float32, device=self.device)
        self.confounder_effect = torch.tensor(confounder_effect, dtype=torch.float32, device=self.device)
        self.base_effect = torch.tensor(base_effect, dtype=torch.float32, device=self.device)

        # Constants.
        self.one = torch.tensor(1., dtype=torch.float32, device=self.device)

    def forward(self, repertoires, seq_counts, confounds):
        # Evaluate presence/absence of the causal motif. Out: N x M x 1
        embed_seq = self.kmer_embed(repertoires).clamp(max=self.one).to(torch.float32)

        # Compute expected motif burden. Out: N
        expect_motif = (seq_counts * embed_seq.squeeze(-1)).sum(dim=1) / seq_counts.sum(dim=1)

        # Check if motif burden above threshold.
        presence_motif = expect_motif > self.motif_threshold

        # Compute confounder burden. Out: N
        expect_confound = confounds.sum(dim=1)

        # Compute log odds of outcome.
        motif_contrib = self.motif_effect * presence_motif
        confound_contrib = self.confounder_effect * expect_confound
        lo = motif_contrib + confound_contrib + self.base_effect

        return lo, motif_contrib, confound_contrib


class InterventionEffect(nn.Module):
    """Compute the effect of a soft intervention that adds a sequence to each patient's repertoire."""
    def __init__(self, repertoire_length, alphabet, intervene_frac, inject_motif,
                 motif_effect, base_effect, motif_contribs, confound_contribs,
                 outcome_type, low_dtype=torch.float32, cuda=False):

        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.low_dtype = low_dtype
        self.outcome_type = outcome_type

        # Kmer extraction.
        inject_motif_flat = inject_motif[None, :].to(dtype=low_dtype, device=self.device)
        self.kmer_embed = KmerEmbed(repertoire_length, alphabet, len(inject_motif), custom_kmers=inject_motif_flat,
                                    dtype=low_dtype, cuda=cuda)

        # Intervention contribution.
        self.motif_effect = torch.tensor(motif_effect, dtype=torch.float32, device=self.device)
        self.intervene_frac = torch.tensor(intervene_frac, dtype=torch.float32, device=self.device)

        # Population distribution over natural (non-intervened sequences) contribution.
        self.natural_contribs = ((1 - intervene_frac) * motif_contribs + confound_contribs + base_effect
                                 ).to(dtype=torch.float32, device=self.device)

        # Constants.
        self.one = torch.tensor(1., dtype=torch.float32, device=self.device)

    def forward(self, candidates):
        # Evaluate presence/absence of the causal motif. In: cN x cM x L x B. Out: cN x cM x 1
        embed_seq = self.kmer_embed(candidates).clamp(max=self.one).to(torch.float32)

        # Compute contribution of added sequence. Out: cN x cM
        candidate_contrib = embed_seq.squeeze(2) * self.motif_effect

        # Compute log odds. Out: (cN*cM) x N
        lo = candidate_contrib.reshape([-1])[:, None] + self.natural_contribs[None, :]

        # Compute effect. Out: (cN*cM)
        if self.outcome_type == 'binary':
            effect = lo.sigmoid().mean(dim=1)
        elif self.outcome_type == 'continuous':
            effect = lo.mean(dim=1)

        return effect


class DataFeaturizer(DataFeatures):
    """One hot encode data from a single repertoire."""
    def __init__(self, repertoire_length, repertoire_alphabet, low_dtype=torch.float32, cuda=False):

        class DFArgs:
            def __init__(self):

                self.pos_encode = False
                self.batch_standardize = False
                self.approx_batch_standardize = False
                self.blosum_encode = False
                self.drc_seq_counts = False
                self.low_dtype = low_dtype
                self.cuda = cuda

        super().__init__(repertoire_length, repertoire_alphabet, DFArgs())



class KmerCounter(nn.Module):
    def __init__(self, args, data):
        super().__init__()

        # Save arguments.
        self.args = args

        if args.cuda:
            self.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Kmer k.
        self.kmer_length = args.motif_length

        # Data featurizer - one-hot encode.
        self.data_featurizer = DataFeaturizer(data.repertoire_length, data.repertoire_alphabet,
                                              low_dtype=args.low_dtype, cuda=args.cuda)

        self.kmer_embed = KmerEmbed(data.repertoire_length, data.repertoire_alphabet, self.kmer_length,
                                    dtype=args.low_dtype, cuda=args.cuda)
        self.n_kmers = self.kmer_embed.n_kmers

    def forward(self, repertoires):
        # Embed repertoire sequences.
        embed_seq = self.kmer_embed(repertoires)

        # Embed repertoire.
        embed_rep = embed_seq.sum(dim=1).to(dtype=torch.float32)

        return embed_rep


def extract_kmers(real_data, args, aim_run):
    """Extract kmer counts from dataset."""
    # Set up dataloader.
    dataload = DataLoader(real_data, batch_size=args.unit_batch, shuffle=False,
                          num_workers=args.num_workers, generator=None, drop_last=False,
                          persistent_workers=False, pin_memory=args.cuda)

    # Set up counter.
    kmercounter = KmerCounter(args, real_data)
    kmer_sums = torch.zeros(kmercounter.n_kmers, device=kmercounter.device)
    step_i = 0
    # Count kmers, in mini batches.
    for matures, naives, outcomes, matures_per, naives_per, seq_counts in dataload:
        # Transfer to cuda and featurize data.
        naives = kmercounter.data_featurizer.featurize_seqs(naives)
        kmer_sums += kmercounter(naives).sum(dim=0)
        step_i += 1
        aim_run.track({'counting_kmers_step': step_i})
        if args.smoke:
            break

    # Log results.
    kmers = kmercounter.kmer_embed.kmers.cpu()
    kmer_counts = kmer_sums.cpu()
    kmer_sum_sorted_table = pd.DataFrame([[j, el] for j, el in enumerate(np.sort(kmer_counts.numpy()))],
                                        columns=['ind', 'counts'])
    kmer_sum_sorted_table.to_csv(os.path.join(aim_run['local_dir'], 'kmer_sum_sorted_table.csv'), index=False)
    plot_sorted_kmer_counts = px.line(data_frame=kmer_sum_sorted_table, x='ind', y='counts', title='sorted kmer counts')
    aim_run.track(Figure(plot_sorted_kmer_counts), name='sorted_kmer_counts')

    return kmers, kmer_counts


def main(args, aim_run):
    # -- Setup --
    # Load data.
    data_gen_M = torch.Generator().manual_seed(args.data_seed)
    real_data = RepertoiresDataset(args.datafile, seq_batch=args.subunit_batch, dtype=args.low_dtype,
                                   generator=data_gen_M, cuda_data=False, cuda=args.cuda, synthetic_test=False)

    # Extract kmers.
    print('Counting kmers...')
    kmers, kmer_counts = extract_kmers(real_data, args, aim_run)
    print('Done.')

    # Set up random number generator for simulations.
    rng = torch.Generator().manual_seed(args.seed)

    # Choose from among rare-ish motifs.
    print('Selecting motifs...')
    rareish_motifs_ind = torch.bitwise_and(kmer_counts >= torch.quantile(kmer_counts, args.motif_lower_quantile),
                                           kmer_counts < torch.quantile(kmer_counts, args.motif_upper_quantile))
    rareish_motifs = kmers[rareish_motifs_ind]
    random_motifs_ind = torch.randperm(rareish_motifs.shape[0], generator=rng)
    # Select causal motif.
    causal_motif = rareish_motifs[random_motifs_ind[0]]
    print(causal_motif)
    # Select motifs for confounding.
    confound_motifs = rareish_motifs[random_motifs_ind[1:(1 + args.nconfounders * args.confounder_motifs)]
                                     ].reshape([args.nconfounders, args.confounder_motifs, args.motif_length])
    print('Done.')

    # -- Simulate --
    # Create output file.
    synthetic = h5py.File(args.out, 'w')
    # Label as synthetic.
    synth_metadata = synthetic.create_group('metadata')
    synth_metadata.attrs['synthetic'] = True
    # Store info about simulation.
    synth_metadata.attrs['naive_source'] = args.datafile
    synth_metadata.attrs['causal_motif'] = causal_motif.numpy()
    synth_metadata.attrs['confound_motifs'] = confound_motifs.numpy()
    for ke in args.__dict__:
        if ke == 'low_dtype':
            continue
        synth_metadata.attrs['synthetic_' + ke] = args.__dict__[ke]


    # Create semisynthetic data.
    print('Initializing synthetic data...')
    semisynth_generate = SemisyntheticGenerator(real_data, synthetic, rng, smoke=args.smoke)

    # Create naive repertoire with injected motif.
    print('Simulating naive repertoire...')
    semisynth_generate.create_naive(causal_motif, args.inject_patient_frac, args.inject_seq_frac, confound_motifs,
                                    args.confound_seq_frac, args.inject_position_start, args.inject_position_end,
                                    aim_run)
    print('Done.')

    # Simulate confounder.
    print('Simulating confounder...')
    semisynth_generate.create_confounder(args.nconfounders, args.confound_probability)
    print('Done.')

    # Simulate mature repertoire.
    print('Simulating mature repertoire...')
    semisynth_generate.create_mature(confound_motifs, args.positive_select_strength, args.negative_select_strength,
                                     aim_run, low_dtype=args.low_dtype, cuda=args.cuda)
    print('Done.')

    # Simulate outcome.
    print('Simulating outcome...')
    semisynth_generate.create_outcome(causal_motif, args.inject_seq_frac/2,
                                      args.motif_effect, args.confounder_effect, args.base_effect,
                                      args.outcome, aim_run,
                                      outcome_noise=args.outcome_noise, low_dtype=args.low_dtype, cuda=args.cuda)
    print('Done.')

    # Close file.
    synthetic.close()


class DefaultArgs:
    def __init__(self):
        # Simulation parameters.
        self.inject_patient_frac = 0.4
        self.inject_seq_frac = 0.01
        self.confound_seq_frac = 0.005
        self.motif_length = 3
        self.inject_position_start = 2  # TODO: revise w/ncut.
        self.inject_position_end = 4
        self.nconfounders = 2
        self.confounder_motifs = 10
        self.confound_probability = 0.4
        self.positive_select_strength = 2.0
        self.negative_select_strength = -2.0
        self.motif_effect = 2 / self.inject_seq_frac
        self.confounder_effect = 2
        self.base_effect = -3.0
        self.outcome = 'binary'
        self.outcome_noise = 0.1

        # Choosing motifs.
        self.motif_lower_quantile = 0.1
        self.motif_upper_quantile = 0.2

        # Evaluation settings.
        self.intervene_frac = 0.1

        # System settings.
        self.out = ''
        self.unit_batch = 1
        self.subunit_batch = -1
        self.unit_batch_eval = 1
        self.subunit_batch_eval = 100
        self.data_seed = 1
        self.seed = 2
        self.num_workers = 4
        self.low_dtype = 'bfloat16'
        self.cuda = False
        self.smoke = False

if __name__ == "__main__":
    # Load default arguments.
    defaults = DefaultArgs()
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Semisynthetic data for evaluating causal immune receptor inference.")
    parser.add_argument('datafile', help='Input data, with repertoires to be injected with synthetic motifs.')
    parser.add_argument('--out', default=defaults.out, help='Output file with semisynthetic data.')
    parser.add_argument('--inject-patient-frac', default=defaults.inject_patient_frac, type=float,
                        help='Per patient probability of injecting the motif.')
    parser.add_argument('--inject-seq-frac', default=defaults.inject_seq_frac, type=float,
                        help='Per sequence probability of injecting the causal motif (within patients with the motif).')
    parser.add_argument('--confound-seq-frac', default=defaults.confound_seq_frac, type=float,
                        help='Per sequence probability of injecting a confounder motif.')
    parser.add_argument('--motif-length', default=defaults.motif_length, type=int,
                        help='Length of motifs to inject and select.')
    parser.add_argument('--inject-position-start', default=defaults.inject_position_start, type=int,
                        help='Starting position where motif can be injected.')
    parser.add_argument('--inject-position-end', default=defaults.inject_position_end, type=int,
                        help='Ending position where motif can be injected.')
    parser.add_argument('--confounder-motifs', default=defaults.confounder_motifs, type=int,
                        help='Number of motifs under selection, per confounder.')
    parser.add_argument('--nconfounders', default=defaults.nconfounders, type=int,
                        help='Number of confounders.')
    parser.add_argument('--confound-probability', default=defaults.confound_probability, type=float,
                        help='Probability each confounder is present in each patient.')
    parser.add_argument('--positive-select-strength', default=defaults.positive_select_strength, type=float,
                        help='Log relative fitness of sequences under positive selection from confounder.')
    parser.add_argument('--negative-select-strength', default=defaults.negative_select_strength, type=float,
                        help='Log relative fitness of sequences under negative selection from confounder.')
    parser.add_argument('--motif-effect', default=defaults.motif_effect, type=float,
                        help='Effect of average motif prevalence on outcome.')
    parser.add_argument('--confounder-effect', default=defaults.confounder_effect, type=float,
                        help='Effect of each confounder on outcome.')
    parser.add_argument('--base-effect', default=defaults.base_effect, type=float,
                        help='Offset for outcome model.')
    parser.add_argument('--outcome', default=defaults.outcome, help='Options: binary OR continuous')
    parser.add_argument('--outcome-noise', default=defaults.outcome_noise, type=float,
                        help='If outcome is continuous, this sets the std of the noise.')
    parser.add_argument('--motif-lower-quantile', default=defaults.motif_lower_quantile, type=float,
                        help='Choose motifs from above this quantile.')
    parser.add_argument('--motif-upper-quantile', default=defaults.motif_upper_quantile, type=float,
                        help='Choose motifs from below this quantile.')
    parser.add_argument('--intervene-frac', default=defaults.intervene_frac, type=float,
                        help='Fraction of repertoire with intervened (candidate therapeutic) sequence.')

    parser.add_argument('--unit-batch', default=defaults.unit_batch, type=int)
    parser.add_argument('--subunit-batch', default=defaults.subunit_batch, type=int)
    parser.add_argument('--unit-batch-eval', default=defaults.unit_batch_eval, type=int)
    parser.add_argument('--subunit-batch-eval', default=defaults.subunit_batch_eval, type=int)
    parser.add_argument('--seed', default=defaults.seed, type=int,
                        help='Random seed for generating synthetic data.')
    parser.add_argument('--data-seed', default=defaults.data_seed, type=int,
                        help='Random seed for subsetting real data.')
    parser.add_argument('--num-workers', default=defaults.num_workers, type=int,
                        help='Number of workers to use in data loading (should only be > 0 if using a gpu).')
    parser.add_argument('--low-dtype', default=defaults.low_dtype, type=str,
                        help='Lower precision dtype (float32, float16 OR bfloat16.')
    parser.add_argument('--cuda', default=defaults.cuda, action="store_true")
    parser.add_argument('--smoke', default=defaults.smoke, action="store_true")
    args0 = parser.parse_args()

    # Set up aim logging.
    aim_run0 = create_run('cir-sim', args0)
    if args0.out == '':
        args0.out = f"{aim_run0['local_dir']}/generated.hdf5"

    # Use gpu w/most available memory.
    free_mem = np.zeros(torch.cuda.device_count())
    for j in range(torch.cuda.device_count()):
        with torch.cuda.device(j):
            free_mem[j] = torch.cuda.mem_get_info()[0]
    torch.cuda.set_device(int(np.argmax(free_mem)))

    # Set pytorch defaults and backend.
    torch.set_default_dtype(torch.float32)
    args0.low_dtype = getattr(torch, args0.low_dtype)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Run simulation.
    main(args0, aim_run0)

    # Print file name (for experiment management).
    print('GeneratedData:', args0.out)