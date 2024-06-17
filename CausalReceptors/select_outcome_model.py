"""
Causal adaptive immune repertoire estimation (CAIRE).

This is the main CAIRE model. It takes in a dataset of immune repertoires and patient outcomes,
and learns a model of the causal effects of adding an individual sequence to the repertoire.

Code organization, and corresponding notation in the paper:
 - Featurizer: Neural network module that extracts sequence features for predicting maturity.
   (notation: h_r)
 - RepertoireFeaturizer: Neural network module that extracts repertoire features for predicting outcome.
   (notation: E[h_a(A;theta)])
 - Encoder: Neural network module used for amortized inference of fitness representations.
   (notation: enc)
 - InterventionEffect: Module used to compute intervention effects, based on the trained model.
   (notation: ATE)
 - CausalRepertoireModel: The main CAIRE model. It includes the complete Pyro model, as well as functions for
   training and evaluation.
 - main: core program flow, implementing setup, training and evaluation.

Additional comments:

In descriptions, we generally use the notation:
N - number of patients (aka "units").
M - number of repertoire sequences per patient (aka "subunits"; this is usually a batch size).
L - max sequence length
D - number of sequence features (corresponding to the amino acid alphabet, plus e.g. position features).
In variable names, we use the term "naive" to refer to the preselection repertoire.
"""
from aim import Figure, Distribution
import argparse
from collections import defaultdict
from datetime import datetime
import numpy as np
import os
import pandas as pd
import plotly.express as px
import pickle
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam, NAdam, SGD
import torch
from torch import nn
from torch.nn.functional import softplus
from torch.utils.data import DataLoader, random_split, Subset, ConcatDataset
from sklearn.linear_model import LogisticRegression

from CausalReceptors.distributions import MissingDataOneHotCategorical, Normal
from CausalReceptors.layers import SeqEmbed, AttentionReduce
from CausalReceptors.dataloader import RepertoiresDataset, DataFeatures, BindingDataset
from CausalReceptors.metrics import roc_auc_score, r2_score, accuracy, pearson_score, average_precision_score, explained_variance_score, r_pvalue
from CausalReceptors.elbos import CudaJitTrace_ELBO, CudaSVI

from CausalReceptors.manager import create_run

from CausalReceptors.semisynthetic import InterventionEffect as InterventionEffectTrue
from CausalReceptors.semisynthetic import KmerEmbed


# --- Featurizer ---
class Featurizer(nn.Module):
    """Extract sequence features.
    This module extracts features from a collection of preselection and mature sequences,
    which are used as part of the relative fitness model.
    Input: preselection and mature sequences (2(N x M x L x D))
    Output: sequence features (2(N x M x features))
    Neural network layer: SeqEmbed
    """
    def __init__(self, num_length, alphabet, args):

        super().__init__()

        # Move module to GPU.
        if args.cuda:
            self.cuda()
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # CNN layer
        self.seq_embed = SeqEmbed(
            num_length, alphabet, args.selection_channels,
            conv_kernel=args.selection_conv_kernel, architecture='cnn',
            pos_encode=args.pos_encode, sum_pool=args.sum_pool, linear_cnn=args.linear_cnn,
            no_pool=False, dtype=args.low_dtype, cuda=args.cuda)
        self.seq_embed.to(dtype=args.low_dtype)

        # Feedforward layer.
        ff = []
        n_input_features = args.selection_channels
        n_output_features = args.n_selection_units
        self.n_selection_layers = args.n_selection_layers
        for layer_i in range(args.n_selection_layers):
            if layer_i == args.n_selection_layers - 1:
                n_output_features = args.select_latent_dim
            linear = nn.Linear(n_input_features, n_output_features)
            linear.weight.data.normal_(0., np.sqrt(1 / np.prod(linear.weight.shape)))
            ff.append(linear)
            if layer_i < args.n_selection_layers - 1:
                ff.append(nn.SELU())
            n_input_features = args.n_selection_units
        self.ff = torch.nn.Sequential(*ff)
        if args.cuda:
            self.ff.cuda()

    def forward(self, matures, naives):
        # Extract sequence features.
        features = torch.cat([self.seq_embed(matures), self.seq_embed(naives)], 1).to(dtype=torch.float32)

        # Feed forward.
        features = self.ff(features)

        return features


# --- Repertoire featurizer ---
class RepertoireFeaturizer(nn.Module):
    """Extract repertoire features.
    This module extracts repertoire-level features from mature repertoires.
    Input: repertoire sequences (N x M x L x D)
    Output: repertoire embedding (N x repertoire_latent_dim)

    Layers:
    SeqEmbed: N x M x L x D -> N x M x repertoire_latent_dim
    attention: N x M x embed -> N x repertoire_latent_dim

    """

    def __init__(self, num_length, alphabet, args):
        """Initialization.
        Inputs
        num_lengths - padded sequence length
        alphabet - str
        """
        super().__init__()
        self.args = args
        if args.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Sequence embedding.
        self.seq_embed = SeqEmbed(
                num_length, alphabet, args.repertoire_latent_dim,
                conv_kernel=args.conv_kernel, architecture='cnn',
                pos_encode=args.pos_encode, sum_pool=args.sum_pool,
                dtype=args.low_dtype, cuda=args.cuda)
        self.seq_embed.to(dtype=args.low_dtype)

        # Repertoire embedding.
        self.repertoire_reduce = AttentionReduce(args.repertoire_latent_dim, args.n_attention_layers,
                                                 args.n_attention_units,
                                                 no_attention=args.no_attention,
                                                 top_fraction=args.top_fraction, use_counts=True, cuda=args.cuda)

        if args.cuda:
            self.cuda()

    def forward(self, repertoires, counts):

        # Embed repertoire sequences.
        embed = self.seq_embed(repertoires).to(dtype=torch.float32)
        N, M, ed = embed.shape

        # Embed entire repertoires.
        enc_hidden, normalizer_ln = self.repertoire_reduce(embed, counts)

        return enc_hidden, normalizer_ln


# --- Encoder ---
class Encoder(nn.Module):
    """Encoder.
    This module is used for amortized inference of the relative fitness model. It takes in
    preselection and mature repertoire sequences, and outputs parameters for a distribution
    over the fitness representation rho_i and the additional local latent variable beta_i.

    Input: preselection and mature sequences (2(N x M x L x D)).
    Output: fitness representation variational parameters (rho_mn, rho_sd, beta_mn, beta_sd)
    (Note when the model is trained using a point estimate, rho_sd and beta_sd will not be used.)

    Layers:
    latent_size = (2 (select_latent_dim + 1))
    SeqEmbed: 2(N x M x L x D) -> 2(N x M x latent_size)
    AttentionReduce: -> 2(N x latent_size)
    multiply by +1 for mature and -1 for naive, sum: -> N x latent_size
    split & softplus: -> N x select_latent_dim, N x select_latent_dim, N x 1, N x 1
    """

    def __init__(self, num_length, alphabet, max_subunits, args):
        super().__init__()
        self.args = args
        if args.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Sequence embedding.
        self.seq_embed = SeqEmbed(
            num_length, alphabet, args.encoder_channels,
            conv_kernel=args.encoder_conv_kernel, architecture='cnn', pos_encode=args.pos_encode,
            no_pool=False, sum_pool=args.sum_pool, dtype=args.low_dtype, cuda=args.cuda)
        self.seq_embed.to(dtype=args.low_dtype)

        # Naive counts.
        # Warning: update this if switch to using only unique sequences for naive.
        self.naive_counts = torch.ones(args.unit_batch, max_subunits, device=self.device, dtype=torch.float32)

        # Repertoire embedding.
        self.repertoire_reduce = AttentionReduce(
            args.encoder_channels, args.n_encoder_attention_layers, args.n_encoder_attention_units,
            no_attention=args.encoder_no_attention,
            top_fraction=args.encoder_top_fraction, use_counts=True, cuda=args.cuda)
        # Feedforward.
        layers = []
        layers.append(nn.Linear(args.encoder_channels, args.select_embed_layer_dim))
        layers[-1].weight.data.normal_(0.0, np.sqrt(1 / np.prod(layers[-1].weight.shape)))
        layers.append(nn.SELU())
        layers.append(nn.Linear(args.select_embed_layer_dim, 2 * args.select_latent_dim + 2))
        layers[-1].weight.data.normal_(0.0, np.sqrt(1 / np.prod(layers[-1].weight.shape)))
        self.layers = nn.Sequential(*layers)

        if args.cuda:
            self.cuda()

    def forward(self, matures, mature_counts, naives):

        # Dimensions.
        N, Mn = matures.shape[0], naives.shape[1]

        # Embed mature repertoire.
        matures_embed = self.seq_embed(matures)
        mature_rep = self.repertoire_reduce(matures_embed.to(dtype=torch.float32), mature_counts)[0]

        # Embed naive repertoire.
        naives_embed = self.seq_embed(naives)
        naive_rep = self.repertoire_reduce(naives_embed.to(dtype=torch.float32), self.naive_counts[:N, :Mn])[0]

        # Full patient embedding.
        rep = mature_rep - naive_rep
        out = self.layers(rep)

        # Split into locations and scales (rho_mn, rho_sd, beta_mn, beta_sd)
        rho_mn = out[:, None, :self.args.select_latent_dim]
        rho_sd = nn.functional.softplus(out[:, None, self.args.select_latent_dim:(2 * self.args.select_latent_dim)])
        beta_mn = out[:, None, 2 * self.args.select_latent_dim]
        beta_sd = nn.functional.softplus(out[:, None, 2 * self.args.select_latent_dim + 1])
        return rho_mn, rho_sd, beta_mn, beta_sd


class InterventionEffect(nn.Module):
    """Interventional effect.

    This module compute the effect of an intervention that adds a sequence to each patient's repertoire.

    Input: Candidate sequences (cN x cM x L x D).
    Output: Effects (cN*cM)
    The candidate sequences take the same shape as repertoires, but this is only for convenience; there is
    nothing distinguishing the first two dimensions. The output is a vector of length cN*cM.

    Note: intervene_frac is notated epsilon in the paper; it is the fraction of the intervened repertoire that
    is the added sequence. It is provided at initialization.
    """
    def __init__(self, intervene_frac, treatment_coeff, treatment_contribs, treatment_norms_ln, confound_contribs,
                 base_contribs, outcome_type, max_batch=1, repertoire_featurizer=None, no_attention=False,
                 low_dtype=torch.float32, cuda=False):

        super().__init__()
        if cuda:
            self.cuda()
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.low_dtype = low_dtype
        self.outcome_type = outcome_type

        # Confounder and baseline contributions.
        self.natural_contribs = (confound_contribs + base_contribs).to(dtype=torch.float32, device=self.device)

        # Observed treatment's contributions.
        self.treatment_contribs = treatment_contribs.to(dtype=torch.float32, device=self.device)
        self.treatment_norms_ln = treatment_norms_ln.to(dtype=torch.float32, device=self.device)

        # Model coefficients (W_A)
        self.treatment_coeff = treatment_coeff.to(dtype=torch.float32, device=self.device)

        # Fraction of repertoire consisting of intervened sequence.
        self.intervene_frac = torch.tensor(intervene_frac, dtype=torch.float32, device=self.device)
        self.intervene_fracs_ln = torch.tensor([1 - self.intervene_frac, self.intervene_frac],
                                               dtype=torch.float32, device=self.device).log()

        # Repertoire low-dimensional features.
        if repertoire_featurizer is not None:
            self.repertoire_featurizer = repertoire_featurizer
        else:
            assert False, 'Need to implement model loading.'
        self.no_attention = no_attention

        # Constants.
        self.one = torch.ones(max_batch, dtype=torch.float32, device=self.device)
        self.N = self.treatment_contribs.shape[0]

    def forward(self, candidates):
        # Size of tensor of candidate intervention sequences.
        cN, cM, L, B = candidates.shape

        # Compute the repertoire embedding and normalizer for the candidate intervention sequences.
        # candidate reshape: (cN*cM) x 1 x L x B. intervene_alpha: (cN*cM) x embed_dim, intervene_norms_ln: (cN*cM)
        intervene_alpha, intervene_norms_ln = self.repertoire_featurizer(candidates.view([cN * cM, L, B]).unsqueeze(1),
                                                                         self.one[:(cN * cM)].unsqueeze(1))

        # intervene_contribs: (cN*cM) x N
        intervene_contribs = torch.einsum('ik,jk->ij', intervene_alpha, self.treatment_coeff)

        # Weighting of the observed treatment and intervention contributions.
        if self.no_attention:
            # inter_rep_contrib: (cN*cM) x N
            inter_rep_contrib = ((1 - self.intervene_frac) * self.treatment_contribs[None, :] +
                                 self.intervene_frac * intervene_contribs)
        else:
            # contrib_score: (cN*cM) x N x 2
            contrib_score = (self.intervene_fracs_ln[None, None, :] +
                             torch.cat([-intervene_norms_ln[:, None, None].expand(-1, self.N, 1),
                                               -self.treatment_norms_ln[None, :, None].expand(cN * cM, -1, 1)], 2))
            contrib_weights = contrib_score.softmax(dim=2)

            # Compute intervened repertoire's contribution to the outcome. (cN*cM) x N
            inter_rep_contrib = (contrib_weights[:, :, 0] * self.treatment_contribs[None, :] +
                                 contrib_weights[:, :, 1] * intervene_contribs)

        # Compute effect.
        if self.outcome_type == 'binary':
            # Binary outcome variable.
            effect_on = (inter_rep_contrib + self.natural_contribs[None, :]).sigmoid().mean(dim=1)
            effect_off = (self.treatment_contribs[None, :] + self.natural_contribs[None, :]).sigmoid().mean(dim=1)
            effect = effect_on - effect_off
        elif self.outcome_type == 'continuous':
            # Continuous outcome variable.
            # Here, using a Gaussian output distribution, the confounder and base/offset terms cancel
            # between _on and _off
            effect_on = inter_rep_contrib.mean(dim=1)
            effect_off = self.treatment_contribs[None, :].mean(dim=1)
            effect = effect_on - effect_off

        return effect


class CausalRepertoireModel(nn.Module):
    """The CAIRE model."""
    def __init__(self, args, data):
        """Initialize"""
        super().__init__()

        # Save arguments.
        self.args = args

        if args.cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # Active model components.
        self.outcome_model = True
        self.selection_model = True
        self.propensity_model = True
        if self.args.no_outcome:
            self.outcome_model = False
            self.propensity_model = False
        elif self.args.no_selection:
            self.selection_model = False
            self.propensity_model = False
        elif self.args.no_propensity:
            self.propensity_model = False

        # Max total subunits.
        # Note: we assume in evaluation we use the same unit and subunit batch as training.
        max_subunits = torch.maximum(data.tot_seqs_per_patient.max(), data.tot_naive_per_patient.max())
        max_total_subunits = np.maximum(args.unit_batch * args.subunit_batch, max_subunits)

        # Initialize data featurizer.
        self.data_featurizer = DataFeatures(data.repertoire_length, data.repertoire_alphabet, args)

        # Initialize selection (aka relative fitness) featurizer.
        self.selection_featurizer = Featurizer(data.repertoire_length, data.repertoire_alphabet, args)

        # Initialize repertoire featurizer
        self.repertoire_featurizer = RepertoireFeaturizer(data.repertoire_length, data.repertoire_alphabet, args)

        # Initialize encoder.
        self.encoder = Encoder(data.repertoire_length, data.repertoire_alphabet, max_total_subunits, args)

        # Save data parameters.
        self.repertoire_length = data.repertoire_length
        self.repertoire_alphabet = data.repertoire_alphabet

        # Initialize model components on gpu.
        if self.propensity_model:
            # Propensity model priors.
            self.propensity_coeff_loc = torch.zeros((self.args.select_latent_dim + 1) * self.args.repertoire_latent_dim,
                                                    device=self.device)
            self.propensity_coeff_scale = torch.tensor(10., device=self.device)
            self.propensity_noise_loc = torch.tensor(-1., device=self.device)
            self.propensity_noise_scale = torch.tensor(2., device=self.device)

        if self.outcome_model:
            # Outcome model priors.
            outcome_coeff_dim = self.args.repertoire_latent_dim + 1 + self.args.select_latent_dim * self.selection_model
            self.outcome_coeff_loc = torch.zeros(outcome_coeff_dim, device=self.device)
            self.outcome_coeff_scale = torch.tensor(100., device=self.device)
            self.outcome_noise_loc = torch.tensor(-1., device=self.device)
            self.outcome_noise_scale = torch.tensor(2., device=self.device)
            self.outcome_noise_init = torch.tensor(0.1, device=self.device)

        if self.selection_model:
            # Latent selection representation
            self.rho_loc = torch.zeros(self.args.select_latent_dim, device=self.device)
            self.rho_scale = torch.tensor(1., device=self.device)
            self.beta_loc = torch.tensor(0., device=self.device)
            self.beta_scale = torch.tensor(10., device=self.device)

        # Initialize constants on GPU.
        self.ones_subunit = torch.ones(max_total_subunits, device=self.device)
        self.zeros_subunit = torch.zeros(max_total_subunits, device=self.device)
        self.one = torch.tensor(1., device=self.device)
        self.zero = torch.tensor(0., device=self.device)

        # Track model performance on validation set.
        self.validation_score = -np.inf
        self.best_model = None
        self.validation_select_accuracy_mean = np.nan
        self.validation_propensity_pearson_mean = np.nan
        self.validation_outcome_score = np.nan
        self.best_model_update = -1

        # Store information for convergence diagnostics.
        self.dfrac = 0.67  # Fraction of training to look backward at when computing diagnostics.
        self.param_diffs = []
        self.convergence_diagnostic = np.nan
        self.elbos = []
        self.elbo_average = np.nan

    def _make_labels(self, num_units, num_subunits_mature, num_subunits_naive):
        mature_labels = self.ones_subunit[:(num_units * num_subunits_mature)].view([num_units, num_subunits_mature])
        naive_labels = self.zeros_subunit[:(num_units * num_subunits_naive)].view([num_units, num_subunits_naive])
        return torch.cat([mature_labels, naive_labels], dim=1)

    def model(self, matures, mature_counts, naives, outcomes, selection_correction,
              unit_correction, local_prior_scale):
        """Generative model and propensity model."""
        # -- Global parameters. --
        if self.propensity_model:
            # Propensity.
            propensity_coeff = pyro.sample('propensity_coeff', Normal(
                    self.propensity_coeff_loc, self.propensity_coeff_scale).to_event(1))
            W_e = propensity_coeff[:self.args.select_latent_dim * self.args.repertoire_latent_dim].view(
                    [self.args.select_latent_dim, self.args.repertoire_latent_dim])
            B_e = propensity_coeff[self.args.select_latent_dim * self.args.repertoire_latent_dim:]
            propensity_tau = pyro.sample('propensity_tau', dist.LogNormal(
                    self.propensity_noise_loc, self.propensity_noise_scale))

        if self.outcome_model:
            # Outcome.
            outcome_coeff = pyro.sample('outcome_coeff',
                                        Normal(self.outcome_coeff_loc, self.outcome_coeff_scale).to_event(1))
            W_A = outcome_coeff[:self.args.repertoire_latent_dim]
            pyro.deterministic('W_A', W_A)
            B_Y = outcome_coeff[self.args.repertoire_latent_dim]
            if self.selection_model:
                W_R = outcome_coeff[(self.args.repertoire_latent_dim+1):]
            if self.args.outcome == 'continuous':
                tau_Y = pyro.sample('tau_Y', dist.LogNormal(self.outcome_noise_loc, self.outcome_noise_scale))

        # -- Mapping between high dimensional data and latent representations --
        if self.selection_model:
            # Selection featurizer.
            pyro.module('selection_featurizer', self.selection_featurizer)
            # Compute selection features for sequences
            feats = self.selection_featurizer(matures, naives)

        if self.outcome_model:
            # Repertoire featurizer.
            pyro.module('repertoire_featurizer', self.repertoire_featurizer)
            # Latent repertoire representation.
            alpha, treatment_normalizer_ln = self.repertoire_featurizer(matures, mature_counts)
            alpha = alpha.unsqueeze(1)
            # Record normalizer for later treatment effect computations.
            pyro.deterministic('treatment_normalizer_ln', treatment_normalizer_ln)

        # -- Generate data --
        # Units (patients) and subunits (sequences).
        num_units, num_subunits_mature, num_subunits_naive = matures.shape[0], matures.shape[1], naives.shape[1]
        labels = self._make_labels(num_units, num_subunits_mature, num_subunits_naive)
        # Sample units.
        with pyro.plate('units', num_units, dim=-2):
            # Scale for unit minibatch.
            with poutine.scale(scale=unit_correction):
                # - Selection model: predict naive vs mature. -
                if self.selection_model:
                    # Scale for annealing the prior on the local latent variables.
                    with poutine.scale(scale=local_prior_scale):
                        # Latent selection representation.
                        rho = pyro.sample('rho', Normal(self.rho_loc, self.rho_scale).to_event(1))
                        # Auxiliary offset variable.
                        beta = pyro.sample('beta', Normal(self.beta_loc, self.beta_scale))

                    # Sample subunits.
                    with pyro.plate('selection', num_subunits_mature + num_subunits_naive, dim=-1):
                        # - Classify naive/mature sequences. -
                        # Scale for subunit minibatch.
                        mature_su_weight = mature_counts / mature_counts.sum(dim=1, keepdim=True)
                        naive_su_weight = (self.ones_subunit[:(num_units * num_subunits_mature)].reshape([num_units, num_subunits_mature])
                                           / num_subunits_naive)
                        sc = selection_correction[:, None] * torch.cat([mature_su_weight, naive_su_weight], dim=1)
                        with poutine.scale(scale=sc):
                            # Compute log odds based on embedding and features.
                            logodds = torch.einsum('ibk,ijk->ij', rho, feats) + beta
                            # Draw labels (naive/mature).
                            pyro.sample('labels', dist.Bernoulli(logits=logodds), obs=labels)

                # Propensity model.
                if self.propensity_model:
                    # Propensity mean.
                    e_rho = torch.einsum('ibk,kl->ibl', rho, W_e) + B_e
                    # Propensity model. Alpha should not be updated based on this log probability, so is
                    # detached from the automatic differentiation graph.
                    pyro.sample('alpha', dist.Normal(e_rho, propensity_tau).to_event(1),
                                 obs=alpha.detach())

                if self.outcome_model:
                    # - Outcome. -
                    # We record each term of the outcome model to enable treatment effect calculations.
                    pyro.deterministic('base_contrib', B_Y)
                    mn_Y = (pyro.deterministic('treatment_contrib', torch.einsum('ibk,k->ib', alpha, W_A))
                            + B_Y)
                    if self.selection_model:
                        confound_contrib = pyro.deterministic('confound_term', torch.einsum('ibk,k->ib', rho, W_R))
                        if self.propensity_model:
                            confound_contrib = pyro.deterministic(
                                    'confound_contrib',
                                    confound_contrib - torch.einsum('ibk,k->ib', e_rho.detach(), W_A))
                        else:
                            pyro.deterministic('confound_contrib', confound_contrib)
                        mn_Y = mn_Y + confound_contrib

                    if self.args.outcome == 'continuous':
                        outcome = pyro.sample('outcome', dist.Normal(mn_Y[:, :, None], tau_Y).to_event(1),
                                              obs=outcomes.unsqueeze(1))
                    elif self.args.outcome == 'binary':
                        # Convert from log(p/1-p) to log(1-p), log(p)
                        out_mn = -softplus(torch.cat([mn_Y[:, :, None], -mn_Y[:, :, None]], dim=2))
                        outcome = pyro.sample('outcome', MissingDataOneHotCategorical(logits=out_mn),
                                              obs=outcomes.unsqueeze(1))

    def guide(self, matures, mature_counts, naives, outcomes, selection_correction,
              unit_correction, local_prior_scale):
        """Variational (or point mass) approximation to the posterior."""
        # -- Global parameters --
        if self.propensity_model:
            # Propensity.
            propensity_coeff_dim =(self.args.select_latent_dim + 1) * self.args.repertoire_latent_dim
            propensity_coeff_loc = pyro.param('propensity_coeff_loc',
                                              lambda: torch.randn(propensity_coeff_dim, device=self.device))
            if self.args.posterior_rank > 0:
                propensity_coeff_cov_factor = pyro.param(
                        'propensity_coeff_cov_factor',
                        lambda: torch.randn(propensity_coeff_dim, self.args.posterior_rank, device=self.device))
                propensity_coeff_cov_diag = pyro.param('propensity_coeff_cov_diag',
                                                       lambda: torch.randn(propensity_coeff_dim, device=self.device))
                pyro.sample('propensity_coeff',
                            dist.LowRankMultivariateNormal(propensity_coeff_loc, propensity_coeff_cov_factor,
                                                           softplus(propensity_coeff_cov_diag)))
            elif self.args.posterior_rank == 0:
                propensity_coeff_cov_diag = pyro.param('propensity_coeff_cov_diag',
                                                       lambda: torch.randn(propensity_coeff_dim, device=self.device))
                pyro.sample('propensity_coeff',
                            Normal(propensity_coeff_loc, softplus(propensity_coeff_cov_diag)).to_event(1))
            elif self.args.posterior_rank == -1:
                pyro.sample('propensity_coeff', dist.Delta(propensity_coeff_loc).to_event(1))
            # propensity_tau
            propensity_tau_loc = pyro.param("propensity_tau_loc",
                                            lambda: torch.tensor(0.1, device=self.device))
            pyro.sample("propensity_tau", dist.Delta(softplus(propensity_tau_loc)))

        if self.outcome_model:
            # Outcome.
            outcome_coeff_dim = self.args.repertoire_latent_dim + 1 + self.args.select_latent_dim * self.selection_model
            outcome_coeff_loc = pyro.param('outcome_coeff_loc',
                                           lambda: (torch.tensor(1./outcome_coeff_dim, device=self.device).sqrt() *
                                                    torch.randn(outcome_coeff_dim, device=self.device)))
            if self.args.posterior_rank > 0:
                outcome_coeff_cov_factor = pyro.param('outcome_coeff_cov_factor',
                                                      lambda: torch.randn(outcome_coeff_dim, self.args.posterior_rank,
                                                                          device=self.device))
                outcome_coeff_cov_diag = pyro.param('outcome_coeff_cov_diag',
                                                    lambda: torch.randn(outcome_coeff_dim, device=self.device))
                pyro.sample('outcome_coeff',
                            dist.LowRankMultivariateNormal(outcome_coeff_loc, outcome_coeff_cov_factor,
                                                           softplus(outcome_coeff_cov_diag)))

            elif self.args.posterior_rank == 0:
                outcome_coeff_cov_diag = pyro.param('outcome_coeff_cov_diag',
                                                    lambda: torch.randn(outcome_coeff_dim, device=self.device))
                pyro.sample('outcome_coeff',
                            Normal(outcome_coeff_loc, softplus(outcome_coeff_cov_diag)).to_event(1))

            elif self.args.posterior_rank == -1:
                pyro.sample('outcome_coeff', dist.Delta(outcome_coeff_loc).to_event(1))

            if self.args.outcome == 'continuous':
                # tau_Y
                tau_Y_loc = pyro.param("tau_Y_loc", lambda: self.outcome_noise_init)
                pyro.sample("tau_Y", dist.Delta(softplus(tau_Y_loc)))

        # -- Local (per-patient) latent variables. --
        if self.selection_model:
            # - Encoder -
            # Layers
            pyro.module('encoder', self.encoder)

            # Sample units.
            num_units = matures.shape[0]
            with pyro.plate('units', num_units, dim=-2):
                # Scale for unit minibatch.
                with poutine.scale(scale=unit_correction):
                    # - Latent representations -
                    # Scale for KL annealing
                    with poutine.scale(scale=local_prior_scale):
                        # Encoder: data -> selection representation.
                        rho_mn, rho_sd, beta_mn, beta_sd = self.encoder(matures, mature_counts, naives)

                        if self.args.select_posterior_rank == 0:
                            # Latent selection representation.
                            rho = pyro.sample('rho', Normal(rho_mn, rho_sd).to_event(1))
                            # Auxiliary offset variable
                            beta = pyro.sample('beta', Normal(beta_mn, beta_sd))
                        elif self.args.select_posterior_rank == -1:
                            # Latent selection representation.
                            rho = pyro.sample('rho', dist.Delta(rho_mn).to_event(1))
                            # Auxiliary offset variable
                            beta = pyro.sample('beta', dist.Delta(beta_mn))

    def _beta_anneal(self, step, patient_batch_correction, anneal_length):
        """Annealing schedule for prior KL term (beta annealing)."""
        anneal_frac = step / (anneal_length * patient_batch_correction)
        return torch.minimum(anneal_frac, self.one)

    def _beta_anneal_time(self, current_time, max_time):
        return torch.minimum(current_time * self.one / max_time, self.one)

    def _evaluate_batch(self, matures, mature_counts, naives, outcomes, selection_correction,
                        unit_correction, local_prior_scale, details=True):
        """Compute key summary statistics for a batch of data."""
        # Get number of units (patients) and subunits (sequences).
        num_units, num_subunits = matures.shape[0], matures.shape[1]
        # Model arguments
        model_args = (matures, mature_counts, naives, outcomes, selection_correction,
                      unit_correction, local_prior_scale)
        summaries = dict()
        with torch.no_grad():
            # Trace the model.
            guide_tr = poutine.trace(self.guide).get_trace(*model_args)
            model_tr = poutine.trace(poutine.replay(self.model, trace=guide_tr)).get_trace(*model_args)
            # Compute ELBO estimate.
            summaries['elbo'] = (model_tr.log_prob_sum() - guide_tr.log_prob_sum()).unsqueeze(0)

            if self.selection_model:
                # Selection classifier accuracy.
                # Note: this is unweighted by counts.
                num_units, num_subunits_mature, num_subunits_naive = matures.shape[0], matures.shape[1], naives.shape[1]
                labels = self._make_labels(num_units, num_subunits_mature, num_subunits_naive)
                summaries['select_accuracy'] = accuracy(labels, model_tr.nodes['labels']['fn'].logits,
                                                        check_missing=False, one=self.one)

                if details:
                    # Selection embedding.
                    summaries['select_embed'] = guide_tr.nodes['rho']['fn'].mean.squeeze(dim=1)
                    summaries['select_embed_scale'] = guide_tr.nodes['rho']['fn'].stddev.squeeze(dim=1)

            if self.propensity_model:
                summaries['propensity'] = model_tr.nodes['alpha']['fn'].mean.squeeze(dim=1)
                summaries['repertoire_embed'] = model_tr.nodes['alpha']['value'].squeeze(dim=1)

            if self.outcome_model:
                # Outcome model prediction.
                if self.args.outcome == 'binary':
                    summaries['outcome_predict'] = torch.diff(model_tr.nodes['outcome']['fn'].logits,
                                                              dim=-1).squeeze(dim=(1, 2))
                elif self.args.outcome == 'continuous':
                    summaries['outcome_predict'] = model_tr.nodes['outcome']['fn'].mean.squeeze(dim=1)

                # Outcome model contributions (for computing causal effects).
                if details:
                    summaries['treatment_normalizer_ln'] = model_tr.nodes['treatment_normalizer_ln']['value']
                    summaries['treatment_contrib'] = model_tr.nodes['treatment_contrib']['value'].squeeze(dim=1)
                    summaries['base_contrib'] = model_tr.nodes['base_contrib']['value'] * torch.ones_like(
                        summaries['treatment_contrib'])
                    summaries['W_A'] = model_tr.nodes['W_A']['value'][None, :] * torch.ones_like(
                        summaries['treatment_contrib'])[:, None]
                    if self.selection_model:
                        summaries['confound_contrib'] = model_tr.nodes['confound_contrib']['value'].squeeze(dim=1)
                        summaries['confound_term'] = model_tr.nodes['confound_term']['value'].squeeze(dim=1)
                    else:
                        summaries['confound_contrib'] = torch.zeros_like(summaries['treatment_contrib'])
                        summaries['confound_term'] = torch.zeros_like(summaries['treatment_contrib'])

                # Repertoire embedding (if not recorded earlier).
                if details and not self.propensity_model:
                    summaries['repertoire_embed'] = self.repertoire_featurizer(matures, mature_counts)[0]

        return summaries

    def _evaluate_set(self, dataload, details=False):
        """Evaluate model on a dataset (e.g. validation or test set)."""
        storage = defaultdict(lambda: [])
        dataset_len = len(dataload.dataset)
        for matures, naives, outcomes, matures_per, naives_per, mature_counts in dataload:
            # -- Setup --
            # Transfer to cuda and featurize data.
            matures = self.data_featurizer.featurize_seqs(matures)
            naives = self.data_featurizer.featurize_seqs(naives)
            unit_correction = dataset_len / matures.shape[0]
            if self.args.cuda:
                mature_counts = mature_counts.cuda(non_blocking=True)
                outcomes = outcomes.cuda(non_blocking=True)
                matures_per = matures_per.cuda(non_blocking=True)
                naives_per = naives_per.cuda(non_blocking=True)
            local_prior_scale = self.one
            selection_correction = (matures_per + naives_per)/2

            # Collect inputs.
            model_args = [matures, mature_counts, naives, outcomes, selection_correction,
                          unit_correction, local_prior_scale]

            # -- Evaluate model on batch. --
            summaries = self._evaluate_batch(*model_args, details=details)

            # -- Compile summaries from each batch. --
            for elem in summaries:
                storage[elem].append(summaries[elem])
            storage['outcomes'].append(outcomes)

        # -- Concatenate results into single tensor. --
        for elem in storage:
            if storage[elem][0].dim() == 0:
                storage[elem] = torch.tensor(storage[elem])
            else:
                storage[elem] = torch.cat(storage[elem], dim=0)

        return storage

    def _log_training_info(self, step_i, loss, dataload_validate, aim_run, record_weights=False, final=False):
        """Record information during training."""

        # Track model stats.
        if step_i % self.args.validate_iter == 0 or final:
            # Evaluate model on the validation set.
            summaries = self._evaluate_set(dataload_validate, details=False)

            # ELBO estimate.
            elbo_mn = summaries['elbo'].mean().cpu()
            aim_run.track({'elbo_validate': elbo_mn}, step=step_i)

            # An overall score for model quality, based on roughly equal weighting of model parts.
            overall_score = torch.zeros(1, device=self.device)[0]

            if self.selection_model:
                # Selection model accuracy.
                select_accuracy_mn = summaries['select_accuracy'].mean().cpu()
                overall_score += select_accuracy_mn
                aim_run.track({'select_accuracy_mn': select_accuracy_mn,
                               'select_accuracy_sd': summaries['select_accuracy'].std().cpu()}, step=step_i)

            if self.propensity_model:
                rep_embed = summaries['repertoire_embed'].detach().cpu()
                propens = summaries['propensity'].detach().cpu()
                propensity_r2_mean = np.mean([r2_score(rep_embed[:, j], propens[:, j])
                                              for j in range(propens.shape[1])])
                propensity_pearson_mean = np.mean([pearson_score(rep_embed[:, j], propens[:, j])
                                                   for j in range(propens.shape[1])])
                propensity_rmse = ((rep_embed - propens)**2).mean().sqrt()
                aim_run.track({'propensity_r2_mean': propensity_r2_mean,
                               'propensity_pearson_mean': propensity_pearson_mean,
                               'propensity_rmse': propensity_rmse}, step=step_i)

            if self.outcome_model:
                # Outcome model performance.
                outcomes = summaries['outcomes']
                if self.args.outcome == 'binary':
                    # Un-one-hot encode (so outcomes_binary are 0, 1 or -1 for missing data).
                    outcomes_binary = (outcomes.argmax(dim=1) + outcomes.sum(dim=1) - self.one).cpu()
                    outcomes_predict = summaries['outcome_predict'].cpu()
                    outcome_accuracy = accuracy(outcomes_binary, outcomes_predict)
                    outcome_auc = roc_auc_score(outcomes_binary, outcomes_predict)
                    overall_score += outcome_auc
                    aim_run.track({'outcome_accuracy': outcome_accuracy,
                                   'outcome_auc': outcome_auc}, step=step_i)
                elif self.args.outcome == 'continuous':
                    outcome_r2 = r2_score(outcomes.cpu(), summaries['outcome_predict'].cpu())
                    overall_score += outcome_r2
                    outcome_rmse = ((outcomes - summaries['outcome_predict'])**2).mean().sqrt()
                    aim_run.track({'outcome_r2': outcome_r2,
                                   'outcome_rmse': outcome_rmse.cpu()}, step=step_i)

            # Checkpoint
            if self.args.elbo_early_stop:
                validation_score = elbo_mn
            else:
                validation_score = overall_score

            # Update record of best model.
            if (not self.args.no_early_stop and validation_score > self.validation_score) or (self.args.no_early_stop and final):
                aim_run.track({'best_validation': validation_score}, step=step_i)
                self.best_model = f"{aim_run['local_dir']}/checkpoint_model"
                self.best_model_update = step_i
                for elem in pyro.get_param_store():
                    torch.save(pyro.get_param_store()[elem].clone().detach().cpu().to(dtype=torch.float32),
                               f"{self.best_model}_{elem}.pt")
                torch.save(self.state_dict(), f"{self.best_model}_state_dict.pt")
                self.validation_score = validation_score
                if self.selection_model:
                    self.validation_select_accuracy_mean = select_accuracy_mn
                if self.propensity_model:
                    self.validation_propensity_pearson_mean = propensity_pearson_mean
                if self.outcome_model:
                    if self.args.outcome == 'binary':
                        self.validation_outcome_score = outcome_auc
                    elif self.args.outcome == 'continuous':
                        self.validation_outcome_score = outcome_r2

        if step_i % self.args.monitor_iter == 0:

            if self.args.weight_average and record_weights:
                # Sum weights
                for elem in pyro.get_param_store():
                    self.weight_average_params[elem] += pyro.get_param_store()[elem].clone().detach()
                # Sum number of steps to average over.
                self.weight_steps += 1

            # Iterate to look back to for computing diagnostics. Note dfrac = 1/q in Pesme et al.'s notation.
            lookback = int(len(self.elbos) * self.dfrac)

            if self.args.monitor_converge:
                # Track change in parameters.
                param_diff_norm = 0
                for elem in pyro.get_param_store():
                    new_param = pyro.get_param_store()[elem].clone().detach()
                    param_diff_norm += (self.init_params[elem] - new_param).pow(2).sum().item()
                param_diff = np.log(param_diff_norm)
                self.param_diffs.append(param_diff)
                aim_run.track({'param_diff': param_diff}, step=step_i)
                # Compute convergence diagnostic from Pesme et al., 2020.
                self.convergence_diagnostic = (self.param_diffs[-1] - self.param_diffs[lookback])/-np.log(self.dfrac)
                aim_run.track({'convergence_diagnostic': self.convergence_diagnostic}, step=step_i)

            # Track loss.
            elbo = -loss.item()
            print(step_i, elbo)
            self.elbos.append(elbo)
            self.elbo_average = np.mean(self.elbos[lookback:])
            aim_run.track({'elbo': elbo, 'elbo_average': self.elbo_average, 'update': step_i}, step=step_i)

    def _init_fn(self, dataload, seed, svi=None):
        """Initialize model."""
        pyro.set_rng_seed(seed)
        pyro.clear_param_store()
        for matures, naives, outcomes, matures_per, naives_per, mature_counts in dataload:
            # Transfer to cuda and featurize data.
            matures = self.data_featurizer.featurize_seqs(matures)
            naives = self.data_featurizer.featurize_seqs(naives)
            if self.args.cuda:
                mature_counts = mature_counts.cuda(non_blocking=True)
                outcomes = outcomes.cuda(non_blocking=True)
                matures_per = matures_per.cuda(non_blocking=True)
                naives_per = naives_per.cuda(non_blocking=True)

            selection_correction = (matures_per + naives_per) / 2
            break

        unit_correction = self.one
        local_prior_scale = self.one
        model_args = [matures, mature_counts, naives, outcomes, selection_correction,
                      unit_correction, local_prior_scale]
        self.guide(*model_args)
        self.model(*model_args)
        if svi is not None:
            loss = svi.step(*model_args)
            return loss.item()
        else:
            return None

    def fit_svi(self, train_data, validate_data, aim_run, generator=None):
        """Infer approximate posterior"""
        # Set up data loader.
        dataload_train = DataLoader(train_data, batch_size=self.args.unit_batch, shuffle=True,
                                    num_workers=self.args.num_workers, pin_memory=self.args.cuda,
                                    generator=generator, drop_last=True, persistent_workers=True)
        dataload_validate = DataLoader(validate_data, batch_size=self.args.unit_batch_eval, shuffle=False,
                                       num_workers=self.args.num_workers, pin_memory=self.args.cuda,
                                       generator=generator, drop_last=False, persistent_workers=False)

        # -- Optimizer(s) setup --
        optimizer_options = {}
        if self.args.optimizer == 'Adam':
            optimizer_alg = Adam
            optimizer_options['amsgrad'] = True
            # Use optimized implementation for cuda.
            if self.args.cuda:
                optimizer_options['fused'] = True
        elif self.args.optimizer == 'NAdam':
            optimizer_alg = NAdam
            if self.args.cuda:
                optimizer_options['foreach'] = True
        elif self.args.optimizer == 'SGD':
            optimizer_alg = SGD
            if self.args.cuda:
                optimizer_options['foreach'] = True
        if self.args.low_dtype in [torch.float16, torch.bfloat16] and self.args.optimizer in ['Adam', 'NAdam']:
            # Increase epsilon for greater numerical stability (at cost of lower accuracy).
            optimizer_options['eps'] = 1e-4
        optimizer_options['weight_decay'] = self.args.weight_decay

        if self.args.separate_propensity:
            # Separate updates for propensity model.
            def subset_params(pm, propensity):
                if 'propensity' in pm:
                    opt_params = {'lr': propensity * self.args.lr}
                    opt_params.update(optimizer_options)
                    return opt_params
                else:
                    opt_params = {'lr': (not propensity) * self.args.lr}
                    opt_params.update(optimizer_options)
                    return opt_params
            propensity_params = lambda pm: subset_params(pm, True)
            non_propensity_params = lambda pm: subset_params(pm, False)
            propensity_optim = optimizer_alg(propensity_params)
            model_optim = optimizer_alg(non_propensity_params)
        else:
            optimizer_options['lr'] = self.args.lr
            model_optim = optimizer_alg(optimizer_options)

        # -- Stochastic variational inference setup --
        if self.args.no_jit:
            # Without compilation.
            elbo = Trace_ELBO(max_plate_nesting=2)
            svi = SVI(self.model, self.guide, model_optim, loss=elbo)
            if self.args.separate_propensity:
                elbo_propensity = Trace_ELBO(max_plate_nesting=2)
                svi_propensity = SVI(self.model, self.guide, propensity_optim, loss=elbo_propensity)
        else:
            # With model compilation, for the GPU.
            elbo = CudaJitTrace_ELBO(max_plate_nesting=2)
            svi = CudaSVI(self.model, self.guide, model_optim, loss=elbo)
            if self.args.separate_propensity:
                elbo_propensity = CudaJitTrace_ELBO(max_plate_nesting=2)
                svi_propensity = CudaSVI(self.model, self.guide, propensity_optim, loss=elbo_propensity)


        # -- Initialize parameters. --
        _, seed = min((self._init_fn(dataload_train, seed, svi), seed)
                         for seed in range(self.args.seed, self.args.seed+self.args.ninit*10000, 10000))
        self._init_fn(dataload_train, seed, svi)

        if self.args.monitor_converge:
            self.init_params = {}
            for elem in pyro.get_param_store():
                self.init_params[elem] = pyro.get_param_store()[elem].clone().detach()

        # -- Stochastic weight averaging setup --
        record_weights = False

        if self.args.weight_average:
            self.weight_average_params = {}
            for elem in pyro.get_param_store():
                self.weight_average_params[elem] = torch.zeros_like(pyro.get_param_store()[elem])
            self.weight_steps = 0

        # -- Profiling setup --
        if self.args.profile:
            prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule = torch.profiler.schedule(wait=8, warmup=1, active=3, repeat=1),
                on_trace_ready = torch.profiler.tensorboard_trace_handler(aim_run['local_dir']),
                with_stack = True, record_shapes=True, profile_memory=True)
            prof.start()

        # -- Main training loop --
        step_i = 1
        start_time = datetime.now()
        halt = False
        if self.args.anneal_time:
            local_prior_scale = self._beta_anneal_time(0., self.args.anneal)
        # Unit batch correction.
        unit_correction = torch.tensor(len(train_data) / self.args.unit_batch, device=self.device)
        for epoch in range(self.args.epochs):
            for matures, naives, outcomes, matures_per, naives_per, mature_counts in dataload_train:
                # Transfer to cuda and featurize data.
                matures = self.data_featurizer.featurize_seqs(matures)
                naives = self.data_featurizer.featurize_seqs(naives)
                if self.args.cuda:
                    mature_counts = mature_counts.cuda(non_blocking=True)
                    outcomes = outcomes.cuda(non_blocking=True)
                    matures_per = matures_per.cuda(non_blocking=True)
                    naives_per = naives_per.cuda(non_blocking=True)

                # We use the same scale for the contribution of the log probability of the mature & naive sequences
                # under the selection (relative fitness) model. This is hacky from a fully Bayesian perspective
                # but allows the classifier/log-odds trick to go through smoothly.
                selection_correction = (matures_per + naives_per)/2

                # Prior/KL annealing.
                if not self.args.anneal_time:
                    local_prior_scale = self._beta_anneal(step_i, unit_correction, self.args.anneal)

                # Collect inputs.
                model_args = [matures, mature_counts, naives, outcomes, selection_correction,
                              unit_correction, local_prior_scale]

                # Update propensity model alone.
                if self.args.separate_propensity and (step_i % self.args.propensity_update == 0):
                    loss = svi_propensity.step(*model_args)

                # Step.
                loss = svi.step(*model_args)

                # Record training information.
                self._log_training_info(step_i, loss, dataload_validate, aim_run, record_weights=record_weights)

                # Update step.
                step_i += 1

                # Profiler update.
                if self.args.profile:
                    torch.cuda.synchronize()
                    prof.step()
                    if step_i == 20:
                        break

                if self.args.smoke and step_i == 6:
                    break
            aim_run.track({'epoch': epoch}, step=step_i)

            # Halt training if exceeded maximum time or reached nan.
            train_time = (datetime.now() - start_time).seconds/60
            if (train_time > self.args.max_time or torch.isnan(loss)
                    or self.args.smoke or self.args.profile):
                break

            if train_time > self.args.weight_fraction * self.args.max_time:
                record_weights = True

            if self.args.anneal_time:
                local_prior_scale = self._beta_anneal_time(train_time, self.args.anneal)


        if self.args.profile:
            prof.stop()
            print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
            print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=5))
            print(prof.key_averages(group_by_stack_n=5).table(sort_by="cpu_time_total", row_limit=5))

        # Final validation set evaluation.
        self._log_training_info(step_i, loss, dataload_validate, aim_run, record_weights=record_weights, final=True)

    def evaluate(self, data, train_ind, validate_ind, test_ind, aim_run, model_params=None,
                 data_eval_effect_dist=None):
        """Evaluation metrics and embeddings for trained model."""
        # Dataloader for evaluation.
        dataload = DataLoader(data, batch_size=self.args.unit_batch_eval, shuffle=False,
                              num_workers=self.args.num_workers, pin_memory=self.args.cuda,
                              drop_last=False, persistent_workers=False)

        # Load model parameters (can be different from current values, e.g. due to early stopping).
        if model_params is not None:
            self._init_fn(dataload, 0)
            for elem in pyro.get_param_store():
                pyro.get_param_store()[elem] = torch.load(f"{model_params}_{elem}.pt").to(
                                                        dtype=pyro.get_param_store()[elem].dtype,
                                                        device=pyro.get_param_store()[elem].device)
            self.load_state_dict(torch.load(f"{model_params}_state_dict.pt"))

        # Extract summary results for each patient.
        # Repeat evaluation
        data_subsets = {'train': train_ind, 'validate': validate_ind, 'test': test_ind}
        r2_scores = {subset: [] for subset in data_subsets}
        for ks in range(self.args.eval_samples):
            summaries_samp = self._evaluate_set(dataload, details=True)
            for el in summaries_samp:
                summaries_samp[el] = summaries_samp[el].detach().cpu()

            # Check r2 variability.
            for subset in data_subsets:
                ind = data_subsets[subset]
                outcomes = summaries_samp['outcomes'][ind]
                outcomes_predict = summaries_samp['outcome_predict'][ind]
                r2_scores[subset].append(r2_score(outcomes, outcomes_predict))

            # Take average.
            if ks == 0:
                summaries = summaries_samp
                for el in summaries:
                    summaries[el] = summaries[el] / self.args.eval_samples
            else:
                for el in summaries_samp:
                    summaries[el] += summaries_samp[el] / self.args.eval_samples
        for subset in r2_scores:
            # Check for substantial prediction variability
            aim_run.track({'r2_score_mean_{}'.format(subset): np.array(r2_scores[subset]).mean(),
                           'r2_score_sd_{}'.format(subset): np.array(r2_scores[subset]).std()})
        # We proceed with the average of the summaries
        # (note this may be inappropriate in some cases, e.g. for non-mean-field variational approximations).

        # Overall results.
        data_subsets = {'train': train_ind, 'validate': validate_ind, 'test': test_ind}
        results = {}
        for subset in data_subsets:
            ind = data_subsets[subset]
            if self.selection_model:
                # Selection model accuracy.
                results['select_accuracy_mn_' + subset] = summaries['select_accuracy'][ind].mean().numpy()
                results['select_accuracy_sd_' + subset] = summaries['select_accuracy'][ind].std().numpy()

            if self.propensity_model:
                rep_embed = summaries['repertoire_embed'][ind].numpy()
                propens = summaries['propensity'][ind].numpy()
                propensity_r2_mean = np.mean([r2_score(rep_embed[:, j], propens[:, j])
                                              for j in range(propens.shape[1])])
                propensity_pearson_mean = np.mean([pearson_score(rep_embed[:, j], propens[:, j])
                                                   for j in range(propens.shape[1])])
                results['propensity_r2_mean_' + subset] = propensity_r2_mean
                results['propensity_pearson_mean_' + subset] = propensity_pearson_mean

            if self.outcome_model:
                # Outcome model performance.
                outcomes = summaries['outcomes']
                outcomes_var = ((outcomes - outcomes.mean())**2).mean()
                outcomes_predict = summaries['outcome_predict'][ind]
                confound_term = summaries['confound_term'][ind]
                treatment_term = (summaries['treatment_contrib'][ind]
                                  + summaries['confound_contrib'][ind] - confound_term)

                if self.args.outcome == 'binary':
                    # Un-one-hot encode (so outcomes_binary are 0, 1 or -1 for missing data).
                    outcomes_binary = (outcomes.argmax(dim=1) + outcomes.sum(dim=1) - 1.)[ind]
                    outcome_accuracy = accuracy(outcomes_binary, outcomes_predict)
                    results['outcome_accuracy_' + subset] = outcome_accuracy
                    results['outcome_auc_' + subset] = roc_auc_score(outcomes_binary, outcomes_predict)
                    results['outcome_treatment_auc_' + subset] = roc_auc_score(outcomes_binary, treatment_term)
                    results['outcome_confound_auc_' + subset] = roc_auc_score(outcomes_binary, confound_term)
                elif self.args.outcome == 'continuous':
                    outcome_r2 = r2_score(outcomes[ind], outcomes_predict, base_var=outcomes_var)
                    results['outcome_r2_' + subset] = outcome_r2
                    results['outcome_r_pvalue_' + subset] = r_pvalue(outcomes[ind].squeeze(), outcomes_predict)
                    results['outcome_treatment_explained_' + subset] = explained_variance_score(
                                                                            outcomes[ind], treatment_term)
                    results['outcome_treatment_r2_' + subset] = r2_score(outcomes[ind], treatment_term, 
                                                                         base_var=outcomes_var)
                    results['outcome_treatment_r_pvalue_' + subset] = r_pvalue(outcomes[ind].squeeze(),
                                                                               treatment_term)
                    results['outcome_confound_explained_' + subset] = explained_variance_score(
                                                                            outcomes[ind], confound_term)
                    results['outcome_confound_r2_' + subset] = r2_score(outcomes[ind], confound_term,
                                                                        base_var=outcomes_var)
                    results['outcome_confound_r_pvalue' + subset] = r_pvalue(outcomes[ind].squeeze(), confound_term)
                    outcome_rmse = ((outcomes[ind] - outcomes_predict)**2).mean().sqrt()
                    results['outcome_rmse_' + subset] = outcome_rmse
        aim_run.track(results)

        # Store embeddings/representations
        if self.selection_model:
            select_embed_table = pd.DataFrame(summaries['select_embed'].numpy(),
                columns=list(map('select_embed_{}'.format, range(self.args.select_latent_dim))))
            select_embed_table.to_csv(os.path.join(aim_run['local_dir'], 'select_embed.csv'), index=False)
            if 'select_embed_1' in select_embed_table:
                select_embed_fig = px.scatter(x=select_embed_table['select_embed_0'],
                                              y=select_embed_table['select_embed_1'])
            else:
                select_embed_fig = px.scatter(x=select_embed_table['select_embed_0'],
                                              y=select_embed_table['select_embed_0'])
            aim_run.track(Figure(select_embed_fig), name='select_embed_fig')
            select_embed_scale_table = pd.DataFrame(
                summaries['select_embed_scale'].numpy(),
                columns=list(map('select_embed_scale_{}'.format, range(self.args.select_latent_dim))))
            select_embed_scale_table.to_csv(os.path.join(aim_run['local_dir'], 'select_embed_scale.csv'), index=False)
        if self.outcome_model:
            if self.args.outcome == 'continuous':
                outcome_predict_table = pd.DataFrame(
                        {'predicted outcome': summaries['outcome_predict'].squeeze(1).numpy(), 'outcome': outcomes.squeeze(1).numpy()})
                outcome_predict_table.to_csv(os.path.join(aim_run['local_dir'], 'outcome_predict.csv'), index=False)
                outcome_predict_fig = px.scatter(x=outcome_predict_table['predicted outcome'], y=outcome_predict_table['outcome'])
                aim_run.track(Figure(outcome_predict_fig), name='outcome_predict_fig')
            repertoire_columns = list(map('repertoire_embed_{}'.format, range(self.args.repertoire_latent_dim)))
            repertoire_embed_table = pd.DataFrame(
                summaries['repertoire_embed'].numpy(), columns=repertoire_columns)
            repertoire_embed_table.to_csv(os.path.join(aim_run['local_dir'], 'repertoire_embed.csv'), index=False)
            repertoire_embed_fig = px.scatter(x=repertoire_embed_table['repertoire_embed_0'], y=repertoire_embed_table['repertoire_embed_1'])
            aim_run.track(Figure(repertoire_embed_fig), name='repertoire_embed_fig')
            if self.propensity_model:
                propensity_columns = list(map('propensity_{}'.format, range(self.args.repertoire_latent_dim)))
                propensity_table = pd.DataFrame(
                    torch.cat([summaries['repertoire_embed'], summaries['propensity']], dim=1).numpy(),
                    columns=(repertoire_columns + propensity_columns))
                propensity_table.to_csv(os.path.join(aim_run['local_dir'], 'propensity.csv'), index=False)
                propensity_fig = px.scatter(x=propensity_table['propensity_0'], y=propensity_table['propensity_1'])
                aim_run.track(Figure(propensity_fig), name='propensity_fig')
                for j in range(self.args.repertoire_latent_dim):
                    alpha_propensity_fig = px.scatter(x=propensity_table['repertoire_embed_{}'.format(j)],
                                                      y=propensity_table['propensity_{}'.format(j)])
                    aim_run.track(Figure(alpha_propensity_fig), name='alpha_propensity_{}'.format(j))

        # Get effect distribution within mature repertoires.
        if self.args.eval_effect_dist:
            # Set up dataloader
            dataload = DataLoader(data_eval_effect_dist, batch_size=self.args.unit_batch_eval, shuffle=False,
                                  num_workers=self.args.num_workers, pin_memory=self.args.cuda,
                                  drop_last=False, persistent_workers=False)
            # Function for computing intervention effects based on learned model.
            intervene_effect_est = InterventionEffect(
                self.args.intervene_frac, summaries['W_A'], summaries['treatment_contrib'],
                summaries['treatment_normalizer_ln'], summaries['confound_contrib'], summaries['base_contrib'],
                self.args.outcome, max_batch=self.args.unit_batch_eval * self.args.subunit_batch_eval,
                repertoire_featurizer=self.repertoire_featurizer, no_attention=self.args.no_attention,
                low_dtype=self.args.low_dtype, cuda=self.args.cuda)

            # Compute effects
            est_effects = []
            candidate_counts = []
            for candidates, _, _, _, _, candidate_count in dataload:
                # -- Setup --
                # Transfer to cuda and featurize data.
                candidates = self.data_featurizer.featurize_seqs(candidates)
                with torch.no_grad():
                    # Compute interventional effect.
                    est_effect = intervene_effect_est(candidates).view([-1, self.args.subunit_batch_eval]).cpu()
                    est_effects.append(est_effect)
                candidate_counts.append(candidate_count)
            # Consolidate.
            est_effects = torch.cat(est_effects, dim=0)
            candidate_counts = torch.cat(candidate_counts, dim=0)
        else:
            est_effects = torch.tensor(torch.nan)
            candidate_counts = torch.tensor(torch.nan)

        return results, summaries, est_effects, candidate_counts

    def evaluate_true(self, data, summaries, train_ind, validate_ind, test_ind, aim_run):
        """Evaluate model against ground truth, for semisynthetic data"""
        # Compare latent simulation variables to model terms.
        inject_ind = torch.tensor(data.h5f['inject_patients'][:], dtype=torch.bool)
        confound_ind = torch.tensor(data.h5f['confounder'][:], dtype=torch.bool)
        confound_total = confound_ind.sum(dim=1)
        aim_run.track({'inject_yes_treatment_mn': summaries['treatment_contrib'][inject_ind].mean().cpu(),
                   'inject_yes_treatment_sd': summaries['treatment_contrib'][inject_ind].std().cpu(),
                   'inject_no_treatment_mn': summaries['treatment_contrib'][~inject_ind].mean().cpu(),
                   'inject_no_treatment_sd': summaries['treatment_contrib'][~inject_ind].std().cpu(),
                   'inject_yes_confound_mn': summaries['confound_contrib'][inject_ind].mean().cpu(),
                   'inject_yes_confound_sd': summaries['confound_contrib'][inject_ind].std().cpu(),
                   'inject_no_confound_mn': summaries['confound_contrib'][~inject_ind].mean().cpu(),
                   'inject_no_confound_sd': summaries['confound_contrib'][~inject_ind].std().cpu(),
                   'confound_yes_treatment_mn': summaries['treatment_contrib'][confound_total > 0].mean().cpu(),
                   'confound_yes_treatment_sd': summaries['treatment_contrib'][confound_total > 0].std().cpu(),
                   'confound_no_treatment_mn': summaries['treatment_contrib'][confound_total == 0].mean().cpu(),
                   'confound_no_treatment_sd': summaries['treatment_contrib'][confound_total == 0].std().cpu(),
                   'confound_yes_confound_mn': summaries['confound_contrib'][confound_total > 0].mean().cpu(),
                   'confound_yes_confound_sd': summaries['confound_contrib'][confound_total > 0].std().cpu(),
                   'confound_no_confound_mn': summaries['confound_contrib'][confound_total == 0].mean().cpu(),
                   'confound_no_confound_sd': summaries['confound_contrib'][confound_total == 0].std().cpu()
                   })
        sim_table = pd.DataFrame(
            {'inject_ind': inject_ind,
             'confound_total': confound_total,
             'treatment_contrib': summaries['treatment_contrib'].cpu(),
             'confound_contrib': summaries['confound_contrib'].cpu()})
        sim_table.to_csv(os.path.join(aim_run['local_dir'], 'sim_table.csv'), index=False)
        inject_v_treatment_fig = px.scatter(x=sim_table['inject_ind'], y=sim_table['treatment_contrib'])
        aim_run.track(Figure(inject_v_treatment_fig), name='inject_v_treatment_fig')
        inject_v_confound_fig = px.scatter(x=sim_table['inject_ind'], y=sim_table['confound_contrib'])
        aim_run.track(Figure(inject_v_confound_fig), name='inject_v_confound_fig')
        confound_tot_v_treatment_fig = px.scatter(x=sim_table['confound_total'], y=sim_table['treatment_contrib'])
        aim_run.track(Figure(confound_tot_v_treatment_fig), name='confound_tot_v_treatment_fig')
        confound_tot_v_confound_fig = px.scatter(x=sim_table['confound_total'], y=sim_table['confound_contrib'])
        aim_run.track(Figure(confound_tot_v_confound_fig), name='confound_tot_v_confound_fig')

        # Compare selection embeddings against true confounders, as well as against injection status.
        if self.selection_model:
            nconfound = confound_ind.shape[1]
            for j in range(nconfound):
                # Plot.
                positive_select_embed_table = pd.DataFrame(
                    summaries['select_embed'][confound_ind[:, j], :].numpy(),
                    columns=list(
                        map(('select_embed_{}_confound_' + str(j)).format, range(self.args.select_latent_dim))))
                positive_select_embed_table.to_csv(os.path.join(aim_run['local_dir'],
                                                                'positive_select_embed_{}.csv'.format(j)), index=False)
                positive_select_embed_fig = px.scatter(x=positive_select_embed_table['select_embed_0_confound_' + str(j)],
                                                       y=positive_select_embed_table['select_embed_1_confound_' + str(j)])
                aim_run.track(Figure(positive_select_embed_fig), name='select_embed_confound_' + str(j) + '_fig')

                # Evaluate whether embeddings can be used to predict confounders.
                mdl_X, mdl_Y = summaries['select_embed'], confound_ind[:, j]
                if len(np.unique(mdl_Y[train_ind].numpy())) > 1:
                    embed_confound_mdl = LogisticRegression().fit(mdl_X[train_ind].numpy(), mdl_Y[train_ind].numpy())
                    mdl_acc_train = embed_confound_mdl.score(mdl_X[train_ind].numpy(), mdl_Y[train_ind].numpy())
                    mdl_acc_val = embed_confound_mdl.score(mdl_X[validate_ind].numpy(), mdl_Y[validate_ind].numpy())
                    mdl_acc_test = embed_confound_mdl.score(mdl_X[test_ind].numpy(), mdl_Y[test_ind].numpy())
                    aim_run.track({'select_embed_confound_{}_accuracy_train'.format(j): mdl_acc_train})
                    aim_run.track({'select_embed_confound_{}_accuracy_validate'.format(j): mdl_acc_val})
                    aim_run.track({'select_embed_confound_{}_accuracy_test'.format(j): mdl_acc_test})

            # Evaluate whether embeddings can predict injection status.
            mdl_X, mdl_Y = summaries['select_embed'], inject_ind
            if len(np.unique(mdl_Y[train_ind].numpy())) > 1:
                embed_inject_mdl = LogisticRegression().fit(mdl_X[train_ind].numpy(), mdl_Y[train_ind].numpy())
                mdl_acc_train = embed_inject_mdl.score(mdl_X[train_ind].numpy(), mdl_Y[train_ind].numpy())
                mdl_acc_val = embed_inject_mdl.score(mdl_X[validate_ind].numpy(), mdl_Y[validate_ind].numpy())
                mdl_acc_test = embed_inject_mdl.score(mdl_X[test_ind].numpy(), mdl_Y[test_ind].numpy())
                aim_run.track({'select_embed_inject_accuracy_train': mdl_acc_train})
                aim_run.track({'select_embed_inject_accuracy_validate': mdl_acc_val})
                aim_run.track({'select_embed_inject_accuracy_test': mdl_acc_test})

        # Dataloader for evaluation.
        dataload = DataLoader(data, batch_size=1, shuffle=False,
                              num_workers=1, pin_memory=self.args.cuda,
                              drop_last=False, persistent_workers=False)

        # Function for computing intervention effects based on learned model.
        # Note that the data has mean outcome of zero, so this gives E[Y|do(...)] = E[Y|do(...)] - E[Y]
        intervene_effect_est = InterventionEffect(
            self.args.intervene_frac, summaries['W_A'], summaries['treatment_contrib'],
            summaries['treatment_normalizer_ln'], summaries['confound_contrib'], summaries['base_contrib'],
            self.args.outcome, max_batch=int(self.args.subunit_batch * self.args.unit_batch),
            repertoire_featurizer=self.repertoire_featurizer, no_attention=self.args.no_attention,
            low_dtype=self.args.low_dtype, cuda=self.args.cuda)

        # Function for computing intervention effects based on true model.
        dattrs = data.h5f['metadata'].attrs
        causal_motif = torch.tensor(dattrs['causal_motif'][:], dtype=torch.float32)
        intervene_effect_true = InterventionEffectTrue(
            self.repertoire_length, self.repertoire_alphabet, self.args.intervene_frac,
            torch.tensor(dattrs['causal_motif'][:], dtype=torch.float32),
            torch.tensor(dattrs['synthetic_motif_effect'], dtype=torch.float32),
            torch.tensor(dattrs['synthetic_base_effect'], dtype=torch.float32),
            torch.tensor(data.h5f['motif_contribs'][:], dtype=torch.float32),
            torch.tensor(data.h5f['confound_contribs'][:], dtype=torch.float32),
            self.args.outcome, low_dtype=self.args.low_dtype, cuda=self.args.cuda)

        # Functions for determining whether sequences are causal or affected by a confounder.
        confound_motifs = torch.tensor(dattrs['confound_motifs'][:], dtype=torch.float32)
        confound_motif_flat = confound_motifs.reshape([-1, confound_motifs.shape[2]]).to(dtype=self.args.low_dtype, device=self.device)
        kmer_embed_confound = KmerEmbed(self.repertoire_length, data.repertoire_alphabet, confound_motif_flat.shape[1],
                                        custom_kmers=confound_motif_flat, dtype=self.args.low_dtype, cuda=self.args.cuda)
        causal_motif_flat = causal_motif[None, :].to(dtype=self.args.low_dtype, device=self.device)
        kmer_embed_inject = KmerEmbed(self.repertoire_length, data.repertoire_alphabet, confound_motif_flat.shape[1],
                                      custom_kmers=causal_motif_flat, dtype=self.args.low_dtype, cuda=self.args.cuda)

        # Iterate over dataset.
        rep_ind = -1
        effect_keys = ['est_effect_causal_mn', 'est_effect_causal_sd', 'true_effect_causal_mn', 'true_effect_causal_sd',
                       'est_effect_noncausal_mn', 'est_effect_noncausal_sd', 'true_effect_noncausal_mn', 'true_effect_noncausal_sd',
                       'effect_causal_error_l1', 'effect_causal_error_l2', 'effect_causal_error_l1_weighted', 'effect_causal_error_l2_weighted',
                       'causal_noncausal_pr_auc', 'causal_noncausal_auc', 'causal_confounder_auc',
                       'est_effect_confounder_mn', 'est_effect_confounder_sd', 'true_effect_confounder_mn', 'true_effect_confounder_sd']
        effect_summaries = {ke: [] for ke in effect_keys}
        for candidates, _, _, _, _, mature_counts in dataload:
            rep_ind += 1

            # Transfer to cuda and featurize data.
            candidates = self.data_featurizer.featurize_seqs(candidates)
            if self.args.cuda:
                candidate_counts = mature_counts.cuda(non_blocking=True).squeeze(0)

            # Initialize storage.
            s = {ke: torch.nan for ke in effect_keys}

            # Compute effects of each sequence if used for intervention.
            with torch.no_grad():
                # Compute estimated intervention effect.
                est_effect = intervene_effect_est(candidates)

                # Compute true intervention effect.
                true_effect = intervene_effect_true(candidates)

                # Get sequences with confounder motif.
                confounder_seqs_ind = kmer_embed_confound(candidates).squeeze(0).sum(dim=-1).clamp(max=self.one).to(
                                            torch.bool)

                # Get sequences with causal motif.
                causal_seqs_ind = kmer_embed_inject(candidates).squeeze(0).sum(dim=-1).clamp(max=self.one).to(
                                            torch.bool)

                # Evaluate causal sequences only on patients with the causal motif injected.
                if data.h5f['inject_patients'][rep_ind] > 0.5 and causal_seqs_ind.sum() > 0:
                    # Get effects of sequences with and without causal motif.
                    est_effect_causal = est_effect[causal_seqs_ind]
                    true_effect_causal = true_effect[causal_seqs_ind]
                    causal_weights = candidate_counts[causal_seqs_ind] / candidate_counts[causal_seqs_ind].sum()
                    est_effect_noncausal = est_effect[~causal_seqs_ind]
                    true_effect_noncausal = true_effect[~causal_seqs_ind]
                    noncausal_weights = candidate_counts[~causal_seqs_ind] / candidate_counts[~causal_seqs_ind].sum()
                    # Compute average effect.
                    s['est_effect_causal_mn'], s['est_effect_causal_sd'] = est_effect_causal.mean().cpu(), est_effect_causal.std().cpu()
                    s['true_effect_causal_mn'], s['true_effect_causal_sd'] = true_effect_causal.mean().cpu(), true_effect_causal.std().cpu()
                    s['est_effect_noncausal_mn'], s['est_effect_noncausal_sd'] = est_effect_noncausal.mean().cpu(), est_effect_noncausal.std().cpu()
                    s['true_effect_noncausal_mn'], s['true_effect_noncausal_sd'] = true_effect_noncausal.mean().cpu(), true_effect_noncausal.std().cpu()
                    # Compute effect error.
                    effect_error = ((est_effect_causal[:, None] - est_effect_noncausal[None, :]) -
                                    (true_effect_causal[:, None] - true_effect_noncausal[None, :]))
                    s['effect_causal_error_l1'] = effect_error.abs().mean().cpu()
                    s['effect_causal_error_l2'] = effect_error.pow(2).mean().sqrt().cpu()
                    error_weight = causal_weights[:, None] * noncausal_weights[None, :]
                    s['effect_causal_error_l1_weighted'] = (effect_error.abs() * error_weight).sum().cpu()
                    s['effect_causal_error_l2_weighted'] = (effect_error.pow(2) * error_weight).sum().cpu()

                    # Evaluate PR-AUC and AUC.
                    causal_seqs_ind_cpu, est_effect_cpu = causal_seqs_ind.cpu(), est_effect.cpu()
                    s['causal_noncausal_pr_auc'] = average_precision_score(causal_seqs_ind_cpu, est_effect_cpu)
                    s['causal_noncausal_auc'] = roc_auc_score(causal_seqs_ind_cpu, est_effect_cpu)
                    # Compare to random baseline.
                    s['causal_noncausal_pr_auc_baseline'] = average_precision_score(causal_seqs_ind_cpu, torch.randn_like(est_effect_cpu))
                    s['causal_noncausal_auc_baseline'] = roc_auc_score(causal_seqs_ind_cpu, torch.randn_like(est_effect_cpu))

                    # Compare to sequences affected by the confounder.
                    if np.sum(data.h5f['confounder'][rep_ind]) > 0.5 and confounder_seqs_ind.sum() > 0:
                        # Get effects of sequences with confounder motif.
                        est_effect_confounder = est_effect[confounder_seqs_ind]
                        true_effect_confounder = true_effect[confounder_seqs_ind]
                        # Compute average effect.
                        s['est_effect_confounder_mn'], s['est_effect_confounder_sd'] = est_effect_confounder.mean().cpu(), est_effect_confounder.std().cpu()
                        s['true_effect_confounder_mn'], s['true_effect_confounder_sd'] = true_effect_confounder.mean().cpu(), true_effect_confounder.std().cpu()

                        # Evaluate AUC.
                        causal_confound_ind = torch.bitwise_or(causal_seqs_ind, confounder_seqs_ind)
                        s['causal_confounder_auc'] = roc_auc_score(causal_seqs_ind[causal_confound_ind].cpu(),
                                                                   est_effect[causal_confound_ind].cpu())
                        s['causal_confounder_auc_baseline'] = roc_auc_score(causal_seqs_ind[causal_confound_ind].cpu(),
                                                                            torch.randn_like(est_effect[causal_confound_ind]).cpu())

            # Store results.
            aim_run.track(s)
            for ke in effect_keys:
                effect_summaries[ke].append(s[ke])

        for ke in effect_keys:
            effect_summaries[ke] = torch.tensor(effect_summaries[ke])

        # Break into train/test, summarize further.
        data_subsets = {'train': train_ind, 'validate': validate_ind, 'test': test_ind, 'all': range(len(data))}
        effect_results = {}
        for subset in data_subsets:
            ind = data_subsets[subset]
            for ke in effect_keys:
                sub = effect_summaries[ke][ind]
                sub = sub[~sub.isnan()]
                effect_results[ke + '_' + subset + '_mn'] = sub.mean().numpy()
        aim_run.track(effect_results)

        return effect_results

    def evaluate_bind(self, datafile, summaries, aim_run):
        """Evaluate predicted intervention effects against laboratory binding data."""
        # Load evaluation data.
        data = BindingDataset(datafile, self.args.eval_bind, self.args.cuda)
        batch_size = self.args.unit_batch_eval * self.args.subunit_batch_eval
        dataload = DataLoader(data, batch_size=batch_size, shuffle=False,
                              num_workers=self.args.num_workers, pin_memory=self.args.cuda, drop_last=False,
                              persistent_workers=False)

        # Function for computing intervention effects based on learned model.
        intervene_effect_est = InterventionEffect(
            self.args.intervene_frac, summaries['W_A'], summaries['treatment_contrib'],
            summaries['treatment_normalizer_ln'], summaries['confound_contrib'], summaries['base_contrib'],
            self.args.outcome, max_batch=batch_size,
            repertoire_featurizer=self.repertoire_featurizer, no_attention=self.args.no_attention,
            low_dtype=self.args.low_dtype, cuda=self.args.cuda)

        # Compute the effect of each sequence, if it was used for an intervention.
        est_effects = torch.zeros(len(data), dtype=torch.float)
        seq_ind = 0
        for candidates, _ in dataload:
            with torch.no_grad():
                # Transfer to cuda and featurize data.
                candidates = self.data_featurizer.featurize_seqs(candidates.unsqueeze(0))
                # Compute estimated intervention effect.
                est_effects[seq_ind:(seq_ind+candidates.shape[1])] = intervene_effect_est(candidates).cpu()
                seq_ind += candidates.shape[1]

            if self.args.smoke:
                break

        # Save effect estimates.
        est_effect_table = pd.DataFrame({'estimated_effect': est_effects.numpy(),
                                         'hit': data.hits.numpy(),
                                         'patient': data.patients.numpy(),
                                         'mhc_class_1_eval': data.mhc_class_eval[1].numpy(),
                                         'mhc_class_2_eval': data.mhc_class_eval[2].numpy()})
        est_effect_table.to_csv(os.path.join(aim_run['local_dir'], 'est_effect.csv'), index=False)

        # Compare effect estimates against hits, aggregated by class.
        hits = data.hits.to(torch.float)
        results = dict()
        for mc in [1, 2]:
            mcind = data.mhc_class_eval[mc]
            hit_sub = hits[mcind]
            effect_sub = est_effects[mcind]
            # AUC.
            results['bind_effect_auc_class_' + str(mc)] = roc_auc_score(hit_sub, effect_sub)
            # Baselines.
            baseline_samples = self.args.baseline_samples
            auc_baselines = np.zeros(baseline_samples)
            for k in range(baseline_samples):
                rand_effects = est_effects[torch.randperm(len(hit_sub))]
                auc_baselines[k] = roc_auc_score(hit_sub, rand_effects)
            results['bind_effect_auc_baseline_mn_class_' + str(mc)] = np.mean(auc_baselines)
            results['bind_effect_auc_baseline_upper_quantile_class_' + str(mc)] = np.quantile(auc_baselines, self.args.baseline_quantile)
            results['bind_effect_auc_baseline_lower_quantile_class_' + str(mc)] = np.quantile(auc_baselines, 1-self.args.baseline_quantile)

        # Compare effect estimates against hits, aggregated by patient.
        patient_aucs = np.zeros(data.num_patients)
        patient_weights = np.zeros(data.num_patients)
        for p_i in range(data.num_patients):
            p_ind = data.patients == p_i
            hit_sub = hits[p_ind]
            effect_sub = est_effects[p_ind]
            patient_weights[p_i] = torch.sum(hit_sub)
            # AUC.
            patient_aucs[p_i] = roc_auc_score(hit_sub, effect_sub)

        patient_weights = patient_weights / np.sum(patient_weights)

        # Weighted sample variance (https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance)
        def weighted_mean_std(val, wgt):
            w_mn = np.sum(val * wgt)
            w_std = np.sqrt(np.sum(wgt * ((val - w_mn) ** 2)) / (1 - np.sum(wgt ** 2)))
            return w_mn, w_std

        w_mean, w_std = weighted_mean_std(patient_aucs, patient_weights)
        results['bind_effect_auc_patient_mean'] = w_mean
        results['bind_effect_auc_patient_std'] = w_std
        results['bind_effect_auc_patient_se'] = w_std / np.sqrt(data.num_patients)

        aim_run.track(results)

        # Plot distribution of estimated effects.
        aim_run.track(Distribution(est_effects[hits < 0.5]), name='nonhit-effects')
        aim_run.track(Distribution(est_effects[hits > 0.5]), name='hit-effects')

        return results


def main(args):

    # Set up logging.
    aim_run = create_run('caire-model', args)
    if args.log_output is not None:
        with open(args.log_output, 'a') as lof:
            lof.write('{},{}\n'.format(args.split_choice, aim_run['local_dir']))

    # Use gpu w/most available memory.
    free_mem = np.zeros(torch.cuda.device_count())
    for j in range(torch.cuda.device_count()):
        with torch.cuda.device(j):
            free_mem[j] = torch.cuda.mem_get_info()[0]
    torch.cuda.set_device(int(np.argmax(free_mem)))

    # Set pytorch defaults and backend.
    torch.set_default_dtype(torch.float32)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32

    # Check distribution arguments. Good for debugging, but slows down code on the gpu.
    pyro.enable_validation(args.no_jit)

    # Store the data set on the gpu or the cpu.
    if args.cuda_data:
        # Deprecated, can remove option.
        data_device = 'cuda'
    else:
        data_device = 'cpu'

    # Set first-layer dtype (can be a lower precision than float32 for faster training.)
    args.low_dtype = getattr(torch, args.low_dtype)

    # Load data for training.
    # Note: this random number generator is used for the subunit (sequence) batching.
    data_gen_M = torch.Generator(device=data_device).manual_seed(args.data_seed)
    train_data = RepertoiresDataset(args.datafile, outcome_type=args.outcome, flip_outcome=args.flip_outcome,
                                    seq_batch=args.subunit_batch, uniform_sample_seq=args.uniform_sample_seq,
                                    dtype=args.low_dtype, generator=data_gen_M, cuda_data=args.cuda_data,
                                    cuda=args.cuda, synthetic_test=args.synthetic_test)
    validate_data = RepertoiresDataset(args.datafile, outcome_type=args.outcome, flip_outcome=args.flip_outcome,
                                       seq_batch=args.subunit_batch_eval, uniform_sample_seq=args.uniform_sample_seq,
                                       dtype=args.low_dtype, generator=data_gen_M, cuda_data=args.cuda_data,
                                       cuda=args.cuda, synthetic_test=args.synthetic_test)

    # Initialize model.
    model = CausalRepertoireModel(args, train_data)

    # Train-validate-test split.
    # Note: this random number generator is used for the unit (patient) batching.
    data_gen_N = torch.Generator().manual_seed(args.data_seed+1000)
    splits = args.splits
    split_choice = args.split_choice
    validate_choice = (split_choice - 1) % splits
    train_choice = [c for c in list(range(splits)) if c not in [split_choice, validate_choice]]
    if args.stratify_cv:
        # Stratify splits so they have roughly even numbers of each outcome.
        test_ind, validate_ind, train_ind = [], [], []
        for oi, rout in enumerate(torch.unique(train_data.outcomes)):
            splits_inds = random_split(torch.arange(len(train_data))[torch.isclose(train_data.outcomes.squeeze(), rout)],
                                       np.ones(splits) / splits, generator=data_gen_N)
            test_ind.append(splits_inds[split_choice])
            validate_ind.append(splits_inds[validate_choice])
            train_ind.append(ConcatDataset([splits_inds[c] for c in train_choice]))
        test_ind = ConcatDataset(test_ind)
        validate_ind = ConcatDataset(validate_ind)
        train_ind = ConcatDataset(train_ind)
    else:
        splits_inds = random_split(range(len(train_data)), np.ones(splits)/splits, generator=data_gen_N)
        test_ind = splits_inds[split_choice]
        validate_ind = splits_inds[validate_choice]
        train_ind = ConcatDataset([splits_inds[c] for c in train_choice])

    train_data = Subset(train_data, train_ind)
    validate_data = Subset(validate_data, validate_ind)

    # Train.
    if args.pretrained_model_params is None:
        model.fit_svi(train_data, validate_data, aim_run, generator=data_gen_N)

        # Record average weights if using SWA.
        if args.weight_average:
            model.best_model = f"{aim_run['local_dir']}/average_model"
            for elem in model.weight_average_params:
                torch.save(model.weight_average_params[elem].cpu().to(dtype=torch.float32)/model.weight_steps,
                           f"{model.best_model}_{elem}.pt")

        # Save validation results.
        result_summary = (float(model.validation_score), float(model.validation_select_accuracy_mean),
                          float(model.validation_propensity_pearson_mean), float(model.validation_outcome_score),
                          float(model.elbo_average), float(model.convergence_diagnostic),)
        validation_scores_file = os.path.join(aim_run['local_dir'], 'validation_scores.npy')
        np.save(validation_scores_file, np.array(result_summary))
        # Return validation results
        print('ValidationResults:', validation_scores_file, model.best_model)

    if not args.no_test_set_evaluation:
        # Load data for embedding and evaluation.
        data = RepertoiresDataset(args.datafile, outcome_type=args.outcome, flip_outcome=args.flip_outcome,
                                  seq_batch=args.subunit_batch_eval, uniform_sample_seq=args.uniform_sample_seq,
                                  dtype=args.low_dtype, generator=data_gen_M, cuda_data=args.cuda_data, cuda=args.cuda,
                                  synthetic_test=args.synthetic_test)

        # Take just-trained model or pre-trained model. If early stopping is on, the best_model parameters aren't
        # necessarily the final model parameters.
        if args.pretrained_model_params is None:
            model_params = model.best_model
        else:
            model_params = args.pretrained_model_params

        # Set up reproducible batch for analyzing effect distribution.
        if args.eval_effect_dist:
            data_eval_effect_dist = RepertoiresDataset(
                    args.datafile, outcome_type=args.outcome, flip_outcome=args.flip_outcome,
                    seq_batch=args.subunit_batch, uniform_sample_seq=args.uniform_sample_seq,
                    dtype=args.low_dtype, generator=data_gen_M, cuda_data=args.cuda_data,
                    cuda=args.cuda, synthetic_test=args.synthetic_test,
                    deterministic_batch=True)
        else:
            data_eval_effect_dist = None

        # Evaluate and embed.
        evaluation_results, evaluation_summaries, est_natural_effects, natural_counts = model.evaluate(
                    data, train_ind, validate_ind, test_ind, aim_run, model_params=model_params,
                    data_eval_effect_dist=data_eval_effect_dist)

        # For semisynthetic data, evaluate against ground truth.
        if args.eval_true:
            eval_results_true = model.evaluate_true(data, evaluation_summaries, train_ind, validate_ind, test_ind,
                                                    aim_run)
            evaluation_results.update(eval_results_true)
        elif args.eval_bind != 'NA':
            eval_results_bind = model.evaluate_bind(args.datafile, evaluation_summaries, aim_run)
            evaluation_results.update(eval_results_bind)

        # Return evaluation results.
        eval_results_file = os.path.join(aim_run['local_dir'], 'evaluation_results.pkl')
        with open(eval_results_file, 'wb') as f:
            pickle.dump(evaluation_results, f)
            pickle.dump(est_natural_effects.numpy(), f)
            pickle.dump(natural_counts.numpy(), f)
            pickle.dump(data.outcomes.numpy(), f)
        print(f'TestResults: {eval_results_file}')

        return evaluation_results


class DefaultArgs:
    def __init__(self):
        self.outcome = 'binary'
        self.flip_outcome = False
        self.select_latent_dim = 2
        self.repertoire_latent_dim = 3
        self.select_embed_layer_dim = 16
        self.posterior_rank = 2
        self.selection_channels = 12
        self.encoder_channels = 16
        self.conv_kernel = 5
        self.selection_conv_kernel = 3
        self.encoder_conv_kernel = 7
        self.n_attention_layers = 2
        self.n_selection_layers = 1
        self.n_encoder_attention_layers = 2
        self.n_attention_units = 32
        self.n_selection_units = 16
        self.n_encoder_attention_units = 32
        self.top_fraction = 1.0
        self.encoder_top_fraction = 1.0
        self.batch_standardize = False
        self.pos_encode = True
        self.blosum_encode = False
        self.optimizer = 'Adam'
        self.unit_batch = 3
        self.subunit_batch = 10
        self.unit_batch_eval = 3
        self.subunit_batch_eval = -1
        self.lr = 0.001
        self.epochs = 10
        self.anneal = 5
        self.cuda = False
        self.seed = 12345
        self.data_seed = 0
        self.splits = 10
        self.split_choice = 0
        self.stratify_cv = False
        self.monitor_iter = 5
        self.validate_iter = 100
        self.ninit = 3
        self.separate_propensity = False
        self.propensity_update = 5
        self.no_outcome = False
        self.no_selection = False
        self.no_propensity = False
        self.profile = False
        self.no_jit = False
        self.low_dtype = 'bfloat16'
        self.cuda_data = False
        self.no_tf32 = False
        self.num_workers = 0
        self.num_torch_threads = 1
        self.elbo_early_stop = False
        self.max_time = 20
        self.smoke = False
        self.no_test_set_evaluation = False
        self.pretrained_model_params = None

        self.synthetic_test = False

        self.sum_pool = False
        self.no_attention = False
        self.encoder_no_attention = False
        self.linear_cnn = False

        self.anneal_time = False
        self.weight_decay = 0.
        self.no_early_stop = False
        self.weight_average = False
        self.weight_fraction = 0.75
        self.monitor_converge = False

        self.select_posterior_rank = 0
        self.approx_batch_standardize = False
        self.uniform_sample_seq = False

        self.intervene_frac = 0.1
        self.eval_true = False
        self.eval_bind = 'NA'
        self.baseline_samples = 5
        self.baseline_quantile = 1.0
        self.eval_samples = 5
        self.eval_effect_dist = False

        self.log_output = None


if __name__ == "__main__":
    # Get default argument values.
    defaults = DefaultArgs()
    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Causal inference for immune receptor repertoires.")
    parser.add_argument('datafile', help='Input data')
    parser.add_argument('--outcome', default=defaults.outcome, help='Options: binary OR continuous')
    parser.add_argument('--flip-outcome', default=defaults.flip_outcome, action="store_true",
                        help='Reverse sign of outcome (not important for model, but does change evaluation).')
    parser.add_argument('--select-latent-dim', default=defaults.select_latent_dim, type=int)
    parser.add_argument('--repertoire-latent-dim', default=defaults.repertoire_latent_dim, type=int)
    parser.add_argument('--select-embed-layer-dim', default=defaults.select_embed_layer_dim, type=int)
    parser.add_argument('--posterior-rank', default=defaults.posterior_rank, type=int)
    parser.add_argument('--select-posterior-rank', default=defaults.select_posterior_rank,
                        type=int, help='Rank of selection latent representation posterior.')
    parser.add_argument('--selection-channels', default=defaults.selection_channels, type=int)
    parser.add_argument('--encoder-channels', default=defaults.encoder_channels, type=int)
    parser.add_argument('--conv-kernel', default=defaults.conv_kernel, type=int, help='Must be odd.')
    parser.add_argument('--selection-conv-kernel', default=defaults.selection_conv_kernel, type=int, help='Must be odd.')
    parser.add_argument('--encoder-conv-kernel', default=defaults.encoder_conv_kernel, type=int,
                        help='Must be odd.')
    parser.add_argument('--no-attention', default=defaults.no_attention, action="store_true",
                        help='Do not weight by attention score when producing repertoire embedding.')
    parser.add_argument('--n-attention-layers', default=defaults.n_attention_layers, type=int)
    parser.add_argument('--n-selection-layers', default=defaults.n_selection_layers, type=int)
    parser.add_argument('--encoder-no-attention', default=defaults.encoder_no_attention, action="store_true",
                        help='Do not weight by attention score when producing repertoire embedding.')
    parser.add_argument('--n-encoder-attention-layers', default=defaults.n_encoder_attention_layers, type=int)
    parser.add_argument('--n-attention-units', default=defaults.n_attention_units, type=int)
    parser.add_argument('--n-selection-units', default=defaults.n_selection_units, type=int)
    parser.add_argument('--n-encoder-attention-units', default=defaults.n_encoder_attention_units, type=int)
    parser.add_argument('--top-fraction', default=defaults.top_fraction, type=float)
    parser.add_argument('--encoder-top-fraction', default=defaults.encoder_top_fraction, type=float)
    parser.add_argument('--pos-encode', default=defaults.pos_encode, action="store_true")
    parser.add_argument('--blosum-encode', default=defaults.blosum_encode, action="store_true")
    parser.add_argument('--sum-pool', default=defaults.sum_pool, action="store_true",
                        help='Use sum pooling instead of max pooling across positions in sequence embedding.')
    parser.add_argument('--linear-cnn', default=defaults.linear_cnn, action="store_true",
                        help='Remove nonlinear activation from CNN.')
    parser.add_argument('--batch-standardize', default=defaults.batch_standardize, action="store_true",
                        help='Divide input batch data by its standard deviation.')
    parser.add_argument('--optimizer', default=defaults.optimizer, help='Options: Adam OR NAdam')
    parser.add_argument('--unit-batch', default=defaults.unit_batch, type=int)
    parser.add_argument('--subunit-batch', default=defaults.subunit_batch, type=int)
    parser.add_argument('--unit-batch-eval', default=defaults.unit_batch_eval, type=int)
    parser.add_argument('--subunit-batch-eval', default=defaults.subunit_batch_eval, type=int)
    parser.add_argument('--lr', default=defaults.lr, type=float)
    parser.add_argument('--epochs', default=defaults.epochs, type=int)
    parser.add_argument('--anneal', default=defaults.anneal, type=float)
    parser.add_argument('--anneal-time', default=defaults.anneal_time, action="store_true",
                        help='Anneal based on time, rather than epochs')
    parser.add_argument('--cuda', default=defaults.cuda, action="store_true")
    parser.add_argument('--data-seed', default=defaults.data_seed, type=int,
                        help='Random seed for data (splitting and batching).')
    parser.add_argument('--splits', default=defaults.splits, type=int,
                        help='Number of splits used for cross validation')
    parser.add_argument('--split-choice', default=defaults.split_choice, type=int,
                        help='Split to use for testing (integer from zero to value of splits).')
    parser.add_argument('--stratify-cv', default=defaults.stratify_cv, action="store_true",
                        help='Stratify splits based on outcome.')
    parser.add_argument('--log-output', default=defaults.log_output, type=str,
                        help='Log path to output to file, along with split-choice, for cross-validation.')
    parser.add_argument('--seed', default=defaults.seed, type=int, help='Random seed for model (initialization and training).')
    parser.add_argument('--monitor-iter', default=defaults.monitor_iter, type=int,
                        help='How often (# of steps) to transfer loss back to cpu and log.')
    parser.add_argument('--validate-iter', default=defaults.validate_iter, type=int,
                        help='How often (# of steps) to run more expensive model evaluation for early stopping.')
    parser.add_argument('--ninit', default=defaults.ninit, type=int,
                        help='Number of initializations to try.')
    parser.add_argument('--separate-propensity', default=defaults.separate_propensity, action="store_true",
                        help='Update propensity model separately.')
    parser.add_argument('--propensity-update', default=defaults.propensity_update, type=int)
    parser.add_argument('--no-outcome', default=defaults.no_outcome, action="store_true",
                        help='Drop outcome model (selection alone).')
    parser.add_argument('--no-selection', default=defaults.no_selection, action="store_true",
                        help='Drop selection model (outcome alone).')
    parser.add_argument('--no-propensity', default=defaults.no_propensity, action="store_true",
                        help='Drop propensity model.')
    parser.add_argument('--profile', default=defaults.profile, action="store_true",
                        help='Profile code performance.')
    parser.add_argument('--no-jit', default=defaults.no_jit, action="store_true",
                        help='No Jit ELBO.')
    parser.add_argument('--low-dtype', default=defaults.low_dtype, type=str,
                        help='Lower precision dtype (float32, float16 OR bfloat16.')
    parser.add_argument('--cuda-data', default=defaults.cuda_data, action="store_true",
                        help='Store data on the gpu.')
    parser.add_argument('--no-tf32', default=defaults.no_tf32, action="store_true",
                        help='Use TensorFloat-32 tensor cores.')
    parser.add_argument('--num-workers', default=defaults.num_workers, type=int,
                        help='Number of workers to use in data loading (should only be > 0 if using a gpu).')
    parser.add_argument('--num-torch-threads', default=defaults.num_torch_threads, type=int,
                        help='Number of threads to use for pytorch intraop parallelism.')
    parser.add_argument('--elbo-early-stop', default=defaults.elbo_early_stop, action="store_true",
                        help='Use ELBO for early stopping, rather than an accuracy-based metric.')
    parser.add_argument('--max-time', default=defaults.max_time, type=float,
                        help='Maximum training time, in minutes.')
    parser.add_argument('--no-test-set-evaluation', default=defaults.no_test_set_evaluation,
                        action="store_true", help='Do not perform final model evaluation on test set.')
    parser.add_argument('--pretrained-model-params', default=defaults.pretrained_model_params,
                        help='Do not train, instead just evaluate a pretrained model with the given parameters.')

    # Additional training parameters.
    parser.add_argument('--weight-decay', default=defaults.weight_decay, type=float)
    parser.add_argument('--no-early-stop', default=defaults.no_early_stop, action="store_true")
    parser.add_argument('--weight-average', default=defaults.weight_average, action="store_true",
                        help='Use weight averaging (SWA).')
    parser.add_argument('--weight-fraction', default=defaults.weight_fraction, type=float,
                        help='Fraction of max training time after which weights will be averaged.')
    parser.add_argument('--monitor-converge', default=defaults.monitor_converge, action="store_true",
                        help='Track difference in parameters to monitor convergence.')
    parser.add_argument('--approx-batch-standardize', default=defaults.approx_batch_standardize,
                        action="store_true",
                        help='Normalize by a constant, rough estimate of typical batch standardization values')
    parser.add_argument('--uniform-sample-seq', default=defaults.uniform_sample_seq, action="store_true",
                        help='Sample mature sequences based on raw distribution, rather than clonotype distribution.')

    # Evaluation against ground truth (for semisynthetic tests) or laboratory binding data
    parser.add_argument('--eval-true', default=defaults.eval_true, action="store_true",
                        help='Evaluate against ground truth effects, available for semisynthetic data.')
    parser.add_argument('--intervene-frac', default=defaults.intervene_frac, type=float,
                        help='Fraction of repertoire with intervened (candidate therapeutic) sequence.')
    parser.add_argument('--eval-bind', default=defaults.eval_bind, type=str,
                        help='Evaluate against laboratory binding data.')
    parser.add_argument('--baseline-samples', default=defaults.baseline_samples, type=int,
                        help='Number of random samples used to get baseline performance.')
    parser.add_argument('--baseline-quantile', default=defaults.baseline_quantile, type=float,
                        help='Quantile of random samples used to get baseline performance.')
    parser.add_argument('--eval-samples', default=defaults.eval_samples, type=int,
                        help='Number of resamples to use in evaluation (randomness over subunit batch).')
    parser.add_argument('--eval-effect-dist', default=defaults.eval_effect_dist, action="store_true",
                        help='Evaluate the distribution of estimated effects for patient repertoires.')

    # Additional tools for testing code.
    parser.add_argument('--smoke', default=defaults.smoke, action="store_true", help='Smoke test.')
    parser.add_argument('--synthetic-test', default=defaults.synthetic_test, action="store_true",
                        help='Perform simple synthetic data test, for sanity checks (inject motif into data).')

    args0 = parser.parse_args()

    main(args0)