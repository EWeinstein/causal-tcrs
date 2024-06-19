"""
Summarize the results of cross-validation of CAIRE.

This file takes as input the results of several runs of CAIRE, with different model splits.
It produces summary statistics and consolidated files.
"""

from aim import Figure
import argparse
import numpy as np
import os
import pandas as pd
import pickle
import plotly.express as px
import subprocess

from CausalReceptors.manager import create_run
from sklearn import metrics as skl_metrics


def main(args):
    # Set up logging.
    aim_run = create_run('cir-cv', args)

    # --- Cross validation ---
    # Set up data structure to store results.
    results = []
    local_dirs = []
    bind_effects = []
    natural_effects = []
    natural_counts = []
    select_embeds = []
    # Iterate over test sets.
    with open(args.file, 'r') as lf:
        for line in lf:

            # Obtain test evaluation file and binding effect file.
            split_choice, local_dir = line.strip('\n').split(',')
            test_eval_file = os.path.join(local_dir, 'evaluation_results.pkl')
            effect_file = os.path.join(local_dir, 'est_effect.csv')
            local_dirs.append(local_dir)

            # Load results.
            with open(test_eval_file, 'rb') as f:
                results.append(pickle.load(f))
                natural_effects.append(pickle.load(f)[:, :, None])
                natural_counts.append(pickle.load(f)[:, :, None])
                outcomes = pickle.load(f)  # (same value for each split choice.)
            est_effect_table = pd.read_csv(os.path.join(local_dir, 'est_effect.csv'))
            bind_effects.append(np.array(est_effect_table['estimated_effect'])[:, None])
            select_embed_table = pd.read_csv(os.path.join(local_dir, 'select_embed.csv'))
            select_embeds.append(np.array(select_embed_table)[:, :, None])

            # Track.
            aim_run.track({'split-choice': int(split_choice)})
    bind_effects = np.concatenate(bind_effects, axis=1)
    natural_effects = np.concatenate(natural_effects, axis=2)
    hits = np.array(est_effect_table['hit'])
    select_embeds = np.concatenate(select_embeds, axis=2)

    # --- Plotting and summaries ---
    # Save results
    results_file = os.path.join(aim_run['local_dir'], 'cv_results.pkl')
    with open(results_file, 'wb') as f:
        pickle.dump(results, f)
        pickle.dump(local_dirs, f)
        pickle.dump(bind_effects, f)
        pickle.dump(hits, f)
        pickle.dump(natural_effects, f)
        pickle.dump(outcomes, f)
        pickle.dump(select_embeds, f)

    # Average, std, and s.e. of test set r^2.
    test_r2 = np.array([result['outcome_r2_test'] for result in results])
    aim_run.track({'outcome_r2_test_mean': np.mean(test_r2),
                   'outcome_r2_test_sd': np.std(test_r2),
                   'outcome_r2_test_se': np.std(test_r2)/np.sqrt(len(test_r2))})
    # Average, std, and s.e. of test set {treatment, confounder} variance explained.
    for val in ['treatment', 'confound']:
        for mtc in ['explained', 'r2']:
            test_explained = np.array([result['outcome_{}_{}_test'.format(val, mtc)] for result in results])
            aim_run.track({'outcome_{}_{}_mean'.format(val, mtc): np.mean(test_explained),
                           'outcome_{}_{}_sd'.format(val, mtc): np.std(test_explained),
                           'outcome_{}_{}_se'.format(val, mtc): np.std(test_explained)/np.sqrt(len(test_explained))})

    # Average, std, and s.e. of bind effect AUC
    # Average effect and std/mean per binder.
    binder_effect_mn = np.mean(bind_effects, axis=1)
    binder_effect_sd = np.std(bind_effects, axis=1)

    # Average effect performance at discriminating hits.
    for mc in [1, 2]:
        mcind = np.array(est_effect_table['mhc_class_{}_eval'.format(mc)])
        hit_sub = hits[mcind]
        bem_sub = binder_effect_mn[mcind]
        aim_run.track({'bind_effect_auc_class_{}'.format(mc): skl_metrics.roc_auc_score(hit_sub, bem_sub),
                       'bind_effect_pr_auc_{}'.format(mc): skl_metrics.average_precision_score(hit_sub, bem_sub),
                       'bind_effect_negative_pr_auc_{}'.format(mc): skl_metrics.average_precision_score(hit_sub, -bem_sub)})

    patients = np.array(est_effect_table['patient'])
    num_patients = np.max(patients) + 1
    patient_aucs = np.zeros(num_patients)
    patient_abs_aucs = np.zeros(num_patients)
    patient_weights = np.zeros(num_patients)
    for p_i in range(num_patients):
        p_ind = patients == p_i
        hit_sub = hits[p_ind]
        effect_sub = binder_effect_mn[p_ind]
        patient_weights[p_i] = np.sum(hit_sub)
        # AUC.
        patient_aucs[p_i] = skl_metrics.roc_auc_score(hit_sub, effect_sub)
        patient_abs_aucs[p_i] = skl_metrics.roc_auc_score(hit_sub, np.abs(effect_sub))
    patient_weights = patient_weights / np.sum(patient_weights)

    def weighted_mean_std(val, wgt):
        w_mn = np.sum(val * wgt)
        w_std = np.sqrt(np.sum(wgt * ((val - w_mn) ** 2)) / (1 - np.sum(wgt ** 2)))
        return w_mn, w_std

    w_mean, w_std = weighted_mean_std(patient_aucs, patient_weights)
    aim_run.track({'bind_effect_auc_patient_mean': w_mean,
                   'bind_effect_auc_patient_std': w_std,
                   'bind_effect_auc_patient_se': w_std / np.sqrt(num_patients)})
    w_mean, w_std = weighted_mean_std(patient_abs_aucs, patient_weights)
    aim_run.track({'bind_effect_abs_auc_patient_mean': w_mean,
                   'bind_effect_abs_auc_patient_std': w_std,
                   'bind_effect_abs_auc_patient_se': w_std / np.sqrt(num_patients)})

    # Volcano plot of binder effects.
    binder_effect_se = binder_effect_sd / np.sqrt(len(test_r2))
    binder_effect_volcano = px.scatter(x=binder_effect_mn[hits > 0.5],
                                       y=np.abs(binder_effect_mn[hits > 0.5]) / binder_effect_sd[hits > 0.5])
    aim_run.track(Figure(binder_effect_volcano), name='binder_effect_volcano')

    # Get natural sequence effect distribution for patients with different outcomes.
    natural_effect_summary = dict()
    natural_effects_mn = np.mean(natural_effects, axis=-1)
    effect_bins = np.linspace(np.min(natural_effects_mn), np.max(natural_effects_mn), 51)
    natural_effect_summary['effect'] = effect_bins[:-1] + np.diff(effect_bins)[0]/2
    legend_names = []
    for rout in np.unique(outcomes):
        out_ind = np.isclose(outcomes.squeeze(), rout)
        legend_name = 'outcome_{}'.format(rout)
        natural_effect_summary[legend_name] = np.histogram(
                natural_effects_mn[out_ind].reshape([-1]), bins=effect_bins, density=True)[0]
        legend_names.append(legend_name)
    natural_effect_distr = px.line(natural_effect_summary, x='effect', y=legend_names)
    aim_run.track(Figure(natural_effect_distr), name='natural_effect_distr')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize results of cross validation of CAIRE.")
    parser.add_argument('file', help='Input file with logs from individual model runs.')
    args0 = parser.parse_args()

    main(args0)
