# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
# import csv
import warnings
from itertools import chain
from statistics import mean

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import scipy

import seaborn as sns

UM_PER_PX = 1.225


def smp_trks_len(tracks, start_frame=None, end_frame=None):
    sampled_tracks = tracks
    for frame in [start_frame, end_frame]:
        if frame is not None:
            sampled_tracks = [t for t in sampled_tracks if any(
                              det.frame_num == frame for det in t.dets)]

    return sampled_tracks


def smp_trks_le_dist(tracks, ldg_es, dist_btm_le=None, dist_top_le=None,
                     num_btm_le=None, num_top_le=None):
    sampled_tracks = []
    top_le, btm_le = ldg_es
    for le_trks, le, num_le, dist_le, op in (
            zip([tracks, tracks], [top_le, btm_le], [num_top_le, num_top_le],
                [dist_top_le, dist_btm_le], [1, -1])):
        le_trks = [t for t in le_trks if op * t.dets[0].centre_y <= op * le]
        if dist_le is not None:
            le_trks = [t for t in le_trks if
                       op * t.dets[0].centre_y >= op * (le - op * dist_le)]
        if num_le is not None:
            le_trks = le_trks[0:num_le]
            # le_trks = rd.sample(le_trks, num_btm_le)
        sampled_tracks += le_trks

    return sampled_tracks


def smp_trks_le_propn(tracks, ldg_es, propn, img_hgt, num_btm_le=None,
                      num_top_le=None):
    top_le, btm_le = ldg_es
    btm_le += ((img_hgt - btm_le) * (1 - propn))
    top_le *= propn
    sampled_tracks = smp_trks_le_dist(tracks, (top_le, btm_le),
                                      num_btm_le=num_btm_le,
                                      num_top_le=num_top_le)

    return sampled_tracks


def get_axes_limits(xy_pts, ax_pad=50):
    lims = []
    for m, i, op in [(min, 0, -1), (min, 1, -1), (max, 0, 1), (max, 1, 1)]:
        lims.append(round(m(list(chain.from_iterable([x[i] for x in xy_pts]))
                            ) + op * ax_pad))

    return lims


def create_traj_axes(xy_pts):
    fig = plt.figure()
    fig.tight_layout()
    ax = plt.gca()
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    lims = get_axes_limits(xy_pts)
    ax.set_xlim(left=lims[0], right=lims[2])
    ax.set_ylim(bottom=lims[1], top=lims[3])

    return ax


def traj_plots(img_cen_y, *tracks, length=None, inc_combined=True, um=True):
    for trks in tracks:
        xy_pts = []
        length = (min([len(t.dets) for t in trks]) if length is None else
                  length)
        for trk in trks:
            pts = [[0], [0]]
            for det in trk.dets[1:length]:
                op = -1 if trk.dets[0].centre[1] < img_cen_y else 1
                x_pt = -(trk.dets[0].centre[0] - det.centre[0])
                y_pt = op * (trk.dets[0].centre[1] - det.centre[1])
                if um:
                    x_pt *= UM_PER_PX
                    y_pt *= UM_PER_PX
                pts[0].append(x_pt)
                pts[1].append(y_pt)
            xy_pts.append(pts)

        ax_label = 'Micrometres (um)' if um else 'No. Pixels (px.)'
        ax = create_traj_axes(xy_pts)
        ax.set_title(f'Individual Trajectories ({trks[0].method})')
        ax.set_xlabel(ax_label)
        ax.set_ylabel(ax_label)

        for x, y in xy_pts:
            plt.plot(x, y, 'g-')

        if not inc_combined:
            return

        avg_xy_pts = [[], []]
        for i in range(0, length):
            pts = [[], []]
            for x, y in xy_pts:
                if x[0] != 0 or y[0] != 0:
                    print()
                pts[0].append(x[i])
                pts[1].append(y[i])
            avg_xy_pts[0].append(mean(pts[0]))
            avg_xy_pts[1].append(mean(pts[1]))

        ax = create_traj_axes(xy_pts)
        ax.set_title(f'Combined Trajectories ({trks[0].method})')
        ax.set_xlabel(ax_label)
        ax.set_ylabel(ax_label)

        plt.plot(avg_xy_pts[0], avg_xy_pts[1], 'g-')


def compare_metric_methods(auto_results, man_results):
    pass  # pvalue = scipy.stats.ttest_ind(param, param1)


def metric_box_plots(*results, show_points=True):
    plt_hgt = round(len(results[0]) / 2)
    fig, axes = plt.subplots(plt_hgt, 2, sharex=True)
    fig.tight_layout(pad=2)
    fig.suptitle('Cell Migration Metrics')
    for i, rs in enumerate(zip(*results)):
        ax = axes.flatten()[i]
        plt.sca(ax)
        ax.set_title(rs[0].name)
        ax.set_xticks([i for i in range(1, len(results) + 1)])
        ax.set_ylabel(rs[0].units)
        series = [pd.Series(r.stored, name=r.method) for r in rs]
        df = pd.concat(series, axis=1)

        sns.boxplot(data=df, showmeans=True)
        # may not show all points
        if show_points:
            with warnings.catch_warnings(record=True) as w:
                sns.swarmplot(data=df, color=".25")
                if (len(w) > 1 or (len(w) == 1 and issubclass(w[0].category,
                                   UserWarning))):
                    warnings.warn(w[0].message, w[0].category)


def metric_mthd_avgs(*results):
    for rs in results:
        plt_hgt = round(len(rs) / 2)
        fig, axes = plt.subplots(plt_hgt, 2, sharex=True)
        fig.tight_layout(pad=2)
        fig.supxlabel('No. Tracks')
        for ax, r in zip(axes.flatten(), rs):
            fig.suptitle(f'Cumulative Metric Averages ({r.method})')
            plt.sca(ax)
            x = list(range(1, len(r.stored) + 1))
            y = [mean(r.stored[0:i]) for i in range(1, len(r.stored) + 1)]
            ax.set_title(f'{r.name}')
            # ax.set_xlabel('No. Tracks')
            ax.set_ylabel(r.units)
            plt.plot(x, y, 'go')


def drnty_vs_euclid_dist(*results):
    for rs in results:
        rs = [r for r in rs if r.name == 'Directionality' or r.name == (
              'Euclidean Distance')]
        if len(rs) != 2:
            continue
        if rs[0].name != 'Directionality':
            rs.reverse()

        fig = plt.figure()
        fig.tight_layout()
        ax = plt.gca()
        x, y = np.array(rs[0].stored), np.array(rs[1].stored)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m * x + b, 'r-')

        p = str(round(scipy.stats.pearsonr(x, y)[0], 2))  # Pearson's r
        s = str(round(scipy.stats.spearmanr(x, y)[0], 2))  # Spearman's rho
        k = str(round(scipy.stats.kendalltau(x, y)[0], 2))  # Kendall's tau
        txt = f'Pearson\'s r = {p}\nSpearman\'s ρ = {s}\nKendall\'s τ = {k}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        ax.set_title(f'Directionality vs Euclidean Distance ({rs[0].method})')
        ax.set_xlabel(f'{rs[0].name} ({rs[0].units})')
        ax.set_ylabel(f'{rs[1].name} ({rs[1].units})')
        plt.plot(rs[0].stored, rs[1].stored, 'ro')
