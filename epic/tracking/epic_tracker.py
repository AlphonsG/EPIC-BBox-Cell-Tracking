# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import warnings
from collections import defaultdict
from itertools import chain
from statistics import mean
from typing import Any

from epic.features.feature_factory import FeatureFactory
from epic.tracking.tracker import Tracker
from epic.tracking.tracklet import Tracklet

from lapsolver import solve_dense

import numpy as np
import numpy.typing as npt

from scipy.spatial import cKDTree


class EpicTracker(Tracker):
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    def track(self, imgs: list[npt.NDArray[Any]], dets: list[list[
            Tracklet]], ldg_es: tuple[int, int] | None = None) -> list[
                Tracklet]:
        tracklets = defaultdict(dict)
        for frame_num, ds in enumerate(dets, start=1):
            for key in ['EF', 'SF']:
                tracklets[key][frame_num] = [d for d in ds]

        self.ldg_es, self.imgs = ldg_es, imgs
        for i in range(0, self.config['max_tracker_runs']):
            self.stage = 0 if i == 0 else 1
            tracklets, tracklets_linked = self.traverse_all_frames(tracklets)
            if not tracklets_linked:
                break

        tracks = list(chain.from_iterable(list(tracklets['EF'].values())))

        return tracks

    def traverse_all_frames(self, tracklets):
        ts1, tracklets_linked = [], False
        frame_nums = iter(tracklets['EF'])
        while True:
            ts2 = self.find_potential_links(ts1, tracklets)

            cost_matrix, asgmts = [], []
            if len(ts1) != 0 and len(ts2) != 0:
                cost_matrix, asgmts = self.link_tracklets(ts1, ts2)
            if self.config['track_refinement']:
                self.refine_tracklets(cost_matrix, asgmts)
            if len(asgmts) != 0 and not tracklets_linked:
                tracklets_linked = True
            for idx1, idx2 in asgmts:
                t1, t2 = ts1[idx1], ts2[idx2]
                tracklets['EF'][t1.end_frame].remove(t1)
                t1.link_tracklet(t2, self.stage)
                tracklets['EF'][t1.end_frame].append(t1)
                tracklets['EF'][t2.end_frame].remove(t2)
                tracklets['SF'][t2.start_frame].remove(t2)

            try:
                ts1 = tracklets['EF'][next(frame_nums)].copy()
            except StopIteration:
                break

        return tracklets, tracklets_linked

    def find_potential_links(self, ts1, tracklets):
        pot_links = []
        if not (len(ts1) == 0):
            pot_start_frames = range(ts1[0].end_frame + 1,
                                     ts1[0].end_frame + 1 + self.config[
                                     'glob_temp_dist'][self.stage])
            pot_links += [tracklets['SF'][start_frame] for start_frame in
                          pot_start_frames if start_frame in tracklets['SF']]

        return list(chain.from_iterable(pot_links))

    def refine_tracklets(self, cost_matrix, asgmts):
        poor_asgmts = []
        for i, j in asgmts:
            with warnings.catch_warnings(record=True) as w:
                min_cost = np.nanmin(cost_matrix[i, :])
                if len(w) == 1 and issubclass(w[-1].category, RuntimeWarning):
                    continue
            if min_cost != cost_matrix[i][j]:
                poor_asgmts.append((i, j))

        while len(poor_asgmts) != 0:
            current_i, poor_j = poor_asgmts.pop()
            asgmts.remove((current_i, poor_j))
            better_j = np.where(cost_matrix[current_i, :] == np.nanmin(
                                cost_matrix[current_i, :]))[0][0]
            asgmts_dict = dict(zip([j for (i, j) in asgmts],
                                   [i for (i, j) in asgmts]))
            assigned_i = asgmts_dict.get(better_j, None)
            if assigned_i is not None:
                if(np.nanmin(cost_matrix[assigned_i, :]) != cost_matrix[
                   assigned_i][better_j]):
                    asgmts.append((current_i, better_j))
                elif(cost_matrix[assigned_i][better_j] > cost_matrix[
                     current_i][better_j]):
                    asgmts.append((current_i, better_j))
                    asgmts.remove((assigned_i, better_j))
            else:
                asgmts.append((current_i, better_j))

    def link_tracklets(self, ts1, ts2):
        ts1_ts2_dists = self.find_nearest_neighbours(ts1, ts2)
        cost_matrix = np.full((len(ts1), len(ts2)), np.nan)
        feat_fcty = FeatureFactory()
        feats = feat_fcty.get_cfgd_feats(self.config, self.stage,
                                         img=self.imgs[0][1])

        for (t1_idx, ts2_idxs, dists) in ts1_ts2_dists:
            t1 = ts1[t1_idx]
            for (t2_idx, dist) in zip(ts2_idxs, dists):
                t2 = ts2[t2_idx]
                affinities = []
                for feat in feats:
                    affinity = (feat.compute_affinity(t1, t2, self.stage,
                                ldg_es=self.ldg_es, dist=dist,
                                glob_temp_dist=self.config['glob_temp_dist']))
                    if affinity is None:
                        continue
                    elif affinity == -1:
                        break
                    else:
                        if feat.weight[self.stage] != 0:
                            affinities.append(affinity * feat.weight[
                                              self.stage])
                else:
                    cost_matrix[t1_idx][t2_idx] = 1 - mean(affinities)

        asgmts = list(zip(*solve_dense(cost_matrix)))

        return cost_matrix, asgmts

    def find_nearest_neighbours(self, ts1, ts2):
        ts1_pts = np.asarray([t.dets[-1].centre for t in ts1])
        ts2_pts = np.asarray([t.dets[0].centre for t in ts2])
        tree = cKDTree(ts2_pts)

        distance_upper_bound = (self.config['feats']['euclid_dist']['thresh'][
                                self.stage] if self.config['glob_euclid_dist'][
                                'thresh'][self.stage] is None else self.config[
                                'glob_euclid_dist']['thresh'][self.stage])
        dists, idxs = (tree.query(ts1_pts,
                       k=self.config['glob_euclid_dist']['num_nns'],
                       distance_upper_bound=distance_upper_bound,  workers=1))
        ts1_ts2_dists = [(i, j[j != tree.n], dist[dist != np.inf]) for i, (j,
                         dist) in enumerate(zip(idxs, dists))]

        return ts1_ts2_dists
