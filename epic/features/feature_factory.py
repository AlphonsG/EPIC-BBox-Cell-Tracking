# Copyright (c) 2021 Alphons Gwatimba
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
from epic.features.appearance_features import GrayscaleHistogram, \
    StructuralSimilarityIndexMeassure
from epic.features.motion_features import (Boundary, EuclideanDistance,
                                           IntersectionOverUnion,
                                           MotionVectors, TemporalDistance)


class FeatureFactory:

    def get_feature(self, feature, **kwargs):
        if feature == 'iou':
            return IntersectionOverUnion(**kwargs)
        elif feature == 'euclid_dist':
            return EuclideanDistance(**kwargs)
        elif feature == 'mot_vects':
            return MotionVectors(**kwargs)
        elif feature == 'boundary':
            return Boundary(**kwargs)
        elif feature == 'struct_sim':
            return StructuralSimilarityIndexMeassure(**kwargs)
        elif feature == 'gray_hist':
            return GrayscaleHistogram(**kwargs)
        elif feature == 'temp_dist':
            return TemporalDistance(**kwargs)
        else:
            msg = f'Chosen affinity function ({feature}) is not supported.'
            raise ValueError(msg)

    def get_cfgd_feats(self, config, stage, **kwargs):
        cfg_features = {'gray_hist': config['feats']['gray_hist'],
                        'struct_sim': config['feats']['struct_sim'],
                        'iou': config['feats']['iou'],
                        'temp_dist': config['feats']['temp_dist'],
                        'mot_vects': config['feats']['mot_vects'],
                        'euclid_dist': config['feats']['euclid_dist'],
                        'boundary': config['feats']['boundary']
                        }

        cfgd_features = [self.get_feature(f, **{**kwargs, **f_cfg}) for f,
                         f_cfg in cfg_features.items() if f_cfg['enabled'][
                         stage]]

        return cfgd_features
