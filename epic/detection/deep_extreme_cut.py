import contextlib
import os
import sys
import warnings
from pathlib import Path

from epic.detection.base_mask_generator import BaseMaskGenerator

import numpy as np

import torch
from torch.nn.functional import upsample

CHECKPOINT_DIR = (Path(__file__).parents[2].resolve() / 'misc/checkpoints/'
                  'etos_deepcut')

class DeepExtremeCut(BaseMaskGenerator):
    def __init__(self, device='cpu', model_name='dextr_pascal-sbd', pad=50,
                 thres=0.8):
        # cite
        curr_dir = Path(__file__).resolve().parents[2]
        pkg_dir = os.path.join(curr_dir, 'third_party', 'etos-deepcut')
        sys.path.insert(1, pkg_dir)
        import networks.deeplab_resnet as resnet
        from dataloaders import helpers as helpers

        #  Create the network and load the weights
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            net = resnet.resnet101(1, nInputChannels=4, classifier='psp')
        state_dict_checkpoint = torch.load(
            os.path.join(CHECKPOINT_DIR, model_name + '.pth'),
            map_location=lambda storage, loc: storage)

        # Remove the prefix .module from the model when it is trained using
        # DataParallel
        if 'module.' in list(state_dict_checkpoint.keys())[0]:
            new_state_dict = {}
            for k, v in state_dict_checkpoint.items():
                name = k[7:]  # remove `module.` from multi-gpu training
                new_state_dict[name] = v
        else:
            new_state_dict = state_dict_checkpoint
        net.load_state_dict(new_state_dict)
        net.eval()
        net.to(device)

        self._net = net
        self._helpers = helpers
        self._pad = pad
        self._thres = thres
        self._device = device

    def gen_mask(self, frame, coords):
        x1, y1, x2, y2 = coords
        centre = ((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)

        extreme_points_ori = [[coords[0], centre[1]],
                              [coords[2], centre[1]],
                              [centre[0], coords[1]],
                              [centre[0], coords[3]]]
        extreme_points_ori = np.array([[round(x), round(y)] for x, y in
                                       extreme_points_ori])

        with torch.no_grad():
            bbox = self._helpers.get_bbox(frame, points=extreme_points_ori,
                                          pad=self._pad, zero_pad=True)
            crop_image = self._helpers.crop_from_bbox(frame, bbox,
                                                      zero_pad=True)
            im_size = frame.shape[:2]

            # Crop image to the bounding box from the extreme points and
            # resize
            resize_image = self._helpers.fixed_resize(crop_image, (
                512, 512)).astype(np.float32)

            # Generate extreme point heat map normalized to image values
            extreme_points = extreme_points_ori - [np.min(
                extreme_points_ori[:, 0]), np.min(
                    extreme_points_ori[:, 1])] + [self._pad, self._pad]
            extreme_points = (512 * extreme_points * [
                1 / crop_image.shape[1],
                1 / crop_image.shape[0]]).astype(int)
            extreme_heatmap = self._helpers.make_gt(
                resize_image, extreme_points, sigma=10)
            extreme_heatmap = self._helpers.cstm_normalize(
                extreme_heatmap, 255)

            # Concatenate inputs and convert to tensor
            input_dextr = np.concatenate((resize_image, extreme_heatmap[
                :, :, np.newaxis]), axis=2)
            inputs = torch.from_numpy(input_dextr.transpose((2, 0, 1))[
                np.newaxis, ...])

            # Run a forward pass
            inputs = inputs.to(self._device)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                outputs = self._net.forward(inputs)
                outputs = upsample(outputs, size=(512, 512), mode='bilinear',
                                   align_corners=True)
            outputs = outputs.to(torch.device('cpu'))
            pred = np.transpose(outputs.data.numpy()[0, ...], (1, 2, 0))
            pred = 1 / (1 + np.exp(-pred))
            pred = np.squeeze(pred)

            result = self._helpers.crop2fullmask(
                pred, bbox, im_size=im_size, zero_pad=True,
                relax=self._pad) > self._thres

        return result
