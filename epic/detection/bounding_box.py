from math import ceil

from epic.detection.point import Point


class BoundingBox(Point):
    """Summary of class here.

    Longer class information....
    Longer class information....

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
    def __init__(self, coords, frame, frame_num, score=None, **kwargs):
        x1, y1, x2, y2 = coords  # broken somehow - x2 can equal max length of image e.g. img of shape 1408 but y2 1408 - came from misc.create tracttlsts fro mdets
        cen_x, cen_y = ((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
        super().__init__((cen_x, cen_y), frame_num, score)
        self._width, self._height = x2 - x1, y2 - y1
        self._area = (y2 - y1) * (x2 - x1)
        self._bbox_img = frame[ceil(y1):ceil(y2), ceil(x1):ceil(x2)]
        self._coords = coords
        # score?

    @property
    def coords(self):
        return self._coords

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def area(self):
        return self._area

    @property
    def bbox_img(self):
        return self._bbox_img
