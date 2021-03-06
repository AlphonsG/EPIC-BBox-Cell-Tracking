from math import ceil

from epic.utils.point import Point


class BoundingBox(Point):
    def __init__(self, coords, frame, frame_num, score=None):
        x1, y1, x2, y2 = coords
        cen_x, cen_y = ((x2 - x1) / 2 + x1, (y2 - y1) / 2 + y1)
        super().__init__((cen_x, cen_y), frame_num, score)
        self._width, self._height = x2 - x1, y2 - y1
        self._area = (y2 - y1) * (x2 - x1)
        self._bbox_img = frame[ceil(y1):ceil(y2), ceil(x1):ceil(x2)]
        self._coords = coords

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
