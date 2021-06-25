class Point:
    def __init__(self, coords, frame_num, score=None):
        self._centre = coords
        self._coords = coords
        self._frame_num = frame_num

    @property
    def coords(self):
        return self._coords

    @property
    def centre(self):
        return self._centre

    @property
    def centre_x(self):
        return self._centre[0]

    @property
    def centre_y(self):
        return self._centre[1]

    @property
    def frame_num(self):
        return self._frame_num
