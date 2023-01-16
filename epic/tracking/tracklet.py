class Tracklet:
    def __init__(self, dets, idn=None, method=None):
        self._dets = dets
        self._num_dets = len(self._dets)
        self._links = []
        self._start_frame = dets[0].frame_num
        self._end_frame = dets[-1].frame_num
        self._idn = idn
        self._method = method

    def link_tracklet(self, new_tracklet, stage=None):
        if stage == 1:
            self._links += [(link[0] + self._num_dets + 1,
                            link[1] + self._num_dets + 1) for link in
                            new_tracklet.links]
            link_tail = len(self._dets) - 1
            link_head = link_tail + 1
            self._links.append((link_tail, link_head))
        self._end_frame = new_tracklet.end_frame
        self._dets += new_tracklet.dets.copy()
        self._num_dets += new_tracklet.num_dets

    def det_at_frame(self, idx):
        det = [det for det in self.dets if det.frame_num == idx]
        det = None if len(det) != 1 else det[0]

        return det

    @property
    def dets(self):
        return self._dets

    @property
    def num_dets(self):
        return self._num_dets

    @property
    def start_frame(self):
        return self._start_frame

    @property
    def end_frame(self):
        return self._end_frame

    @property
    def first_det(self):
        return self._dets[0]

    @property
    def last_det(self):
        return self._dets[-1]

    @property
    def links(self):
        return self._links

    @property
    def method(self):
        return self._method

    @property
    def idn(self):
        return self._idn

    @idn.setter
    def idn(self, idn):
        self._idn = idn
