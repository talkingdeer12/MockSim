class Event:
    """Discrete event object used by :class:`SimulatorEngine`."""

    def __init__(self, src, dst, cycle, data_size=0, program=None,
                 event_type=None, payload=None, priority=0):
        self.src = src
        self.dst = dst
        self.cycle = cycle
        self.data_size = data_size
        self.program = program
        self.event_type = event_type
        self.payload = payload or {}
        # ``priority`` is used to break ties between events scheduled for the
        # same cycle.  Higher priority events (lower value) will be popped from
        # the queue first.
        self.priority = priority
        self.time = 0.0  # filled by SimulatorEngine when scheduled

    def __lt__(self, other):
        if self.time == other.time:
            return self.priority < other.priority
        return self.time < other.time

    def handle(self):
        if self.dst is not None:
            self.dst._process_event(self)
