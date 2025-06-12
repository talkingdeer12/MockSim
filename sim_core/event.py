class Event:
    def __init__(self, src, dst, cycle, data_size=0, identifier=None,
                 event_type=None, payload=None, priority=0):
        self.src = src
        self.dst = dst
        self.cycle = cycle
        self.data_size = data_size
        self.identifier = identifier
        self.event_type = event_type
        self.payload = payload or {}
        # higher priority events are handled first when cycle is equal
        self.priority = priority

    def __lt__(self, other):
        if self.cycle == other.cycle:
            return self.priority > other.priority
        return self.cycle < other.cycle

    def handle(self):
        if self.dst is not None:
            self.dst.handle_event(self)
