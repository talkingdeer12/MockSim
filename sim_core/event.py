class Event:
    def __init__(self, src, dst, cycle, data_size=0, identifier=None, event_type=None, payload=None):
        self.src = src
        self.dst = dst
        self.cycle = cycle
        self.data_size = data_size
        self.identifier = identifier
        self.event_type = event_type
        self.payload = payload or {}

    def __lt__(self, other):
        return self.cycle < other.cycle

    def handle(self):
        if self.dst is not None:
            self.dst.handle_event(self)
