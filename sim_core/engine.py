import heapq
import math
from sim_core.logger import EventLogger

class SimulatorEngine:
    def __init__(self):
        self.current_cycle = 0
        self.current_time = 0.0  # microseconds
        self.event_queue = []
        self.modules = {}
        self.module_freqs = {}
        self.module_times = {}
        self.module_cycles = {}
        self.logger = EventLogger() # Initialize logger here
        self._order = 0
    
    def register_module(self, module):
        self.modules[module.name] = module
        self.module_freqs[module.name] = getattr(module, "frequency", 1000)
        self.module_times[module.name] = 0.0
        self.module_cycles[module.name] = 0

    def set_logger(self, logger):
        """Attach an EventLogger instance."""
        self.logger = logger

    def push_event(self, event):
        src = event.src or event.dst
        if src and src.name in self.module_freqs:
            freq = self.module_freqs[src.name]
            src_time = self.module_times[src.name]
            src_cycle = self.module_cycles[src.name]
        else:
            freq = 1000
            src_time = self.current_time
            src_cycle = self.current_cycle
        delta_cycles = max(0, event.cycle - src_cycle)
        event_time = src_time + delta_cycles / freq
        event.time = event_time
        heapq.heappush(self.event_queue, (event_time, event.priority, event))

    def tick(self):
        if not self.event_queue:
            return
        event_time, _, event = heapq.heappop(self.event_queue)
        self.current_time = event_time
        if event.dst and event.dst.name in self.module_freqs:
            freq = self.module_freqs[event.dst.name]
            cycle = math.ceil(event_time * freq)
            self.module_cycles[event.dst.name] = cycle
            self.module_times[event.dst.name] = event_time
            self.current_cycle = cycle
        event.handle()

    def run_until_idle(self, max_tick=None):
        tick_count = 0
        while self.event_queue:
            self.tick()
            tick_count += 1
            if max_tick and tick_count >= max_tick:
                print(f"[Engine] 최대 {max_tick} tick 도달, 강제 종료")
                break
        print(f"[Engine] 모든 이벤트 처리 완료, 총 tick: {tick_count}")
