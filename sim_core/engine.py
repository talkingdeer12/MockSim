import heapq

class SimulatorEngine:
    def __init__(self):
        self.current_cycle = 0
        self.event_queue = []
        self.modules = {}
    
    def register_module(self, module):
        self.modules[module.name] = module

    def push_event(self, event):
        heapq.heappush(self.event_queue, event)
    
    def tick(self):
        self.current_cycle += 1
        events_to_handle = []
        while self.event_queue and self.event_queue[0].cycle <= self.current_cycle:
            event = heapq.heappop(self.event_queue)
            events_to_handle.append(event)
        for event in events_to_handle:
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
