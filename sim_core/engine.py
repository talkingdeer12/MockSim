import heapq

class SimulatorEngine:
    def __init__(self):
        self.current_cycle = 0
        self.event_queue = []
        self.modules = {}
        self.logger = None
    
    def register_module(self, module):
        self.modules[module.name] = module

    def set_logger(self, logger):
        """Attach an EventLogger instance."""
        self.logger = logger

    def push_event(self, event):
        heapq.heappush(self.event_queue, event)
    
    def tick(self):
        self.current_cycle += 1
        sync_map = {
            "NPU_DMA_IN_SYNC": 0,
            "NPU_CMD_SYNC": 1,
            "NPU_DMA_OUT_SYNC": 2,
        }

        while self.event_queue and self.event_queue[0].cycle <= self.current_cycle:
            event = heapq.heappop(self.event_queue)

            if event.event_type in sync_map:
                cp = event.dst
                sync_type = sync_map[event.event_type]
                targets = event.payload.get("sync_targets")
                if not cp._is_sync_ready(event.program, sync_type, targets):
                    event.cycle += 1
                    heapq.heappush(self.event_queue, event)

                    # Delay any other pending events destined for the same CP this cycle
                    temp = []
                    while self.event_queue and self.event_queue[0].cycle <= self.current_cycle:
                        nxt = heapq.heappop(self.event_queue)
                        if nxt.dst is cp and nxt.event_type not in (
                            "RETRY_SEND",
                            "NPU_DMA_IN_DONE",
                            "NPU_CMD_DONE",
                            "NPU_DMA_OUT_DONE",
                        ):
                            nxt.cycle += 1
                        temp.append(nxt)
                    for nxt in temp:
                        heapq.heappush(self.event_queue, nxt)
                    continue

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
