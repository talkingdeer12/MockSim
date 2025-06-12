from sim_core.module import HardwareModule
from sim_core.event import Event

class DRAM(HardwareModule):
    def __init__(self, engine, name, mesh_info, latency=10):
        super().__init__(engine, name, mesh_info)
        self.latency = latency
        self.read_queue = []
        self.write_queue = []
        self.processing = False

    def handle_event(self, event):
        if event.event_type in ("DMA_WRITE", "DMA_READ"):
            self.enqueue_request(event)
        elif event.event_type == "_PROCESS_QUEUE":
            self.process_next()
        else:
            print(f"[{self.name}] 알 수 없는 이벤트: {event.event_type}")

    def enqueue_request(self, event):
        if event.event_type == "DMA_WRITE":
            self.write_queue.append(event)
            print(f"[{self.name}] DMA_WRITE 대기열 추가: {event.identifier} (len={len(self.write_queue)})")
        else:  # DMA_READ
            self.read_queue.append(event)
            print(f"[{self.name}] DMA_READ 대기열 추가: {event.identifier} (len={len(self.read_queue)})")

        if not self.processing:
            self.processing = True
            proc_evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                             event_type="_PROCESS_QUEUE")
            self.send_event(proc_evt)

    def process_next(self):
        req = None
        if self.write_queue:
            req = self.write_queue.pop(0)
        elif self.read_queue:
            req = self.read_queue.pop(0)

        if req is None:
            self.processing = False
            return

        if req.event_type == "DMA_WRITE":
            print(f"[{self.name}] DMA_WRITE 처리 시작: {req.identifier}")
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + self.latency,
                data_size=4,
                identifier=req.identifier,
                event_type="DMA_WRITE_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][req.payload["pe_name"]],
                    "cp_name": req.payload["cp_name"]
                }
            )
            self.send_event(reply_event)
        else:  # DMA_READ
            print(f"[{self.name}] DMA_READ 처리 시작: {req.identifier}")
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + self.latency,
                data_size=req.data_size,
                identifier=req.identifier,
                event_type="DMA_READ_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][req.payload["pe_name"]],
                    "cp_name": req.payload.get("cp_name", "")
                }
            )
            self.send_event(reply_event)

        if self.write_queue or self.read_queue:
            proc_evt = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1,
                             event_type="_PROCESS_QUEUE")
            self.send_event(proc_evt)
        else:
            self.processing = False

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
