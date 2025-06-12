from sim_core.module import HardwareModule
from sim_core.event import Event

class DRAM(HardwareModule):
    def __init__(self, engine, name, mesh_info, num_stages=0):
        super().__init__(engine, name, mesh_info, num_stages)

    def handle_event(self, event):
        if event.event_type == "DMA_WRITE":
            print(f"[{self.name}] Write 시작: {event.identifier} size={event.data_size} bytes")
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + 10,
                data_size=4,
                identifier=event.identifier,
                event_type="WRITE_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][event.payload["pe_name"]],
                    "cp_name": event.payload["cp_name"]
                }
            )
            self.send_event(reply_event)
        elif event.event_type == "DMA_READ":
            print(f"[{self.name}] Read 완료: {event.identifier} size={event.data_size} bytes")
        else:
            print(f"[{self.name}] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
