from sim_core.module import HardwareModule
from sim_core.event import Event

class DRAM(HardwareModule):
    def __init__(self, engine, name, mesh_info, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)

    def handle_event(self, event):
        if event.event_type == "DMA_WRITE":
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + 5,
                data_size=4,
                identifier=event.identifier,
                event_type="WRITE_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][event.payload["pe_name"]],
                    "cp_name": event.payload["cp_name"],
                },
            )
            self.send_event(reply_event)
        elif event.event_type == "DMA_READ":
            reply_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + 5,
                data_size=4,
                identifier=event.identifier,
                event_type="DMA_READ_REPLY",
                payload={
                    "dst_coords": self.mesh_info["pe_coords"][event.payload["pe_name"]],
                    "cp_name": event.payload["cp_name"],
                },
            )
            self.send_event(reply_event)
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["dram_coords"][self.name]
        return self.mesh_info["router_map"][coords]
