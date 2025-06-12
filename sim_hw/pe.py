from sim_core.module import HardwareModule
from sim_core.event import Event

class PE(HardwareModule):
    def __init__(self, engine, name, mesh_info, mac_units=32, mac_width=32):
        super().__init__(engine, name, mesh_info)
        self.mac_units = mac_units
        self.mac_width = mac_width

    def handle_event(self, event):
        if event.event_type == "PE_CTRL":
            gemm_shape = event.payload["gemm_shape"]
            M, N, K = gemm_shape
            ops = 2 * M * N * K
            latency = (ops + self.mac_units - 1) // self.mac_units
            print(f"[{self.name}] 연산 시작. (MAC:{self.mac_units}, 연산량:{ops} → {latency} 사이클)")

            dram_write_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + latency,
                data_size=event.payload["weights_size"],
                identifier=event.identifier,
                event_type="DMA_WRITE",
                payload={
                    "dst_coords": self.mesh_info["dram_coords"]["DRAM"],
                    "pe_name": self.name,
                    "cp_name": event.payload["cp_name"]
                }
            )
            self.send_event(dram_write_event)

        elif event.event_type == "WRITE_REPLY":
            done_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                identifier=event.identifier,
                event_type="PE_DONE",
                payload={
                    "dst_coords": self.mesh_info["cp_coords"][event.payload["cp_name"]],
                    "pe_name": self.name
                }
            )
            self.send_event(done_event)
        else:
            print(f"[{self.name}] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["pe_coords"][self.name]
        return self.mesh_info["router_map"][coords]
