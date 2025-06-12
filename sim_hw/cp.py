from sim_core.module import HardwareModule
from sim_core.event import Event

class ControlProcessor(HardwareModule):
    def __init__(self, engine, name, mesh_info, pes, dram):
        super().__init__(engine, name, mesh_info)
        self.pes = pes
        self.dram = dram
        self.waiting_pes = set()

    def handle_event_module(self, event):
        if event.event_type == "GEMM":
            print(f"[CP] GEMM 시작: {event.identifier}, shape={event.payload['gemm_shape']}")
            self.waiting_pes = set(pe.name for pe in self.pes)
            for pe in self.pes:
                ctrl_event = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="PE_CTRL",
                    payload={
                        "dst_coords": self.mesh_info["pe_coords"][pe.name],
                        "gemm_shape": event.payload["gemm_shape"],
                        "weights_size": event.payload["weights_size"],
                        "act_size": event.payload["act_size"],
                        "cp_name": self.name,
                        "dram_name": self.dram.name,
                    }
                )
                self.send_event(ctrl_event)
        elif event.event_type == "PE_DONE":
            pe_name = event.payload.get("pe_name", event.src.name)
            print(f"[CP] PE 완료 신호 수신: {pe_name}")
            self.waiting_pes.discard(pe_name)
            if not self.waiting_pes:
                print("[CP] 모든 PE 완료 → GEMM 작업 종료")
        else:
            print(f"[CP] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
