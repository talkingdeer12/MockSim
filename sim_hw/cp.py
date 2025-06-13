from sim_core.module import HardwareModule
from sim_core.event import Event

class ControlProcessor(HardwareModule):
    def __init__(self, engine, name, mesh_info, pes, dram, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.pes = pes
        self.dram = dram
        self.active_gemms = {}

    def handle_event(self, event):
        if event.event_type == "GEMM":
            print(f"[CP] GEMM 시작: {event.identifier}, shape={event.payload['gemm_shape']}")
            state = {
                "waiting_dma_in": set(pe.name for pe in self.pes),
                "waiting_gemm": set(pe.name for pe in self.pes),
                "waiting_dma_out": set(pe.name for pe in self.pes),
                "gemm_shape": event.payload["gemm_shape"],
                "weights_size": event.payload["weights_size"],
                "act_size": event.payload["act_size"],
            }
            self.active_gemms[event.identifier] = state
            for pe in self.pes:
                dma_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=state["weights_size"] + state["act_size"],
                    identifier=event.identifier,
                    event_type="PE_DMA_IN",
                    payload={
                        "dst_coords": self.mesh_info["pe_coords"][pe.name],
                        "data_size": state["weights_size"] + state["act_size"],
                        "cp_name": self.name,
                        "dram_name": self.dram.name,
                        "pe_name": pe.name,
                    },
                )
                self.send_event(dma_evt)

        elif event.event_type == "PE_DMA_IN_DONE":
            state = self.active_gemms.get(event.identifier)
            if not state:
                return
            pe_name = event.payload["pe_name"]
            state["waiting_dma_in"].discard(pe_name)
            if not state["waiting_dma_in"]:
                for pe in self.pes:
                    gemm_evt = Event(
                        src=self,
                        dst=self.get_my_router(),
                        cycle=self.engine.current_cycle,
                        data_size=4,
                        identifier=event.identifier,
                        event_type="PE_GEMM",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "gemm_shape": state["gemm_shape"],
                            "cp_name": self.name,
                        },
                    )
                    self.send_event(gemm_evt)

        elif event.event_type == "PE_GEMM_DONE":
            state = self.active_gemms.get(event.identifier)
            if not state:
                return
            pe_name = event.payload["pe_name"]
            state["waiting_gemm"].discard(pe_name)
            if not state["waiting_gemm"]:
                out_size = state["gemm_shape"][0] * state["gemm_shape"][1] * 4
                for pe in self.pes:
                    dma_evt = Event(
                        src=self,
                        dst=self.get_my_router(),
                        cycle=self.engine.current_cycle,
                        data_size=out_size,
                        identifier=event.identifier,
                        event_type="PE_DMA_OUT",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "data_size": out_size,
                            "cp_name": self.name,
                            "pe_name": pe.name,
                            "dram_name": self.dram.name,
                        },
                    )
                    self.send_event(dma_evt)

        elif event.event_type == "PE_DMA_OUT_DONE":
            state = self.active_gemms.get(event.identifier)
            if not state:
                return
            pe_name = event.payload["pe_name"]
            state["waiting_dma_out"].discard(pe_name)
            if not state["waiting_dma_out"]:
                print(f"[CP] GEMM {event.identifier} 작업 완료")
                self.active_gemms.pop(event.identifier, None)

        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
