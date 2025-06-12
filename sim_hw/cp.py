from sim_core.module import HardwareModule
from sim_core.event import Event

class ControlProcessor(HardwareModule):
    def __init__(self, engine, name, mesh_info, pes, dram):
        super().__init__(engine, name, mesh_info)
        self.pes = pes
        self.dram = dram
        self.waiting_dma_in = set()
        self.waiting_gemm = set()
        self.waiting_dma_out = set()

    def handle_event(self, event):
        if event.event_type == "GEMM":
            print(f"[CP] GEMM 시작: {event.identifier}, shape={event.payload['gemm_shape']}")
            self.waiting_dma_in = set(pe.name for pe in self.pes)
            for pe in self.pes:
                dma_in_event = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="PE_DMA_IN",
                    payload={
                        "dst_coords": self.mesh_info["pe_coords"][pe.name],
                        "gemm_shape": event.payload["gemm_shape"],
                        "weights_size": event.payload["weights_size"],
                        "act_size": event.payload["act_size"],
                        "cp_name": self.name,
                        "dram_name": self.dram.name,
                    }
                )
                self.send_event(dma_in_event)
        elif event.event_type == "PE_DMA_IN_DONE":
            pe_name = event.payload.get("pe_name")
            self.waiting_dma_in.discard(pe_name)
            if not self.waiting_dma_in:
                print("[CP] 모든 PE DMA_IN 완료 → GEMM 단계 시작")
                self.waiting_gemm = set(pe.name for pe in self.pes)
                for pe in self.pes:
                    gemm_event = Event(
                        src=self,
                        dst=self.get_my_router(),
                        cycle=self.engine.current_cycle,
                        data_size=4,
                        identifier=event.identifier,
                        event_type="PE_GEMM",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "gemm_shape": event.payload["gemm_shape"],
                            "weights_size": event.payload["weights_size"],
                            "act_size": event.payload["act_size"],
                            "cp_name": self.name,
                            "dram_name": self.dram.name,
                        }
                    )
                    self.send_event(gemm_event)
        elif event.event_type == "PE_GEMM_DONE":
            pe_name = event.payload.get("pe_name")
            self.waiting_gemm.discard(pe_name)
            if not self.waiting_gemm:
                print("[CP] 모든 PE GEMM 완료 → DMA_OUT 단계 시작")
                self.waiting_dma_out = set(pe.name for pe in self.pes)
                result_size = event.payload.get("result_size")
                for pe in self.pes:
                    dma_out_event = Event(
                        src=self,
                        dst=self.get_my_router(),
                        cycle=self.engine.current_cycle,
                        data_size=4,
                        identifier=event.identifier,
                        event_type="PE_DMA_OUT",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "gemm_shape": event.payload["gemm_shape"],
                            "weights_size": event.payload["weights_size"],
                            "act_size": event.payload["act_size"],
                            "result_size": result_size,
                            "cp_name": self.name,
                            "dram_name": self.dram.name,
                        }
                    )
                    self.send_event(dma_out_event)
        elif event.event_type == "PE_DMA_OUT_DONE":
            pe_name = event.payload.get("pe_name")
            self.waiting_dma_out.discard(pe_name)
            if not self.waiting_dma_out:
                print("[CP] 모든 PE DMA_OUT 완료 → GEMM 작업 종료")
        else:
            print(f"[CP] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
