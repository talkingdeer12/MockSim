from sim_core.module import HardwareModule
from sim_core.event import Event
import random


class PipelineStage:
    """Simple 1-cycle pipeline stage."""

    def __init__(self, stage_id):
        self.stage_id = stage_id
        self.slot = None

    def empty(self):
        return self.slot is None

    def push(self, token):
        if self.slot is None:
            self.slot = token
            return True
        return False

    def pop(self):
        token = self.slot
        self.slot = None
        return token

class PE(HardwareModule):
    def __init__(self, engine, name, mesh_info, mac_units=32, mac_width=32):
        super().__init__(engine, name, mesh_info)
        self.mac_units = mac_units
        self.mac_width = mac_width
        self.num_stages = 5
        self.pipeline = [PipelineStage(i) for i in range(self.num_stages)]
        self.pending_iters = 0
        self.completed_iters = 0
        self.total_iters = 0
        self.iter_identifier = None
        self.gemm_cp_name = None
        self.stalled = False

        self.pending_dma_reads = 0
        self.pending_dma_writes = 0
        self.dma_cp_name = None

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
                    "cp_name": event.payload.get("cp_name"),
                },
            )
            self.send_event(dram_write_event)

        elif event.event_type == "PE_DMA_IN":
            words = event.data_size // 4
            self.pending_dma_reads = words
            self.dma_cp_name = event.payload.get("cp_name")
            dram = event.payload.get("dram_name", "DRAM")
            for i in range(words):
                read_ev = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="DMA_READ",
                    payload={
                        "dst_coords": self.mesh_info["dram_coords"][dram],
                        "pe_name": self.name,
                        "cp_name": self.dma_cp_name,
                    },
                )
                self.send_event(read_ev)

        elif event.event_type == "DMA_READ_REPLY":
            self.pending_dma_reads -= 1
            if self.pending_dma_reads == 0:
                done_event = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="PE_DMA_IN_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.dma_cp_name],
                        "pe_name": self.name,
                    },
                )
                self.send_event(done_event)

        elif event.event_type == "PE_GEMM":
            M, N, K = event.payload["gemm_shape"]
            self.total_iters = max(1, (M * N * K) // self.mac_units)
            self.pending_iters = self.total_iters
            self.completed_iters = 0
            self.iter_identifier = event.identifier
            self.gemm_cp_name = event.payload.get("cp_name")
            self.pipeline = [PipelineStage(i) for i in range(self.num_stages)]
            tick = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle + 1,
                event_type="PIPELINE_TICK",
                identifier=event.identifier,
                data_size=0,
                payload={"dst_coords": self.mesh_info["pe_coords"][self.name]},
            )
            self.send_event(tick)

        elif event.event_type == "PIPELINE_TICK":
            self.pipeline_step(event.identifier)
            if self.pending_iters > 0 or any(not s.empty() for s in self.pipeline):
                tick = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + 1,
                    event_type="PIPELINE_TICK",
                    identifier=event.identifier,
                    data_size=0,
                    payload={"dst_coords": self.mesh_info["pe_coords"][self.name]},
                )
                self.send_event(tick)

        elif event.event_type == "PE_DMA_OUT":
            words = event.data_size // 4
            self.pending_dma_writes = words
            self.dma_cp_name = event.payload.get("cp_name")
            dram = event.payload.get("dram_name", "DRAM")
            for i in range(words):
                wr_ev = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle + i,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="DMA_WRITE",
                    payload={
                        "dst_coords": self.mesh_info["dram_coords"][dram],
                        "pe_name": self.name,
                        "cp_name": self.dma_cp_name,
                    },
                )
                self.send_event(wr_ev)

        elif event.event_type == "DMA_WRITE_REPLY":
            self.pending_dma_writes -= 1
            if self.pending_dma_writes == 0:
                done_event = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    identifier=event.identifier,
                    event_type="PE_DMA_OUT_DONE",
                    payload={
                        "dst_coords": self.mesh_info["cp_coords"][self.dma_cp_name],
                        "pe_name": self.name,
                    },
                )
                self.send_event(done_event)

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
                    "pe_name": self.name,
                },
            )
            self.send_event(done_event)
        else:
            print(f"[{self.name}] 알 수 없는 이벤트: {event.event_type}")

    def get_my_router(self):
        coords = self.mesh_info["pe_coords"][self.name]
        return self.mesh_info["router_map"][coords]

    def pipeline_step(self, identifier):
        """Advance the GEMM pipeline by one cycle."""
        # Determine stalls for each stage
        stall_flags = [random.random() < 0.1 for _ in range(self.num_stages)]
        self.stalled = any(stall_flags)

        # Pop the last stage
        if not stall_flags[-1] and not self.pipeline[-1].empty():
            self.pipeline_output(identifier)
            self.pipeline[-1].pop()

        # Move tokens backwards so we don't overwrite
        for idx in reversed(range(self.num_stages - 1)):
            if stall_flags[idx] or stall_flags[idx + 1]:
                continue
            if not self.pipeline[idx].empty() and self.pipeline[idx + 1].empty():
                tok = self.pipeline[idx].pop()
                self.pipeline[idx + 1].push(tok)

        # Feed new iteration
        if self.pending_iters > 0 and not stall_flags[0] and self.pipeline[0].empty():
            self.pipeline[0].push(object())
            self.pending_iters -= 1

    def pipeline_output(self, identifier):
        self.completed_iters += 1
        if self.completed_iters >= self.total_iters:
            # All iterations have exited the last stage
            done_event = Event(
                src=self,
                dst=self.get_my_router(),
                cycle=self.engine.current_cycle,
                data_size=4,
                identifier=identifier,
                event_type="PE_GEMM_DONE",
                payload={
                    "dst_coords": self.mesh_info["cp_coords"][self.gemm_cp_name],
                    "pe_name": self.name,
                },
            )
            self.send_event(done_event)
