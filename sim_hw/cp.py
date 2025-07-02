from sim_core.module import HardwareModule
from sim_core.event import Event

class ControlProcessor(HardwareModule):
    def __init__(self, engine, name, mesh_info, pes, dram, npus=None, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.pes = pes
        self.npus = npus or []
        self.dram = dram
        self.active_gemms = {}
        self.active_npu_programs = {}
        # Track synchronization state of NPU commands so external modules can
        # poll progress.  Each dict maps a program name to a boolean flag.
        self.npu_dma_in_opcode_done = {}
        self.npu_cmd_opcode_done = {}
        self.npu_dma_out_opcode_done = {}

    def _create_program_state(self, payload):
        return {
            "waiting_dma_in": set(n.name for n in self.npus),
            "waiting_op": set(n.name for n in self.npus),
            "waiting_dma_out": set(n.name for n in self.npus),
            "program_cycles": payload["program_cycles"],
            "in_size": payload["in_size"],
            "out_size": payload["out_size"],
            "dma_in_opcode_cycles": payload.get("dma_in_opcode_cycles", 5),
            "dma_out_opcode_cycles": payload.get("dma_out_opcode_cycles", 5),
            "cmd_opcode_cycles": payload.get("cmd_opcode_cycles", payload["program_cycles"]),
        }

    def _is_sync_ready(self, program, sync_type, targets=None):
        """Return True if the given sync type has completed for ``program``.

        ``targets`` may be an iterable of NPU names to check. If ``None`` it
        checks all NPUs involved in the task.
        """
        state = self.active_npu_programs.get(program)
        if not state:
            return True

        if sync_type == 0:
            pending = state["waiting_dma_in"]
        elif sync_type == 1:
            pending = state["waiting_op"]
        else:
            pending = state["waiting_dma_out"]

        if targets is None:
            return not pending
        return not pending.intersection(targets)

    def _gate_by_sync(self, event):
        """Reschedule ``event`` if its ``sync_type`` dependency isn't ready."""
        sync_type = event.payload.get("sync_type")
        if sync_type is None:
            return False

        targets = event.payload.get("sync_targets")
        if not self._is_sync_ready(event.program, sync_type, targets):
            retry_evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                data_size=0,
                program=event.program,
                event_type=event.event_type,
                payload=event.payload,
            )
            self.send_event(retry_evt)
            return True
        return False

    def handle_event(self, event):
        if event.event_type == "GEMM":
            print(f"[CP] GEMM 시작: {event.program}, shape={event.payload['gemm_shape']}")
            state = {
                "waiting_dma_in": set(pe.name for pe in self.pes),
                "waiting_gemm": set(pe.name for pe in self.pes),
                "waiting_dma_out": set(pe.name for pe in self.pes),
                "gemm_shape": event.payload["gemm_shape"],
                "weights_size": event.payload["weights_size"],
                "act_size": event.payload["act_size"],
            }
            self.active_gemms[event.program] = state
            for pe in self.pes:
                dma_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=state["weights_size"] + state["act_size"],
                    program=event.program,
                    event_type="PE_DMA_IN",
                    payload={
                        "dst_coords": self.mesh_info["pe_coords"][pe.name],
                        "data_size": state["weights_size"] + state["act_size"],
                        "src_name": self.name,
                        "need_reply": True,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(dma_evt)

        elif event.event_type == "PE_DMA_IN_DONE":
            state = self.active_gemms.get(event.program)
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
                        program=event.program,
                        event_type="PE_GEMM",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "gemm_shape": state["gemm_shape"],
                            "src_name": self.name,
                            "need_reply": True,
                            "input_port": 0,
                            "vc": 0,
                        },
                    )
                    self.send_event(gemm_evt)

        elif event.event_type == "PE_GEMM_DONE":
            state = self.active_gemms.get(event.program)
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
                        program=event.program,
                        event_type="PE_DMA_OUT",
                        payload={
                            "dst_coords": self.mesh_info["pe_coords"][pe.name],
                            "data_size": out_size,
                            "src_name": self.name,
                            "need_reply": True,
                            "pe_name": pe.name,
                        },
                    )
                    self.send_event(dma_evt)

        elif event.event_type == "PE_DMA_OUT_DONE":
            state = self.active_gemms.get(event.program)
            if not state:
                return
            pe_name = event.payload["pe_name"]
            state["waiting_dma_out"].discard(pe_name)
            if not state["waiting_dma_out"]:
                print(f"[CP] GEMM {event.program} 작업 완료")
                self.active_gemms.pop(event.program, None)

        elif event.event_type == "NPU_DMA_IN":
            if self._gate_by_sync(event):
                return

            prog_state = self._create_program_state(event.payload)
            self.active_npu_programs[event.program] = prog_state
            self.npu_dma_in_opcode_done[event.program] = False
            self.npu_cmd_opcode_done.setdefault(event.program, True)
            self.npu_dma_out_opcode_done.setdefault(event.program, True)
            for npu in self.npus:
                dma_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=prog_state["in_size"],
                    program=event.program,
                    event_type="NPU_DMA_IN",
                    payload={
                        "dst_coords": self.mesh_info["npu_coords"][npu.name],
                        "data_size": prog_state["in_size"],
                        "src_name": self.name,
                        "need_reply": True,
                        "opcode_cycles": prog_state["dma_in_opcode_cycles"],
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(dma_evt)

        elif event.event_type == "NPU_CMD":
            if self._gate_by_sync(event):
                return

            program = self.active_npu_programs.get(event.program)
            if not program:
                return
            program["waiting_op"] = set(n.name for n in self.npus)
            self.npu_cmd_opcode_done[event.program] = False
            for npu in self.npus:
                cmd_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=4,
                    program=event.program,
                    event_type="NPU_CMD",
                    payload={
                        "dst_coords": self.mesh_info["npu_coords"][npu.name],
                        "opcode_cycles": program["cmd_opcode_cycles"],
                        "src_name": self.name,
                        "need_reply": True,
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(cmd_evt)

        elif event.event_type == "NPU_DMA_OUT":
            if self._gate_by_sync(event):
                return

            program = self.active_npu_programs.get(event.program)
            if not program:
                return
            program["waiting_dma_out"] = set(n.name for n in self.npus)
            self.npu_dma_out_opcode_done[event.program] = False
            for npu in self.npus:
                out_evt = Event(
                    src=self,
                    dst=self.get_my_router(),
                    cycle=self.engine.current_cycle,
                    data_size=program["out_size"],
                    program=event.program,
                    event_type="NPU_DMA_OUT",
                    payload={
                        "dst_coords": self.mesh_info["npu_coords"][npu.name],
                        "data_size": program["out_size"],
                        "src_name": self.name,
                        "need_reply": True,
                        "opcode_cycles": program["dma_out_opcode_cycles"],
                        "input_port": 0,
                        "vc": 0,
                    },
                )
                self.send_event(out_evt)


        elif event.event_type == "NPU_DMA_IN_DONE":
            prog_state = self.active_npu_programs.get(event.program)
            if not prog_state:
                return
            npu_name = event.payload["npu_name"]
            prog_state["waiting_dma_in"].discard(npu_name)
            if not prog_state["waiting_dma_in"]:
                # Mark completion so external modules can trigger the next phase
                self.npu_dma_in_opcode_done[event.program] = True

        elif event.event_type == "NPU_CMD_DONE":
            prog_state = self.active_npu_programs.get(event.program)
            if not prog_state:
                return
            npu_name = event.payload["npu_name"]
            prog_state["waiting_op"].discard(npu_name)
            if not prog_state["waiting_op"]:
                # Command phase finished
                self.npu_cmd_opcode_done[event.program] = True

        elif event.event_type == "NPU_DMA_OUT_DONE":
            prog_state = self.active_npu_programs.get(event.program)
            if not prog_state:
                return
            npu_name = event.payload["npu_name"]
            prog_state["waiting_dma_out"].discard(npu_name)
            if not prog_state["waiting_dma_out"]:
                print(f"[CP] NPU task {event.program} 완료")
                self.npu_dma_out_opcode_done[event.program] = True
                self.active_npu_programs.pop(event.program, None)

        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
