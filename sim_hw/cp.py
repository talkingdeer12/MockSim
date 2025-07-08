from sim_core.module import SyncModule
from sim_core.event import Event


class ControlProcessor(SyncModule):
    def __init__(
        self, engine, name, mesh_info, pes, dram, npus=None, buffer_capacity=4
    ):
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
        # Per-program NPU synchronization state
        # Each entry keeps a set of remaining phase names and the cycle the sync
        # was released.  We keep the entry around for one extra cycle so that
        # events scheduled in the same cycle as the *_DONE that released the
        # sync are still blocked.
        # synchronization tracking for NPU operations
        self.npu_sync_wait = self.sync_wait
        # Store control programs and per-program execution state
        self.program_store = {}
        self.program_state = {}
        # Pre-created NPU program templates
        self.npu_program_templates = {}
        # Per-program instruction scoreboards
        self.program_scoreboards = {}
        # Event handler dispatch table. New instructions can be registered via
        # :func:`register_handler` so ``handle_event`` remains simple.
        self.event_handlers = {}
        self._register_default_handlers()

    def _resume_program(self, program):
        state = self.program_state.get(program)
        if state:
            state["waiting"] = False
            resume = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                program=program,
                event_type="RUN_PROGRAM",
            )
            self.engine.push_event(resume)
            # Mark the latest issued sync as done
            board = self.program_scoreboards.get(program)
            if board:
                for entry in board["entries"]:
                    if entry["event_type"] == "NPU_SYNC" and entry["status"] == "issued":
                        entry["status"] = "done"
                        break

    def _scoreboard_mark_done(self, program, event_type):
        board = self.program_scoreboards.get(program)
        if not board:
            return
        for entry in board["entries"]:
            if entry["event_type"] == event_type and entry["status"] == "issued":
                entry["status"] = "done"
                break

    def _create_program_state(self, payload):
        """Initialize per-program state used to track outstanding work units."""
        return {
            # Each waiting_* entry maps task_id -> set of NPU names.  Using a
            # dict allows multiple units from the same program to execute
            # concurrently.
            "waiting_dma_in": {},
            "waiting_op": {},
            "waiting_dma_out": {},
            "program_cycles": payload["program_cycles"],
            "in_size": payload["in_size"],
            "out_size": payload["out_size"],
            "dma_in_opcode_cycles": payload.get("dma_in_opcode_cycles", 5),
            "dma_out_opcode_cycles": payload.get("dma_out_opcode_cycles", 5),
            "cmd_opcode_cycles": payload.get(
                "cmd_opcode_cycles", payload["program_cycles"]
            ),
        }

    def load_program(self, name, instructions):
        """Register a list of instructions for sequential execution."""
        self.program_store[name] = instructions
        self.program_state[name] = {"pc": 0, "waiting": False}
        cfg = None
        for instr in instructions:
            payload = instr.get("payload")
            if isinstance(payload, dict) and "program_cycles" in payload:
                cfg = payload
                break
        if cfg:
            tmpl = self._create_program_state(cfg)
            self.npu_program_templates[name] = {
                k: set(v) if isinstance(v, set) else v for k, v in tmpl.items()
            }
            # Pre-register active state so DMA handlers assume existence
            self.active_npu_programs[name] = {
                k: set(v) if isinstance(v, set) else v for k, v in tmpl.items()
            }
            # Initialize opcode completion flags
            self.npu_dma_in_opcode_done[name] = True
            self.npu_cmd_opcode_done[name] = True
            self.npu_dma_out_opcode_done[name] = True

        # Build initial scoreboard
        sb_entries = []
        for idx, instr in enumerate(instructions):
            etype = instr["event_type"]
            if etype in ("NPU_DMA_IN", "PE_DMA_IN"):
                op_type = "read"
            elif etype in ("NPU_DMA_OUT", "PE_DMA_OUT"):
                op_type = "write"
            elif etype in ("NPU_CMD", "PE_GEMM", "GEMM"):
                op_type = "compute"
            else:
                op_type = "sync"
            entry = {
                "id": idx,
                "event_type": etype,
                "payload": instr.get("payload", {}),
                "op_type": op_type,
                "status": "pending",
                "deps": set(),
            }
            if etype == "NPU_SYNC":
                entry["deps"] = set(instr.get("payload", {}).get("sync_types", []))
            sb_entries.append(entry)
        self.program_scoreboards[name] = {"entries": sb_entries, "commit": 0}


    def register_handler(self, evt_type, fn):
        """Register a handler for ``evt_type``."""
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("RUN_PROGRAM", self._handle_run_program)
        self.register_handler("GEMM", self._handle_gemm)
        self.register_handler("PE_DMA_IN_DONE", self._handle_pe_dma_in_done)
        self.register_handler("PE_GEMM_DONE", self._handle_pe_gemm_done)
        self.register_handler("PE_DMA_OUT_DONE", self._handle_pe_dma_out_done)
        self.register_handler("NPU_SYNC", self._handle_npu_sync)
        self.register_handler("NPU_DMA_IN", self._handle_npu_dma_in)
        self.register_handler("NPU_CMD", self._handle_npu_cmd)
        self.register_handler("NPU_DMA_OUT", self._handle_npu_dma_out)
        self.register_handler("NPU_DMA_IN_DONE", self._handle_npu_dma_in_done)
        self.register_handler("NPU_CMD_DONE", self._handle_npu_cmd_done)
        self.register_handler("NPU_DMA_OUT_DONE", self._handle_npu_dma_out_done)

    # ------------------------------------------------------------------
    # Default handlers

    def _handle_run_program(self, event):
        prog = self.program_store.get(event.program)
        state = self.program_state.get(event.program)
        board = self.program_scoreboards.get(event.program)
        if not prog or not state or not board:
            return

        # If all instructions completed, retire program
        if all(e["status"] == "done" for e in board["entries"]):
            if not state.get("waiting"):
                if (
                    self.npu_dma_in_opcode_done.get(event.program, True)
                    and self.npu_cmd_opcode_done.get(event.program, True)
                    and self.npu_dma_out_opcode_done.get(event.program, True)
                ):
                    print(f"[CP] NPU task {event.program} 완료")
                    self.active_npu_programs.pop(event.program, None)
            return

        if state.get("waiting"):
            retry = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                program=event.program,
                event_type="RUN_PROGRAM",
            )
            self.engine.push_event(retry)
            return

        issued = False
        for entry in board["entries"]:
            if entry["status"] != "pending":
                continue
            etype = entry["event_type"]
            if etype == "NPU_DMA_IN":
                task_id = entry.get("payload", {}).get("task_id")
                conflict = False
                for e in board["entries"]:
                    if e is entry:
                        break
                    if (
                        e["event_type"] == "NPU_DMA_IN"
                        and e.get("payload", {}).get("task_id") == task_id
                        and e["status"] != "done"
                    ):
                        conflict = True
                        break
                if conflict:
                    continue
            if etype == "NPU_CMD" and self.active_npu_programs.get(event.program, {}).get("waiting_dma_in"):
                continue
            if etype == "NPU_DMA_OUT" and self.active_npu_programs.get(event.program, {}).get("waiting_op"):
                continue
            # Issue the instruction
            entry["status"] = "issued"
            state["pc"] = entry["id"] + 1
            if etype == "NPU_SYNC":
                state["waiting"] = True
            elif etype == "NPU_DMA_IN":
                self.npu_dma_in_opcode_done[event.program] = False
            elif etype == "NPU_CMD":
                self.npu_cmd_opcode_done[event.program] = False
            elif etype == "NPU_DMA_OUT":
                self.npu_dma_out_opcode_done[event.program] = False

            instr_evt = Event(
                src=None,
                dst=self,
                cycle=self.engine.current_cycle,
                program=event.program,
                event_type=etype,
                payload=entry.get("payload", {}),
            )
            tsk = instr_evt.payload.get("task_id")
            print(
                f"[CP] Dispatch {instr_evt.event_type} task={tsk} cycle={self.engine.current_cycle}"
            )
            self.engine.push_event(instr_evt)
            issued = True
            break

        if issued or state.get("waiting"):
            nxt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                program=event.program,
                event_type="RUN_PROGRAM",
            )
            self.engine.push_event(nxt)

    def _handle_gemm(self, event):
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

    def _handle_pe_dma_in_done(self, event):
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

    def _handle_pe_gemm_done(self, event):
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

    def _handle_pe_dma_out_done(self, event):
        state = self.active_gemms.get(event.program)
        if not state:
            return
        pe_name = event.payload["pe_name"]
        state["waiting_dma_out"].discard(pe_name)
        if not state["waiting_dma_out"]:
            print(f"[CP] GEMM {event.program} 작업 완료")
            self.active_gemms.pop(event.program, None)

    def _handle_npu_sync(self, event):
        self.npu_sync_wait[event.program] = {
            "types": set(event.payload.get("sync_types", [])),
            "release_cycle": None,
        }

    def _handle_npu_dma_in(self, event):
        prog_state = self.active_npu_programs.get(event.program)
        if not prog_state:
            raise KeyError(f"Unknown NPU program {event.program}")
        task_id = event.payload.get("task_id")
        prog_state.setdefault("waiting_dma_in", {})[task_id] = set(n.name for n in self.npus)
        self.npu_dma_in_opcode_done[event.program] = False
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
                    "task_id": task_id,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(dma_evt)

    def _handle_npu_cmd(self, event):
        program = self.active_npu_programs.get(event.program)
        if not program:
            raise KeyError(f"Unknown NPU program {event.program}")
        task_id = event.payload.get("task_id")
        program.setdefault("waiting_op", {})[task_id] = set(n.name for n in self.npus)
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
                    "task_id": task_id,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(cmd_evt)

    def _handle_npu_dma_out(self, event):
        program = self.active_npu_programs.get(event.program)
        if not program:
            raise KeyError(f"Unknown NPU program {event.program}")
        task_id = event.payload.get("task_id")
        program.setdefault("waiting_dma_out", {})[task_id] = set(n.name for n in self.npus)
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
                    "task_id": task_id,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(out_evt)

    def _handle_npu_dma_in_done(self, event):
        print(
            f"[CP] DMA_IN_DONE task={event.payload.get('task_id')} cycle={self.engine.current_cycle}"
        )
        self.process_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_dma_in",
            self.npu_dma_in_opcode_done,
            "dma_in",
            lambda: self._resume_program(event.program),
            task_id=event.payload.get("task_id"),
        )
        self._scoreboard_mark_done(event.program, "NPU_DMA_IN")

    def _handle_npu_cmd_done(self, event):
        print(
            f"[CP] CMD_DONE task={event.payload.get('task_id')} cycle={self.engine.current_cycle}"
        )
        self.process_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_op",
            self.npu_cmd_opcode_done,
            "cmd",
            lambda: self._resume_program(event.program),
            task_id=event.payload.get("task_id"),
        )
        self._scoreboard_mark_done(event.program, "NPU_CMD")

    def _handle_npu_dma_out_done(self, event):
        print(
            f"[CP] DMA_OUT_DONE task={event.payload.get('task_id')} cycle={self.engine.current_cycle}"
        )
        self.process_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_dma_out",
            self.npu_dma_out_opcode_done,
            "dma_out",
            lambda: self._resume_program(event.program),
            task_id=event.payload.get("task_id"),
        )
        self._scoreboard_mark_done(event.program, "NPU_DMA_OUT")

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
