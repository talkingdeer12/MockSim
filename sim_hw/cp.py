from sim_core.module import HardwareModule
from sim_core.event import Event


class ControlProcessor(HardwareModule):
    def __init__(self, engine, name, mesh_info, npus=None, buffer_capacity=4, frequency=1000):
        super().__init__(engine, name, mesh_info, buffer_capacity, frequency)
        self.npus = npus or []
        # Control programs manage only NPUs.
        self.active_npu_programs = {}
        # Track synchronization state of NPU commands so external modules can
        # poll progress.  Each dict maps a program name to a boolean flag.
        self.npu_dma_in_opcode_done = {}
        self.npu_cmd_opcode_done = {}
        self.npu_dma_out_opcode_done = {}
        # DMA structural hazard flag
        self.dma_busy = False
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

    def _schedule_run(self, program):
        evt = Event(
            src=self,
            dst=self,
            cycle=self.engine.current_cycle + 1,
            program=program,
            event_type="RUN_PROGRAM",
        )
        self.engine.push_event(evt)

    def _scoreboard_mark_done(self, program, event_type, stream_id):
        board = self.program_scoreboards.get(program)
        if not board:
            return
        for entry in board["entries"]:
            if (
                entry["event_type"] == event_type
                and entry.get("payload", {}).get("stream_id") == stream_id
                and entry["status"] == "issued"
            ):
                entry["status"] = "done"
                break
        # Advance commit pointer
        while (
            board["commit"] < len(board["entries"])
            and board["entries"][board["commit"]]["status"] == "done"
        ):
            board["commit"] += 1

    def _update_phase_done(
        self,
        program,
        actor,
        active_states,
        waiting_field,
        done_dict,
        stream_id=None,
    ):
        state = active_states.get(program)
        if state:
            waiting = state.get(waiting_field)
            if isinstance(waiting, dict):
                task_set = waiting.get(stream_id)
                if task_set is not None:
                    task_set.discard(actor)
                    if not task_set:
                        waiting.pop(stream_id, None)
                if waiting:
                    return False
            else:
                waiting.discard(actor)
                if waiting:
                    return False

        done_dict[program] = True
        return True

    def _has_stream_dependency(self, board, entry):
        """Return True if an earlier instruction with the same stream is not done."""
        sid = entry.get("payload", {}).get("stream_id")
        for e in board["entries"]:
            if e is entry:
                break
            if e.get("payload", {}).get("stream_id") == sid and e["status"] != "done":
                return True
        return False

    def _can_issue(self, program, entry, board):
        if self._has_stream_dependency(board, entry):
            return False
        if entry["op_type"] in ("read", "write") and self.dma_busy:
            return False
        return True

    def _issue_instruction(self, program, entry):
        """Mark scoreboard and dispatch the instruction event."""
        state = self.program_state[program]
        etype = entry["event_type"]

        entry["status"] = "issued"
        state["pc"] = entry["id"] + 1

        if etype == "NPU_DMA_IN":
            self.npu_dma_in_opcode_done[program] = False
            self.dma_busy = True
        elif etype == "NPU_CMD":
            self.npu_cmd_opcode_done[program] = False
        elif etype == "NPU_DMA_OUT":
            self.npu_dma_out_opcode_done[program] = False
            self.dma_busy = True

        instr_evt = Event(
            src=None,
            dst=self,
            cycle=self.engine.current_cycle,
            program=program,
            event_type=etype,
            payload=entry.get("payload", {}),
        )
        sid = instr_evt.payload.get("stream_id")
        print(
            f"[CP] Dispatch {instr_evt.event_type} stream={sid} cycle={self.engine.current_cycle}"
        )
        self.engine.push_event(instr_evt)

    def _create_program_state(self, payload):
        """Initialize per-program state used to track outstanding work units."""
        return {
            # Each waiting_* entry maps ``stream_id`` -> set of NPU names.
            # ``stream_id`` represents an independent execution stream inside a
            # control program, typically corresponding to a logical layer or
            # iteration.  Multiple streams can be in flight, so we track each
            # stream separately to allow interleaving of DMA/compute phases.
            #
            #    [stream 0] DMA_IN -> CMD -> DMA_OUT
            #    [stream 1] DMA_IN -> CMD -> DMA_OUT
            #
            # Streams progress independently but share the same program.
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
        self.program_state[name] = {"pc": 0}
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
            if etype == "NPU_DMA_IN":
                op_type = "read"
            elif etype == "NPU_DMA_OUT":
                op_type = "write"
            elif etype == "NPU_CMD":
                op_type = "compute"
            else:
                op_type = "other"
            entry = {
                "id": idx,
                "event_type": etype,
                "payload": instr.get("payload", {}),
                "op_type": op_type,
                "status": "pending",
                "deps": set(),
            }
            sb_entries.append(entry)
        self.program_scoreboards[name] = {"entries": sb_entries, "commit": 0}


    def register_handler(self, evt_type, fn):
        """Register a handler for ``evt_type``."""
        self.event_handlers[evt_type] = fn

    def _register_default_handlers(self):
        self.register_handler("RUN_PROGRAM", self._handle_run_program)
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
            if (
                self.npu_dma_in_opcode_done.get(event.program, True)
                and self.npu_cmd_opcode_done.get(event.program, True)
                and self.npu_dma_out_opcode_done.get(event.program, True)
            ):
                print(f"[CP] NPU task {event.program} 완료")
                self.active_npu_programs.pop(event.program, None)
            return

        issued = False
        for entry in board["entries"]:
            if entry["status"] != "pending":
                continue
            if not self._can_issue(event.program, entry, board):
                continue
            self._issue_instruction(event.program, entry)
            issued = True
            break

        if issued:
            self._schedule_run(event.program)

    def _handle_npu_dma_in(self, event):
        prog_state = self.active_npu_programs.get(event.program)
        if not prog_state:
            raise KeyError(f"Unknown NPU program {event.program}")
        sid = event.payload.get("stream_id")
        prog_state.setdefault("waiting_dma_in", {})[sid] = set(n.name for n in self.npus)
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
                    "stream_id": sid,
                    "eaddr": event.payload.get("eaddr"),
                    "iaddr": event.payload.get("iaddr"),
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(dma_evt)

    def _handle_npu_cmd(self, event):
        program = self.active_npu_programs.get(event.program)
        if not program:
            raise KeyError(f"Unknown NPU program {event.program}")
        sid = event.payload.get("stream_id")
        program.setdefault("waiting_op", {})[sid] = set(n.name for n in self.npus)
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
                    "stream_id": sid,
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(cmd_evt)

    def _handle_npu_dma_out(self, event):
        program = self.active_npu_programs.get(event.program)
        if not program:
            raise KeyError(f"Unknown NPU program {event.program}")
        sid = event.payload.get("stream_id")
        program.setdefault("waiting_dma_out", {})[sid] = set(n.name for n in self.npus)
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
                    "stream_id": sid,
                    "eaddr": event.payload.get("eaddr"),
                    "iaddr": event.payload.get("iaddr"),
                    "input_port": 0,
                    "vc": 0,
                },
            )
            self.send_event(out_evt)

    def _handle_npu_dma_in_done(self, event):
        sid = event.payload.get("stream_id")
        print(
            f"[CP] DMA_IN_DONE stream={sid} cycle={self.engine.current_cycle}"
        )
        self._update_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_dma_in",
            self.npu_dma_in_opcode_done,
            stream_id=sid,
        )
        self._scoreboard_mark_done(event.program, "NPU_DMA_IN", sid)
        self.dma_busy = False
        self._schedule_run(event.program)

    def _handle_npu_cmd_done(self, event):
        sid = event.payload.get("stream_id")
        print(
            f"[CP] CMD_DONE stream={sid} cycle={self.engine.current_cycle}"
        )
        self._update_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_op",
            self.npu_cmd_opcode_done,
            stream_id=sid,
        )
        self._scoreboard_mark_done(event.program, "NPU_CMD", sid)
        self._schedule_run(event.program)

    def _handle_npu_dma_out_done(self, event):
        sid = event.payload.get("stream_id")
        print(
            f"[CP] DMA_OUT_DONE stream={sid} cycle={self.engine.current_cycle}"
        )
        self._update_phase_done(
            event.program,
            event.payload["npu_name"],
            self.active_npu_programs,
            "waiting_dma_out",
            self.npu_dma_out_opcode_done,
            stream_id=sid,
        )
        self._scoreboard_mark_done(event.program, "NPU_DMA_OUT", sid)
        self.dma_busy = False
        self._schedule_run(event.program)

    def handle_event(self, event):
        handler = self.event_handlers.get(event.event_type)
        if handler:
            handler(event)
        else:
            super().handle_event(event)

    def get_my_router(self):
        coords = self.mesh_info["cp_coords"][self.name]
        return self.mesh_info["router_map"][coords]
