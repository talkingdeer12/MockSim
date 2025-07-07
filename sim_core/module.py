from .event import Event


class HardwareModule:
    def __init__(self, engine, name, mesh_info, buffer_capacity=4):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
        self.buffer_capacity = buffer_capacity
        self.buffer_occupancy = 0

    # Credit based buffer bookkeeping
    def _reserve_slot(self, event=None):
        if self.buffer_occupancy >= self.buffer_capacity:
            return False
        self.buffer_occupancy += 1
        return True

    def _release_slot(self, event=None):
        if self.buffer_occupancy > 0:
            self.buffer_occupancy -= 1

    def can_accept_event(self, event=None):
        return self.buffer_occupancy < self.buffer_capacity

    def _process_event(self, event):
        try:
            if self.engine.logger:
                stage = (
                    event.payload.get("stage_idx", 0)
                    if isinstance(event.payload, dict)
                    else 0
                )
                self.engine.logger.log_event(
                    self.engine.current_cycle, self.name, stage, event.event_type
                )
            self.handle_event(event)
        finally:
            self._release_slot(event)

    def handle_event(self, event):
        if event.event_type == "RETRY_SEND":
            retry_evt = event.payload["event"]
            self.send_event(retry_evt)

    def send_event(self, event):
        if not event.dst._reserve_slot(event):
            # Destination buffer full; stall and retry next cycle
            if hasattr(self, "set_stall"):
                self.set_stall(1)
            retry = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="RETRY_SEND",
                payload={"event": event},
            )
            self.engine.push_event(retry)
        else:
            self.engine.push_event(event)


class PipelineModule(HardwareModule):
    """Base class for modules with event driven pipelined execution."""

    def __init__(self, engine, name, mesh_info, num_stages, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        self.num_stages = num_stages
        self.stage_funcs = [lambda m, d: (d, i + 1, False) for i in range(num_stages)]
        self.stage_queues = [list() for _ in range(num_stages)]
        self.stage_scheduled = [False for _ in range(num_stages)]
        self.stage_capacity = buffer_capacity

    def set_stage_funcs(self, funcs):
        if len(funcs) != self.num_stages:
            raise ValueError("stage_funcs length must match num_stages")
        self.stage_funcs = funcs

    def add_data(self, data, stage_idx=0):
        if stage_idx >= self.num_stages:
            raise ValueError("stage_idx out of range")
        self.stage_queues[stage_idx].append(data)
        self._schedule_stage(stage_idx)

    def _schedule_stage(self, idx):
        if not self.stage_scheduled[idx]:
            evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="PIPE_STAGE",
                payload={"stage_idx": idx},
                priority=-idx,
            )
            self.send_event(evt)
            self.stage_scheduled[idx] = True

    def handle_event(self, event):
        if event.event_type == "PIPE_STAGE":
            idx = event.payload["stage_idx"]
            self.stage_scheduled[idx] = False
            self._on_stage_execute(idx)
            self._execute_stage(idx)
        else:
            super().handle_event(event)

    def set_stall(self, cycles):
        # Backwards compatibility for send_event based stalling.
        pass

    def _on_stage_execute(self, idx):
        """Hook called before processing a stage."""
        pass

    def _execute_stage(self, idx):
        if not self.stage_queues[idx]:
            return
        data = self.stage_queues[idx][0]
        func = self.stage_funcs[idx]
        out_data, next_stage, do_stall = func(self, data)
        if do_stall:
            self._schedule_stage(idx)
            return

        if (
            next_stage < self.num_stages
            and len(self.stage_queues[next_stage]) >= self.stage_capacity
        ):
            # downstream stage full - retry later
            self._schedule_stage(idx)
            return

        self.stage_queues[idx].pop(0)

        if next_stage >= self.num_stages:
            self.handle_pipeline_output(out_data)
        else:
            self.stage_queues[next_stage].append(out_data)
            self._schedule_stage(next_stage)

        if self.stage_queues[idx]:
            self._schedule_stage(idx)

    def handle_pipeline_output(self, data):
        pass


class SyncModule(HardwareModule):
    """Mixin providing synchronization utilities for *_SYNC instructions."""

    def __init__(self, engine, name, mesh_info, buffer_capacity=4):
        super().__init__(engine, name, mesh_info, buffer_capacity)
        # Map program -> {"types": set(str), "release_cycle": int}
        self.sync_wait = {}

    def gate_by_sync(self, event, allowed=()):
        """Return True if the event should be retried due to an active sync."""
        wait = self.sync_wait.get(event.program)
        if not wait:
            return False

        if event.event_type not in allowed and (
            wait["types"] or wait.get("release_cycle") == self.engine.current_cycle
        ):
            retry_evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                program=event.program,
                event_type=event.event_type,
                payload=event.payload,
                data_size=getattr(event, "data_size", 0),
            )
            self.send_event(retry_evt)
            return True

        if (
            not wait["types"]
            and wait.get("release_cycle", -1) < self.engine.current_cycle
        ):
            del self.sync_wait[event.program]
        return False

    def process_phase_done(
        self,
        program,
        actor,
        active_states,
        waiting_field,
        done_dict,
        wait_type,
        resume_fn,
        task_id=None,
    ):
        """Common logic for handling *_DONE events under synchronization.

        Parameters
        ----------
        program : str
            Program identifier.
        actor : str
            Name of the module that completed the phase.
        active_states : dict
            Dictionary mapping program names to per-program state.
        waiting_field : str
            Field in the per-program state that tracks remaining actors. The
            value may be either a ``set`` or ``dict`` mapping ``task_id`` to a
            set.
        done_dict : dict
            Boolean flags keyed by program indicating whether the phase has no
            outstanding work.
        wait_type : str
            Name of the phase (e.g. ``"dma_in"``) used for sync logic.
        resume_fn : callable
            Function invoked when a sync is released to resume program
            execution.
        task_id : optional
            Generic identifier for the unit of work that generated the
            ``*_DONE`` event when ``waiting_field`` stores per-task
            information.
        """
        state = active_states.get(program)
        if state:
            waiting = state.get(waiting_field)
            if isinstance(waiting, dict):
                task_set = waiting.get(task_id)
                if task_set is not None:
                    task_set.discard(actor)
                    if not task_set:
                        waiting.pop(task_id, None)
                if waiting:
                    return False
            else:
                waiting.discard(actor)
                if waiting:
                    return False

        done_dict[program] = True

        wait = self.sync_wait.get(program)
        should_resume = False
        if wait and wait_type in wait["types"]:
            wait["types"].discard(wait_type)
            if not wait["types"]:
                wait["release_cycle"] = self.engine.current_cycle
                should_resume = True
        else:
            # No sync for this phase; still resume so RUN_PROGRAM can check
            should_resume = True

        if should_resume:
            resume_fn()

        return True
