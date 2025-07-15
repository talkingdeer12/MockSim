from .event import Event
import collections.abc
import copy


def _nested_list_of_lists(shape, maxlen=None):
    if not shape:
        return []
    if len(shape) == 1:
        return [collections.deque(maxlen=maxlen) for _ in range(shape[0])]
    return [_nested_list_of_lists(shape[1:], maxlen) for _ in range(shape[0])]


def _get_from_nested_list(nested_list, indices):
    for index in indices:
        nested_list = nested_list[index]
    return nested_list


def _is_nested_list_empty(nested_list):
    if not isinstance(nested_list, list):
        return False  # A non-list item means the container is not empty
    if not nested_list:
        return True
    return all(_is_nested_list_empty(item) for item in nested_list)


def _is_any_deque_not_empty(nested_list):
    if isinstance(nested_list, collections.deque):
        return bool(nested_list)
    if isinstance(nested_list, list):
        return any(_is_any_deque_not_empty(item) for item in nested_list)
    return False


class HardwareModule:
    def __init__(self, engine, name, mesh_info, buffer_capacity=4, frequency=1000):
        self.engine = engine
        self.name = name
        self.mesh_info = mesh_info
        self.buffer_capacity = buffer_capacity
        self.buffer_occupancy = 0
        self.frequency = frequency  # MHz

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
                evt_type = event.event_type
                if isinstance(event.payload, dict) and event.payload.get("op_type"):
                    evt_type = f"{evt_type}-{event.payload['op_type']}"
                self.engine.logger.log_event(
                    self.engine.current_cycle, self.name, stage, evt_type
                )
            self.handle_event(event)
        finally:
            if not isinstance(self, PipelineModule):
                self._release_slot(event)

    def handle_event(self, event):
        if event.event_type == "RETRY_SEND":
            retry_evt = event.payload["event"]
            self.send_event(retry_evt)

    def send_event(self, event):
        if not event.dst._reserve_slot(event):
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
    def __init__(
        self, engine, name, mesh_info, num_stages, buffer_shapes, buffer_capacity=4, frequency=1000
    ):
        super().__init__(engine, name, mesh_info, buffer_capacity, frequency)
        if len(buffer_shapes) != num_stages + 1:
            raise ValueError("buffer_shapes length must be num_stages + 1")

        self.num_stages = num_stages
        self.stage_funcs = [lambda: None for _ in range(num_stages)]  # Stage funcs now take no args
        self.stage_buffers = [_nested_list_of_lists(shape, buffer_capacity) for shape in buffer_shapes]
        self.buffer_shapes = buffer_shapes
        self.pipeline_scheduled = False

    def set_stage_funcs(self, funcs):
        if len(funcs) != self.num_stages:
            raise ValueError("stage_funcs length must match num_stages")
        self.stage_funcs = funcs

    def add_data(self, data, indices=None):
        indices = indices or []
        input_buffer = _get_from_nested_list(self.stage_buffers[0], indices)
        if len(input_buffer) < self.buffer_capacity:
            input_buffer.append(data)
            self._reserve_slot()  # Manually increment occupancy
            self._schedule_pipeline()
            return True
        return False

    def _schedule_pipeline(self):
        if not self.pipeline_scheduled:
            evt = Event(
                src=self,
                dst=self,
                cycle=self.engine.current_cycle + 1,
                event_type="PIPELINE_TICK",
            )
            self.engine.push_event(evt)
            self.pipeline_scheduled = True

    def handle_event(self, event):
        if event.event_type == "PIPELINE_TICK":
            self.pipeline_scheduled = False
            self._large_pipeline_func()
        else:
            super().handle_event(event)

    def _large_pipeline_func(self):
        self.engine.logger.log(f"Module {self.name}: Cycle {self.engine.current_cycle} - Starting _large_pipeline_func")
        if hasattr(self, 'credit_counts'):
            self.engine.logger.log(f"Module {self.name}: Current credit_counts: {self.credit_counts}")
        for i, buf in enumerate(self.stage_buffers):
            self.engine.logger.log(f"Module {self.name}: Stage {i} buffer: {buf}")

        # Iterate stages in reverse order to propagate backpressure
        for stage_idx in range(self.num_stages - 1, -1, -1):
            # Each stage function will now directly process its input buffer
            # and move data to the next stage's input buffer if not stalled.
            self.stage_funcs[stage_idx]()

        # Handle pipeline output from the last stage buffer
        output_buffer = self.stage_buffers[self.num_stages]
        if not _is_nested_list_empty(output_buffer):
            self._handle_output_recursive(output_buffer)

        # Check if any data is left in any stage buffer (excluding the final output buffer)
        data_left = False
        for i in range(self.num_stages):  # Check all stage input buffers
            if _is_any_deque_not_empty(self.stage_buffers[i]):
                data_left = True
                break

        if data_left:
            self._schedule_pipeline()

    def _handle_output_recursive(self, output_buffer):
        if isinstance(output_buffer, collections.deque): # Base case: it's a deque
            while output_buffer:
                item = output_buffer.popleft()
                self.handle_pipeline_output(item)
                self._release_slot()
        elif isinstance(output_buffer, list): # Recursive case: it's a list of nested buffers
            for sub_buffer in output_buffer:
                self._handle_output_recursive(sub_buffer)

    def handle_pipeline_output(self, data):
        pass

    def _reserve_slot(self, event=None):
        # Override default behavior, as add_data handles capacity check.
        # This is just for incrementing the occupancy counter.
        if self.buffer_occupancy >= self.buffer_capacity:
            return False  # Should not happen if add_data is used correctly
        self.buffer_occupancy += 1
        return True

    def can_accept_event(self, event=None):
        # This check is now effectively managed by add_data.
        # We can make it more accurate by checking the input buffer specifically.
        def count_items(nested_list):
            count = 0
            if isinstance(nested_list, list):
                for item in nested_list:
                    count += count_items(item) if isinstance(item, list) else 1
            return count

        return count_items(self.stage_buffers[0]) < self.buffer_capacity
