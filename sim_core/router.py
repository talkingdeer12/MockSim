from sim_core.module import PipelineModule, _nested_list_of_lists, _get_from_nested_list, _is_nested_list_empty
from sim_core.event import Event
import collections

def _arbitrate_lrg(candidates, counter):
    if not candidates:
        return None, counter
    # Sort candidates to make arbitration deterministic
    sorted_candidates = sorted(list(candidates))
    for i in range(len(sorted_candidates)):
        pick = (counter + i) % len(sorted_candidates)
        if sorted_candidates[pick] in candidates:
            winner = sorted_candidates[pick]
            counter = (pick + 1) % len(sorted_candidates)
            return winner, counter
    return None, counter


class Router(PipelineModule):
    def __init__(self, engine, name, mesh_x, mesh_y, mesh_info,
                 num_ports=5, num_vcs=2, buffer_capacity=4, frequency=1000):
        # Define buffer shapes for each stage (RC_in, VA_in, SA_in, ST_in, ST_out)
        buffer_shapes = [
            [num_ports, num_vcs],  # RC input buffer
            [num_ports, num_vcs],  # VA input buffer
            [num_ports, num_vcs],  # SA input buffer
            [num_ports],           # ST input buffer
            [num_ports]            # ST output buffer (for handle_pipeline_output)
        ]
        super().__init__(engine, name, mesh_info, num_stages=4, buffer_shapes=buffer_shapes, buffer_capacity=buffer_capacity, frequency=frequency)

        self.x = mesh_x
        self.y = mesh_y
        self.num_ports = num_ports
        self.num_vcs = num_vcs
        self.port_num_vcs = [1] + [num_vcs] * (num_ports - 1)

        # Output link and credit management
        self.output_links = [None] * num_ports
        self.credit_counts = [[buffer_capacity] * self.port_num_vcs[i] for i in range(num_ports)]

        # Arbitration state
        self.va_lrg_counters = [0] * self.num_ports # Input port
        self.va_out_vc_lrg_counters = [0] * self.num_ports # Output port
        self.sa_lrg_counters = [0] * num_ports # Output port

        self.attached_module = None

        # Set stage functions for PipelineModule
        self.set_stage_funcs([
            self._stage_rc, # Stage 0
            self._stage_va, # Stage 1
            self._stage_sa, # Stage 2
            self._stage_st  # Stage 3
        ])
    def _reserve_slot(self, event=None):
        return True

    def set_neighbors(self, neighbor_dict):
        for d, n in neighbor_dict.items():
            out_port = DIRS.index(d)
            in_port = DIRS.index(OPPOSITE[d])
            self.output_links[out_port] = (n, in_port)
            self.engine.logger.log(f"Router {self.name}: Set neighbor {n.name} on out_port {out_port} (in_port {in_port})")

    def attach_module(self, mod):
        self.attached_module = mod
        self.output_links[DIRS.index("LOCAL")] = (mod, 0)
        self.engine.logger.log(f"Router {self.name}: Attached module {mod.name} on LOCAL port.")

    def handle_event(self, event):
        event_types_to_route = {
            "PACKET", "NPU_DMA_IN", "NPU_CMD", "NPU_DMA_OUT",
            "DMA_READ", "DMA_WRITE", "WRITE_REPLY", "DMA_READ_REPLY",
            "NPU_DMA_IN_DONE", "NPU_CMD_DONE", "NPU_DMA_OUT_DONE"
        }

        if event.event_type in event_types_to_route:
            in_port = event.payload.get("input_port", 0)
            in_vc = event.payload.get("input_vc", 0)

            # If packet is from an attached module, set credit return info
            if event.src == self.attached_module:
                event.payload["prev_out_port"] = in_port
                event.payload["prev_out_vc"] = in_vc

            # Store upstream port and VC for credit return
            event.payload["last_hop_src"] = event.src

            # add_data returns True if successful, False if buffer full
            if not self.add_data(event, indices=[in_port, in_vc]):
                # If buffer is full, retry next cycle
                retry = Event(src=self, dst=self, cycle=self.engine.current_cycle + 1, event_type="RETRY_SEND", payload={"event": event})
                self.engine.push_event(retry)
            self.engine.logger.log(f"Router {self.name}: Received packet {event.payload.get('id', event.event_type)} on port {in_port}, vc {in_vc}")
        elif event.event_type == "RECV_CRED":
            port = event.payload.get("prev_out_port", 0)
            vc = event.payload.get("prev_out_vc", 0)
            self.credit_counts[port][vc] += 1
            self.engine.logger.log(
                f"Router {self.name}: Received credit for port {port}, vc {vc}. New count: {self.credit_counts[port][vc]}"
            )
            self._schedule_pipeline() # Schedule pipeline to process stalled packets
            return # No _release_slot for credit events
        else:
            super().handle_event(event)

    def _stage_rc(self):
        # RC stage processes packets from stage_buffers[0] (RC input)
        # and tries to move them to stage_buffers[1] (VA input)
        for in_port_idx in range(self.num_ports):
            for in_vc_idx in range(self.port_num_vcs[in_port_idx]):
                rc_input_buffer = self.stage_buffers[0][in_port_idx][in_vc_idx]
                if rc_input_buffer:
                    pkt = rc_input_buffer[0]  # Peek at the head of the queue
                    self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - RC - Processing packet {pkt.payload.get('id', pkt.event_type)} from in_port {in_port_idx}, in_vc {in_vc_idx}")

                    dst_coords = pkt.payload.get("dst_coords")
                    if (self.x, self.y) == tuple(dst_coords):
                        direction = "LOCAL"
                    else:
                        dx = dst_coords[0] - self.x
                        dy = dst_coords[1] - self.y
                        # Dimension order routing
                        if dx != 0:
                            direction = "E" if dx > 0 else "W"
                        elif dy != 0:
                            direction = "S" if dy > 0 else "N"
                        else:  # Should not happen if not at destination
                            direction = "LOCAL"

                    out_port = DIRS.index(direction)
                    pkt.payload["out_port"] = out_port

                    # Check if the corresponding VA input buffer is full
                    va_input_buffer = self.stage_buffers[1][in_port_idx][in_vc_idx]
                    if len(va_input_buffer) < self.buffer_capacity:
                        # Not stalled, move packet to next stage
                        rc_input_buffer.popleft()
                        va_input_buffer.append(pkt)
                        self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - RC - Moved packet {pkt.payload.get('id', pkt.event_type)} to VA input buffer. out_port: {out_port}")
                    else:
                        self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - RC - Stalled packet {pkt.payload.get('id', pkt.event_type)}, VA input buffer full. Current VA buffer occupancy: {len(va_input_buffer)}/{self.buffer_capacity}")
                    # Else: stalled, packet remains in RC input buffer for next cycle

    def _stage_va(self):
        # VA stage processes packets from stage_buffers[1] (VA input)
        # and tries to move them to stage_buffers[2] (SA input)

        # First, collect all candidates for VA arbitration
        # Candidates are (in_port_idx, in_vc_idx) for packets at the head of VA input buffers
        # that have a valid out_port from RC stage and credit available.
        va_candidates = collections.defaultdict(list)  # Key: (out_port), Value: list of (in_port_idx, in_vc_idx)

        for in_port_idx in range(self.num_ports):
            for in_vc_idx in range(self.port_num_vcs[in_port_idx]):
                va_input_buffer = self.stage_buffers[1][in_port_idx][in_vc_idx]
                if va_input_buffer:
                    pkt = va_input_buffer[0]
                    out_port = pkt.payload["out_port"]
                    self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - Processing packet {pkt.payload.get('id', pkt.event_type)} from in_port {in_port_idx}, in_vc {in_vc_idx} for out_port {out_port}")

                    # Check if credit is available for any VC on the chosen output port
                    # Only consider VCs that have credit > 0
                    available_vcs_for_out_port = [
                        ovc
                        for ovc in range(self.port_num_vcs[out_port])
                        if self.credit_counts[out_port][ovc] > 0
                    ]

                    if available_vcs_for_out_port:
                        va_candidates[out_port].append((in_port_idx, in_vc_idx))
                    else:
                        self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - Stalled packet {pkt.payload.get('id', pkt.event_type)}, no credit available for out_port {out_port}. Current credits: {self.credit_counts[out_port]}")

        # Perform VA arbitration for each output port
        for out_port, candidates_list in va_candidates.items():
            # Group candidates by (in_port_idx, out_port) to apply LRG for out_vc selection
            # This is the first level of arbitration: within an input port, which VC gets to pick an output VC
            in_port_to_in_vc_candidates = collections.defaultdict(list)
            for in_p, in_vc in candidates_list:
                in_port_to_in_vc_candidates[in_p].append(in_vc)

            va_winners_for_out_port = []  # List of (in_port_idx, in_vc_idx, chosen_out_vc)

            for in_port_idx, in_vcs_for_this_in_port in in_port_to_in_vc_candidates.items():
                # Arbitrate among in_vcs for this in_port to select one to proceed
                # This is the LRG for input VCs within the same input port
                winner_in_vc, self.va_lrg_counters[in_port_idx] = _arbitrate_lrg(
                    set(in_vcs_for_this_in_port), self.va_lrg_counters[in_port_idx]
                )

                if winner_in_vc is not None:
                    pkt = self.stage_buffers[1][in_port_idx][winner_in_vc][0]

                    # Now, for the winner (in_port_idx, winner_in_vc), select an output VC
                    # This is the second level of arbitration: which output VC to use

                    # Collect available output VCs for this out_port
                    available_out_vcs = [
                        ovc
                        for ovc in range(self.port_num_vcs[out_port])
                        if self.credit_counts[out_port][ovc] > 0
                    ]

                    if available_out_vcs:
                        # LRG arbitration for output VCs
                        chosen_out_vc, self.va_out_vc_lrg_counters[out_port] = _arbitrate_lrg(
                            set(available_out_vcs), self.va_out_vc_lrg_counters[out_port]
                        )

                        if chosen_out_vc is not None:
                            va_winners_for_out_port.append(
                                (in_port_idx, winner_in_vc, chosen_out_vc)
                            )
                            pkt.payload["out_vc"] = chosen_out_vc
                            pkt.payload["va_granted"] = True  # Mark as VA granted
                            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - Granted out_vc {chosen_out_vc} for packet {pkt.payload.get('id', pkt.event_type)}. Current credits: {self.credit_counts[out_port]}")
                        else:
                            pkt.payload["va_granted"] = False  # No output VC granted
                            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - No output VC granted for packet {pkt.payload.get('id', pkt.event_type)}. Available VCs: {available_out_vcs}")
                    else:
                        pkt.payload["va_granted"] = False  # No output VC available
                        self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - No available output VCs for packet {pkt.payload.get('id', pkt.event_type)}. Current credits: {self.credit_counts[out_port]}")
                else:
                    pkt.payload["va_granted"] = False  # No input VC granted
                    self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - No winner_in_vc for packet {pkt.payload.get('id', pkt.event_type)}. Candidates: {in_vcs_for_this_in_port}")

        # Now, process the winners and move packets to SA stage
        for in_port_idx in range(self.num_ports):
            for in_vc_idx in range(self.port_num_vcs[in_port_idx]):
                va_input_buffer = self.stage_buffers[1][in_port_idx][in_vc_idx]
                if va_input_buffer:
                    pkt = va_input_buffer[0]
                    if pkt.payload.get("va_granted"):
                        out_port = pkt.payload["out_port"]
                        chosen_out_vc = pkt.payload["out_vc"]

                        # Check if next stage (SA) input buffer is full
                        sa_input_buffer = self.stage_buffers[2][in_port_idx][in_vc_idx]
                        if len(sa_input_buffer) < self.buffer_capacity:
                            # Not stalled, move packet to next stage
                            va_input_buffer.popleft()
                            sa_input_buffer.append(pkt)
                            self.credit_counts[out_port][chosen_out_vc] -= 1  # Consume credit
                            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - Moved packet {pkt.payload.get('id', pkt.event_type)} to SA input buffer. Consumed credit for out_port {out_port}, out_vc {chosen_out_vc}. New credit: {self.credit_counts[out_port][chosen_out_vc]}")
                        else:
                            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - VA - Stalled packet {pkt.payload.get('id', pkt.event_type)}, SA input buffer full. Current SA buffer occupancy: {len(sa_input_buffer)}/{self.buffer_capacity}")
                    # Else: not VA granted, packet remains in VA input buffer for next cycle
                    pkt.payload["va_granted"] = False  # Reset for next cycle

    def _stage_sa(self):
        # SA stage processes packets from stage_buffers[2] (SA input)
        # and tries to move them to stage_buffers[3] (ST input)

        # Collect all candidates for SA arbitration
        # Candidates are (in_port_idx, in_vc_idx) for packets at the head of SA input buffers
        sa_candidates = collections.defaultdict(list)  # Key: out_port, Value: list of (in_port_idx, in_vc_idx)

        for in_port_idx in range(self.num_ports):
            for in_vc_idx in range(self.port_num_vcs[in_port_idx]):
                sa_input_buffer = self.stage_buffers[2][in_port_idx][in_vc_idx]
                if sa_input_buffer:
                    pkt = sa_input_buffer[0]
                    out_port = pkt.payload["out_port"]
                    self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - SA - Processing packet {pkt.payload.get('id', pkt.event_type)} from in_port {in_port_idx}, in_vc {in_vc_idx} for out_port {out_port}")
                    sa_candidates[out_port].append((in_port_idx, in_vc_idx))

        sa_winners = {}  # Key: out_port, Value: (in_port_idx, in_vc_idx)

        # Perform SA arbitration for each output port
        for out_port, candidates_list in sa_candidates.items():
            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - SA - Arbitrating for out_port {out_port}. Candidates: {candidates_list}")
            # LRG arbitration among (in_port_idx, in_vc_idx) pairs contending for this out_port
            winner_tuple, self.sa_lrg_counters[out_port] = _arbitrate_lrg(
                set(candidates_list), self.sa_lrg_counters[out_port]
            )
            self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - SA - Winner for out_port {out_port}: {winner_tuple}. New LRG counter: {self.sa_lrg_counters[out_port]}")
            if winner_tuple is not None:
                sa_winners[out_port] = winner_tuple

        # Now, process the winners and move packets to ST stage
        for out_port, (in_port_idx, in_vc_idx) in sa_winners.items():
            sa_input_buffer = self.stage_buffers[2][in_port_idx][in_vc_idx]
            pkt = sa_input_buffer[0]  # Get the winning packet

            # Check if next stage (ST) input buffer is full
            st_input_buffer = self.stage_buffers[3][out_port]
            if len(st_input_buffer) < self.buffer_capacity:
                # Not stalled, move packet to next stage
                sa_input_buffer.popleft()
                st_input_buffer.append(pkt)
                self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - SA - Moved packet {pkt.payload.get('id', pkt.event_type)} to ST input buffer.")
            else:
                self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - SA - Stalled packet {pkt.payload.get('id', pkt.event_type)}, ST input buffer full. Current ST buffer occupancy: {len(st_input_buffer)}/{self.buffer_capacity}")
            # Else: stalled, packet remains in SA input buffer for next cycle

    def _stage_st(self):
        # ST stage processes packets from stage_buffers[3] (ST input)
        # and sends them out.

        for out_port in range(self.num_ports):
            st_input_buffer = self.stage_buffers[3][out_port]
            if st_input_buffer:
                pkt = st_input_buffer[0]  # Peek at the head of the queue
                self.engine.logger.log(f"Router {self.name}: Cycle {self.engine.current_cycle} - ST - Processing packet {pkt.payload.get('id', pkt.event_type)} for out_port {out_port}")

                dest_mod, dest_port = self.output_links[out_port]

                # Update payload for the next hop
                pkt.payload["input_port"] = dest_port
                pkt.payload["input_vc"] = pkt.payload.get("out_vc", 0)
                
                # Prepare credit info for upstream router before updating for next hop
                prev_out_port = pkt.payload.get("prev_out_port")
                prev_out_vc = pkt.payload.get("prev_out_vc")

                # Send the packet
                new_event = Event(
                    src=self,
                    dst=dest_mod,
                    cycle=self.engine.current_cycle + 1,
                    data_size=pkt.data_size,
                    program=pkt.program,
                    event_type=pkt.event_type,
                    payload=pkt.payload,
                    priority=pkt.priority,
                )

                st_input_buffer.popleft()  # Packet leaves the router
                self.send_event(new_event)
                self.engine.logger.log(
                    f"Router {self.name}: Cycle {self.engine.current_cycle} - ST - Sent packet {pkt.payload.get('id', pkt.event_type)} to {dest_mod.name} via port {out_port}"
                )

                # Return credit to upstream router
                if prev_out_port is not None and prev_out_vc is not None:
                    up_module = pkt.payload["last_hop_src"]

                    # Always send credit back to the upstream module, regardless of its type
                    cred_evt = Event(
                        src=self,
                        dst=up_module,
                        cycle=self.engine.current_cycle + 1,
                        event_type="RECV_CRED",
                        payload={"prev_out_port": prev_out_port, "prev_out_vc": prev_out_vc},
                    )
                    self.engine.push_event(cred_evt)
                    self.engine.logger.log(
                        f"Router {self.name}: Cycle {self.engine.current_cycle} - ST - Returned credit for port {prev_out_port}, vc {prev_out_vc} to {up_module.name}"
                    )
                # Else: no previous port/vc info; nothing to return

                # Update for credit return from next hop
                pkt.payload["prev_out_port"] = out_port
                pkt.payload["prev_out_vc"] = pkt.payload.get("out_vc", 0)
                pkt.payload["last_hop_src"] = self
                self._release_slot()

    def handle_pipeline_output(self, data):
        # This method is called by PipelineModule when data exits the last stage (ST_out buffer)
        # For router, the actual packet sending is done in _stage_st, so this can be empty
        pass

# Constants for routing
DIRS = ["LOCAL", "E", "W", "N", "S"]
OPPOSITE = {"E": "W", "W": "E", "N": "S", "S": "N", "LOCAL": "LOCAL"}
