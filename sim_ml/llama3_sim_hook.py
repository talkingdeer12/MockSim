from sim_core.event import Event

def linear_gemm_hook(cp, mesh_info):
    def hook(module, input, output):
        in_tensor = input[0]
        out_tensor = output
        M = in_tensor.shape[0]
        K = in_tensor.shape[1]
        N = out_tensor.shape[1]
        gemm_shape = (M, N, K)
        weights_size = K * N * 4
        act_size = M * K * 4
        event = Event(
            src=None,
            dst=cp,
            cycle=cp.engine.current_cycle + 1,
            program=f"Linear_GEMM_{module.sim_layer_idx}",
            event_type="GEMM",
            payload={
                "gemm_shape": gemm_shape,
                "weights_size": weights_size,
                "act_size": act_size,
            }
        )
        cp.send_event(event)
        print(f"[Hook] nn.Linear at layer {module.sim_layer_idx}: shape={gemm_shape}")
    return hook
