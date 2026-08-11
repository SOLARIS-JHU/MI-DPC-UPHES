"""Build one-shot computational graph.

    build_oneshot_system() — Node chain (all-at-once, one-shot)
"""

from neuromancer.system import Node


# ===================================================================
# One-shot (Node chain) — single forward pass for all 24 hours
# ===================================================================

def build_oneshot_system(dynamics_batch, net_cont, net_int, ste_fn):
    """Construct a one-shot system as a list of Nodes.

    One-shot flow (single forward pass):
        [x(B,1,2), d(B,24,1)] -> pi_cont  -> u_c(B,24,2)
        [x(B,1,2), d(B,24,1)] -> pi_int   -> mode_logits(B,24,3)
        [u_c(B,24,2), mode_logits(B,24,3)] -> ste -> u(B,24,3)
        [x(B,1,2), u(B,24,3)] -> dynamics  -> x(B,25,2), aux(B,24,4)

    Returns a list of Nodes (used directly in Problem, not wrapped in System).
    """
    cont_node = Node(net_cont, ['x', 'd'], ['u_c'], name='pi_cont')
    int_node = Node(net_int, ['x', 'd'], ['mode_logits'], name='pi_int')
    round_node = Node(ste_fn, ['u_c', 'mode_logits'], ['u'], name='ste_round')
    dyn_node = Node(dynamics_batch, ['x', 'u'], ['x', 'aux'], name='dynamics')

    return [cont_node, int_node, round_node, dyn_node]
