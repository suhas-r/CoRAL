import logging

from typing import List, Optional
from coral.breakpoint_graph import BreakpointGraph
import pyomo.environ as pyo
import pyomo.core
import pyomo.opt
import pyomo.util.infeasible

from coral.models import constraints

logger = logging.getLogger(__name__)


def get_objective(model: pyo.Model, bp_graph: BreakpointGraph, is_post: bool, is_greedy: bool, pc_list: List, unsatisfied_pc: List, pp: float) -> pyo.Objective:
    if not is_greedy:
      obj_value = 1.0 if not is_post else 2.0
      for i in range(k):
          obj_value += model.z[i]
          for seqi in range(lseg):
              obj_value -= model.x[seqi, i] * model.w[i] * bp_graph.sequence_edges[seqi][-2] / total_weights
      if is_post:
        obj_value -= sum(model.R[pi] / len(pc_list) for pi in range(len(pc_list)))
      return pyo.Objective(sense=pyo.minimize, expr=obj_value)
    
    # Greedy objective
    obj_value = 0.0
    for seqi in range(bp_graph.num_seq_edges):
        obj_value += model.x[seqi] * model.w[0] * bp_graph.sequence_edges[seqi][-2]
    for pi in range(len(pc_list)):
      if unsatisfied_pc[pi] >= 0:
        obj_value += model.r[pi] * max(pp, 1.0)
    return pyo.Objective(sense=pyo.maximize, expr=obj_value)


def get_minimize_cycle_model(bp_graph: BreakpointGraph, k: int, total_weights: float, node_order,
    pc_list,model_name: str,
    is_post: bool = False,
    is_greedy: bool = False,
    p_total_weight=0.9,
    resolution = 0.1, # Used only in post
    p_path_constraints: Optional[float] = None, # Used only in post
    p_bp_cn=0.9, # Used only in non-post
  ) -> pyo.ConcreteModel:
    lseg = len(bp_graph.sequence_edges)
    lc = len(bp_graph.concordant_edges)
    ld = len(bp_graph.discordant_edges)
    lsrc = len(bp_graph.source_edges)
    nnodes = len(bp_graph.nodes)
    nedges = lseg + lc + ld + 2 * lsrc + 2 * len(bp_graph.endnodes)
    endnode_list = [node for node in bp_graph.endnodes.keys()]
    logger.debug(f"Num nodes to be used in QP = {nnodes}")
    logger.debug(f"Num edges to be used in QP = {nedges}")

    model = pyo.ConcreteModel(name=f"{model_name}_{k}")

    #region Indices
    model.k = pyo.RangeSet(0, k - 1)
    model.edge_idx = pyo.RangeSet(0, nedges - 1)
    model.seq_edge_idx = pyo.RangeSet(0, len(bp_graph.sequence_edges) - 1)
    model.conc_edge_idx = pyo.RangeSet(0, len(bp_graph.concordant_edges) - 1)
    model.disc_edge_idx = pyo.RangeSet(0, len(bp_graph.discordant_edges) - 1)
    model.src_edge_idx = pyo.RangeSet(0, len(bp_graph.source_edges) - 1)
    model.node_idx = pyo.RangeSet(0, len(bp_graph.nodes) - 1)
    #endregion

    #region Variables
    # z[i]: indicating whether cycle or path i exists
    model.z = pyo.Var(model.k, domain=pyo.Binary)
    # w[i]: the weight of cycle or path i, continuous variable
    model.w = pyo.Var(model.k, domain=pyo.NonNegativeReals, bounds=(0.0, bp_graph.max_cn))
    # x: the number of times an edge occur in cycle or path i
    model.x = pyo.Var(model.edge_idx, model.k, domain=pyo.Integers, bounds=(0.0, 10.0))

    # Subpath constraints
    model.pc_idx = pyo.RangeSet(0, len(pc_list) - 1)
    model.r = pyo.Var(model.pc_idx, model.k, within=pyo.Binary)
    if is_post:
      model.R = pyo.Var(model.pc_idx, within=pyo.Binary)

    # c: decomposition i is a cycle, and start at particular node
    model.c = pyo.Var(model.node_idx, model.k, within=pyo.Binary)

    # d: BFS/spanning tree order of the nodes in decomposition i (A4.15-A4.17)
    model.d = pyo.Var(model.node_idx, model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))
    model.ds = pyo.Var(model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))
    model.dt = pyo.Var(model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))

    # y: spanning tree indicator (directed)
    model.y1 = pyo.Var(model.edge_idx, model.k, domain=pyo.Binary) # Positive direction
    model.y2 = pyo.Var(model.edge_idx, model.k, domain=pyo.Binary) # Negative direction
    #endregion

    # Relationship between w[i] and z[i]
    # Below constraint is shared by `minimize_cycles` and `minimize_cycles_post`
    model.ConstraintWZ = pyo.Constraint(model.k, rule=lambda model, i: model.w[i] <= model.z[i] * bp_graph.max_cn)
    if is_post or is_greedy:
       model.ConstraintWZResolution = pyo.Constraint(model.k, rule=lambda model, i: model.w[i] >= model.z[i] * resolution)
    
    # Objective: minimize the total number of cycles
    obj_value = 1.0 if not is_post else 2.0
    for i in range(k):
        obj_value += model.z[i]
        for seqi in range(lseg):
            obj_value -= model.x[seqi, i] * model.w[i] * bp_graph.sequence_edges[seqi][-2] / total_weights
    if is_post:
      obj_value -= sum(model.R[pi] / len(pc_list) for pi in range(len(pc_list)))
    model.Objective = get_objective(model, bp_graph, is_post, is_greedy, pc_list, unsatisfied_pc, pp)

    # Must include at least 0.9 * total CN weights (bilinear constraint)
    model.ConstraintTotalWeights = constraints.get_total_weight_constraint(model,k, bp_graph,  p_total_weight * total_weights)

    model.ConstraintEulerianPath = pyo.Constraint(model.k, rule=constraints.get_eulerian_path_constraint(endnode_list, bp_graph))
    model.ConstraintEulerianNodes = pyo.ConstraintList()
    for constraint in constraints.get_eulerian_node_constraints(model, k, endnode_list, lseg, lc, ld, lsrc, bp_graph):
        model.ConstraintEulerianNodes.add(constraint)

    constraints.set_copy_number_constraints(model, k, bp_graph, p_bp_cn)

    model.ConstraintBPOccurrence = constraints.get_bp_occurrence_constraint(model, bp_graph)
    model.ConstraintCycleWeight = constraints.get_cycle_weight_constraint(model, bp_graph, endnode_list)
    model.ConstraintSingularBPEdge = constraints.get_single_bp_edge_constraint(model, bp_graph, node_order)

    constraints.set_cycle_tree_constraints(model, bp_graph)

    # Relationship between y and z:
    model.ConstraintY1Z = pyo.Constraint(model.k, model.edge_idx, rule=lambda model, i, j: model.y1[j, i] <= model.z[i])
    model.ConstraintY2Z = pyo.Constraint(model.k, model.edge_idx, rule=lambda model, i, j: model.y2[j, i] <= model.z[i])

    # Relationship between x, y and d
    model.ConstraintXY = pyo.Constraint(
        model.k, model.edge_idx, rule=lambda model, i, j: model.y1[j, i] + model.y2[j, i] <= model.x[j, i]
    )
    model.ConstraintXY1D = pyo.ConstraintList()
    model.ConstraintXY2D = pyo.ConstraintList()
    for i in range(k):
        for di in range(ld):
            dedge = bp_graph.discordant_edges[di]
            if dedge[0] == dedge[3] and dedge[1] == dedge[4] and dedge[2] == dedge[5]:  # exclude self loops
                model.ConstraintXY1D.add(model.y1[(lseg + lc + di), i] == 0)
                model.ConstraintXY2D.add(model.y2[(lseg + lc + di), i] == 0)

    model.ConstraintsXYD = pyo.ConstraintList()
    for constraint in constraints.get_spanning_tree_constraints(model,k, bp_graph, endnode_list, node_order):
      model.ConstraintsXYD.add(constraint)
    for constraint in constraints.get_spanning_tree_constraints__non_endnode(model,k, bp_graph, endnode_list, node_order):
      model.ConstraintsXYD.add(constraint)

    model.ConstraintSubpathEdges = pyo.ConstraintList()
    for constraint in constraints.get_shared_subpath_edge_constraints(model, k, pc_list, bp_graph):
      model.ConstraintSubpathEdges.add(constraint)

    if pc_list:
      model.ConstraintSubpathEdgesAddtl = pyo.ConstraintList()
      for constraint in constraints.get_addtl_subpath_edge_constraints(model, k, pc_list, is_post, p_path_constraints):
        model.ConstraintSubpathEdgesAddtl.add(constraint)

    return model