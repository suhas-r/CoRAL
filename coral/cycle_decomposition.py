"""Functions used for cycle decomposition"""

from __future__ import annotations
from dataclasses import dataclass, field

import pyomo
import pyomo.environ as pyo
import pyomo.core
import pyomo.opt
import pyomo.util.infeasible
import logging
import math
import os
import random
import time
from typing import Any, Dict, List, Set

import gurobipy as gp
from gurobipy import GRB
import pyomo.solvers
import pyomo.solvers.plugins
import pyomo.solvers.plugins.solvers
import pyomo.solvers.plugins.solvers.GUROBI

from coral import constants, infer_breakpoint_graph, state_provider
from coral.breakpoint_graph import BreakpointGraph
from coral.constants import CHR_TAG_TO_IDX
from coral.path_constraints import longest_path_dict

logger = logging.getLogger(__name__)


@dataclass
class ParsedLPSolution:
    total_weights_included: float = 0.0
    cycles: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    cycle_weights: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    path_constraints_satisfied: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    path_constraints_satisfied_set: Set[int] = field(default_factory=set)


def minimize_cycles(
    amplicon_id: int,
    bp_graph: BreakpointGraph,
    k: int,
    total_weights: float,
    node_order,
    pc_list,
    p_total_weight=0.9,
    p_bp_cn=0.9,
    num_threads=-1,
    time_limit=7200,
    model_prefix="",
):
    """Cycle decomposition by minimizing the number of cycles/paths

    amplicon_id: integer, amplicon ID
    g: breakpoint graph (object)
    k: integer, maximum mumber of cycles/paths allowed in cycle decomposition
    total_weights: float, total length-weighted CN in breakpoint graph g
    node_order: dict maps each node in the input breakpoint graph to a distinct integer, indicating a total order of the nodes in g
    pc_list: list of subpath constraints to be satisfied, each as a dict that maps an edge to its multiplicity
            *** note that all subpath constraints in this list are required to be satisfied ***
            *** otherwise will return infeasible ***
    p_total_weight: float between (0, 1), minimum proportion of length-weighted CN to be covered by the resulting cycles or paths,
                    default value is 0.9
    p_bp_cn: float float between (0, 1), minimum proportion of CN for each discordant edge to be covered by the resulting cycles or paths,
                    default value is 0.9
    num_threads: integer, number of working threads for gurobipy, by default it tries to use up all available cores
    time_limit: integer, maximum allowed running time, in seconds, default is 7200 (2 hour)
    model_prefix: output prefix for gurobi *.lp model

    Returns: (1) Status of gurobi optimization model (usually 2 - optimal; 3 - infeasible; 9 - suboptimal/reached time limit)
            (2) Total length weighted CN in resulting cycles/paths
            (3) Total num subpath constraints satisfied by resulting cycles/paths
            (4) List of cycles, each as a dict which maps an edge to its multiplicity in the cycle
            (5) List of the corresponding CN of the above cycles
            (6) Subpath constraints (indices) satisfied by each cycle
            (7) List of paths, each as a dict which maps an edge to its multiplicity in the path
            (8) List of the corresponding CN of the above paths
            (9) Subpath constraints (indices) satisfied by each path
    """
    logger.debug(f"Regular cycle decomposition with at most {k} cycles/paths allowed.")
    lseg = len(bp_graph.sequence_edges)
    lc = len(bp_graph.concordant_edges)
    ld = len(bp_graph.discordant_edges)
    lsrc = len(bp_graph.source_edges)
    nnodes = len(bp_graph.nodes)
    nedges = lseg + lc + ld + 2 * lsrc + 2 * len(bp_graph.endnodes)
    endnode_list = [node for node in bp_graph.endnodes.keys()]
    logger.debug(f"Num nodes to be used in QP = {nnodes}")
    logger.debug(f"Num edges to be used in QP = {nedges}")
    print(f"Running non-post with {k} cycles/paths, {pc_list}")

    model = pyo.ConcreteModel(name=f"{model_prefix}/amplicon_{amplicon_id}_cycle_decomposition_{k=}")
    # model.k = pyo.Param(initialize=k, within=pyo.NonNegativeIntegers)

    model.k = pyo.RangeSet(0, k - 1)
    model.edge_idx = pyo.RangeSet(0, nedges - 1)
    model.seq_edge_idx = pyo.RangeSet(0, len(bp_graph.sequence_edges) - 1)
    model.conc_edge_idx = pyo.RangeSet(0, len(bp_graph.concordant_edges) - 1)
    model.disc_edge_idx = pyo.RangeSet(0, len(bp_graph.discordant_edges) - 1)
    model.src_edge_idx = pyo.RangeSet(0, len(bp_graph.source_edges) - 1)
    model.node_idx = pyo.RangeSet(0, len(bp_graph.nodes) - 1)
    # model.K = pyo.RangeSet(0,k-1)
    # model.k = pyo.Param(model.K, domain=pyo.NonNegativeIntegers)

    # z[i]: indicating whether cycle or path i exists
    model.z = pyo.Var(model.k, domain=pyo.Binary)

    # w[i]: the weight of cycle or path i, continuous variable
    model.w = pyo.Var(model.k, domain=pyo.NonNegativeReals, bounds=(0.0, bp_graph.max_cn))

    # Relationship between w[i] and z[i]\
    model.ConstraintWZList = pyo.ConstraintList()
    for i in range(k):
        model.ConstraintWZList.add(model.w[i] <= model.z[i] * bp_graph.max_cn)
    # model.ConstraintWZ = pyo.Constraint(model.k, rule=lambda model, i: model.w[i] <= model.z[i] * bp_graph.max_cn)

    # x: the number of times an edge occur in cycle or path i
    model.x = pyo.Var(model.edge_idx, model.k, domain=pyo.Integers, bounds=(0.0, 10.0))

    # Objective: minimize the total number of cycles
    obj_value = 1.0
    for i in range(k):
        obj_value += model.z[i]
        for seqi in range(lseg):
            obj_value -= model.x[seqi, i] * model.w[i] * bp_graph.sequence_edges[seqi][-2] / total_weights
    model.ObjectiveMinCycles = pyo.Objective(sense=pyo.minimize, expr=obj_value)

    # Must include at least 0.9 * total CN weights (bilinear constraint)
    total_weights_expr = 0.0
    for i in range(k):
        for seqi in range(lseg):
            total_weights_expr += model.x[seqi, i] * model.w[i] * bp_graph.sequence_edges[seqi][-2]
    model.ConstraintTotalWeights = pyo.Constraint(expr=(total_weights_expr >= p_total_weight * total_weights))

    # Eulerian constraint
    model.ConstraintEulerianEnd = pyo.ConstraintList()
    model.ConstraintEulerian = pyo.ConstraintList()
    for node in bp_graph.nodes:
        if node in endnode_list:
            for i in range(k):
                edge_idx = (lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node))
                model.ConstraintEulerianEnd.add(model.x[edge_idx, i] + model.x[edge_idx + 1, i] == model.x[bp_graph.nodes[node][0][0], i])
        else:
            for i in range(k):
                ec_expr = 0.0
                for seqi in bp_graph.nodes[node][0]:
                    ec_expr += model.x[seqi, i]
                for ci in bp_graph.nodes[node][1]:
                    ec_expr -= model.x[(lseg + ci), i]
                for di in bp_graph.nodes[node][2]:
                    ec_expr -= model.x[(lseg + lc + di), i]
                for srci in bp_graph.nodes[node][3]:
                    ec_expr -= model.x[(lseg + lc + ld + 2 * srci), i]  # connected to s
                    ec_expr -= model.x[(lseg + lc + ld + 2 * srci) + 1, +i]  # connected to t
                model.ConstraintEulerian.add(ec_expr == 0.0)

    def constrain_eulerian_path(model: pyo.Model, i: int) -> pyo.Expression:
        path_expr = 0.0
        for enodei in range(len(endnode_list)):
            path_expr += model.x[(lseg + lc + ld + 2 * lsrc + 2 * enodei), i]  # (s, v)
            path_expr -= model.x[(lseg + lc + ld + 2 * lsrc + 2 * enodei) + 1, i]  # (v, t)
        for srci in range(lsrc):
            path_expr += model.x[(lseg + lc + ld + 2 * srci), i]  # (s, v)
            path_expr -= model.x[(lseg + lc + ld + 2 * srci) + 1, i]  # (v, t)
        return path_expr == 0.0

    model.ConstraintEulerianPath = pyo.Constraint(model.k, rule=constrain_eulerian_path)

    # CN constraint (all quadratic)
    model.ConstraintCNSequence = pyo.Constraint(
        model.seq_edge_idx,
        rule=lambda model, seqi: sum(model.w[i] * model.x[seqi, i] for i in range(k))
        <= bp_graph.sequence_edges[seqi][-1],
    )
    model.ConstraintCNConcordant = pyo.Constraint(
        model.conc_edge_idx,
        rule=lambda model, ci: sum(model.w[i] * model.x[(lseg + ci), i] for i in range(k))
        <= bp_graph.concordant_edges[ci][-1],
    )
    model.ConstraintCNDiscordant1 = pyo.Constraint(
        model.disc_edge_idx,
        rule=lambda model, di: sum(model.w[i] * model.x[(lseg + lc + di), i] for i in range(k))
        <= bp_graph.discordant_edges[di][-1],
    )
    model.ConstraintCNDiscordant2 = pyo.Constraint(
        model.disc_edge_idx,
        rule=lambda model, di: sum(model.w[i] * model.x[(lseg + lc + di), i] for i in range(k))
        >= p_bp_cn * bp_graph.discordant_edges[di][-1],
    )
    model.ConstraintCNSource = pyo.Constraint(
        model.src_edge_idx,
        rule=lambda model, srci: sum(
            model.w[i] * (model.x[(lseg + lc + ld + 2 * srci), i] + model.x[(lseg + lc + ld + 2 * srci) + 1, i])
            for i in range(k)
        )
        <= bp_graph.source_edges[srci][-1],
    )

    # Occurrence of breakpoints in each cycle/path
    discordant_multiplicities = bp_graph.infer_discordant_edge_multiplicities()

    def constraint_bp_occurence(model: pyo.Model, i: int, di: int) -> pyo.Expression:
        return model.x[(lseg + lc + di), i] <= discordant_multiplicities[di]

    model.ConstraintBPOccurrence = pyo.Constraint(model.k, model.disc_edge_idx, rule=constraint_bp_occurence)

    # c: decomposition i is a cycle, and start at particular node
    model.c = pyo.Var(model.node_idx, model.k, within=pyo.Binary)  # TODO: figure out way to name with nnode,i

    # Relationship between c and x
    def constrain_c_x(model: pyo.Model, i: int) -> pyo.Expression:
        cycle_expr = 0.0
        cycle_expr += sum(model.c[ni, i] for ni in range(nnodes))
        cycle_expr += sum(model.x[(lseg + lc + ld + 2 * lsrc + 2 * enodei), i] for enodei in range(len(endnode_list)))
        cycle_expr += sum(model.x[(lseg + lc + ld + 2 * srci), i] for srci in range(lsrc))
        return cycle_expr <= 1.0

    model.ConstraintCX = pyo.Constraint(model.k, rule=constrain_c_x)

    # There must be a concordant/discordant edge occuring one time
    def constrain_singular_bp_edge(model: pyo.Model, i: int) -> pyo.Expression:
        expr_xc = 0.0
        if not bp_graph.nodes:
            return pyo.Constraint.Skip  #  need to skip if no nodes to avoid trivial constraint error
        for node in bp_graph.nodes:
            for ci in set(bp_graph.nodes[node][1]):
                expr_xc += model.c[node_order[node], i] * model.x[(lseg + ci), i]
            for di in set(bp_graph.nodes[node][2]):
                expr_xc += model.c[node_order[node], i] * model.x[(lseg + lc + di), i]
        # Skip trivial constraints when all components have 0 coeff
        return expr_xc <= 1.0 if expr_xc else pyo.Constraint.Skip

    model.ConstraintSingularBPEdge = pyo.Constraint(model.k, rule=constrain_singular_bp_edge)

    # d: BFS/spanning tree order of the nodes in decomposition i
    model.d = pyo.Var(model.node_idx, model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))
    model.ds = pyo.Var(model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))
    model.dt = pyo.Var(model.k, within=pyo.NonNegativeIntegers, bounds=(0.0, nnodes + 2))

    # y: spanning tree indicator (directed)
    model.y1 = pyo.Var(model.edge_idx, model.k, domain=pyo.Binary)
    model.y2 = pyo.Var(model.edge_idx, model.k, domain=pyo.Binary)

    # Relationship between c and d
    model.ConstraintCD1 = pyo.Constraint(
        model.k, model.node_idx, rule=lambda model, i, ni: model.d[ni, i] >= model.c[ni, i]
    )
    model.ConstraintCD2 = pyo.Constraint(
        model.k, rule=lambda model, i: (sum(model.c[ni, i] for ni in range(nnodes)) + model.ds[i]) <= 1.0
    )

    # Relationship between y and z:
    model.ConstraintY1Z = pyo.Constraint(
        model.k, model.edge_idx, rule=lambda model, i, j: model.y1[j, i] <= model.z[i]
    )
    model.ConstraintY2Z = pyo.Constraint(
        model.k, model.edge_idx, rule=lambda model, i, j: model.y2[j, i] <= model.z[i]
    )

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
    model.ConstraintsXYD2 = pyo.ConstraintList()
    model.ConstraintsXYD3 = pyo.ConstraintList()
    model.ConstraintsXYD4 = pyo.ConstraintList()
    for i in range(k):
        t_expr_x = 0.0  # linear
        t_expr_y = 0.0  # linear
        t_expr_yd = 0.0  # quad
        for node in endnode_list:
            t_expr_x += model.x[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)) + 1, i]
            t_expr_y += model.y1[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)) + 1, i]
            t_expr_yd += model.y1[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)) + 1, i] * (
                model.dt[i] - model.d[node_order[node], i]
            )  # node -> t
            expr_x = 0.0  # linear
            expr_y = 0.0  # linear
            expr_xc = 0.0  # quad
            expr_yd = 0.0  # quad
            for seqi in bp_graph.nodes[node][0]:
                sseg = bp_graph.sequence_edges[seqi]
                node_ = (sseg[0], sseg[1], "-")
                if node_ == node:
                    node_ = (sseg[0], sseg[2], "+")
                expr_x += model.x[seqi, i]
                expr_xc += model.x[seqi, i] * model.c[node_order[node], i]
                if node_order[node_] <= node_order[node]:
                    expr_y += model.y1[seqi, i]
                    expr_yd += model.y1[seqi, i] * (model.d[node_order[node], i] - model.d[node_order[node_], i])
                else:
                    expr_y += model.y2[seqi, i]
                    expr_yd += model.y2[seqi, i] * (model.d[node_order[node], i] - model.d[node_order[node_], i])

            expr_x += model.x[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)), i]  # from s
            expr_xc += (
                model.x[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)), i] * model.c[node_order[node], i]
            )
            expr_x += model.x[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)) + 1, i]  # to t
            expr_xc += (
                model.x[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)) + 1, i]
                * model.c[node_order[node], i]
            )
            expr_y += model.y1[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)), i]  # from s
            expr_yd += model.y1[(lseg + lc + ld + 2 * lsrc + 2 * endnode_list.index(node)), i] * (
                model.d[node_order[node], i] - model.ds[i]
            )
            model.ConstraintsXYD.add(expr_x * (nnodes + 2) >= model.d[node_order[node], i])
            model.ConstraintsXYD.add(expr_y <= 1.0)
            model.ConstraintsXYD2.add(expr_y * nedges * k + expr_xc >= expr_x)
            model.ConstraintsXYD2.add(expr_yd * nedges * k + expr_xc >= expr_x)

        for srci in range(lsrc):
            srce = bp_graph.source_edges[srci]
            t_expr_x += model.x[(lseg + lc + ld + 2 * srci) + 1, i]
            t_expr_y += model.y1[(lseg + lc + ld + 2 * srci) + 1, i]
            t_expr_yd += model.y1[(lseg + lc + ld + 2 * srci) + 1, i] * (
                model.dt[i] - model.d[node_order[(srce[3], srce[4], srce[5])], i]
            )
        model.ConstraintsXYD3.add(t_expr_x * (nnodes + 2) >= model.dt[i])
        model.ConstraintsXYD3.add(t_expr_y <= 1.0)
        model.ConstraintsXYD4.add(t_expr_y * nedges * k >= t_expr_x)
        model.ConstraintsXYD4.add(t_expr_yd >= t_expr_x)

        for node in bp_graph.nodes.keys():
            if node not in endnode_list:
                expr_x = 0.0
                expr_y = 0.0
                expr_xc = 0.0
                expr_yd = 0.0
                for seqi in bp_graph.nodes[node][0]:
                    sseg = bp_graph.sequence_edges[seqi]
                    node_ = (sseg[0], sseg[1], "-")
                    if node_ == node:
                        node_ = (sseg[0], sseg[2], "+")
                    expr_x += model.x[seqi, i]
                    expr_xc += model.x[seqi, i] * model.c[node_order[node], i]
                    if node_order[node_] <= node_order[node]:
                        expr_y += model.y1[seqi, i]
                        expr_yd += model.y1[seqi, i] * (model.d[node_order[node], i] - model.d[node_order[node_], i])
                    else:
                        expr_y += model.y2[seqi, i]
                        expr_yd += model.y2[seqi, i] * (model.d[node_order[node], i] - model.d[node_order[node_], i])
                for ci in bp_graph.nodes[node][1]:
                    cedge = bp_graph.concordant_edges[ci]
                    node_ = (cedge[0], cedge[1], cedge[2])
                    if node_ == node:
                        node_ = (cedge[3], cedge[4], cedge[5])
                    expr_x += model.x[(lseg + ci), i]
                    expr_xc += model.x[(lseg + ci), i] * model.c[node_order[node], i]
                    if node_order[node_] <= node_order[node]:
                        expr_y += model.y1[(lseg + ci), i]
                        expr_yd += model.y1[(lseg + ci), i] * (
                            model.d[node_order[node], i] - model.d[node_order[node_], i]
                        )
                    else:
                        expr_y += model.y2[(lseg + ci), i]
                        expr_yd += model.y2[(lseg + ci), i] * (
                            model.d[node_order[node], i] - model.d[node_order[node_], i]
                        )
                for di in bp_graph.nodes[node][2]:
                    dedge = bp_graph.discordant_edges[di]
                    node_ = (dedge[0], dedge[1], dedge[2])
                    if node_ == node:
                        node_ = (dedge[3], dedge[4], dedge[5])
                    expr_x += model.x[(lseg + lc + di), i]
                    expr_xc += model.x[(lseg + lc + di), i] * model.c[node_order[node], i]
                    if node_order[node_] <= node_order[node]:
                        expr_y += model.y1[(lseg + lc + di), i]
                        expr_yd += model.y1[(lseg + lc + di), i] * (
                            model.d[node_order[node], i] - model.d[node_order[node_], i]
                        )
                    else:
                        expr_y += model.y2[(lseg + lc + di), i]
                        expr_yd += model.y2[(lseg + lc + di), i] * (
                            model.d[node_order[node], i] - model.d[node_order[node_], i]
                        )
                for srci in bp_graph.nodes[node][3]:
                    expr_x += model.x[(lseg + lc + ld + 2 * srci), i]
                    expr_x += model.x[(lseg + lc + ld + 2 * srci) + 1, i]
                    expr_xc += model.x[(lseg + lc + ld + 2 * srci), i] * model.c[node_order[node], i]
                    expr_xc += model.x[(lseg + lc + ld + 2 * srci) + 1, i] * model.c[node_order[node], i]
                    expr_y += model.y1[(lseg + lc + ld + 2 * srci), i]
                    expr_yd += model.y1[(lseg + lc + ld + 2 * srci), i] * (model.d[node_order[node], i] - model.ds[i])
                model.ConstraintsXYD.add(expr_x * (nnodes + 2) >= model.d[node_order[node], i])
                model.ConstraintsXYD.add(expr_y <= 1.0)
                model.ConstraintsXYD.add(expr_y * nedges * k + expr_xc >= expr_x)
                model.ConstraintsXYD.add(expr_yd * nedges * k + expr_xc >= expr_x)

    # Subpath constraints
    model.pc_idx = pyo.RangeSet(0, len(pc_list) - 1)
    model.r = pyo.Var(model.pc_idx, model.k, within=pyo.Binary)

    model.ConstraintSubpath1 = pyo.Constraint(
        model.pc_idx, rule=lambda model, pi: sum(model.r[pi, i] for i in range(k)) >= 1.0
    )
    model.ConstraintSubpathEdges = pyo.ConstraintList()
    # breakpoint()
    for pi, path_constraint_ in enumerate(pc_list):
        for edge in path_constraint_:
            for i in range(k):
                if edge[0] == "s":
                    model.ConstraintSubpathEdges.add(model.x[edge[1], i] >= model.r[pi, i] * path_constraint_[edge])
                elif edge[0] == "c":
                    model.ConstraintSubpathEdges.add(
                        model.x[(lseg + edge[1]), i] >= model.r[pi, i] * path_constraint_[edge]
                    )
                else:
                    model.ConstraintSubpathEdges.add(
                        model.x[(lseg + lc + edge[1]), i] >= model.r[pi, i] * path_constraint_[edge]
                    )

    model_name = f"{model_prefix}/amplicon_{amplicon_id}_model"
    solver: pyomo.solvers.plugins.solvers.GUROBI = pyo.SolverFactory("gurobi", solver_io="python")

    if num_threads > 0:
        solver.options["threads"] = num_threads
    solver.options["NonConvex"] = 2
    solver.options["timelimit"] = max(time_limit, ld * 300)  # each breakpoint edge is assigned 5 minutes
    solver.options["LogFile"] = f"{model_name}.log"

    model.write(f"{model_name}.lp", io_options={"symbolic_solver_labels": True})
    logger.debug(f"Completed model setup, wrote to {model_name}.lp.")

    results: pyomo.opt.SolverResults = solver.solve(model, tee=True)

    logger.debug(f"Completed optimization with status {results.solver.status}")
    if (results.solver.status == pyo.SolverStatus.ok) and (
        results.solver.termination_condition == pyo.TerminationCondition.optimal
    ):
        logger.debug("Found optimal solution.")
    elif results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        logger.debug("Model is infeasible.")
        pyomo.util.infeasible.log_infeasible_constraints(model, log_expression=True, log_variables=True)

    # model.display()
    # sol_z = solver.getAttr("X", model.z)
    # sol_w = solver.getAttr("X", model.w)
    # sol_d = solver.getAttr("X", model.d)
    # sol_r = solver.getAttr("X", model.r)
    parsed_sol = parse_lp_solution(model, bp_graph, k, pc_list)
    logger.debug(f"Total length weighted CN from cycles/paths = {parsed_sol.total_weights_included}/{total_weights}.")
    logger.debug(
        f"Total num subpath constraints satisfied = {len(parsed_sol.path_constraints_satisfied_set)}/{len(pc_list)}."
    )
    return (
        results.solver.status,
        parsed_sol.total_weights_included,  # TODO: just return parsed sol
        len(parsed_sol.path_constraints_satisfied_set),
        parsed_sol.cycles,
        parsed_sol.cycle_weights,
        parsed_sol.path_constraints_satisfied,
    )


def cycle_decomposition(
    bb: infer_breakpoint_graph.BamToBreakpointNanopore,
    alpha: float = 0.01,
    p_total_weight: float = 0.9,
    resolution: float = 0.1,
    num_threads: int = -1,
    postprocess: int = 0,
    time_limit: int = 7200,
    model_prefix: str = "",
):
    """Caller for cycle decomposition functions"""
    for amplicon_idx in range(len(bb.lr_graph)):
        bp_graph = bb.lr_graph[amplicon_idx]
        lseg = len(bp_graph.sequence_edges)
        lc = len(bp_graph.concordant_edges)
        ld = len(bp_graph.discordant_edges)
        lsrc = len(bp_graph.source_edges)

        total_weights = 0.0
        for sseg in bp_graph.sequence_edges:
            total_weights += sseg[7] * sseg[-1]  # type: ignore[operator]
        logger.info(f"Begin cycle decomposition for amplicon{amplicon_idx +1}.")
        logger.info(f"Total CN weights = {total_weights}.")

        bb.longest_path_constraints[amplicon_idx] = longest_path_dict(
            bb.path_constraints[amplicon_idx],
        )
        logger.info(f"Total num maximal subpath constraints = {len(bb.longest_path_constraints[amplicon_idx][0])}.")
        try:
            for pathi in bb.longest_path_constraints[amplicon_idx][1]:
                logger.debug(f"Subpath constraint {pathi} = {bb.longest_path_constraints[amplicon_idx][0][pathi]}")
        except:
            breakpoint()

        k = max(10, ld // 2)  # Initial num cycles/paths
        logger.info(f"Initial num cycles/paths = {k}.")
        nnodes = len(bp_graph.nodes)  # Does not include s and t
        nedges = lseg + lc + ld + 2 * lsrc + 2 * len(bp_graph.endnodes)
        node_order = {}
        ni_ = 0
        for node in bb.lr_graph[amplicon_idx].nodes.keys():
            node_order[node] = ni_
            ni_ += 1
        if nedges < k:
            k = nedges
            logger.info(f"Reset num cycles/paths to {k}.")
        sol_flag = 0
        while k <= nedges:
            if (
                nedges > 100
                or (3 * k + 3 * k * nedges + 2 * k * nnodes + k * len(bb.longest_path_constraints[amplicon_idx][0]))
                >= 10000
            ):
                (
                    total_cycle_weights_init,
                    total_path_satisfied_init,
                    cycles_init,
                    cycle_weights_init,
                    path_constraints_satisfied_init,
                ) = maximize_weights_greedy(
                    amplicon_idx + 1,
                    bb.lr_graph[amplicon_idx],
                    total_weights,
                    node_order,
                    bb.longest_path_constraints[amplicon_idx][0],
                    alpha,
                    p_total_weight,
                    resolution,
                    0.005,
                    0.9,
                    num_threads,
                    postprocess,
                    time_limit,
                    model_prefix,
                )
                logger.info("Completed greedy cycle decomposition.")
                logger.info(f"Num cycles = {len(cycles_init[0])}; num paths = {len(cycles_init[1])}.")
                logger.info(f"Total length weighted CN = {total_cycle_weights_init}/{total_weights}.")
                logger.info(
                    f"Total num subpath constraints satisfied = {total_path_satisfied_init}/{len(bb.longest_path_constraints[amplicon_idx][0])}."
                )
                if postprocess == 1:
                    (
                        status_post,
                        total_cycle_weights_post,
                        total_path_satisfied_post,
                        cycles_post,
                        cycle_weights_post,
                        path_constraints_satisfied_post,
                    ) = minimize_cycles_post(
                        amplicon_idx + 1,
                        bb.lr_graph[amplicon_idx],
                        total_weights,
                        node_order,
                        bb.longest_path_constraints[amplicon_idx][0],
                        [cycles_init, cycle_weights_init, path_constraints_satisfied_init],
                        min(total_cycle_weights_init / total_weights * 0.9999, p_total_weight),
                        resolution,
                        num_threads,
                        time_limit,
                        model_prefix,
                    )
                    logger.info("Completed postprocessing of the greedy solution.")
                    logger.info(f"Num cycles = {len(cycles_post[0])}; num paths = {len(cycles_post[1])}.")
                    logger.info(f"Total length weighted CN = {total_cycle_weights_post}/{total_weights}.")
                    logger.info(
                        f"Total num subpath constraints satisfied = {total_path_satisfied_post}/{len(bb.longest_path_constraints[amplicon_idx][0])}."
                    )
                    bb.cycles[amplicon_idx] = cycles_post
                    bb.cycle_weights[amplicon_idx] = cycle_weights_post
                    bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_post
                else:
                    print(f"{cycles_init=}")
                    bb.cycles[amplicon_idx] = cycles_init
                    bb.cycle_weights[amplicon_idx] = cycle_weights_init
                    bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_init
                sol_flag = 1
                break
            (
                status_,
                total_cycle_weights_,
                total_path_satisfied_,
                cycles_,
                cycle_weights_,
                path_constraints_satisfied_,
            ) = minimize_cycles(
                amplicon_idx + 1,
                bb.lr_graph[amplicon_idx],
                k,
                total_weights,
                node_order,
                bb.longest_path_constraints[amplicon_idx][0],
                p_total_weight,
                0.9,
                num_threads,
                time_limit,
                model_prefix,
            )
            if status_ == GRB.INFEASIBLE:
                logger.info("Cycle decomposition is infeasible.")
                logger.info(f"Doubling k from {k} to {k * 2}.")
                k *= 2
            else:
                logger.info(f"Completed cycle decomposition with k = {k}.")
                logger.info(f"Num cycles = {len(cycles_[0])}; num paths = {len(cycles_[1])}.")
                logger.info(f"Total length weighted CN = {total_cycle_weights_}/{total_weights}.")
                logger.info(
                    f"Total num subpath constraints satisfied = {total_path_satisfied_}/{len(bb.longest_path_constraints[amplicon_idx][0])}."
                )

                print(f"{cycles_=}")
                bb.cycles[amplicon_idx] = cycles_
                bb.cycle_weights[amplicon_idx] = cycle_weights_
                bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_
                sol_flag = 1
                break
        if sol_flag == 0:
            logger.info("Cycle decomposition is infeasible, switch to greedy cycle decomposition.")
            (
                total_cycle_weights_init,
                total_path_satisfied_init,
                cycles_init,
                cycle_weights_init,
                path_constraints_satisfied_init,
            ) = maximize_weights_greedy(
                amplicon_idx + 1,
                bb.lr_graph[amplicon_idx],
                total_weights,
                node_order,
                bb.longest_path_constraints[amplicon_idx][0],
                alpha,
                p_total_weight,
                resolution,
                0.005,
                0.9,
                num_threads,
                postprocess,
                time_limit,
                model_prefix,
            )
            logger.info(
                "Completed greedy cycle decomposition.",
            )
            logger.info(
                "\tNum cycles = %d; num paths = %d." % (len(cycles_init[0]), len(cycles_init[1])),
            )
            logger.info(
                "\tTotal length weighted CN = %f/%f." % (total_cycle_weights_init, total_weights),
            )
            logger.info(
                "\tTotal num subpath constraints satisfied = %d/%d."
                % (total_path_satisfied_init, len(bb.longest_path_constraints[amplicon_idx][0])),
            )
            if postprocess == 1:
                (
                    status_post,
                    total_cycle_weights_post,
                    total_path_satisfied_post,
                    cycles_post,
                    cycle_weights_post,
                    path_constraints_satisfied_post,
                ) = minimize_cycles_post(
                    amplicon_idx + 1,
                    bb.lr_graph[amplicon_idx],
                    total_weights,
                    node_order,
                    bb.longest_path_constraints[amplicon_idx][0],
                    [cycles_init, cycle_weights_init, path_constraints_satisfied_init],
                    min(total_cycle_weights_init / total_weights * 0.9999, p_total_weight),
                    resolution,
                    num_threads,
                    time_limit,
                    model_prefix,
                )
                logger.info("Completed postprocessing of the greedy solution.")
                logger.info(
                    "Num cycles = %d; num paths = %d." % (len(cycles_post[0]), len(cycles_post[1])),
                )
                logger.info(
                    "Total length weighted CN = %f/%f." % (total_cycle_weights_post, total_weights),
                )
                logger.info(
                    "Total num subpath constraints satisfied = %d/%d."
                    % (
                        total_path_satisfied_post,
                        len(bb.longest_path_constraints[amplicon_idx][0]),
                    ),
                )
                bb.cycles[amplicon_idx] = cycles_post
                bb.cycle_weights[amplicon_idx] = cycle_weights_post
                bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_post
            else:
                bb.cycles[amplicon_idx] = cycles_init
                bb.cycle_weights[amplicon_idx] = cycle_weights_init
                bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_init


def eulerian_cycle_t(g, edges_next_cycle, path_constraints_next_cycle, path_constraints_support):
    """Return an eulerian traversal of a cycle, represented by a dict of edges

    g: breakpoint graph (object)
    edges_next_cycle: subgraph induced by the cycle, as a dict that maps an edge to its multiplicity
    path_constraints_next_cycle: list of subpath constraints to be satisfied,
            each as a list of alternating nodes and edges
            ***
            Note: the traversal may not satisfy all subpath constraints
            in case not all subpath constraints are satisfied, return the eulerian traversal satisfying the
            maximum number of subpath constraints
            ***
    path_constraints_support: num long reads supporting each subpath constraint
    """
    lseg = len(g.sequence_edges)

    eulerian_cycle = []  # A cycle is edge - node list starting and ending with the same edge
    # Since Eulerian, there could be subcycles in the middle of a cycle
    eulerian_cycle_ = []  # Cycle in AA cycle format
    best_cycle = []  # Cycle in AA cycle format
    valid = 0
    num_trials = 0
    l = len(path_constraints_next_cycle)
    unsatisfied_path_metric = [range(l), 100 * l, 100 * max(path_constraints_support + [0])]
    while valid <= 0 and num_trials < 1000:
        valid = 1
        num_trials += 1
        eulerian_cycle = []
        eulerian_cycle_ = []
        edges_cur = edges_next_cycle.copy()
        last_seq_edge = lseg  # Start with the edge with smallest index and on the positive strand
        for edge in edges_cur.keys():
            if edge[0] == "e":
                last_seq_edge = min(last_seq_edge, edge[1])
        last_edge_dir = "+"
        eulerian_cycle.append(("s", last_seq_edge))
        eulerian_cycle_.append(str(last_seq_edge + 1) + "+")
        while len(edges_cur) > 0:
            seq_edge = g.sequence_edges[last_seq_edge]
            node = (seq_edge[0], seq_edge[2], "+")
            if last_edge_dir == "-":
                node = (seq_edge[0], seq_edge[1], "-")
            eulerian_cycle.append(node)
            next_bp_edges = []  # Since cycle, only consider discordant edges and concordant edges
            for ci in g.nodes[node][1]:
                next_bp_edges.append(("c", ci))
            for di in g.nodes[node][2]:
                next_bp_edges.append(("d", di))
            del_list = [i for i in range(len(next_bp_edges)) if next_bp_edges[i] not in edges_cur]
            for i in del_list[::-1]:
                del next_bp_edges[i]
            if len(next_bp_edges) == 0:
                valid = 0
                break
            if len(next_bp_edges) == 1:  # No branching on the path
                eulerian_cycle.append(next_bp_edges[0])
                edges_cur[next_bp_edges[0]] = int(edges_cur[next_bp_edges[0]]) - 1
                if edges_cur[next_bp_edges[0]] == 0:
                    del edges_cur[next_bp_edges[0]]
                bp_edge = []
                if next_bp_edges[0][0] == "c":
                    bp_edge = g.concordant_edges[next_bp_edges[0][1]][:6]
                else:
                    bp_edge = g.discordant_edges[next_bp_edges[0][1]][:6]
                node_ = (bp_edge[0], bp_edge[1], bp_edge[2])
                if node == (bp_edge[0], bp_edge[1], bp_edge[2]):
                    node_ = (bp_edge[3], bp_edge[4], bp_edge[5])
                eulerian_cycle.append(node_)
                last_seq_edge = g.nodes[node_][0][0]
                eulerian_cycle.append(("s", last_seq_edge))
                if node_[2] == "-":
                    last_edge_dir = "+"
                    eulerian_cycle_.append(str(last_seq_edge + 1) + "+")
                else:
                    last_edge_dir = "-"
                    eulerian_cycle_.append(str(last_seq_edge + 1) + "-")
                edges_cur[("e", last_seq_edge)] = int(edges_cur[("e", last_seq_edge)]) - 1
                if edges_cur[("e", last_seq_edge)] == 0:
                    del edges_cur[("e", last_seq_edge)]
            else:
                r = random.randint(0, len(next_bp_edges) - 1)
                eulerian_cycle.append(next_bp_edges[r])
                edges_cur[next_bp_edges[r]] = int(edges_cur[next_bp_edges[r]]) - 1
                if edges_cur[next_bp_edges[r]] == 0:
                    del edges_cur[next_bp_edges[r]]
                bp_edge = []
                if next_bp_edges[r][0] == "c":
                    bp_edge = g.concordant_edges[next_bp_edges[r][1]][:6]
                else:
                    bp_edge = g.discordant_edges[next_bp_edges[r][1]][:6]
                node_ = (bp_edge[0], bp_edge[1], bp_edge[2])
                if node == (bp_edge[0], bp_edge[1], bp_edge[2]):
                    node_ = (bp_edge[3], bp_edge[4], bp_edge[5])
                eulerian_cycle.append(node_)
                last_seq_edge = g.nodes[node_][0][0]
                eulerian_cycle.append(("s", last_seq_edge))
                if node_[2] == "-":
                    last_edge_dir = "+"
                    eulerian_cycle_.append(str(last_seq_edge + 1) + "+")
                else:
                    last_edge_dir = "-"
                    eulerian_cycle_.append(str(last_seq_edge + 1) + "-")
                edges_cur[("e", last_seq_edge)] = int(edges_cur[("e", last_seq_edge)]) - 1
                if edges_cur[("e", last_seq_edge)] == 0:
                    del edges_cur[("e", last_seq_edge)]
        if valid == 1 and len(best_cycle) == 0:
            best_cycle = eulerian_cycle_
        path_metric = [[], 0, 0]
        # check if the remaining path constraints are satisfied
        for pathi in range(len(path_constraints_next_cycle)):
            path_ = path_constraints_next_cycle[pathi]
            path0 = path_[0]
            s = 0
            for ei in range(len(eulerian_cycle) - 1):
                obj = eulerian_cycle[ei]
                if obj == path0:
                    s_ = 1
                    for i in range(len(path_)):
                        if eulerian_cycle[:-1][(ei + i) % (len(eulerian_cycle) - 1)] != path_[i]:
                            s_ = 0
                            break
                    if s_ == 1:
                        s = 1
                        break
                    s_ = 1
                    for i in range(len(path_)):
                        if eulerian_cycle[:-1][ei - i] != path_[i]:
                            s_ = 0
                            break
                    if s_ == 1:
                        s = 1
                        break
            if s == 0 and valid == 1:
                path_metric[0].append(pathi)
                path_metric[1] += len(path_)
                path_metric[2] += path_constraints_support[pathi]
        if valid == 1 and len(path_metric[0]) > 0:
            valid = -1
        if (
            valid != 0
            and (len(path_metric[0]) < len(unsatisfied_path_metric[0]))
            or (len(path_metric[0]) == len(unsatisfied_path_metric[0]) and path_metric[1] < unsatisfied_path_metric[1])
            or (
                len(path_metric[0]) == len(unsatisfied_path_metric[0])
                and path_metric[1] == unsatisfied_path_metric[1]
                and path_metric[2] < unsatisfied_path_metric[2]
            )
        ):
            unsatisfied_path_metric[0] = path_metric[0]
            unsatisfied_path_metric[1] = path_metric[1]
            unsatisfied_path_metric[2] = path_metric[2]
            best_cycle = eulerian_cycle_
    if len(unsatisfied_path_metric[0]) == 0:
        logger.debug(
            "Cycle satisfies all subpath constraints.",
        )
    else:
        logger.debug("The following path constraints are not satisfied:")
        for pathi in unsatisfied_path_metric[0]:
            logger.debug(f"{path_constraints_next_cycle[pathi]}")
    return best_cycle


def eulerian_path_t(g: BreakpointGraph, edges_next_path, path_constraints_next_path, path_constraints_support):
    """Return an eulerian traversal of an s-t walk, represented by a dict of edges

    g: breakpoint graph (object)
    edges_next_path: subgraph induced by the s-t walk, as a dict that maps an edge to its multiplicity
            ***
            must include s and t in the dict
            ***
    path_constraints_next_path: list of subpath constraints to be satisfied,
            each as a list of alternating nodes and edges
            ***
            Note: the traversal may not satisfy all subpath constraints
            in case not all subpath constraints are satisfied, return the eulerian traversal satisfying the
            maximum number of subpath constraints
            ***
    path_constraints_support: num long reads supporting each subpath constraint
    """
    lseg = len(g.sequence_edges)
    endnode_list = [node for node in g.endnodes.keys()]

    eulerian_path = []  # A path is edge - node list starting and ending with edges
    # Since Eulerian, there could be subcycles in the middle of a path
    eulerian_path_ = []  # Path in AA cycle format
    best_path = []  # Path in AA cycle format
    valid = 0
    num_trials = 0
    l = len(path_constraints_next_path)
    unsatisfied_path_metric = [range(l), 100 * l, 100 * max(path_constraints_support + [0])]
    while valid <= 0 and num_trials < 1000:
        valid = 1
        num_trials += 1
        eulerian_path = []
        eulerian_path_ = []
        edges_cur = edges_next_path.copy()
        src_edge = ()
        last_seq_edge = lseg
        last_edge_dir = "+"
        for edge in edges_cur.keys():  # Start with the edge with smallest index
            if edge[0] == "s" or edge[0] == "t":
                src_edge = edge
                node = (
                    g.source_edges[edge[1]][3],
                    g.source_edges[edge[1]][4],
                    g.source_edges[edge[1]][5],
                )
                if len(eulerian_path) == 0:
                    last_edge_dir = constants.INVERT_STRAND_DIRECTION[node[2]]
                    eulerian_path.append(("$", -1))
                    eulerian_path.append(node)
                    last_seq_edge = g.nodes[node][0][0]
                elif g.nodes[node][0][0] < last_seq_edge:
                    last_edge_dir = constants.INVERT_STRAND_DIRECTION[node[2]]
                    eulerian_path[-1] = node
                    last_seq_edge = g.nodes[node][0][0]
            elif edge[0] == "ns" or edge[0] == "nt":
                src_edge = edge
                node = endnode_list[edge[1]]
                if len(eulerian_path) == 0:
                    last_edge_dir = constants.INVERT_STRAND_DIRECTION[node[2]]
                    eulerian_path.append(("$", -1))
                    eulerian_path.append(node)
                    last_seq_edge = g.nodes[node][0][0]
                elif g.nodes[node][0][0] < last_seq_edge:
                    last_edge_dir = constants.INVERT_STRAND_DIRECTION[node[2]]
                    eulerian_path[-1] = node
                    last_seq_edge = g.nodes[node][0][0]
        del edges_cur[src_edge]
        eulerian_path.append(("s", last_seq_edge))
        if last_edge_dir == "+":
            eulerian_path_.append(str(last_seq_edge + 1) + "+")
        else:
            eulerian_path_.append(str(last_seq_edge + 1) + "-")
        edges_cur[("e", last_seq_edge)] = int(edges_cur[("e", last_seq_edge)]) - 1
        if edges_cur[("e", last_seq_edge)] == 0:
            del edges_cur[("e", last_seq_edge)]
        while len(edges_cur) > 0:
            seq_edge = g.sequence_edges[last_seq_edge]
            node = (seq_edge[0], seq_edge[2], "+")
            if last_edge_dir == "-":
                node = (seq_edge[0], seq_edge[1], "-")
            eulerian_path.append(node)
            if len(edges_cur) == 1 and (
                list(edges_cur.keys())[0][0] == "s"
                or list(edges_cur.keys())[0][0] == "ns"
                or list(edges_cur.keys())[0][0] == "t"
                or list(edges_cur.keys())[0][0] == "nt"
            ):
                eulerian_path.append(("$", -1))
                break
            next_bp_edges = []  # Since cycle, only consider discordant edges and concordant edges
            for ci in g.nodes[node][1]:
                next_bp_edges.append(("c", ci))
            for di in g.nodes[node][2]:
                next_bp_edges.append(("d", di))
            del_list = [i for i in range(len(next_bp_edges)) if next_bp_edges[i] not in edges_cur]
            for i in del_list[::-1]:
                del next_bp_edges[i]
            if len(next_bp_edges) == 0:
                valid = 0
                break
            if len(next_bp_edges) == 1:  # No branching on the path
                eulerian_path.append(next_bp_edges[0])
                edges_cur[next_bp_edges[0]] = int(edges_cur[next_bp_edges[0]]) - 1
                if edges_cur[next_bp_edges[0]] == 0:
                    del edges_cur[next_bp_edges[0]]
                bp_edge = []
                if next_bp_edges[0][0] == "c":
                    bp_edge = g.concordant_edges[next_bp_edges[0][1]][:6]
                else:
                    bp_edge = g.discordant_edges[next_bp_edges[0][1]][:6]
                node_ = (bp_edge[0], bp_edge[1], bp_edge[2])
                if node == (bp_edge[0], bp_edge[1], bp_edge[2]):
                    node_ = (bp_edge[3], bp_edge[4], bp_edge[5])
                eulerian_path.append(node_)
                last_seq_edge = g.nodes[node_][0][0]
                eulerian_path.append(("s", last_seq_edge))
                if node_[2] == "-":
                    last_edge_dir = "+"
                    eulerian_path_.append(str(last_seq_edge + 1) + "+")
                else:
                    last_edge_dir = "-"
                    eulerian_path_.append(str(last_seq_edge + 1) + "-")
                edges_cur[("e", last_seq_edge)] = int(edges_cur[("e", last_seq_edge)]) - 1
                if edges_cur[("e", last_seq_edge)] == 0:
                    del edges_cur[("e", last_seq_edge)]
            else:
                r = random.randint(0, len(next_bp_edges) - 1)
                eulerian_path.append(next_bp_edges[r])
                edges_cur[next_bp_edges[r]] = int(edges_cur[next_bp_edges[r]]) - 1
                if edges_cur[next_bp_edges[r]] == 0:
                    del edges_cur[next_bp_edges[r]]
                bp_edge = []
                if next_bp_edges[r][0] == "c":
                    bp_edge = g.concordant_edges[next_bp_edges[r][1]][:6]
                else:
                    bp_edge = g.discordant_edges[next_bp_edges[r][1]][:6]
                node_ = (bp_edge[0], bp_edge[1], bp_edge[2])
                if node == (bp_edge[0], bp_edge[1], bp_edge[2]):
                    node_ = (bp_edge[3], bp_edge[4], bp_edge[5])
                eulerian_path.append(node_)
                last_seq_edge = g.nodes[node_][0][0]
                eulerian_path.append(("s", last_seq_edge))
                if node_[2] == "-":
                    last_edge_dir = "+"
                    eulerian_path_.append(str(last_seq_edge + 1) + "+")
                else:
                    last_edge_dir = "-"
                    eulerian_path_.append(str(last_seq_edge + 1) + "-")
                edges_cur[("e", last_seq_edge)] = int(edges_cur[("e", last_seq_edge)]) - 1
                if edges_cur[("e", last_seq_edge)] == 0:
                    del edges_cur[("e", last_seq_edge)]
        if valid == 1 and len(best_path) == 0:
            best_path = eulerian_path_
        path_metric = [[], 0, 0]
        # check if the remaining path constraints are satisfied
        for pathi in range(len(path_constraints_next_path)):
            path_ = path_constraints_next_path[pathi]
            s = 0
            for ei in range(2, len(eulerian_path) - 1 - len(path_)):
                if (
                    eulerian_path[ei : ei + len(path_)] == path_[:]
                    or eulerian_path[ei : ei + len(path_)] == path_[::-1]
                ):
                    s = 1
                    break
            if s == 0 and valid == 1:
                path_metric[0].append(pathi)
                path_metric[1] += len(path_)
                path_metric[2] += path_constraints_support[pathi]
        if valid == 1 and len(path_metric[0]) > 0:
            valid = -1
        if (
            valid != 0
            and (len(path_metric[0]) < len(unsatisfied_path_metric[0]))
            or (len(path_metric[0]) == len(unsatisfied_path_metric[0]) and path_metric[1] < unsatisfied_path_metric[1])
            or (
                len(path_metric[0]) == len(unsatisfied_path_metric[0])
                and path_metric[1] == unsatisfied_path_metric[1]
                and path_metric[2] < unsatisfied_path_metric[2]
            )
        ):
            unsatisfied_path_metric[0] = path_metric[0]
            unsatisfied_path_metric[1] = path_metric[1]
            unsatisfied_path_metric[2] = path_metric[2]
            best_path = eulerian_path_
    if len(unsatisfied_path_metric[0]) == 0:
        logger.debug("Path satisfies all subpath constraints.")
    else:
        logger.debug("The following path constraints are not satisfied:")
        for pathi in unsatisfied_path_metric[0]:
            logger.debug(f"{path_constraints_next_path[pathi]}")
    return best_path


def output_cycles(
    bb: infer_breakpoint_graph.BamToBreakpointNanopore, cycle_file_prefix: str, output_all_paths=False
) -> None:
    """Write the result from cycle decomposition into *.cycles files"""
    for amplicon_idx in range(len(bb.lr_graph)):
        logger.info(f"Output cycles for amplicon {amplicon_idx+1}.")
        cycle_path = f"{cycle_file_prefix}/amplicon{amplicon_idx + 1}_cycles.txt"
        fp = open(cycle_path, "w")
        interval_num = 1
        ai_amplicon = [ai for ai in bb.amplicon_intervals if bb.ccid2id[ai[3]] == amplicon_idx + 1]
        ai_amplicon = sorted(ai_amplicon, key=lambda ai: (CHR_TAG_TO_IDX[ai[0]], ai[1]))
        for ai in ai_amplicon:
            fp.write(f"Interval\t{interval_num}\t{ai[0]}\t{ai[1]}\t{ai[2]}\n")
            interval_num += 1
        fp.write("List of cycle segments\n")
        for seqi in range(len(bb.lr_graph[amplicon_idx].sequence_edges)):
            sseg = bb.lr_graph[amplicon_idx].sequence_edges[seqi]
            fp.write(f"Segment\t{seqi + 1}\t{sseg[0]}\t{sseg[1]}\t{sseg[2]}\n")
        if output_all_paths:
            fp.write("List of all subpath constraints\n")
            for pathi in range(len(bb.path_constraints[amplicon_idx][0])):
                fp.write("Path constraint\t%d\t" % (pathi + 1))
                path_ = bb.path_constraints[amplicon_idx][0][pathi]
                if path_[0][1] > path_[-1][1]:
                    path_ = path_[::-1]
                for i in range(len(path_)):
                    if i % 4 == 0:
                        if i < len(path_) - 1:
                            if path_[i + 1][2] == "+":
                                fp.write("%d+," % (path_[i][1] + 1))
                            else:
                                fp.write("%d-," % (path_[i][1] + 1))
                        elif path_[i - 1][2] == "+":
                            fp.write("%d-\t" % (path_[i][1] + 1))
                        else:
                            fp.write("%d+\t" % (path_[i][1] + 1))
                fp.write("Support=%d\n" % (bb.path_constraints[amplicon_idx][1][pathi]))
        else:
            fp.write("List of longest subpath constraints\n")
            path_constraint_indices_ = []
            for paths in (
                bb.path_constraints_satisfied[amplicon_idx][0] + bb.path_constraints_satisfied[amplicon_idx][1]
            ):
                for pathi in paths:
                    if pathi not in path_constraint_indices_:
                        path_constraint_indices_.append(pathi)
            for constraint_i in range(len(bb.longest_path_constraints[amplicon_idx][1])):
                fp.write("Path constraint\t%d\t" % (constraint_i + 1))
                pathi = bb.longest_path_constraints[amplicon_idx][1][constraint_i]
                path_ = bb.path_constraints[amplicon_idx][0][pathi]
                if path_[0][1] > path_[-1][1]:
                    path_ = path_[::-1]
                for i in range(len(path_)):
                    if i % 4 == 0:
                        if i < len(path_) - 1:
                            if path_[i + 1][2] == "+":
                                fp.write("%d+," % (path_[i][1] + 1))
                            else:
                                fp.write("%d-," % (path_[i][1] + 1))
                        elif path_[i - 1][2] == "+":
                            fp.write("%d-\t" % (path_[i][1] + 1))
                        else:
                            fp.write("%d+\t" % (path_[i][1] + 1))
                fp.write(
                    "Support=%d\t" % (bb.longest_path_constraints[amplicon_idx][2][constraint_i]),
                )
                if constraint_i in path_constraint_indices_:
                    fp.write("Satisfied\n")
                else:
                    fp.write("Unsatisfied\n")

        # sort cycles according to weights
        cycle_indices = sorted(
            [(0, i) for i in range(len(bb.cycle_weights[amplicon_idx][0]))]
            + [(1, i) for i in range(len(bb.cycle_weights[amplicon_idx][1]))],
            key=lambda item: bb.cycle_weights[amplicon_idx][item[0]][item[1]],
            reverse=True,
        )

        print(cycle_indices)
        for cycle_i in cycle_indices:
            cycle_edge_list: list = []
            if cycle_i[0] == 0:  # cycles
                logger.debug(f"Traversing next cycle, CN = {bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]]}")
                path_constraints_satisfied_cycle = []
                path_constraints_support_cycle = []
                for pathi in bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]:
                    pathi_ = bb.longest_path_constraints[amplicon_idx][1][pathi]
                    path_constraints_satisfied_cycle.append(
                        bb.path_constraints[amplicon_idx][0][pathi_],
                    )
                    path_constraints_support_cycle.append(
                        bb.longest_path_constraints[amplicon_idx][2][pathi],
                    )
                cycle_seg_list = eulerian_cycle_t(
                    bb.lr_graph[amplicon_idx],
                    bb.cycles[amplicon_idx][cycle_i[0]][cycle_i[1]],
                    path_constraints_satisfied_cycle,
                    path_constraints_support_cycle,
                )
                assert cycle_seg_list[0] == cycle_seg_list[-1]
                fp.write("Cycle=%d;" % (cycle_indices.index(cycle_i) + 1))
                fp.write(
                    "Copy_count=%s;" % str(bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]]),
                )
                fp.write("Segments=")
                for segi in range(len(cycle_seg_list) - 2):
                    fp.write(f"{int(cycle_seg_list[segi][:-1])}{cycle_seg_list[segi][-1]}")
                fp.write("%d%s" % (int(cycle_seg_list[-2][:-1]), cycle_seg_list[-2][-1]))
                if not output_all_paths:
                    fp.write(";Path_constraints_satisfied=")
                    for pathi in range(
                        len(bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]) - 1,
                    ):
                        fp.write(
                            "%d," % (bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]][pathi] + 1),
                        )
                    if len(bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]) > 0:
                        fp.write(
                            "%d\n" % (bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]][-1] + 1),
                        )
                    else:
                        fp.write("\n")
                else:
                    fp.write("\n")
            else:  # paths
                logger.debug(f"Traversing next path, CN = {bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]]}")
                path_constraints_satisfied_path = []
                path_constraints_support_path = []
                for pathi in bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]:
                    pathi_ = bb.longest_path_constraints[amplicon_idx][1][pathi]
                    path_constraints_satisfied_path.append(
                        bb.path_constraints[amplicon_idx][0][pathi_],
                    )
                    path_constraints_support_path.append(
                        bb.longest_path_constraints[amplicon_idx][2][pathi],
                    )
                cycle_seg_list = eulerian_path_t(
                    bb.lr_graph[amplicon_idx],
                    bb.cycles[amplicon_idx][cycle_i[0]][cycle_i[1]],
                    path_constraints_satisfied_path,
                    path_constraints_support_path,
                )
                print(cycle_seg_list, bb.cycles[amplicon_idx][cycle_i[0]][cycle_i[1]])
                fp.write("Cycle=%d;" % (cycle_indices.index(cycle_i) + 1))
                fp.write(
                    "Copy_count=%s;" % str(bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]]),
                )
                fp.write("Segments=0+,")
                for segi in range(len(cycle_seg_list) - 1):
                    fp.write("%d%s," % (int(cycle_seg_list[segi][:-1]), cycle_seg_list[segi][-1]))

                # breakpoint()
                fp.write("%d%s,0-" % (int(cycle_seg_list[-1][:-1]), cycle_seg_list[-1][-1]))
                if not output_all_paths:
                    fp.write(";Path_constraints_satisfied=")
                    for pathi in range(
                        len(bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]) - 1,
                    ):
                        fp.write(
                            "%d," % (bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]][pathi] + 1),
                        )
                    if len(bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]]) > 0:
                        fp.write(
                            "%d\n" % (bb.path_constraints_satisfied[amplicon_idx][cycle_i[0]][cycle_i[1]][-1] + 1),
                        )
                    else:
                        fp.write("\n")
                else:
                    fp.write("\n")
        fp.close()


def parse_lp_solution(model: pyo.Model, bp_graph: BreakpointGraph, k: int, pc_list: List) -> ParsedLPSolution:
    parsed_sol = ParsedLPSolution()

    lseg = len(bp_graph.sequence_edges)
    lc = len(bp_graph.concordant_edges)
    ld = len(bp_graph.discordant_edges)
    lsrc = len(bp_graph.source_edges)
    nnodes = len(bp_graph.nodes)  # Does not include s and t
    nedges = lseg + lc + ld + 2 * lsrc + 2 * len(bp_graph.endnodes)

    for i in range(k):
        logger.debug(f"Walk {i} checking ; CN = {model.w[i].value}.")
        if model.z[i].value >= 0.9:
            # if pyo.value(model.z[i] >= 0.9):
            logger.debug(f"Cycle/Path {i} exists; CN = {model.w[i].value}.")
            # sol_x = m.getAttr("X", model.x)
            # sol_c = m.getAttr("X", model.c)
            cycle_flag = -1
            # for ci in range(len(model.c)):
            for node_idx in range(nnodes):
                if model.c[node_idx, i].value >= 0.9:
                    cycle_flag = node_idx
                    break
            if cycle_flag == -1:
                cycle = dict()
                path_constraints_s = []
                # for xi in range(len(model.x)):
                for edge_i in range(nedges):
                    if model.x[edge_i, i].value >= 0.9:
                        # xi_ = xi // k
                        x_xi = int(round(model.x[edge_i, i].value))
                        if edge_i < lseg:
                            cycle[("e", edge_i)] = x_xi
                        elif edge_i < lseg + lc:
                            cycle[("c", edge_i - lseg)] = x_xi
                        elif edge_i < lseg + lc + ld:
                            cycle[("d", edge_i - lseg - lc)] = x_xi
                        elif edge_i < lseg + lc + ld + 2 * lsrc:
                            assert x_xi == 1
                            if (edge_i - lseg - lc - ld) % 2 == 0:
                                cycle[("s", (edge_i - lseg - lc - ld) // 2)] = 1  # source edge connected to s
                            else:
                                cycle[("t", (edge_i - lseg - lc - ld - 1) // 2)] = 1  # source edge connected to t
                        else:
                            assert x_xi == 1
                            if (edge_i - lseg - lc - ld - 2 * lsrc) % 2 == 0:
                                nsi = (edge_i - lseg - lc - ld - 2 * lsrc) // 2
                                cycle[("ns", nsi)] = 1  # source edge connected to s
                            else:
                                nti = (edge_i - lseg - lc - ld - 2 * lsrc - 1) // 2
                                cycle[("nt", nti)] = 1  # source edge connected to t
                for pi in range(len(pc_list)):
                    if model.r[pi, i].value >= 0.9:
                        path_constraints_s.append(pi)
                if model.w[i].value > 0.0:
                    parsed_sol.cycles[1].append(cycle)
                    parsed_sol.cycle_weights[1].append(model.w[i].value)
                    parsed_sol.path_constraints_satisfied[1].append(path_constraints_s)
                    parsed_sol.path_constraints_satisfied_set |= set(path_constraints_s)
            else:
                cycle = {}
                path_constraints_s = []
                for xi in range(len(model.x)):
                    if xi % k == i and model.x[xi] >= 0.9:
                        xi_ = xi // k
                        x_xi = int(round(model.x[xi]))
                        if xi_ < lseg:
                            cycle[("e", xi_)] = x_xi
                        elif xi_ < lseg + lc:
                            cycle[("c", xi_ - lseg)] = x_xi
                        elif xi_ < lseg + lc + ld:
                            cycle[("d", xi_ - lseg - lc)] = x_xi
                        else:
                            logger.debug(f"Error: Cyclic path cannot connect to source nodes.")
                            logger.debug("Aborted.")
                            os.abort()
                for pi in range(len(pc_list)):
                    if model.r[pi * k + i] >= 0.9:
                        path_constraints_s.append(pi)
                if model.w[i] > 0.0:
                    parsed_sol.cycles[0].append(cycle)
                    parsed_sol.cycle_weights[0].append(model.w[i].value)
                    parsed_sol.path_constraints_satisfied[0].append(path_constraints_s)
                    parsed_sol.path_constraints_satisfied_set |= set(path_constraints_s)
            for seqi in range(lseg):
                parsed_sol.total_weights_included += (
                    model.x[seqi, i].value * model.w[i].value * bp_graph.sequence_edges[seqi][-2]
                )
    return parsed_sol


def reconstruct_cycles(
    output_prefix: str,
    output_all_path_constraints: bool,
    cycle_decomp_alpha: float,
    cycle_decomp_time_limit: int,
    cycle_decomp_threads: int,
    postprocess_greedy_sol: bool,
    bb: infer_breakpoint_graph.BamToBreakpointNanopore,
):
    logging.basicConfig(
        filename=f"{output_prefix}/cycle_decomp.log",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )
    bb.compute_path_constraints()
    logger.info("Computed all subpath constraints.")

    alpha_ = 0.01
    postprocess_ = 0
    nthreads = -1
    time_limit_ = 7200
    if cycle_decomp_alpha:
        alpha_ = cycle_decomp_alpha
    if postprocess_greedy_sol:
        postprocess_ = 1
    if cycle_decomp_threads:
        nthreads = cycle_decomp_threads
    if cycle_decomp_time_limit:
        time_limit_ = cycle_decomp_time_limit
    cycle_decomposition(
        bb,
        alpha=alpha_,
        num_threads=nthreads,
        postprocess=postprocess_,
        time_limit=time_limit_,
        model_prefix=output_prefix,
    )
    logger.info("Completed cycle decomposition for all amplicons.")
    if output_all_path_constraints:
        output_cycles(bb, output_prefix, output_all_paths=True)
    else:
        output_cycles(bb, output_prefix)
    logger.info(f"Wrote cycles for all complicons to {output_prefix}_amplicon*_cycles.txt.")
