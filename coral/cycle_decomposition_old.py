"""Functions used for cycle decomposition"""

from __future__ import annotations

import logging
import math
import os
import random
import time
from typing import Any, Dict, List

import gurobipy as gp
from gurobipy import GRB

from coral import constants, infer_breakpoint_graph, state_provider
from coral.breakpoint_graph import BreakpointGraph
from coral.constants import CHR_TAG_TO_IDX
from coral.path_constraints import longest_path_dict

logger = logging.getLogger(__name__)

def minimize_cycles():
    cycles: List[List[Any]] = [[], []]  # cycles, paths
    cycle_weights: List[List[Any]] = [[], []]  # cycles, paths
    path_constraints_satisfied: List[List[Any]] = [[], []]  # cycles, paths
    path_constraints_satisfied_set = set([])

    sol_z = m.getAttr("X", z)
    sol_w = m.getAttr("X", w)
    sol_d = m.getAttr("X", d)
    sol_r = m.getAttr("X", r)
    sol_x = m.getAttr("X", x)
    sol_c = m.getAttr("X", c)
    total_weights_included = 0.0
    for i in range(k):
        if sol_z[i] >= 0.9:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tCycle/Path %d exists; CN = %f." % (i, sol_w[i]),
            )
            cycle_flag = -1
            for ci in range(len(sol_c)):
                if ci % k == i and sol_c[ci] >= 0.9:
                    cycle_flag = ci // k
                    break
            if cycle_flag == -1:
                cycle = dict()
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if xi % k == i and sol_x[xi] >= 0.9:
                        xi_ = xi // k
                        x_xi = int(round(sol_x[xi]))
                        if xi_ < lseg:
                            cycle[("e", xi_)] = x_xi
                        elif xi_ < lseg + lc:
                            cycle[("c", xi_ - lseg)] = x_xi
                        elif xi_ < lseg + lc + ld:
                            cycle[("d", xi_ - lseg - lc)] = x_xi
                        elif xi_ < lseg + lc + ld + 2 * lsrc:
                            assert x_xi == 1
                            if (xi_ - lseg - lc - ld) % 2 == 0:
                                cycle[("s", (xi_ - lseg - lc - ld) // 2)] = 1  # source edge connected to s
                            else:
                                cycle[("t", (xi_ - lseg - lc - ld - 1) // 2)] = 1  # source edge connected to t
                        else:
                            assert x_xi == 1
                            if (xi_ - lseg - lc - ld - 2 * lsrc) % 2 == 0:
                                nsi = (xi_ - lseg - lc - ld - 2 * lsrc) // 2
                                cycle[("ns", nsi)] = 1  # source edge connected to s
                            else:
                                nti = (xi_ - lseg - lc - ld - 2 * lsrc - 1) // 2
                                cycle[("nt", nti)] = 1  # source edge connected to t
                for pi in range(len(pc_list)):
                    if sol_r[pi * k + i] >= 0.9:
                        path_constraints_s.append(pi)
                if sol_w[i] > 0.0:
                    cycles[1].append(cycle)
                    cycle_weights[1].append(sol_w[i])
                    path_constraints_satisfied[1].append(path_constraints_s)
                    path_constraints_satisfied_set |= set(path_constraints_s)
            else:
                cycle = dict()
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if xi % k == i and sol_x[xi] >= 0.9:
                        xi_ = xi // k
                        x_xi = int(round(sol_x[xi]))
                        if xi_ < lseg:
                            cycle[("e", xi_)] = x_xi
                        elif xi_ < lseg + lc:
                            cycle[("c", xi_ - lseg)] = x_xi
                        elif xi_ < lseg + lc + ld:
                            cycle[("d", xi_ - lseg - lc)] = x_xi
                        else:
                            logger.debug(
                                "#TIME "
                                + "%.4f\t" % (time.time() - state_provider.TSTART)
                                + "\tError: Cyclic path cannot connect to source nodes.",
                            )
                            logger.debug(
                                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tAborted.",
                            )
                            os.abort()
                for pi in range(len(pc_list)):
                    if sol_r[pi * k + i] >= 0.9:
                        path_constraints_s.append(pi)
                if sol_w[i] > 0.0:
                    cycles[0].append(cycle)
                    cycle_weights[0].append(sol_w[i])
                    path_constraints_satisfied[0].append(path_constraints_s)
                    path_constraints_satisfied_set |= set(path_constraints_s)
            for seqi in range(lseg):
                total_weights_included += sol_x[seqi * k + i] * sol_w[i] * bp_graph.sequence_edges[seqi][-2]
    logger.debug(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "Total length weighted CN from cycles/paths = %f/%f." % (total_weights_included, total_weights),
    )
    logger.debug(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "Total num subpath constraints satisfied = %d/%d." % (len(path_constraints_satisfied_set), len(pc_list)),
    )
    return (
        m.Status,
        total_weights_included,
        len(path_constraints_satisfied_set),
        cycles,
        cycle_weights,
        path_constraints_satisfied,
    )


def minimize_cycles_post(
    amplicon_id: int,
    g: BreakpointGraph,
    total_weights: float,
    node_order: Dict[tuple[Any, Any, Any], int],
    pc_list,
    init_sol,
    p_total_weight=0.9,
    resolution=0.1,
    num_threads=-1,
    time_limit=7200,
    model_prefix="",
):
    """Cycle decomposition by postprocessing the greedy solution

    g: breakpoint graph (object)
    total_weights: float, total length-weighted CN in breakpoint graph g
    node_order: dict maps each node in the input breakpoint graphg to a distinct integer, indicating a total order of the nodes in g
    pc_list: list of subpath constraints to be satisfied, each as a dict that maps an edge to its multiplicity
    init_sol: initial solution returned by maximize_weights_greedy
    p_total_weight: float between (0, 1), minimum proportion of length-weighted CN to be covered by the resulting cycles or paths,
                    default value is 0.9
    resolution: float, minimum CN for each cycle or path, default value is 0.1
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


    # Initialize variables
    m.update()

    m.setParam(GRB.Param.LogToConsole, 0)
    if num_threads > 0:
        m.setParam(GRB.Param.Threads, num_threads)
    m.setParam(GRB.Param.NonConvex, 2)
    m.setParam(GRB.Param.Heuristics, 0.25)
    m.setParam(
        GRB.Param.TimeLimit,
        max(time_limit, ld * 300),
    )  # each breakpoint edge is assigned 5 minutes
    logger.debug(
        "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tCompleted gurobi model setup.",
    )
    lp_fn = model_prefix + "_amplicon" + str(amplicon_id) + "_postprocessing_model.lp"
    m.write(lp_fn)
    logger.debug(
        "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tWrote model to file: %s." % lp_fn,
    )
    log_fn = lp_fn[:-2] + "log"
    m.setParam(GRB.Param.LogFile, log_fn)
    m.optimize()
    logger.debug(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "\tCompleted optimization with status %d." % m.Status,
    )

    if m.Status == GRB.INFEASIBLE or m.SolCount == 0:
        logger.debug(
            "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tModel is infeasible.",
        )
        return GRB.INFEASIBLE, 0.0, 0, [[], []], [[], []], [[], []]
    cycles: list[list] = [[], []]  # cycles, paths
    cycle_weights: list[list] = [[], []]  # cycles, paths
    path_constraints_satisfied: list[list] = [[], []]  # cycles, paths
    path_constraints_satisfied_set = set([])

    sol_z = m.getAttr("X", z)
    sol_w = m.getAttr("X", w)
    sol_d = m.getAttr("X", d)
    sol_r = m.getAttr("X", r)
    total_weights_included = 0.0
    for i in range(k):
        if sol_z[i] >= 0.9:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tCycle/Path %d exists; CN = %f." % (i, sol_w[i]),
            )
            sol_x = m.getAttr("X", x)
            sol_c = m.getAttr("X", c)
            cycle_flag = -1
            for ci in range(len(sol_c)):
                if ci % k == i and sol_c[ci] >= 0.9:
                    cycle_flag = ci // k
                    break
            if cycle_flag == -1:
                cycle = dict()
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if xi % k == i and sol_x[xi] >= 0.9:
                        xi_ = xi // k
                        x_xi = int(round(sol_x[xi]))
                        if xi_ < lseg:
                            cycle[("e", xi_)] = x_xi
                        elif xi_ < lseg + lc:
                            cycle[("c", xi_ - lseg)] = x_xi
                        elif xi_ < lseg + lc + ld:
                            cycle[("d", xi_ - lseg - lc)] = x_xi
                        elif xi_ < lseg + lc + ld + 2 * lsrc:
                            assert x_xi == 1
                            if (xi_ - lseg - lc - ld) % 2 == 0:
                                cycle[("s", (xi_ - lseg - lc - ld) // 2)] = 1  # source edge connected to s
                            else:
                                cycle[("t", (xi_ - lseg - lc - ld - 1) // 2)] = 1  # source edge connected to t
                        else:
                            assert x_xi == 1
                            if (xi_ - lseg - lc - ld - 2 * lsrc) % 2 == 0:
                                nsi = (xi_ - lseg - lc - ld - 2 * lsrc) // 2
                                cycle[("ns", nsi)] = 1  # source edge connected to s
                            else:
                                nti = (xi_ - lseg - lc - ld - 2 * lsrc - 1) // 2
                                cycle[("nt", nti)] = 1  # source edge connected to t
                for pi in range(len(pc_list)):
                    if sol_r[pi * k + i] >= 0.9:
                        path_constraints_s.append(pi)
                cycles[1].append(cycle)
                cycle_weights[1].append(sol_w[i])
                path_constraints_satisfied[1].append(path_constraints_s)
                path_constraints_satisfied_set |= set(path_constraints_s)
            else:
                cycle = dict()
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if xi % k == i and sol_x[xi] >= 0.9:
                        xi_ = xi // k
                        x_xi = int(round(sol_x[xi]))
                        if xi_ < lseg:
                            cycle[("e", xi_)] = x_xi
                        elif xi_ < lseg + lc:
                            cycle[("c", xi_ - lseg)] = x_xi
                        elif xi_ < lseg + lc + ld:
                            cycle[("d", xi_ - lseg - lc)] = x_xi
                        else:
                            logger.debug(
                                "#TIME "
                                + "%.4f\t" % (time.time() - state_provider.TSTART)
                                + "\tError: Cyclic path cannot connect to source nodes.",
                            )
                            logger.debug(
                                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tAborted.",
                            )
                            os.abort()
                for pi in range(len(pc_list)):
                    if sol_r[pi * k + i] >= 0.9:
                        path_constraints_s.append(pi)
                cycles[0].append(cycle)
                cycle_weights[0].append(sol_w[i])
                path_constraints_satisfied[0].append(path_constraints_s)
                path_constraints_satisfied_set |= set(path_constraints_s)
            for seqi in range(lseg):
                total_weights_included += sol_x[seqi * k + i] * sol_w[i] * g.sequence_edges[seqi][-2]
    logger.debug(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "Total length weighted CN from cycles/paths = %f/%f." % (total_weights_included, total_weights),
    )
    logger.debug(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "Total num subpath constraints satisfied = %d/%d." % (len(path_constraints_satisfied_set), len(pc_list)),
    )
    return (
        m.Status,
        total_weights_included,
        len(path_constraints_satisfied_set),
        cycles,
        cycle_weights,
        path_constraints_satisfied,
    )


def maximize_weights_greedy(
    amplicon_id,
    g: BreakpointGraph,
    total_weights,
    node_order,
    pc_list,
    alpha=0.01,
    p_total_weight=0.9,
    resolution=0.1,
    cn_tol=0.005,
    p_subpaths=0.9,
    num_threads=-1,
    postprocess=0,
    time_limit=7200,
    model_prefix="",
):
    """Greedy cycle decomposition by maximizing the length-weighted CN of a single cycle/path

    amplicon_id: integer, amplicon ID
    g: breakpoint graph (object)
    total_weights: float, total length-weighted CN in breakpoint graph g
    node_order: dict maps each node in the input breakpoint graphg to a distinct integer, indicating a total order of the nodes in g
    pc_list: list of subpath constraints to be satisfied, each as a dict that maps an edge to its multiplicity
    alpha: float, parameter for multi-objective optimization, default value is 0.01
            maximizing total length-weighted CN +
            (alpha * remaining length-weighted CN in the graph / num remaining unsatisfied subpath constraints) *
            num subpath constraints satisfied by the next cycle or path
            *** when alpha < 0, just maximizing total length-weighted CN
    p_total_weight: float between (0, 1), minimum proportion of length-weighted CN to be covered by the resulting cycles or paths,
                    default value is 0.9
    resolution: float, minimum CN for each cycle or path, default value is 0.1
    cn_tol: float between (0, 1), terminate greedy cycle decomposition when the next cycle/path has total length weighted CN
            < cn_tol * total_weights, default value is 0.005
    p_subpaths: float between (0, 1), minimum proportion of subpath constraints to be satisfied by the resulting cycles or paths,
            default value is 0.9
    num_threads: integer, number of working threads for gurobipy, by default it tries to use up all available cores
    time_limit: integer, maximum allowed running time, in seconds, default is 7200 (2 hour)
    model_prefix: output prefix for gurobi *.lp model

    Returns: (1) Total length weighted CN in resulting cycles/paths
            (2) Total num subpath constraints satisfied by resulting cycles/paths
            (3) List of cycles, each as a dict which maps an edge to its multiplicity in the cycle
            (4) List of the corresponding CN of the above cycles
            (5) Subpath constraints (indices) satisfied by each cycle
            (6) List of paths, each as a dict which maps an edge to its multiplicity in the path
            (7) List of the corresponding CN of the above paths
            (8) Subpath constraints (indices) satisfied by each path
    """
    logger.debug("Integer program too large, perform greedy cycle decomposition.")

    remaining_weights = total_weights
    unsatisfied_pc = [i for i in range(len(pc_list))]
    discordant_multiplicities = g.infer_discordant_edge_multiplicities()
    remaining_CN = dict()
    for segi in range(g.num_seq_edges):
        remaining_CN[("s", segi)] = g.sequence_edges[segi][-1]
    for ci in range(g.num_conc_edges):
        remaining_CN[("c", ci)] = g.concordant_edges[ci][-1]
    for di in range(g.num_disc_edges):
        remaining_CN[("d", di)] = g.discordant_edges[di][-1]
    for srci in range(g.num_src_edges):
        remaining_CN[("src", srci)] = g.source_edges[srci][-1]
    next_w = resolution * 1.1
    cycle_id = 0
    num_unsatisfied_pc = len(pc_list)
    cycles = [[], []]  # cycles, paths
    cycle_weights = [[], []]  # cycles, paths
    path_constraints_satisfied = [[], []]  # cycles, paths
    logger.debug(f"Greedy cycle decomposition with length weighted CN = {remaining_weights} and num subpath constraints = {num_unsatisfied_pc}.")


    while next_w >= resolution and (
        remaining_weights > (1.0 - p_total_weight) * total_weights
        or num_unsatisfied_pc > math.floor((1.0 - p_subpaths) * len(pc_list))
    ):
        pp = 1.0
        if alpha > 0 and num_unsatisfied_pc > 0:
            pp = alpha * remaining_weights / num_unsatisfied_pc  # multi - objective optimization parameter
        logger.debug(f"Iteration {(cycle_id + 1)}")
        logger.debug(f"Remaining length weighted CN: {remaining_weights}/{total_weights}.")
        logger.debug(f"Remaining subpath constraints to be satisfied: {num_unsatisfied_pc}/{len(pc_list)}")
        logger.debug(f"Multiplication factor for subpath constraints: {pp}".)

        # Gurobi model
        m = gp.Model(
            model_prefix + "_amplicon" + str(amplicon_id) + "_cycle_decomposition_greedy_" + str(cycle_id + 1),
        )


        # TODO: z/w/x/r/d/c/y essentially have k = 1

        # Objective: maximize the total weight + num subpath constraints satisfied
        obj = gp.QuadExpr(0.0)
        for seqi in range(lseg):
            obj += x[seqi] * w[0] * g.sequence_edges[seqi][-2]
        for pi in range(len(pc_list)):
            if unsatisfied_pc[pi] >= 0:
                obj += r[pi] * max(pp, 1.0)
        m.setObjective(obj, GRB.MAXIMIZE)

        # TODO: set CN Constraint
        # TODO: add EulerianNode + EulerianPath Constraints
        # TODO: BPOccurrence Constraint
        # TODO: CycleWeight Constraint
        # TODO: SingularBPEdge Constraint
        # TODO: CycleTree Constraints
        # TODO: Relationship between y and z:
        # TODO: SpanningTree Constraints + Relationship between x, y and d
        # Subpath constraints ( no addtl)

        m.setParam(GRB.Param.LogToConsole, 0)
        if num_threads > 0:
            m.setParam(GRB.Param.Threads, num_threads)
        m.setParam(GRB.Param.NonConvex, 2)
        m.setParam(GRB.Param.TimeLimit, time_limit)  # each breakpoint edge is assigned 5 minutes
        logger.debug(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "\tCompleted gurobi setup for model %d." % (cycle_id + 1),
        )
        lp_fn = (
            model_prefix
            + "_amplicon"
            + str(amplicon_id)
            + "_greedy_model_"
            + str(cycle_id + 1)
            + "_alpha="
            + str(alpha)
            + ".lp"
        )
        m.write(lp_fn)
        logger.debug(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "\tWrote model to %d file: %s." % (cycle_id + 1, lp_fn),
        )
        log_fn = lp_fn[:-2] + "log"
        m.setParam(GRB.Param.LogFile, log_fn)
        m.optimize()
        logger.debug(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "\tCompleted optimization of model %d with status %d." % (cycle_id + 1, m.Status),
        )

        if m.Status == GRB.INFEASIBLE or m.SolCount == 0:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tModel %d is infeasible." % (cycle_id + 1),
            )
            logger.debug(
                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tIteration terminated.",
            )
            break
        cycle_id += 1
        sol_z = m.getAttr("X", z)
        sol_w = m.getAttr("X", w)
        sol_d = m.getAttr("X", d)
        sol_r = m.getAttr("X", r)
        total_weights_included = 0.0
        if sol_z[0] >= 0.9:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tNext cycle/path exists; CN = %f." % (sol_w[0]),
            )
            next_w = sol_w[0]
            if next_w < resolution:
                logger.debug(
                    "\tCN less than resolution, iteration terminated successfully.",
                )
                break
            sol_x = m.getAttr("X", x)
            sol_y1 = m.getAttr("X", y1)
            sol_y2 = m.getAttr("X", y2)
            sol_c = m.getAttr("X", c)
            cycle_flag = -1
            for ci in range(len(sol_c)):
                if sol_c[ci] >= 0.9:
                    cycle_flag = ci
                    break
            cycle = dict()
            cycle_for_postprocess = dict()
            for xi in range(len(sol_x)):
                cycle_for_postprocess[("x", xi)] = sol_x[xi]
            for ci in range(len(sol_c)):
                cycle_for_postprocess[("c", ci)] = sol_c[ci]
            for di in range(len(sol_d)):
                cycle_for_postprocess[("d", di)] = sol_d[di]
            for yi in range(len(sol_y1)):
                cycle_for_postprocess[("y1", yi)] = sol_y1[yi]
            for yi in range(len(sol_y2)):
                cycle_for_postprocess[("y2", yi)] = sol_y2[yi]
            if cycle_flag == -1:
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if sol_x[xi] >= 0.9:
                        x_xi = int(round(sol_x[xi]))
                        if xi < lseg:
                            cycle[("e", xi)] = x_xi
                            remaining_CN[("s", xi)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("s", xi)] < resolution:
                                remaining_CN[("s", xi)] = 0.0
                        elif xi < lseg + lc:
                            cycle[("c", xi - lseg)] = x_xi
                            remaining_CN[("c", xi - lseg)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("c", xi - lseg)] < resolution:
                                remaining_CN[("c", xi - lseg)] = 0.0
                        elif xi < lseg + lc + ld:
                            cycle[("d", xi - lseg - lc)] = x_xi
                            remaining_CN[("d", xi - lseg - lc)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("d", xi - lseg - lc)] < resolution:
                                remaining_CN[("d", xi - lseg - lc)] = 0.0
                        elif xi < lseg + lc + ld + 2 * lsrc:
                            assert x_xi == 1
                            if (xi - lseg - lc - ld) % 2 == 0:
                                cycle[("s", (xi_ - lseg - lc - ld) // 2)] = 1  # source edge connected to s
                                remaining_CN[("src", (xi - lseg - lc - ld) // 2)] -= sol_w[0]
                                if remaining_CN[("src", (xi - lseg - lc - ld) // 2)] < resolution:
                                    remaining_CN[("src", (xi - lseg - lc - ld) // 2)] = 0.0
                            else:
                                cycle[("t", (xi_ - lseg - lc - ld - 1) // 2)] = 1  # source edge connected to t
                                remaining_CN[("src", (xi - lseg - lc - ld - 1) // 2)] -= sol_w[0]
                                if remaining_CN[("src", (xi - lseg - lc - ld - 1) // 2)] < resolution:
                                    remaining_CN[("src", (xi - lseg - lc - ld - 1) // 2)] = 0.0
                        else:
                            assert x_xi == 1
                            if (xi - lseg - lc - ld - 2 * lsrc) % 2 == 0:
                                nsi = (xi - lseg - lc - ld - 2 * lsrc) // 2
                                cycle[("ns", nsi)] = 1  # source edge connected to s
                            else:
                                nti = (xi - lseg - lc - ld - 2 * lsrc - 1) // 2
                                cycle[("nt", nti)] = 1  # source edge connected to t
                for pi in range(len(pc_list)):
                    if sol_r[pi] >= 0.9:
                        path_constraints_s.append(pi)
                        unsatisfied_pc[pi] = -1
                if postprocess == 1:
                    cycles[1].append(cycle_for_postprocess)
                else:
                    cycles[1].append(cycle)
                cycle_weights[1].append(sol_w[0])
                path_constraints_satisfied[1].append(path_constraints_s)
            else:
                path_constraints_s = []
                for xi in range(len(sol_x)):
                    if sol_x[xi] >= 0.9:
                        x_xi = int(round(sol_x[xi]))
                        if xi < lseg:
                            cycle[("e", xi)] = x_xi
                            remaining_CN[("s", xi)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("s", xi)] < resolution:
                                remaining_CN[("s", xi)] = 0.0
                        elif xi < lseg + lc:
                            cycle[("c", xi - lseg)] = x_xi
                            remaining_CN[("c", xi - lseg)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("c", xi - lseg)] < resolution:
                                remaining_CN[("c", xi - lseg)] = 0.0
                        elif xi < lseg + lc + ld:
                            cycle[("d", xi - lseg - lc)] = x_xi
                            remaining_CN[("d", xi - lseg - lc)] -= sol_x[xi] * sol_w[0]
                            if remaining_CN[("d", xi - lseg - lc)] < resolution:
                                remaining_CN[("d", xi - lseg - lc)] = 0.0
                        else:
                            logger.debug(
                                "#TIME "
                                + "%.4f\t" % (time.time() - state_provider.TSTART)
                                + "\tError: Cyclic path cannot connect to source nodes.",
                            )
                            logger.debug(
                                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tAborted.",
                            )
                            os.abort()
                for pi in range(len(pc_list)):
                    if sol_r[pi] >= 0.9:
                        path_constraints_s.append(pi)
                        unsatisfied_pc[pi] = -1
                if postprocess == 1:
                    cycles[0].append(cycle_for_postprocess)
                else:
                    cycles[0].append(cycle)
                cycle_weights[0].append(sol_w[0])
                path_constraints_satisfied[0].append(path_constraints_s)
            for seqi in range(lseg):
                total_weights_included += sol_x[seqi] * sol_w[0] * g.sequence_edges[seqi][-2]
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "Total length weighted CN from cycle/path %d = %f/%f."
                % (cycle_id, total_weights_included, total_weights),
            )
            remaining_weights -= total_weights_included
            if total_weights_included < cn_tol * total_weights:
                num_unsatisfied_pc = 0
                for i in range(len(pc_list)):
                    if unsatisfied_pc[i] >= 0:
                        num_unsatisfied_pc += 1
                logger.debug(
                    "\tProportion of length weighted CN less than cn_tol, iteration terminated.",
                )
                break
        else:
            break
        num_unsatisfied_pc = 0
        for i in range(len(pc_list)):
            if unsatisfied_pc[i] >= 0:
                num_unsatisfied_pc += 1
        logger.debug(f"Remaining subpath constraints to be satisfied: {num_unsatisfied_pc}/{len(pc_list)}.")
        logger.debug("Proceed to next iteration.")
    if next_w < resolution:
        logger.debug("Cycle/path reaches CN resolution, greedy iteration completed.")
    elif (num_unsatisfied_pc <= math.floor((1.0 - p_subpaths) * len(pc_list))) and (
        remaining_weights <= (1.0 - p_total_weight) * total_weights
    ):
        logger.debug("Cycles/paths reaches CN resolution, greedy iteration completed.")
    else:
        logger.debug(f"Proportion of length weighted CN less than cn_tol, greedy iteration completed.")
    logger.debug(f"Total length weighted CN from cycles/paths = {total_weights - remaining_weights}/{total_weights}.")
    logger.debug(f"Total num subpath constraints satisfied = {len(pc_list) - num_unsatisfied_pc}/{len(pc_list)}.")
    return (
        total_weights - remaining_weights,
        len(pc_list) - num_unsatisfied_pc,
        cycles,
        cycle_weights,
        path_constraints_satisfied,
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
        lseg = len(bb.lr_graph[amplicon_idx].sequence_edges)
        lc = len(bb.lr_graph[amplicon_idx].concordant_edges)
        ld = len(bb.lr_graph[amplicon_idx].discordant_edges)
        lsrc = len(bb.lr_graph[amplicon_idx].source_edges)

        total_weights = 0.0
        for sseg in bb.lr_graph[amplicon_idx].sequence_edges:
            total_weights += sseg[7] * sseg[-1]  # type: ignore[operator]
        logger.info(f"Begin cycle decomposition for amplicon {amplicon_idx + 1}.")
        logger.info(f"Total CN weights = {total_weights}.")

        bb.longest_path_constraints[amplicon_idx] = longest_path_dict(
            bb.path_constraints[amplicon_idx],
        )
        logger.info(f"Total num maximal subpath constraints = {len(bb.longest_path_constraints[amplicon_idx][0])}.")
        for pathi in bb.longest_path_constraints[amplicon_idx][1]:
            logger.debug(f"Subpath constraints {pathi} = {bb.longest_path_constraints[amplicon_idx][0][pathi]}")

        k = max(10, ld // 2)  # Initial num cycles/paths
        logger.info(f"Total num initial cycles / paths = {k}.")
        nnodes = len(bb.lr_graph[amplicon_idx].nodes)  # Does not include s and t
        node_order = dict()
        ni_ = 0
        for node in bb.lr_graph[amplicon_idx].nodes.keys():
            node_order[node] = ni_
            ni_ += 1
        nedges = lseg + lc + ld + 2 * lsrc + 2 * len(bb.lr_graph[amplicon_idx].endnodes)
        if nedges < k:
            k = nedges
            logger.info(
                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "Reset num cycles/paths to %d." % k,
            )
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
                    % (
                        total_path_satisfied_init,
                        len(bb.longest_path_constraints[amplicon_idx][0]),
                    ),
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
                    logger.info(
                        "#TIME "
                        + "%.4f\t" % (time.time() - state_provider.TSTART)
                        + "Completed postprocessing of the greedy solution.",
                    )
                    logger.info(
                        "#TIME "
                        + "%.4f\t" % (time.time() - state_provider.TSTART)
                        + "\tNum cycles = %d; num paths = %d." % (len(cycles_post[0]), len(cycles_post[1])),
                    )
                    logger.info(
                        "#TIME "
                        + "%.4f\t" % (time.time() - state_provider.TSTART)
                        + "\tTotal length weighted CN = %f/%f." % (total_cycle_weights_post, total_weights),
                    )
                    logger.info(
                        "#TIME "
                        + "%.4f\t" % (time.time() - state_provider.TSTART)
                        + "\tTotal num subpath constraints satisfied = %d/%d."
                        % (
                            total_path_satisfied_post,
                            len(bb.longest_path_constraints[amplicon_idx][0]),
                        ),
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
                logger.info(
                    "Cycle decomposition is infeasible.",
                )
                logger.info(
                    "Doubling k from %d to %d." % (k, k * 2),
                )
                k *= 2
            else:
                logger.info(
                    "Completed cycle decomposition with k = %d." % k,
                )
                logger.info(
                    "\tNum cycles = %d; num paths = %d." % (len(cycles_[0]), len(cycles_[1])),
                )
                logger.info(
                    "\tTotal length weighted CN = %f/%f." % (total_cycle_weights_, total_weights),
                )
                logger.info(
                    "\tTotal num subpath constraints satisfied = %d/%d."
                    % (total_path_satisfied_, len(bb.longest_path_constraints[amplicon_idx][0])),
                )
                print(f"{cycles_=}")
                bb.cycles[amplicon_idx] = cycles_
                bb.cycle_weights[amplicon_idx] = cycle_weights_
                bb.path_constraints_satisfied[amplicon_idx] = path_constraints_satisfied_
                sol_flag = 1
                break
        if sol_flag == 0:
            logger.info(
                "Cycle decomposition is infeasible, switch to greedy cycle decomposition.",
            )
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
                "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "Completed greedy cycle decomposition.",
            )
            logger.info(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tNum cycles = %d; num paths = %d." % (len(cycles_init[0]), len(cycles_init[1])),
            )
            logger.info(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tTotal length weighted CN = %f/%f." % (total_cycle_weights_init, total_weights),
            )
            logger.info(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\tTotal num subpath constraints satisfied = %d/%d."
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
                logger.info(
                    "Completed postprocessing of the greedy solution.",
                )
                logger.info(
                    "\tNum cycles = %d; num paths = %d." % (len(cycles_post[0]), len(cycles_post[1])),
                )
                logger.info(
                    "\tTotal length weighted CN = %f/%f." % (total_cycle_weights_post, total_weights),
                )
                logger.info(
                    "\tTotal num subpath constraints satisfied = %d/%d."
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
            "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tCycle satisfies all subpath constraints.",
        )
    else:
        logger.debug(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "\tThe following path constraints are not satisfied:",
        )
        for pathi in unsatisfied_path_metric[0]:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\t%s" % path_constraints_next_cycle[pathi],
            )
    return best_cycle


def eulerian_path_t(g, edges_next_path, path_constraints_next_path, path_constraints_support):
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
        logger.debug(
            "#TIME " + "%.4f\t" % (time.time() - state_provider.TSTART) + "\tPath satisfies all subpath constraints.",
        )
    else:
        logger.debug(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "\tThe following path constraints are not satisfied:",
        )
        for pathi in unsatisfied_path_metric[0]:
            logger.debug(
                "#TIME "
                + "%.4f\t" % (time.time() - state_provider.TSTART)
                + "\t%s" % path_constraints_next_path[pathi],
            )
    return best_path


def output_cycles(
    bb: infer_breakpoint_graph.BamToBreakpointNanopore, cycle_file_prefix: str, output_all_paths=False
) -> None:
    """Write the result from cycle decomposition into *.cycles files"""
    for amplicon_idx in range(len(bb.lr_graph)):
        logger.info(
            "#TIME "
            + "%.4f\t" % (time.time() - state_provider.TSTART)
            + "Output cycles for amplicon %d." % (amplicon_idx + 1),
        )
        fp = open(cycle_file_prefix + "/amplicon" + str(amplicon_idx + 1) + "_cycles.txt", "w")
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
                logger.debug(
                    "\tTraversing next cycle, CN = %f." % bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]],
                )
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
                    fp.write("%d%s," % (int(cycle_seg_list[segi][:-1]), cycle_seg_list[segi][-1]))
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
                logger.debug(
                    "\tTraversing next path, CN = %f." % bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]],
                )
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
                print(f"heres {cycle_seg_list}, {bb.cycles[amplicon_idx][cycle_i[0]][cycle_i[1]]}")
                print(f"and pcs {path_constraints_satisfied_path}, {path_constraints_support_path}")
                fp.write("Cycle=%d;" % (cycle_indices.index(cycle_i) + 1))
                fp.write(
                    "Copy_count=%s;" % str(bb.cycle_weights[amplicon_idx][cycle_i[0]][cycle_i[1]]),
                )
                fp.write("Segments=0+,")
                for segi in range(len(cycle_seg_list) - 1):
                    fp.write("%d%s," % (int(cycle_seg_list[segi][:-1]), cycle_seg_list[segi][-1]))
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
        format="%(asctime)s:%(levelname)-4s [%(filename)s:%(lineno)d] %(message)s",
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
    logger.info(
        "#TIME "
        + "%.4f\t" % (time.time() - state_provider.TSTART)
        + "Wrote cycles for all complicons to %s." % (output_prefix + "_amplicon*_cycles.txt"),
    )
