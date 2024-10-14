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
import pyomo.contrib.appsi
from coral import constants, infer_breakpoint_graph, models, state_provider
from coral.breakpoint_graph import BreakpointGraph
from coral.constants import CHR_TAG_TO_IDX
from coral.models import output, utils
from coral.path_constraints import longest_path_dict

logger = logging.getLogger(__name__)



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
    solver_to_use="gurobi",
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

    model = models.concrete.get_minimize_cycle_model(
        bp_graph,
        k,
        total_weights,
        node_order,
        pc_list,
        model_name=f"{model_prefix}/amplicon_{amplicon_id}_cycle_decomposition_{k=}")

    model_name = f"{model_prefix}/amplicon_{amplicon_id}_model"
    model.write(f"{model_name}.lp", io_options={"symbolic_solver_labels": True})
    logger.debug(f"Completed model setup, wrote to {model_name}.lp.")

    solver = utils.get_solver(
        solver_name=solver_to_use,
        num_threads=num_threads,
        time_limit_s=max(time_limit, bp_graph.num_disc_edges * 300),  # each breakpoint edge is assigned 5 minutes
    )
    results: pyomo.opt.SolverResults = solver.solve(model, tee=True)

    logger.debug(f"Completed optimization with status {results.solver.status}, condition {results.solver.termination_condition}.")
    if results.solver.termination_condition == pyo.TerminationCondition.infeasible:
        pyomo.util.infeasible.log_infeasible_constraints(model, log_expression=True, log_variables=True)

    parsed_sol = utils.parse_lp_solution(model, bp_graph, k, pc_list)
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
    logger.debug("Cycle decomposition with initial solution from the greedy strategy.")

    k = len(init_sol[0][0]) + len(init_sol[0][1])
    logger.debug(f"Reset k (num cycles) to {k}.")
    p_path_constraints = 0.0
    path_constraint_indices_ = []
    for paths in init_sol[2][0] + init_sol[2][1]:
        for pathi in paths:
            if pathi not in path_constraint_indices_:
                path_constraint_indices_.append(pathi)
    if len(pc_list) > 0:
        p_path_constraints = len(path_constraint_indices_) * 0.9999 / len(pc_list)
        logger.debug(f"Required proportion of subpath constraints to be satisfied: {p_path_constraints}.")
    else:
        logger.debug("Proceed without subpath constraints.")


    model_name = f"{model_prefix}/amplicon_{amplicon_id}_cycle_decomposition_postprocessing_{k=}"

def initialize_post_processing_solver(
    model: pyo.ConcreteModel,
    bp_graph: BreakpointGraph,
    init_sol
):
    """
        bp_graph: breakpoint graph (object)
        init_sol: initial solution returned by maximize_weights_greedy
    """

    for i in range(len(init_sol[0][0])):
        z[i].start = 1
        w[i].start = init_sol[1][0][i]
        for v, vi in init_sol[0][0][i].keys():
            if v == "x":
                x[vi * k + i].start = init_sol[0][0][i][(v, vi)]
            elif v == "c":
                c[vi * k + i].start = init_sol[0][0][i][(v, vi)]
            elif v == "d":
                d[vi * k + i].start = init_sol[0][0][i][(v, vi)]
            elif v == "y1":
                y1[vi * k + i].start = init_sol[0][0][i][(v, vi)]
            elif v == "y2":
                y2[vi * k + i].start = init_sol[0][0][i][(v, vi)]
    for i in range(len(init_sol[0][1])):
        i_ = i + len(init_sol[0][0])
        z[i_].start = 1
        w[i_].start = init_sol[1][1][i]
        for v, vi in init_sol[0][1][i].keys():
            if v == "x":
                x[vi * k + i_].start = init_sol[0][1][i][(v, vi)]
            elif v == "c":
                c[vi * k + i_].start = init_sol[0][1][i][(v, vi)]
            elif v == "d":
                d[vi * k + i_].start = init_sol[0][1][i][(v, vi)]
            elif v == "y1":
                y1[vi * k + i_].start = init_sol[0][1][i][(v, vi)]
            elif v == "y2":
                y2[vi * k + i_].start = init_sol[0][1][i][(v, vi)]
    for i in range(len(init_sol[2][0])):
        for pi in init_sol[2][0][i]:
            r[pi * k + i].start = 1
            R[pi].start = 1
    for i in range(len(init_sol[2][1])):
        i_ = i + len(init_sol[2][0])
        for pi in init_sol[2][1][i]:
            r[pi * k + i_].start = 1
            R[pi].start = 1

def maximize_weights_greedy(
    amplicon_id: int,
    g: BreakpointGraph,
    total_weights: float,
    node_order: Dict[tuple[Any, Any, Any], int],
    pc_list,
    alpha=0.01,
    p_total_weight=0.9,
    resolution=0.1,
    cn_tol=0.005,
    p_subpaths=0.9,
    num_threads=-1,
    postprocess=0,
    time_limit=7200,
    model_prefix=""
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
    solver_to_use: str = "gurobi",
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
        for pathi in bb.longest_path_constraints[amplicon_idx][1]:
            logger.debug(f"Subpath constraint {pathi} = {bb.longest_path_constraints[amplicon_idx][0][pathi]}")


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
                solver_to_use,
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




def reconstruct_cycles(
    output_prefix: str,
    output_all_path_constraints: bool,
    cycle_decomp_alpha: float,
    cycle_decomp_time_limit: int,
    cycle_decomp_threads: int,
    solver_to_use: str,
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
        solver_to_use=solver_to_use,
    )
    logger.info("Completed cycle decomposition for all amplicons.")
    if output_all_path_constraints:
        output.output_cycles(bb, output_prefix, output_all_paths=True)
    else:
        output.output_cycles(bb, output_prefix)
    logger.info(f"Wrote cycles for all complicons to {output_prefix}_amplicon*_cycles.txt.")
