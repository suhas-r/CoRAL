
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
from coral.path_constraints import longest_path_dict

logger = logging.getLogger(__name__)

@dataclass
class EdgeToCN:
    sequence: Dict[int, float] = field(default_factory=dict)
    concordant: Dict[int, float] = field(default_factory=dict)
    discordant: Dict[int, float] = field(default_factory=dict)
    source: Dict[int, float] = field(default_factory=dict)

    @staticmethod
    def from_graph(self, bp_graph: BreakpointGraph):
        return EdgeToCN(
            sequence={i: edge[-1] for i, edge in enumerate(bp_graph.sequence_edges)},
            concordant={i: edge[-1] for i, edge in enumerate(bp_graph.concordant_edges)},
            discordant={i: edge[-1] for i, edge in enumerate(bp_graph.discordant_edges)},
            source={i: edge[-1] for i, edge in enumerate(bp_graph.source_edges)},
        )

@dataclass
class ParsedLPSolution:
    total_weights_included: float = 0.0
    cycles: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    cycle_weights: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    path_constraints_satisfied: List[List[Any]] = field(default_factory=lambda: [[], []])  # cycles, paths
    path_constraints_satisfied_set: Set[int] = field(default_factory=set)

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


def get_solver(solver_name: str, num_threads: int, time_limit_s: int) -> pyomo.solvers.plugins.solvers:
    if solver_name == "gurobi":
        solver = pyo.SolverFactory(solver_name, solver_io="python")
        if num_threads > 0:
            solver.options["threads"] = num_threads
        solver.options["NonConvex"] = 2
        solver.options["timelimit"] = time_limit_s
    else:
        solver = pyo.SolverFactory(solver_name)
    return solver