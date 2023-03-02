"""
Tests for BreakpointGraph class.
"""
import unittest

from long_read_aa.breakpoints import (
    BreakpointEdge,
    BreakpointGraph,
    SequenceEdge,
)


class TestBreakpointGraph(unittest.TestCase):
    def setUp(self):
        self.breakpoint1 = BreakpointEdge.BreakpointEdge(
            "chr1", 2000, "+", "chr2", 4000, "-", annotation="discordant"
        )
        self.breakpoint2 = BreakpointEdge.BreakpointEdge(
            "chr10", 1201, "-", "chr10", 20000, "+", annotation="concordant"
        )
        self.breakpoint3 = BreakpointEdge.BreakpointEdge(
            "chr1", 2000, "+", "chr8", 6000, "-", annotation="discordant"
        )

        self.sequence_edge = SequenceEdge.SequenceEdge("chr1", 10000, 120000)

    def test_breakpoint_edge_constructor(self):
        """Test BreakpointEdge constructor."""

        self.assertEqual(self.breakpoint1.chromosome1, "chr1")
        self.assertEqual(self.breakpoint1.position1, 2000)
        self.assertEqual(self.breakpoint1.orientation1, "+")
        self.assertEqual(self.breakpoint1.chromosome2, "chr2")
        self.assertEqual(self.breakpoint1.position2, 4000)
        self.assertEqual(self.breakpoint1.orientation2, "-")
        self.assertEqual(self.breakpoint1.support, None)
        self.assertEqual(self.breakpoint1.copy_number, None)
        self.assertEqual(self.breakpoint1.annotation, "discordant")

        self.assertEqual(self.breakpoint1.left_breakpoint, ("chr1", 2000, "+"))
        self.assertEqual(self.breakpoint1.right_breakpoint, ("chr2", 4000, "-"))

        # test error catches
        self.assertRaises(
            Exception,
            BreakpointEdge.BreakpointEdge,
            "chr1",
            1,
            "+",
            "chr2",
            2,
            "-",
            None,
            -1,
            0,
        )
        self.assertRaises(
            Exception,
            BreakpointEdge.BreakpointEdge,
            "chr1",
            1,
            "+",
            "chr2",
            2,
            "-",
            None,
            0,
            -1,
        )

        # test print
        self.assertEqual(
            str(self.breakpoint1), "('chr1', 2000, '+') -> ('chr2', 4000, '-')"
        )

    def test_sequence_edge_constructor(self):
        """Test SequenceEdge constructor."""

        self.assertEqual(self.sequence_edge.chromosome, "chr1")
        self.assertEqual(self.sequence_edge.start, 10000)
        self.assertEqual(self.sequence_edge.end, 120000)
        self.assertEqual(self.sequence_edge.length, 120000 - 10000 + 1)

        # test error statements
        self.assertRaises(
            Exception, SequenceEdge.SequenceEdge, "chr1", 1000, 500
        )
        self.assertRaises(
            Exception,
            SequenceEdge.SequenceEdge,
            "chr1",
            1000,
            2000,
            {"long_read": -1},
            {"long_read": 0},
            100,
        )
        self.assertRaises(
            Exception,
            SequenceEdge.SequenceEdge,
            "chr1",
            1000,
            2000,
            {"long_read": 0},
            {"long_read": -1},
            100,
        )

        self.assertEqual(str(self.sequence_edge), "chr1:10000-120000")

    def test_breakpoint_graph_constructor(self):
        """Test BreakpointGraph constructor."""
        breakpoint_graph = BreakpointGraph.BreakpointGraph(
            sequence_edges=[self.sequence_edge],
            discordant_edges=[self.breakpoint1],
            concordant_edges=[self.breakpoint2],
        )

        self.assertEqual(breakpoint_graph.discordant_edges, [self.breakpoint1])
        self.assertEqual(breakpoint_graph.concordant_edges, [self.breakpoint2])
        self.assertEqual(len(breakpoint_graph.source_edges), 0)
        self.assertEqual(breakpoint_graph.sequence_edges, [self.sequence_edge])

        expected_nodes = [
            self.sequence_edge.left_breakpoint,
            self.sequence_edge.right_breakpoint,
            self.breakpoint1.left_breakpoint,
            self.breakpoint1.right_breakpoint,
            self.breakpoint2.left_breakpoint,
            self.breakpoint2.right_breakpoint,
        ]
        for node in expected_nodes:
            self.assertTrue(node in breakpoint_graph)

    def test_add_edge(self):
        breakpoint_graph = BreakpointGraph.BreakpointGraph(
            sequence_edges=[self.sequence_edge],
            discordant_edges=[],
            concordant_edges=[self.breakpoint2],
        )

        # test edge addition
        breakpoint_graph.add_discordant_edge(self.breakpoint1)
        self.assertTrue(("chr1", 2000, "+") in breakpoint_graph.nodes)

    def test_breakpoint_graph_adjacency_list(self):
        breakpoint_graph = BreakpointGraph.BreakpointGraph(
            sequence_edges=[self.sequence_edge],
            discordant_edges=[self.breakpoint1],
            concordant_edges=[self.breakpoint2],
        )

        expected_breakpoint_left = self.breakpoint1.left_breakpoint
        expected_breakpoint_right = self.breakpoint1.right_breakpoint

        self.assertEqual(
            len(breakpoint_graph.get_edges(("chr1", 2000, "+"))), 1
        )

        observed_breakpoint = breakpoint_graph.get_edges(("chr1", 2000, "+"))[0]

        self.assertEqual(
            observed_breakpoint.left_breakpoint, expected_breakpoint_left
        )
        self.assertEqual(
            observed_breakpoint.right_breakpoint, expected_breakpoint_right
        )

        # test errors
        self.assertRaises(Exception, breakpoint_graph.add_node, "chr1")

        self.assertRaises(
            Exception, breakpoint_graph.get_edges, ("chr1", 1111, "+")
        )

        self.assertRaises(Exception, breakpoint_graph.get_edges, 1111)

    def test_remove_edge(self):
        breakpoint_graph = BreakpointGraph.BreakpointGraph(
            sequence_edges=[self.sequence_edge],
            discordant_edges=[self.breakpoint1, self.breakpoint3],
            concordant_edges=[self.breakpoint2],
        )

        self.assertTrue(self.breakpoint1 in breakpoint_graph.discordant_edges)

        breakpoint_graph.remove_edge(self.breakpoint1)

        # ensure edge does not exist anymore
        self.assertFalse(self.breakpoint1 in breakpoint_graph.discordant_edges)
        self.assertEqual(
            len(breakpoint_graph.get_edges(self.breakpoint1.left_breakpoint)), 1
        )

    def test_and_merge_edge(self):
        
        seq_edge1 = SequenceEdge.SequenceEdge('chr1', 1000, 2000, {'lib1': 10}, {'lib1': 2}, 1000)
        seq_edge2 = SequenceEdge.SequenceEdge('chr1', 2000, 7000, {'lib1': 20}, {'lib1': 4}, 4030)
        breakpoint_edge = BreakpointEdge.BreakpointEdge("chr1", 2000, "+", "chr2", 4000, "-", annotation="discordant")

        breakpoint_graph = BreakpointGraph.BreakpointGraph(
            sequence_edges=[seq_edge1, seq_edge2],
            discordant_edges=[breakpoint_edge],
            concordant_edges=[],
        )

        print(breakpoint_graph.get_edges(('chr1', 1000, "+")))

        self.assertTrue(breakpoint_edge in breakpoint_graph.discordant_edges)
        self.assertTrue(('chr1', 2000, '+') in breakpoint_graph.nodes)

        breakpoint_graph.remove_edge(breakpoint_edge)

        for node in breakpoint_graph.nodes:
            print(node)

        # ccheck breakpoint edges and sequenece edges were removed
        self.assertFalse(breakpoint_edge in breakpoint_graph.discordant_edges)
        self.assertFalse(seq_edge1 in breakpoint_graph.sequence_edges)
        self.assertFalse(seq_edge2 in breakpoint_graph.sequence_edges)
        self.assertFalse(('chr1', 2000, "+") in breakpoint_graph.nodes)

        # check nodes that should be there 
        self.assertTrue(('chr1', 1000, "+") in breakpoint_graph.nodes)
        self.assertTrue(('chr1', 7000, "+") in breakpoint_graph.nodes)

        # ensure merge was done successfully
        new_sequence_edge = breakpoint_graph.sequence_edges[0]
        self.assertEqual(new_sequence_edge.start, 1000)
        self.assertEqual(new_sequence_edge.end, 7000)
        self.assertEqual(new_sequence_edge.support['lib1'], 30)
        self.assertEqual(new_sequence_edge.copy_number['lib1'], 3.0)
        self.assertEqual(new_sequence_edge.average_read_length, 2515.0)

        # check new sequence edge is on expected nodes
        edges = breakpoint_graph.get_edges(('chr1', 1000, "+"))
        self.assertTrue(len(edges) == 1)
        self.assertTrue(edges[0].left_breakpoint == ('chr1', 1000, "+"))
        self.assertTrue(edges[0].right_breakpoint == ('chr1', 7000, "+"))

        edges = breakpoint_graph.get_edges(('chr1', 7000, "+"))
        self.assertTrue(len(edges) == 1)
        self.assertTrue(edges[0].left_breakpoint == ('chr1', 1000, "+"))
        self.assertTrue(edges[0].right_breakpoint == ('chr1', 7000, "+"))


if __name__ == "__main__":
    unittest.main()
