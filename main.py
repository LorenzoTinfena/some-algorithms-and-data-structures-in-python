from queue import Queue
from queue import LifoQueue  # is a stack
import sys


class Node:
    def __init__(self, id, next: 'list[Node] | list[WeightedGraph.Edge]' = []):
        assert next is not None
        self.id = id
        self.next = next

    def __str__(self):
        return str(self.id)


class Tree:
    def print_bfs(root: Node): Tree._print(root, data_structure=Queue)
    def print_dfs(root: Node): Tree._print(root, data_structure=LifoQueue)

    def search_bfs(root: Node, id): return Tree._search_by_id(
        root, data_structure=Queue, id=id)

    def search_dfs(root: Node, id): return Tree._search_by_id(
        root, data_structure=LifoQueue, id=id)

    def _print(root: Node, data_structure):
        # create buffer
        buff = data_structure()
        # add the first element, the root
        buff.put(root)
        # while I have nodes to discover
        while not buff.empty():
            # get a node
            current_node: Node = buff.get()
            # make it some use
            print(current_node)
            # put next nodes in buffer
            for node in current_node.next:
                buff.put(node)

    def _search_by_id(root: Node, data_structure, id):
        # create buffer
        buff = data_structure()
        # add the first element, the root
        buff.put(root)
        # while true, cause is assumed that a node with that id exist in this tree
        while True:
            # get a node
            current_node: Node = buff.get()
            # make it some use
            if current_node.id == id:
                return current_node
            # put next nodes in buffer
            for node in current_node.next:
                buff.put(node)


class Graph:  # not weighed
    def print_bfs(root: Node, all_nodes): Graph._print(
        root, data_structure=Queue, all_nodes=all_nodes)

    def print_dfs(root: Node, all_nodes): Graph._print(
        root, data_structure=LifoQueue, all_nodes=all_nodes)

    def search_bfs(root: Node, id, all_nodes): return Graph._search_by_id(
        root, data_structure=Queue, id=id, all_nodes=all_nodes)

    def search_dfs(root: Node, id, all_nodes): return Graph._search_by_id(
        root, data_structure=LifoQueue, id=id, all_nodes=all_nodes)

    def _print(root: Node, data_structure, all_nodes):
        for node in all_nodes:
            node.discovered = False
        # create buffer
        buff = data_structure()
        # make root discovered
        root.discovered = True
        # add the first element, the root
        buff.put(root)
        # while I have nodes to discover
        while not buff.empty():
            # get a node
            current_node: Node = buff.get()
            # make it some use
            print(current_node)
            # put next nodes in buffer
            for node in current_node.next:
                if not node.discovered:
                    node.discovered = True
                    buff.put(node)

    def _search_by_id(root: Node, data_structure, id, all_nodes):
        for node in all_nodes:
            node.discovered = False
        # create buffer
        buff = data_structure()
        # make root discovered
        root.discovered = True
        # add the first element, the root
        buff.put(root)
        # while true, cause is assumed that a node with that id exist in this tree
        while True:
            # get a node
            current_node: Node = buff.get()
            # make it some use
            if current_node.id == id:
                return current_node
            # put next nodes in buffer
            for node in current_node.next:
                if not node.discovered:
                    node.discovered = True
                    buff.put(node)


class WeighedGraph:
    # different implementation, I don't need to interface to nodes directly, but only with their ids
    class Edge:
        def __init__(self, next_node, cost: 'uint'):
            self.cost = cost
            self.next_node = next_node

    def __init__(self):
        self.nodes = []

    def add_node(self, id):
        self.nodes.append(Node(id, []))

    def add_edge(self, from_node_id, to_node_id, cost):
        self._find(from_node_id).next.append(
            WeighedGraph.Edge(self._find(to_node_id), cost))

    def _find(self, id):
        for node in self.nodes:
            if node.id == id:
                return node

    def dijkstra(self, start_node_id, end_node_id):
        # is assumed that the end node is reachable
        if start_node_id == end_node_id:
            return [start_node_id], 0
        for node in self.nodes:
            node.discovered = False
            node.cost = sys.maxsize
            node.prev = None
        start_node = self._find(start_node_id)
        end_node = self._find(end_node_id)
        start_node.discovered = True
        start_node.cost = 0
        for edge in start_node.next:
            edge.next_node.cost = edge.cost
            edge.next_node.prev = start_node
        while True:
            min_cost = sys.maxsize
            for node in self.nodes:
                if not node.discovered and node.cost < min_cost:
                    min_cost = node.cost
                    best_node = node
            if best_node == end_node:
                # return the complete path going back to the start node
                path = []
                cost = best_node.cost
                while best_node.prev != None:
                    path.append(best_node.id)
                    best_node = best_node.prev
                path.append(best_node.id)
                return list(reversed(path)), cost
            best_node.discovered = True
            for edge in best_node.next:
                # update adjacent nodes
                if best_node.cost + edge.cost < edge.next_node.cost:
                    edge.next_node.cost = best_node.cost + edge.cost
                    edge.next_node.prev = best_node


def main():
    # tree algorithms
    root = Node(0, [Node(1, [Node(3), Node(4), Node(5)]),
                Node(2, [Node(6, [Node(7)])])])

    print('print all nodes with breadth first search')
    Tree.print_bfs(root)
    print('print all nodes with deep first search')
    Tree.print_dfs(root)
    print('Search for node with id with BFS')
    print(Tree.search_bfs(root, 6))
    print('Search for node with id with DFS')
    print(Tree.search_dfs(root, 6))

    # graph algorithms
    # you can use bfs in a not weighted graph in order to get the shortest path from a to b, and if from b to a, you can make the search reversed
    # you can use dfs in a not weighted graph in order to solve mazes, is like the rule "always go left/right"

    nodes = []
    nodes.append(Node(0))
    nodes.append(Node(1))
    nodes.append(Node(2))
    nodes.append(Node(3))
    nodes.append(Node(4))
    nodes.append(Node(5))
    nodes.append(Node(6))
    nodes[0].next = [nodes[1], nodes[2]]
    nodes[1].next = [nodes[0], nodes[3]]
    nodes[2].next = [nodes[1], nodes[4]]
    nodes[3].next = [nodes[4], nodes[5]]
    nodes[4].next = [nodes[2], nodes[3]]
    nodes[5].next = []
    nodes[6].next = [nodes[3]]

    print('print all nodes with breadth first search')
    Graph.print_bfs(nodes[0], nodes)
    print('print all nodes with deep first search')
    Graph.print_dfs(nodes[0], nodes)
    print('Search for node with id with BFS')
    print(Graph.search_bfs(nodes[0], 3, nodes))
    print('Search for node with id with DFS')
    print(Graph.search_dfs(nodes[0], 3, nodes))

    # dijkstra in a weighed graph
    w_graph = WeighedGraph()
    for i in range(7):
        w_graph.add_node(i)
    w_graph.add_edge(0, 1, 1)
    w_graph.add_edge(0, 2, 2)
    w_graph.add_edge(1, 0, 1)
    w_graph.add_edge(1, 3, 0)
    w_graph.add_edge(2, 1, 1)
    w_graph.add_edge(2, 4, 3)
    w_graph.add_edge(3, 4, 2)
    w_graph.add_edge(3, 5, 4)
    w_graph.add_edge(4, 2, 3)
    w_graph.add_edge(4, 3, 2)
    w_graph.add_edge(6, 3, 3)
    print('Best path with dijkstra in weighed graph')
    path, total_cost = w_graph.dijkstra(0, 4)
    print(path)
    print(f'total cost: {total_cost}')


if __name__ == '__main__':
    main()

'''
bfs and dfs in tree, print all nodes, and search for a node
and for a not weighed graph
dijkstra for weighed graph
'''
