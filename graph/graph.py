class Node(object):
    """ Vertex data structure
    """
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return "node_" + str(self.content)

    def __repr__(self):
        return "node_" + str(self.content)

class Edge(object):
    """ Edge data structure
    """
    def __init__(self, start, end, weight=1):
        self.start = start
        self.end = end
        self.weight = weight

    def __str__(self):
        return "edge_%s_%s_%s" %(str(self.start), str(self.end), str(self.weight))

    def __repr__(self):
        return "edge_%s_%s_%s" %(str(self.start), str(self.end), str(self.weight))

class Graph(object):
    """ Graph data structure
    """
    def __init__(self, edges=list()):
        """ Construct graph from edges
        A graph is stored by ajacency list, which is implemented by dict.
        """
        self.adj = dict() # adjacency list
        for edg in edges:
            if issubclass(type(edg), Edge):
                # edges represented in Edge
                nw = {edg.end : edg.weight}
                ver = edg.start # vertes
            elif type(edg) == tuple:
                # edges represented in tuple
                if len(edg) == 3:
                    nw = {edg[1] : edg[2]}
                elif len(edg) == 2:
                    nw = {edg[1] : 1} # default weight 1
                else:
                    continue
                ver = edg[0] # vertex
            else:
                continue # unsupported items

            if ver not in self.adj.keys():
                self.adj[ver] = [nw]
            else:
                if nw not in self.adj[ver]:
                    self.adj[ver].append(nw)

    def __str__(self):
        ocnt = 0
        g = '{'
        for node in self.adj.keys():
            g += (str(node) + ' : [')
            icnt = 0
            for d in self.adj[node]:
                if icnt > 0:
                    g += ', '
                g += (str(d))
                icnt += 1
            if ocnt > 0:
                g += '], '
            ocnt += 1
        g += '}'

        return g

    def __repr__(self):
        return "graph_" + str(hash(self))

    def add_edge(self, edge):
        pass # TODO

    def del_edge(self, edge):
        pass # TODO

    def add_node(self, node):
        pass # TODO

    def del_node(self, node):
        pass # TODO

    def bfs(self, start, goal):
        """ Breadth first search
        """
        pass # TODO
