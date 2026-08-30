import scipy.sparse as sp
from models.base_op import GraphOp
from config import args
from utils import adj_to_symmetric_norm


class LaplacianGraphOp(GraphOp):
    def __init__(self, prop_steps, r=0.5):
        super(LaplacianGraphOp, self).__init__(prop_steps)
        self.r = r

    def construct_adj(self, adj):
        if isinstance(adj, sp.csr_matrix):
            adj = adj.tocoo()
        elif not isinstance(adj, sp.coo_matrix):
            raise TypeError("The adjacency matrix must be a scipy.sparse.coo_matrix/csr_matrix!")

        adj_normalized = adj_to_symmetric_norm(adj, self.r)


        return adj_normalized.tocsr()