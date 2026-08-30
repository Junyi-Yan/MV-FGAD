from config import args
from models.gat import GAT
from models.gcn import ChebNet, GCN, GraphTransformer


def build_model(model_name, input_dim, hidden_dim, output_dim):
    common_kwargs = {"feat_dim": input_dim,
        "hidden_dim": hidden_dim,
        "output_dim": output_dim,
        "dropout": args.drop}

    if model_name == "GCN":
        return GCN(**common_kwargs)

    if model_name == "GraphTransformer":
        return GraphTransformer(**common_kwargs)

    if model_name == "GAT":
        return GAT(**common_kwargs)

    if model_name == "ChebNet":
        return ChebNet(**common_kwargs,
            bn=False,
            ln=False)

    supported_models = ["GCN","GraphTransformer","GAT","ChebNet"]

    raise ValueError(
        f"Unsupported model_name: {model_name}. "
        f"Choose from: {', '.join(supported_models)}")


class ClientsManager:

    def __init__(self,model_name,datasets,num_clients,device,eval_single_client=False):
        self.model_name = model_name
        self.input_dim = datasets.input_dim
        self.output_dim = datasets.output_dim
        self.hidden_dim = args.hidden_dim
        self.subgraphs = datasets.subgraphs
        self.device = device
        self.num_clients = num_clients

        self.clients = self._initialize_clients()
        self.tot_nodes = sum(
            client.num_nodes
            for client in self.clients)

        if eval_single_client:
            raise NotImplementedError("eval_single_client=True is not supported because "
                "evaluate_data_isolate() is not implemented.")

    def _initialize_clients(self):
        return [Client(model_name=self.model_name,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                client_id=client_id,
                local_subgraph=self.subgraphs[client_id],
                hidden_dim=self.hidden_dim)
            for client_id in range(self.num_clients)]


class Client:
    """A federated client containing a local graph and local model."""

    def __init__(self,model_name,input_dim,output_dim,client_id,local_subgraph,hidden_dim):
        self.model_name = model_name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.client_id = client_id
        self.local_subgraph = local_subgraph
        self.num_nodes = local_subgraph.num_nodes
        self.hidden_dim = hidden_dim

        self.init_model()

    def init_model(self):
        self.model = build_model(model_name=self.model_name,
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim)

    def set_state_dict(self, model):
        """Synchronize the local model with the global model."""
        self.model.load_state_dict(model.state_dict())