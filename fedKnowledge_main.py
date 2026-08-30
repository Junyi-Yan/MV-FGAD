import torch
from config import args
from datasets.graph_fl_dataset import GraphFLDataset
from roles.client import ClientsManager
from roles.server import ServerManager


BYTES_PER_FLOAT32 = 4
NORMALIZE_TRAINS = 1


def get_device():
    if torch.cuda.is_available():
        return torch.device(f"cuda:{args.gpu_id}")

    return torch.device("cpu")


def estimate_communication_cost(model, num_clients):
    total_params = sum(parameter.numel() for parameter in model.parameters())
    single_model_mb = total_params * BYTES_PER_FLOAT32 / (1024 ** 2)
    total_round_mb = 2 * single_model_mb * num_clients

    return single_model_mb, total_round_mb


def step1_main():
    device = get_device()

    datasets = GraphFLDataset(root="./datasets",name=args.data_name,sampling=args.partition,num_clients=args.num_clients,analysis_local_subgraph=False,analysis_global_graph=False)
    server = ServerManager(model_name=args.gmodel_name,datasets=datasets,num_clients=args.num_clients,device=device,num_rounds=args.num_rounds,client_sample_ratio=1)
    client_manager = ClientsManager(model_name=args.gmodel_name,datasets=datasets,num_clients=args.num_clients,device=device,eval_single_client=False)
    print(f"| ★ Data simulation: {args.partition}, "f"Client: {args.num_clients}, "f"Model name: {args.gmodel_name}")
    single_model_mb, total_round_mb = estimate_communication_cost(server.model,args.num_clients)

    print(f"| ★ Single Model Size: {single_model_mb:.3f} MB")
    print(f"| ★ Total Comm per Round "f"(All {args.num_clients} clients): {total_round_mb:.2f} MB")

    server.collaborative_training_model(client_manager.clients,datasets.name,datasets.num_clients,datasets.sampling,model_name=args.gmodel_name,normalize_trains=NORMALIZE_TRAINS)