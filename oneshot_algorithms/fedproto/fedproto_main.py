from oneshot_algorithms.utils import prepare_checkpoint_dir, compute_local_model_variance, test_acc, local_training, visualize_pic
from dataset_helper import get_supervised_transform, get_client_dataloader
from models_lib import get_train_models
from common_libs import *

from oneshot_algorithms.fedproto.fedproto_eval import eval_with_proto

def parameter_averaging(local_models, weights):
    assert len(local_models) == len(weights)
    
    global_model = copy.deepcopy(local_models[0])
    global_model_param = parameters_to_vector(global_model.parameters()) * weights[0]
    
    for i in range(1, len(local_models)):
        local_model_param = parameters_to_vector(local_models[i].parameters())
        global_model_param += weights[i] * local_model_param
        
    vector_to_parameters(global_model_param, global_model.parameters())
    
    return global_model
    

# ensemble
# Key component of Ensemble    
class WEnsembleProto(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleProto, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x):
        protos_total = 0
        for model, weight in zip(self.models, self.weight_list):
            origin_proto = model.encoder(x)
            proto = origin_proto * weight    
            protos_total += proto
        return protos_total



def protos_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])

    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()

    return proto_clusters

def compute_protos_distance(num_classes, device, avg_protos):
    # calculate class-wise minimum distance
    gap = torch.ones(num_classes, device=device) * 1e9
    for k1 in avg_protos.keys():
        for k2 in avg_protos.keys():
            if k1 > k2:
                dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                gap[k1] = torch.min(gap[k1], dis)
                gap[k2] = torch.min(gap[k2], dis)
    min_gap = torch.min(gap)
    for i in range(len(gap)):
        if gap[i] > torch.tensor(1e8, device=device):
            gap[i] = min_gap
    max_gap = torch.max(gap)
    logger.info(f'class-wise minimum distance {gap.tolist()}, min_gap {min_gap}, max_gap {max_gap}')

def agg_protos(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


def collect_protos(model, trainloader, device):
    model.eval()
    protos = defaultdict(list)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(trainloader):
            if type(data) == type([]):
                data[0] = data[0].to(device)
            else:
                data = data.to(device)
            target = target.to(device) 

            rep = model.encoder(data)

            for i, yy in enumerate(target):
                y_c = yy.item()
                protos[y_c].append(rep[i, :].detach().data)

    local_protos = agg_protos(protos)

    return local_protos  

def OneshotFedProto(trainset, test_loader, client_idx_map, config, device):
    logger.info('OneshotFedProto')
    
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='supervised',
        in_channel=config['dataset'].get('channels', 3)
    )

    global_model.to(device)
    global_model.train()

    method_results = defaultdict(list)
    save_path, _ = prepare_checkpoint_dir(config)
    # save the config to file
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    # all clients perform local training phrase, the global model is used as the initial model
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    # get the weight of each client
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    # weight
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    # visualization
    supervised_transformer = get_supervised_transform(config['dataset']['data_name'])
    vis_loader = torch.utils.data.DataLoader(copy.deepcopy(test_loader.dataset), config['visualization']['vis_size'], shuffle=False)
    vis_iter = iter(vis_loader)
    vis_data, vis_label = next(vis_iter)
    vis_data = supervised_transformer(vis_data)
    vis_folder = config['visualization']['save_path'] +"/oneshotfedproto/"
    
    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):

            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
            
            
            # local training
            local_model_c = local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=0,
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                history_acc_list=[],
                best_acc=-1,
                best_epoch=-1,
                client_model_dir=None,
                device=device,
                num_classes=config['dataset']['num_classes'],
                save_freq=config['checkpoint']['save_freq']
            )
            
            local_models[c] = local_model_c
            
            logger.info(f"Client {c} Finish Local Training--------|")

            # collecting the local prototypes
            local_proto_c = collect_protos(copy.deepcopy(local_model_c), client_dataloader, device)
            local_protos.append(local_proto_c)
            logger.info(f"Client {c} Collecting Local Prototypes--------|")

            # visualize_pic(local_model_c.encoder, vis_data, target_layers=[local_model_c.encoder.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/local_model_{c}.png', device=device)
                
            # logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/local_model_{c}.png")



        logger.info(f"Round {cr} Finish--------|")
        # local training is finished, start to aggregate the local models, one-shot aggregation and test
    
        # some statistics of the current local models
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")
        

        # OneshotFedProto
        method_name = 'OneShotFedProto'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)

        # visualize_pic(aggregated_model.encoder, vis_data, target_layers=[aggregated_model.encoder.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/global_model.png', device=device)         
        # logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/global_model.png")

        # proto eval
        global_proto = protos_cluster(local_protos)
        compute_protos_distance(config['dataset']['num_classes'], device, global_proto)
        proto_acc = eval_with_proto(copy.deepcopy(aggregated_model), global_proto, test_loader, config['dataset']['num_classes'], device)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {proto_acc}")
        method_results[method_name+"_P"].append(proto_acc)

        method_name = 'OneShotFedProto+Ensemble'
        ensemble_model = WEnsembleProto(model_list=local_models, weight_list=weights)
        # acc = test_acc(ensemble_model, test_loader, device)
        # logger.info(f"The test accuracy of {method_name}: {acc}")
        # method_results[method_name].append(acc)

        # proto eval

        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), global_proto, test_loader, config['dataset']['num_classes'], device, mode='ensemble')
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name+"_P"].append(ens_proto_acc)


        method_name = 'OneShotFedProto'
        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)












            


         



