from common_libs import *
from oneshot_algorithms.utils import prepare_checkpoint_dir, local_training, test_acc, compute_local_model_variance
from dataset_helper import get_client_dataloader
from models_lib import get_train_models

# Key component of Ensemble    
class WEnsemble(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsemble, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x):
        logits_total = 0
        for model, weight in zip(self.models, self.weight_list):
            logit = model(x) * weight
            logits_total += logit
        return logits_total

class EnsembleFeature(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(EnsembleFeature, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x):
        feature_total = 0
        for model, weight in zip(self.models, self.weight_list):
            feature = model.encoder(x) * weight
            feature_total += feature

        return feature_total

# Key function of OneshotFedAvg
def parameter_averaging(local_models, weights):
    assert len(local_models) == len(weights)
    
    global_model = copy.deepcopy(local_models[0])
    global_model_param = parameters_to_vector(global_model.parameters()) * weights[0]
    
    for i in range(1, len(local_models)):
        local_model_param = parameters_to_vector(local_models[i].parameters())
        global_model_param += weights[i] * local_model_param
        
    vector_to_parameters(global_model_param, global_model.parameters())
    
    return global_model


def OneshotEnsemble(trainset, test_loader, client_idx_map, config, device):
    logger.info('OneshotEnsemble')
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
    
    
    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
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

        logger.info(f"Round {cr} Finish--------|")
        # local training is finished, start to aggregate the local models, one-shot aggregation and test
    
        # some statistics of the current local models
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")
        

        # Ensemble
        method_name = 'Ensemble'
        aggregated_model = WEnsemble(model_list=copy.deepcopy(local_models), weight_list=weights)
        acc = test_acc(aggregated_model, test_loader, device)    
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)

        # ensemble feature
        method_name = 'Ensemble_Feature'
    
        cls_list = [copy.deepcopy(m.fc) for m in local_models]
        g_cls = parameter_averaging(cls_list, weights)
        ens_feature_model = EnsembleFeature(local_models, weights)

        g_model = torch.nn.Sequential(ens_feature_model, g_cls)

        ens_f_acc = test_acc(g_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {ens_f_acc}")
        method_results[method_name].append(ens_f_acc)        

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)