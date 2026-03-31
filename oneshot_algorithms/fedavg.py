from oneshot_algorithms.utils import prepare_checkpoint_dir, local_training, test_acc, compute_local_model_variance, visualize_pic
from dataset_helper import get_client_dataloader, get_supervised_transform

from common_libs import *
from models_lib import get_train_models

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

def OneshotFedAvg(trainset, test_loader, client_idx_map, config, device):
    logger.info('OneshotFedAvg')
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
    save_path, local_model_dir = prepare_checkpoint_dir(config)
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
    vis_folder = config['visualization']['save_path'] +"/oneshotfedavg/"
    
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
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                history_acc_list=[],
                best_acc=-1,
                best_epoch=-1,
                client_model_dir=local_model_dir + f"/client_{c}",
                device=device,
                num_classes=config['dataset']['num_classes'],
                save_freq=config['checkpoint']['save_freq'],
                
            )
            
            local_models[c] = local_model_c


            logger.info(f"Client {c} Finish Local Training--------|")

            # visualize_pic(local_model_c.encoder, vis_data, target_layers=[local_model_c.encoder.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/local_model_{c}.png', device=device)
                
            logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/local_model_{c}.png")



        logger.info(f"Round {cr} Finish--------|")
        # local training is finished, start to aggregate the local models, one-shot aggregation and test
    
        # some statistics of the current local models
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")
        

        # OneshotFedAvg
        method_name = 'OneShotFedAvg'
        aggregated_model = parameter_averaging(local_models, weights)
        acc = test_acc(aggregated_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")
        method_results[method_name].append(acc)

        # visualize_pic(aggregated_model.encoder, vis_data, target_layers=[aggregated_model.encoder.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/global_model.png', device=device)
            
        logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/global_model.png")


        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)