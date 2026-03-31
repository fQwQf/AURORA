from dataset_helper import get_fl_dataset

from models_lib.models import get_model

from models_lib import get_train_models
from args import args_parser

from common_libs import *

from oneshot_algorithms import *

config_args = args_parser()
config = load_yaml_config(config_args.cfp)
logger.info(f"config: {config}")

setup_seed(config['seed'])


# Determine if we should normalize on CPU (for standard algos) or GPU (for Ours)
normalize_train = not (config_args.algo.startswith('Ours') or config_args.algo in ['FAFIFedAvg', 'AURORAFedAvg'])
normalize_test = True

trainset, testset, client_idx_map = get_fl_dataset(
    config["dataset"]["data_name"], 
    config["dataset"]["root_path"], 
    config['client']['num_clients'], 
    config['distribution']['type'], 
    config['distribution']['label_num_per_client'], 
    config['distribution']['alpha'],
    normalize_train=normalize_train,
    normalize_test=normalize_test,
    max_samples_per_client=config['distribution'].get('max_samples_per_client', None))

test_loader = torch.utils.data.DataLoader(testset, batch_size=config['dataset']['test_batch_size'], shuffle=True)

# If you want to use the FedAvg model, uncomment the following lines
# global_model = get_model(
#     model_name=config['server']['model_name'],
#     num_classes=config['dataset']['num_classes'],
#     channels=config['dataset']['channels'],
# )

global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )

device = config['device']

if config_args.algo == 'FedAvg':
    OneshotFedAvg(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'Ensemble':
    OneshotEnsemble(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OTFusion':
    OTFusion(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'FedProto':
    OneshotFedProto(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'FedETF':
    OneshotFedETF(trainset, test_loader, client_idx_map, config, device)
# elif config_args.algo == 'OursV1':
#     FedBCD(trainset, test_loader, client_idx_map, config, global_model, device)
# elif config_args.algo == 'OursV2':
#     FedBCD2(trainset, test_loader, client_idx_map, config, global_model, device)
# elif config_args.algo == 'OursV3':
#     FedBCD3(trainset, test_loader, client_idx_map, config, global_model, device)
elif config_args.algo == 'OursV4':
    OneshotOurs(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV5':
    OneshotOursV5(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV6':
    OneshotOursV6(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV7':
    OneshotOursV7(trainset, test_loader, client_idx_map, config, device, lambda_val=config_args.lambdaval)
elif config_args.algo == 'OursV8':
    OneshotOursV8(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV9':
    OneshotOursV9(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV10':
    OneshotOursV10(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV11':
    OneshotOursV11(trainset, test_loader, client_idx_map, config, device,annealing_strategy=config_args.annealing_strategy)
elif config_args.algo == 'OursV12':
    OneshotOursV12(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OursV13':
    OneshotOursV13(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV14':
    OneshotOursV14(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV6IFFI':
    OneshotOursV6(trainset, test_loader, client_idx_map, config, device, use_simple_server=False)
elif config_args.algo == 'OursV4IFFI':
    OneshotOurs(trainset, test_loader, client_idx_map, config, device, server_strategy='advanced_iffi')
elif config_args.algo == 'OursV7IFFI':
    OneshotOursV7(trainset, test_loader, client_idx_map, config, device, server_strategy='advanced_iffi')
elif config_args.algo == 'OursV5IFFI':
    OneshotOursV5(trainset, test_loader, client_idx_map, config, device, use_simple_server=False)
elif config_args.algo == 'OursV4SIMPLE':
    OneshotOurs(trainset, test_loader, client_idx_map, config, device, server_strategy='true_simple_output')
elif config_args.algo == 'OursV7SIMPLE':
    OneshotOursV7(trainset, test_loader, client_idx_map, config, device, server_strategy='true_simple_output')
elif config_args.algo == 'FAFIFedAvg':
    OneshotFAFIFedAvg(trainset, test_loader, client_idx_map, config, device, lambda_val=config_args.lambdaval)
elif config_args.algo == 'AURORAFedAvg':
    OneshotAURORAFedAvg(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV15':
    OneshotOursV15(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV15FedAvg':
    OneshotOursV15(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV16':
    OneshotOursV16(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV17':
    OneshotOursV17(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV18':
    OneshotOursV18(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'OursV19':
    OneshotOursV19(trainset, test_loader, client_idx_map, config, device, gamma_reg=config_args.gamma_reg, lambda_max=config_args.lambda_max)
elif config_args.algo == 'Ours_FeatureCollapse_Ablation':
    OneshotOurs_FeatureCollapse_Ablation(trainset, test_loader, client_idx_map, config, device, lambda_val=config_args.lambdaval)
else:
    raise NotImplementedError(f"Algorithm {config_args.algo} is not implemented.")   




