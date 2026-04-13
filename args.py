import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfp', type=str, required=True, help='Path to the experiment configuration file (YAML)')
    parser.add_argument('--algo', type=str, required=True, help='Name of the algorithm', choices=['FedAvg', 'Ensemble', 'OTFusion', 'FedProto', 'FedETF', 'OursV1', 'OursV2', 'OursV3', 'OursV4', 'OursV5', 'OursV6', 'OursV7', 'OursV8', 'OursV4IFFI', 'OursV7IFFI','OursV6IFFI','OursV5IFFI','OursV4SIMPLE','OursV7SIMPLE','OursV9', 'OursV10','OursV11','OursV12','OursV13','OursV14', 'OursV15', 'OursV16', 'OursV17', 'OursV18', 'OursV19', 'OursV20', 'OursV21', 'OursV22', 'OursV23', 'OursV24', 'OursV25', 'OursV24FedAvg', 'OursV24Projector', 'Ablation_AugCEAugSupCon', 'Ablation_CEOnly', 'Ablation_SupConOnly', 'Ablation_AugCEAugSupCon', 'FAFIFedAvg', 'AURORAFedAvg', 'OursV15FedAvg','Ours_FeatureCollapse_Ablation'])
    parser.add_argument('--lambdaval', type=float, default=0, help='Alignment loss weight')
    parser.add_argument('--annealing_strategy', type=str, default='none', help='Select the annealing strategy for the alignment loss weight (lambda).')
    parser.add_argument('--gamma_reg', type=float, default=1e-5, help='Regularization factor for the alignment loss weight sigma_sq_align in V12.')
    parser.add_argument('--lambda_max', type=float, default=50.0, help='Maximum threshold for effective lambda in stability regularization (OursV13).')

    args = parser.parse_args()

    return args

