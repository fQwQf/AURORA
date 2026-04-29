from common_libs import *

from dataset_helper import NORMALIZE_DICT
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam import GuidedBackpropReLUModel
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def save_perf_records(save_path, save_file_name, data_dict, save_mode='w'):
    header = data_dict.keys()

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    with open(os.path.join(save_path, save_file_name)+'.csv', save_mode, encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        if f.tell() == 0:
            w.writerow(header)
        w.writerows(zip(*data_dict.values()))

def save_best_local_model(save_path, model, save_name, keep_only_last=False):

    if save_path is None:
        logger.warning(f"save_best_local_model called with save_path=None for {save_name}; skipping save.")
        return

    os.makedirs(save_path, exist_ok=True)
    save_path_full = os.path.join(save_path, save_name)
    
    # If keep_only_last is True, remove previous checkpoints to save disk space
    if keep_only_last:
        # Remove previous epoch checkpoints in the same directory
        for f in os.listdir(save_path):
            if f.startswith('epoch_') and f.endswith('.pth'):
                try:
                    os.remove(os.path.join(save_path, f))
                    logger.debug(f"Removed old checkpoint: {f}")
                except OSError:
                    pass
    
    torch.save(model.state_dict(), save_path_full)
    logger.info(f"Save the model in {save_path_full}")

def load_best_local_model(save_path, save_name):
    best_model = torch.load(os.path.join(save_path, save_name), map_location='cpu')
    return best_model


# save files
def save_checkpoint(save_path, model, best_model_dict, rounds, best_acc, best_round, acc_list):
    os.makedirs(save_path, exist_ok=True)
    
    checkpoint_data = {
        'rounds': rounds,
        'best_acc': best_acc,
        'best_round': best_round
    }

    save_yaml_config(save_path + "/checkpoint.yaml", checkpoint_data)
    torch.save(model.state_dict(), save_path + "/current_model.pth")
    torch.save(best_model_dict, save_path + "/best_model.pth")
    save_perf_records(save_path=save_path, save_file_name='acc_list', data_dict={'Accuracy': acc_list})

def prepare_checkpoint_dir(config):
    # prepare the save path
    save_path = os.path.join(config['checkpoint']['save_path'], config['exp_name'])

    local_model_dir = os.path.join(save_path, 'local_models')

    os.makedirs(local_model_dir, exist_ok=True)

    return save_path, local_model_dir

def read_perf_records(save_path, save_file_name):
    with open(os.path.join(save_path, save_file_name)+'.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        data_dict = {key: [] for key in header}
        for row in reader:
            for i, value in enumerate(row):
                try:
                    data_dict[header[i]].append(float(value))
                except ValueError:
                    data_dict[header[i]].append(value)
    return data_dict

# load files
def load_checkpoint(save_path):
    checkpoint_data = load_yaml_config(save_path + "/checkpoint.yaml")
    model_state_dict = torch.load(save_path + "/current_model.pth", map_location='cpu')
    best_model_state_dict = torch.load(save_path + "/best_model.pth", map_location='cpu')
    acc_list = read_perf_records(save_path=save_path, save_file_name='acc_list')['Accuracy']

    return checkpoint_data, model_state_dict, best_model_state_dict, acc_list
  
def prepare_client_checkpoint(config, client_idx, global_model):

    _, local_model_dir = prepare_checkpoint_dir(config)

    client_local_model_dir = os.path.join(local_model_dir, f"{config['server']['model_name']}_local_model_client_{client_idx}/")  
    os.makedirs(client_local_model_dir, exist_ok=True)
    # whether read checkpoint
    if config['resume'] and \
        os.path.exists(client_local_model_dir + "/checkpoint.yaml") and \
        os.path.exists(client_local_model_dir + "/current_model.pth") and \
        os.path.exists(client_local_model_dir + "/best_model.pth") and \
        os.path.exists(client_local_model_dir + "/acc_list.csv"):

        checkpoint_data, model_state_dict, best_state_dict, acc_list = load_checkpoint(client_local_model_dir)

        start_round = checkpoint_data['rounds']
        best_acc = checkpoint_data['best_acc']
        best_round = checkpoint_data['best_round']
        
        local_model = copy.deepcopy(global_model)
        
        if config['resume_best']:
            local_model.load_state_dict(best_state_dict)
        else:
            local_model.load_state_dict(model_state_dict)
    
    else:
        start_round = 0
        best_acc = 0.0
        best_round = 0
        
        local_model = copy.deepcopy(global_model)
        acc_list = []


    return client_local_model_dir, start_round, best_acc, best_round, acc_list, local_model

# common training functions
def init_optimizer(model, optim_name, lr, momentum):
    if optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer {optim_name} is not implemented.")
    return optimizer

def init_loss_fn(loss_name):
    if loss_name == 'ce':
        criterion = torch.nn.CrossEntropyLoss()
    elif loss_name == 'mse':
        criterion = torch.nn.MSELoss()
    elif loss_name == 'nll':
        criterion = torch.nn.NLLLoss()
    elif loss_name == 'l1':
        criterion = torch.nn.L1Loss()

    else:
        raise NotImplementedError(f"Loss function {loss_name} is not implemented.")
    return criterion

# common evaluation methods
def test_acc(model, testloader, device='cpu', mode='normal'):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            
            images, labels = images.to(device), labels.to(device)
            if mode == 'etf':
                outputs, feature = model(images)
                # outputs = torch.matmul(feature, model.proto_classifier.proto)
            else:
                outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total if total > 0 else 0.0
    return acc

# local training functions
def local_training(model, training_data, test_dataloader, 
                   start_epoch, local_epochs, 
                   optim_name, lr, momentum, 
                   loss_name, history_acc_list, best_acc, best_epoch, client_model_dir,
                   device='cpu', num_classes=10, save_freq=1):
    model.train()

    model.to(device)

    optimizer = init_optimizer(model, optim_name, lr, momentum)
    criterion = init_loss_fn(loss_name)


    for e in range(start_epoch, local_epochs + start_epoch):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(training_data):
            
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            output = model(data)
            
            if loss_name in ['mse', 'l1']:
                target = torch.nn.functional.one_hot(target, num_classes=num_classes).float()
            loss = criterion(output, target)
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item()

        

        train_acc = test_acc(model, testloader=test_dataloader, device=device)
        train_set_acc = test_acc(model, testloader=training_data, device=device)
        history_acc_list.append(train_acc)
        
        logger.info(f"Epoch {e} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_acc}")

        if train_acc > best_acc:
            best_acc = train_acc
            best_epoch = e
            best_model_state_dict = copy.deepcopy(model.state_dict())
        
        # save the results
        if (e+1) % save_freq == 0:
            # save_checkpoint(client_model_dir, 
            #                 model=copy.deepcopy(model), 
            #                 best_model_dict=best_model_state_dict, 
            #                 rounds=e+1, 
            #                 best_acc=best_acc, 
            #                 best_round=best_epoch, 
            #                 acc_list=history_acc_list)
            save_best_local_model(client_model_dir, model, f'epoch_{e}.pth', keep_only_last=True)

    return model

def compute_local_model_variance(local_models):
    # param: 
    #   local_models: the list of local models, split the local models into layers and compute the variance of each layer
    # return:
    #   the variance of the local models
    
    # stack all local models and use the torch lib to compute the variance
    local_models_vec = [parameters_to_vector(local_model.parameters()) for local_model in local_models]
    local_models_tensor = torch.stack(local_models_vec)
    local_models_variance = torch.std(local_models_tensor, dim=0)
    
    variance_mean, variance_sum = torch.mean(local_models_variance).item(), torch.sum(local_models_variance).item()
    
    
    
    return variance_mean, variance_sum


class NormalizeInverse(torchvision.transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain.
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())



def convert_tensor_rgb(input_tensor, mean, std):
    img = NormalizeInverse(mean=mean, std=std)(input_tensor)
    # img = input_tensor
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    else:
        img = np.array(img)
    # img = img / 2 + 0.5
    img = np.transpose(img, (0, 2, 3, 1))
    img = np.uint8(img * 255)
    return img


def visualize_pic(model, images, target_layers, dataset_name, save_file_name, target_class=None, device='cpu', onlyrgb=False, save_source=False):
    
    # check the file folder
    if not os.path.exists(os.path.dirname(save_file_name)):
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)

    # convert image to rgb_image, 
    rgb_images = convert_tensor_rgb(images, **NORMALIZE_DICT[dataset_name])
    
    
    if isinstance(target_class, int):
        targets = [ClassifierOutputTarget(target_class)]
    elif isinstance(target_class, list) and len(target_class) > 1:
        targets = [ClassifierOutputTarget(label) for label in target_class]
    else:
        targets = None
        
    
        
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=images, targets=targets, aug_smooth=False, eigen_smooth=False)
        # grayscale_cam = grayscale_cam[0, :]
        
        gb_model = GuidedBackpropReLUModel(model=model, device=device)
        
        
        final_img = []
        for rgb_image, g_cam, image in zip(rgb_images, grayscale_cam, images):
            cam_image = show_cam_on_image(rgb_image/255, g_cam, use_rgb=True)
            
            gb = gb_model(image.unsqueeze(0), target_category=None)
            cam_mask = cv2.merge([g_cam, g_cam, g_cam])
            
            cam_gb = deprocess_image(cam_mask * gb)
            gb = deprocess_image(gb)
            
            if onlyrgb:
                con_img = cam_image
            else:
                con_img = np.concatenate((rgb_image, cam_image, gb, cam_gb), axis=1)

            final_img.append(con_img)
        
        final_img = np.concatenate(final_img, axis=0)
    
        pil_img = Image.fromarray(final_img)
        pil_img.save(save_file_name)
        logger.info(f"Save the visualization in {save_file_name}")
    
    if save_source:
        source_img = Image.fromarray(np.concatenate(rgb_images, axis=0))
        source_img.save(save_file_name.replace('.png', '_source.png'))
        logger.info(f"Save the visualization in {save_file_name.replace('.png', '_source.png')}")