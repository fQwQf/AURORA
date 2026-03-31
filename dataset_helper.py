import torch.utils
import torchvision.transforms as transforms
from pathlib import Path
from itertools import combinations
from collections import defaultdict


from common_libs import *

NORMALIZE_DICT = {
    'MNIST': dict(mean=(0.1307,), std=(0.3081,)),
    'FMNIST': dict(mean=(0.2860,), std=(0.3530,)),
    'CIFAR10': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'CIFAR100': dict(mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)),
    'Tiny-ImageNet': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'SVHN': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), 
    'EMNIST_digits': dict(mean=(0.5,), std=(0.5,)),
    'FEMNIST': dict(mean=(0.5,), std=(0.5,)),
    'PathMNIST': dict(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
}

def load_dataset(dataset_name, data_path, normalize_train=True, normalize_test=True):

    if dataset_name == 'MNIST':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)
        
        train_dataset = torchvision.datasets.MNIST(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.MNIST(root=data_path, train=False, transform=test_transform, download=True)

    elif dataset_name == 'FMNIST':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)
        
        train_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.FashionMNIST(root=data_path, train=False, transform=test_transform, download=True)

    elif dataset_name == 'CIFAR10':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)
        
        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, transform=test_transform, download=True)

    elif dataset_name == 'CIFAR100':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = torchvision.datasets.CIFAR100(root=data_path, train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.CIFAR100(root=data_path, train=False, transform=test_transform, download=True)

    elif dataset_name == 'Tiny-ImageNet':
        train_transform = [
            # transforms.RandomRotation(20),
            # transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ]
        if normalize_train:
            train_transform.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform)

        test_transform_list = [
            transforms.ToTensor(),
        ]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/train'), transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_path, 'tiny-imagenet-200/val'), transform=test_transform)

    elif dataset_name == 'SVHN':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = torchvision.datasets.SVHN(root=os.path.join(data_path, 'SVHN'), split='train', download=True, transform=train_transform)
        test_dataset = torchvision.datasets.SVHN(root=os.path.join(data_path, 'SVHN'), split='test', download=True, transform=test_transform)

    elif dataset_name == 'EMNIST_digits':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        train_dataset = torchvision.datasets.EMNIST(root=data_path, split='digits', train=True, transform=train_transform, download=True)
        test_dataset = torchvision.datasets.EMNIST(root=data_path, split='digits', train=False, transform=test_transform, download=True)

    elif dataset_name == 'PathMNIST':
        try:
            from medmnist import PathMNIST
        except ImportError:
            raise ImportError("Please install medmnist to use PathMNIST dataset: pip install medmnist")

        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        target_transform = transforms.Lambda(lambda y: torch.tensor(y).long().squeeze())

        os.makedirs(data_path, exist_ok=True)
        train_dataset = PathMNIST(root=data_path, split='train', download=True, transform=train_transform, target_transform=target_transform)
        test_dataset = PathMNIST(root=data_path, split='test', download=True, transform=test_transform, target_transform=target_transform)

    elif dataset_name == 'FEMNIST':
        train_transform_list = [transforms.ToTensor()]
        if normalize_train:
            train_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        train_transform = transforms.Compose(train_transform_list)

        test_transform_list = [transforms.ToTensor()]
        if normalize_test:
            test_transform_list.append(transforms.Normalize(**NORMALIZE_DICT[dataset_name]))
        test_transform = transforms.Compose(test_transform_list)

        emnist_train = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=True, transform=train_transform, download=True)
        emnist_test = torchvision.datasets.EMNIST(root=data_path, split='byclass', train=False, transform=test_transform, download=True)
        
        train_dataset = emnist_train
        test_dataset = emnist_test

    else:
        logger.error(f'Dataset {dataset_name} not supported')

    logger.info(f'Dataset {dataset_name} loaded')

    return train_dataset, test_dataset


def build_dataset_idxs(dataset, dataset_name):
    preprocess_path = Path('./preprocessed_data/')
    os.makedirs(preprocess_path, exist_ok=True)
    preprocess_path = preprocess_path / f'{dataset_name}.pt'

    if not os.path.exists(preprocess_path):     
        idx_dict = {}
        for idx, data in enumerate(tqdm(dataset, desc=f'Preprocessing data, Build dataset index dict')):
            _, label = data
            if hasattr(label, 'item'):
                label = label.item()
            
            if label in idx_dict:
                idx_dict[label].append(idx)
            else:
                idx_dict[label] = [idx]
        torch.save(idx_dict, preprocess_path)

        logger.info(f'Dataset index dict saved to {preprocess_path}')
    else:
        idx_dict = torch.load(preprocess_path)

        logger.info(f'Dataset index dict loaded from {preprocess_path}')

    return idx_dict 


def iid(data_size, num_users):

    idxs = np.random.permutation(data_size)
    batch_idxs = np.array_split(idxs, num_users)
    client_idx_map = {i: batch_idxs[i] for i in range(num_users)}

    return client_idx_map


def non_iid(data_idxs_dict, num_users, label_num_per_client):
    
    client_idx_map = defaultdict(list)
    label_num = len(data_idxs_dict)

    data_num_per_class = [len(data_idxs_dict[i]) for i in data_idxs_dict.keys()]

    datasize_per_client = size_of_division(num_users, sum(data_num_per_class))
    class_comb = generate_class_comb(num_users, label_num, label_num_per_client)
    
    datasize_per_client_per_class = [size_of_division(len(class_comb[i]), datasize_per_client[i]) for i in range(num_users)]

    temp = [set(i) for i in data_idxs_dict.values()]
    for c in range(num_users):
        cur_client_id = c
        for i in range(label_num_per_client):
            num_data_per_cli = datasize_per_client_per_class[c][i]
            cur_client_class_ = class_comb[c][i]

            if len(temp[cur_client_class_]) < num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=True)
            elif len(temp[cur_client_class_]) == num_data_per_cli:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
            else:
                rand_set = np.random.choice(list(temp[cur_client_class_]), num_data_per_cli, replace=False)
                temp[cur_client_class_] = temp[cur_client_class_] - set(rand_set)  

            client_idx_map[cur_client_id].extend(rand_set)    

    return client_idx_map


def size_of_division(num_groups, size):
    if isinstance(num_groups, int):
        num_per_group = [size // num_groups] * num_groups
    elif isinstance(num_groups, list):
        num_per_group = [math.floor(size * w) for w in num_groups]

    for i in np.random.choice(len(num_per_group), size - sum(num_per_group), replace=False):
        num_per_group[i] += 1
    
    return num_per_group


def generate_class_comb(num_groups, num_class, num_class_each_comb):
    class_comb = list(combinations(range(num_class), num_class_each_comb))
    np.random.shuffle(class_comb)
    if len(class_comb) <= num_groups:
        for _ in range(num_groups//len(class_comb)):
            class_comb += class_comb
    class_comb = class_comb[:num_groups]
    class_comb = class_comb[:num_groups]

    return class_comb


# In dataset_helper.py
# PLEASE REPLACE YOUR ENTIRE dirichlet FUNCTION WITH THIS ONE

def dirichlet(data_idxs_dict, num_users, alpha):
    """
    Distributes data among clients using a Dirichlet distribution, ensuring
    all data is used and sampled without replacement.
    This is the scientifically sound "card dealer" or "cake cutting" model.
    """
    num_classes = len(data_idxs_dict)
    client_idx_map = {i: [] for i in range(num_users)}

    for k in range(num_classes):
        # Get the shuffled indices for the current class
        idx_k = data_idxs_dict[k]
        random.shuffle(idx_k)
        
        # Generate proportions for each client for this class
        proportions = np.random.dirichlet(np.repeat(alpha, num_users))
        
        # Ensure the sum of proportions doesn't lead to losing data due to rounding
        # This calculates the exact number of samples for each client
        num_samples_per_client = (proportions * len(idx_k)).astype(int)
        
        # Because of integer rounding, there might be a few samples left.
        # Distribute the remainder samples one by one to some random clients.
        rem = len(idx_k) - num_samples_per_client.sum()
        add_for_clients = np.random.choice(range(num_users), rem, replace=False)
        for client_id in add_for_clients:
            num_samples_per_client[client_id] += 1
            
        # --- The CRITICAL FIX: Slicing and consuming the data ---
        current_pos = 0
        for i in range(num_users):
            num_samples = num_samples_per_client[i]
            # Get the slice of data for the current client
            client_idx_map[i].extend(idx_k[current_pos : current_pos + num_samples])
            # Move the pointer forward
            current_pos += num_samples

    # Final shuffle for each client's combined data
    for i in range(num_users):
        random.shuffle(client_idx_map[i])
        
    logger.info("Data distribution generated correctly using the corrected dirichlet function.")
    total_assigned = sum(len(v) for v in client_idx_map.values())
    logger.info(f"Total assigned samples: {total_assigned}")
    
    return client_idx_map


def natural_partition(dataset, num_users, alpha=0.5, max_samples_per_client=None):
    total_samples = len(dataset)
    client_idx_map = defaultdict(list)
    class_to_idxs = defaultdict(list)
    for idx in range(total_samples):
        _, label = dataset[idx]
        if hasattr(label, 'item'):
            label = label.item()
        class_to_idxs[label].append(idx)
    
    num_classes = len(class_to_idxs)
    target_per_client = total_samples // num_users
    if max_samples_per_client is not None:
        target_per_client = min(target_per_client, max_samples_per_client)
    
    for client_id in range(num_users):
        class_probs = np.random.dirichlet([alpha] * num_classes)
        samples_per_class = (class_probs * target_per_client).astype(int)
        
        for class_id, num_samples in enumerate(samples_per_class):
            if num_samples > 0 and len(class_to_idxs[class_id]) > 0:
                if num_samples > len(class_to_idxs[class_id]):
                    sampled_idxs = np.random.choice(class_to_idxs[class_id], num_samples, replace=True)
                else:
                    sampled_idxs = np.random.choice(class_to_idxs[class_id], num_samples, replace=False)
                client_idx_map[client_id].extend(sampled_idxs.tolist())
        
        random.shuffle(client_idx_map[client_id])
    
    logger.info(f"Natural partition generated for {num_users} clients with alpha={alpha}")
    if max_samples_per_client is not None:
        for cid in client_idx_map:
            logger.info(f"  Client {cid}: {len(client_idx_map[cid])} samples (capped at {max_samples_per_client})")
    return dict(client_idx_map)


def get_fl_dataset(dataset_name, dataset_path, num_users, distribution, distribution_params=2, alpha=0.1, normalize_train=True, normalize_test=True, max_samples_per_client=None):
    train_set, test_set = load_dataset(dataset_name, dataset_path, normalize_train=normalize_train, normalize_test=normalize_test)
    data_idx_dict = build_dataset_idxs(train_set, dataset_name)

    if distribution == 'iid':
        client_idx_map = iid(len(train_set), num_users)
    elif distribution == 'noniid':
        client_idx_map = non_iid(data_idx_dict, num_users, distribution_params)
    elif distribution == 'dirichlet':
        client_idx_map = dirichlet(data_idx_dict, num_users, alpha)
    elif distribution == 'natural':
        client_idx_map = natural_partition(train_set, num_users, alpha, max_samples_per_client=max_samples_per_client)
    else:
        raise NotImplementedError(f"Distribution type '{distribution}' not supported")
    
    return train_set, test_set, client_idx_map


def get_client_dataloader(client_idxs, trainset, batch_size):
    client_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(client_idxs),
        pin_memory=True, drop_last=False)

    return client_loader


def get_aux_data(aux_dataset_name, dataset_path, split):
    train_set, test_set = load_dataset(aux_dataset_name, dataset_path)
    if split == 'train':
        return train_set
    elif split == 'test':
        return test_set
    elif split == 'both':
        return train_set, test_set
    else:
        raise ValueError('Invalid split type')

def get_aux_dataloader(aux_dataset_name, dataset_path, split, batchsize):
    if split != 'both':
        aux_dataset = get_aux_data(aux_dataset_name, dataset_path, split)
        aux_loader = torch.utils.data.DataLoader(aux_dataset, batch_size=batchsize, shuffle=True)
        
        return aux_loader
    else:
        aux_train_dataset, aux_test_dataset = get_aux_data(aux_dataset_name, dataset_path, split)
        aux_train_loader = torch.utils.data.DataLoader(aux_train_dataset, batch_size=batchsize, shuffle=True)
        aux_test_loader = torch.utils.data.DataLoader(aux_test_dataset, batch_size=batchsize, shuffle=True)

        return aux_train_loader, aux_test_loader
    

def get_uniform_subset_dataloader(trainset, subsize, batch_size):
    # random select a subset of the dataset
    idxs = np.random.permutation(len(trainset))
    idxs = idxs[:subsize]
    subset_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(idxs),
        pin_memory=True, drop_last=False)

    return subset_loader


class AugmentPairDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[index]
        x1 = self.transform(x)
        x2 = self.transform(x)
        
        return x1, x2, y

    def __len__(self):
        return len(self.dataset)
    
def get_supervised_transform(dataset_name):
    return torchvision.transforms.Compose([
        torchvision.transforms.Normalize(**NORMALIZE_DICT[dataset_name])
    ])




