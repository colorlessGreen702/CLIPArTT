import numpy as np
import gc

import torch.multiprocessing as mp
import clip
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import os
import json
import csv
from models.sam import SAM

import configuration
from models import tent, lame, eata, sar
from utils import prepare_dataset


def setup_tent(model, name_model, niter = 10, method = 'clip'):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    LN = True
    if LN == True:
        model.visual = tent.configure_model(model.visual, name_model)
        #extractor = [model.net.conv1, model.net.bn1, nn.ReLU(inplace=True), model.net.layer1, model.net.layer2]
        #extractor = nn.Sequential(*extractor)
        params, param_names = tent.collect_params(model.visual, name_model)
    else:
        params = model.visual.parameters()
    optimizer = torch.optim.SGD(params, 0.001/64, momentum=0.9) 
    tent_model = tent.Tent(model, optimizer,
                           steps=niter,  ### Iterations
                           method=method,
                           episodic=False)
    return tent_model

def setup_eata(model, params, fishers):
    optimizer = torch.optim.SGD(params, 0.001/64, momentum=0.9)
    eata_model = eata.EATA(model, optimizer,
                           fishers=fishers,
                           episodic=False)
    return eata_model

def setup_sar(model, name_model):
    model.visual = sar.configure_model(model.visual, name_model)
    params, param_names = sar.collect_params(model.visual)
    optimizer = setup_sam(params)
    sar_model = sar.SAR(model, optimizer, episodic=False)
    return sar_model

def setup_sam(params):
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=0.001/32, momentum=0.9)
    return optimizer

def setup_optimizer(params):
    """Set up optimizer for tent adaptation.
    Tent needs an optimizer for test-time entropy minimization.
    In principle, tent could make use of any gradient optimizer.
    In practice, we advise choosing Adam or SGD+momentum.
    For optimization settings, we advise to use the settings from the end of
    trainig, if known, or start with a low learning rate (like 0.001) if not.
    For best results, try tuning the learning rate and batch size.
    """
    # if cfg.OPTIM.METHOD == 'Adam':
    return optim.Adam(params,
                lr=1e-3,
                betas=(0.9, 0.999),
                weight_decay=0.0)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, image_features, text_features, labels):
        # Calculate cosine similarity
        similarities = torch.matmul(image_features, text_features.T) / self.temperature
        # Use log-softmax for numerical stability
        loss = nn.functional.cross_entropy(similarities, labels)
        return loss

def calc_fisher(net, device, params, fisher_loader, fisher_dataset):
    ewc_optimizer = torch.optim.SGD(params, 0.001)
    fishers = {}
    train_loss_fn = ContrastiveLoss().to(device)

    for iter_, (inputs, labels) in enumerate(tqdm(fisher_loader, desc="calculating fishers:")):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in fisher_dataset.classes]).to(device)

        _, image_features, text_features = net(inputs, text_inputs)

        loss = train_loss_fn(image_features, text_features, labels)
        loss.backward()

        for name, param in net.named_parameters():
            if param.grad is not None:
                if iter_ > 1:
                    fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                else:
                    fisher = param.grad.data.clone().detach() ** 2
                if iter_ == len(fisher_loader):
                    fisher = fisher / iter_
                fishers.update({name: [fisher, param.data.clone().detach()]})
        ewc_optimizer.zero_grad()
    print("compute fisher matrices finished")
    del ewc_optimizer
    return fishers

# Function to write data to JSON file
def write_to_json(filename, data, lock):
    with lock:
        # Read existing data from the file
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    file_data = json.load(file)
                except json.JSONDecodeError:
                    file_data = []  # In case the file is empty or corrupted
        else:
            file_data = []

        updated = False
        for entry in file_data:
            if entry['method'] == data['method'] and entry['benchmark'] == data['benchmark']:
                # Update the existing entry
                entry.update(data)
                updated = True
                break

        # If no match was found, add the new entry
        if not updated:
            file_data.append(data)

        # Write updated data back to the file
        with open(filename, 'w') as file:
            json.dump(file_data, file, indent=4)
        print(f"Data written to {filename} successfully.")


def check_entry_exists(file_name, lock, method, benchmark):
    with lock:
        try:
            # Read the existing data from the JSON file
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        except FileNotFoundError:
            # If the file does not exist, return False
            return False

        # Check if an entry with the same method and benchmark exists
        for entry in existing_data:
            if entry.get('method') == method and entry.get('benchmark') == benchmark:
                return True

        # Return False if no match is found
        return False













def run_dataset(args, lock, method, model, device, dataset, pos, teloader, teset):
    correct = 0
    try:
        model.reset()
    except:
        print("not resetting model!")
        
    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in teset.classes]).to(device)
    with torch.no_grad():
        text_features = model.model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    for batch_idx, (inputs, labels) in enumerate(tqdm(teloader, desc=f"{method} {dataset}", position=pos)):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        if args.adapt:
            if method in ['clipartt', 'tent']:
                Y = model(inputs, text_inputs, teset, device, K = args.K, target_method = args.target_method)  # infer and adapt
            elif method in ['sar', 'eata']:
                Y = model(inputs, text_inputs)

        if method in ['clipartt', 'tent', 'sar', 'eata'] or not args.adapt:
            # Calculate features
            with torch.no_grad():
                image_features = model.model.encode_image(inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, pred = similarity.topk(1, 1, True, True)
            
        elif method == 'lame':
            pred = Y.argmax(1)
            pred = pred.unsqueeze(1)

        pred = pred.t()
        correctness = pred.eq(labels.view(1, -1).expand_as(pred))
        correct += correctness.sum().item()

    data = {
    "method": method,
    "benchmark": dataset,
    "accuracy": round(correct / len(teloader.dataset), 4),  # in percentage
    }

    write_to_json('data.json', data, lock)

    # return (round(correct / len(teloader.dataset), 4))    



def run_method(method, lock, pos, args, datasets, common_corruptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Set random seed
    random.seed(1)
    torch.manual_seed(1)
    
    # Load the model
    if method != 'eata':
        base_model, _ = clip.load(args.model, device)
    if method == 'sar':
        model = setup_sar(base_model, args.model)
    elif method == 'tent': 
        model = setup_tent(base_model, args.model, niter=1, method=method)
    elif method == 'clipartt': 
        model = setup_tent(base_model, args.model, niter=args.niter, method=method)
    elif method == 'clip':
        model = setup_tent(base_model, args.model, niter=args.niter, method='clipartt')
    
    for dataset in datasets:
        if dataset in ['cifar10', 'cifar100']:
            if method == 'eata':
                base_model, _ = clip.load(args.model, device)
                base_model.visual = eata.configure_model(base_model.visual, args.model)
                params, _ = eata.collect_params(base_model.visual, args.model)
                fisher_loader, _, fisher_dataset = prepare_dataset.prepare_test_data(args, dataset, 'original')
                fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
                model = setup_eata(base_model, params, fishers)
                del fisher_dataset, fisher_loader, _


            for corruption in common_corruptions:
                if check_entry_exists('data.json', lock, method, dataset+' '+corruption):
                    continue
                teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset, corruption)
                run_dataset(args, lock, method, model, device, dataset+' '+corruption, pos, teloader, teset)
                del teloader, teset, _
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

            if method == 'eata':
                del model, base_model
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()
        else:
            if check_entry_exists('data.json', lock, method, dataset):
                    continue

            if method == 'eata':
                base_model, _ = clip.load(args.model, device)
                base_model.visual = eata.configure_model(base_model.visual, args.model)
                params, _ = eata.collect_params(base_model.visual, args.model)
                fisher_loader, _, fisher_dataset = prepare_dataset.prepare_test_data(args, dataset)
                fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
                model = setup_eata(base_model, params, fishers)
                del fisher_loader, fisher_dataset, _
                
            teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset)
            run_dataset(args, lock, method, model, device, dataset, pos, teloader, teset)
            del teloader, teset, _
            torch.cuda.empty_cache()
            gc.collect()
            torch.cuda.synchronize()

            if method == 'eata':
                del model
                torch.cuda.empty_cache()
                gc.collect()
                torch.cuda.synchronize()

def main():
    args = configuration.argparser()
    
    datasets = ['cifar100','caltech101', 'dtd', 'oxford_pets', 'ucf101', 'imagenet-a', 'imagenet-v']

    common_corruptions = ['original','gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                          'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                          'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    # common_corruptions = ['gaussian_noise']

    # methods = ['sar']

    # pool = mp.Pool(len(methods))
    lock = mp.Manager().Lock()
    # pool.starmap(run_method, [(method, lock, idx, args, datasets, common_corruptions) for idx, method in enumerate(methods)])
    # pool.close()
    # pool.join()
    run_method('eata', lock, 0, args, datasets, common_corruptions)


if __name__ == "__main__":
    main()



# def main():
#     args = configuration.argparser()
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     datasets = ['cifar10', 'cifar100', 'caltech101', 'dtd', 'oxford_pets', 'ucf101', 'imagenet-a', 'imagenet-v']

#     common_corruptions = ['original', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
#                         'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
#                         'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
#     header = make_header(datasets, common_corruptions)

#     for method in ['tent','eata','sar','clipartt']:

        
#         data = [method]
#         # Set random seed
#         random.seed(1)
#         torch.manual_seed(1)
#         # Load the model
#         base_model, _ = clip.load(args.model, device)

#         if method == 'eata':
#             base_model.visual = eata.configure_model(base_model.visual, args.model)
#             params, _ = eata.collect_params(base_model.visual, args.model)
#         elif method == 'sar':
#             model = setup_sar(base_model, args.model)
#         elif method == 'tent': 
#             model = setup_tent(base_model, args.model, niter=1, method = method)
#         elif method == 'clipartt': 
#             model = setup_tent(base_model, args.model, niter=args.niter, method = method)
                            
#         for dataset in datasets:
#             if dataset in ['cifar10', 'cifar100']:
#                 if args.method == 'eata':
#                     fisher_loader, _, fisher_dataset = prepare_dataset.prepare_test_data(args, dataset, 'original')
#                     fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
#                     model = setup_eata(base_model, args.model, fishers)

#                 for corruption in common_corruptions:
#                     teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset, corruption)
#                     acc = run_dataset(args, model, device, teloader, teset)
#                     data.append(round(acc*100, 4))
#             else:
#                 if args.method == 'eata':
#                     fisher_loader, _, fisher_dataset = prepare_dataset.prepare_test_data(args, dataset)
#                     fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
#                     model = setup_eata(base_model, args.model, fishers)

#                 teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset)
#                 acc = run_dataset(args, model, device, teloader, teset)
#                 data.append(round(acc*100, 4))

#         append_to_csv('accuracy.csv', data, header)


# if __name__ == "__main__":
#     main()