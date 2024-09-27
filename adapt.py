import numpy as np

import clip
import random
import torch
from torch import nn
import torch.optim as optim
from tqdm import tqdm
import logging
import json
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
    optimizer = setup_optimizer(params)
    tent_model = tent.Tent(model, optimizer,
                           steps=niter,  ### Iterations
                           method=method,
                           episodic=True)
    return tent_model

def setup_eata(model, name_model, fishers):
    model.visual = eata.configure_model(model.visual, name_model)
    params, param_names = eata.collect_params(model.visual, name_model)
    optimizer = setup_optimizer(params)
    eata_model = eata.EATA(model, optimizer,
                           fishers=fishers,
                           episodic=True)
    return eata_model

def setup_sar(model, name_model):
    model.visual = sar.configure_model(model.visual, name_model)
    params, param_names = sar.collect_params(model.visual)
    optimizer = setup_sam(params)
    sar_model = sar.SAR(model, optimizer, episodic=True)
    return sar_model

def setup_sam(params):
    base_optimizer = torch.optim.SGD
    optimizer = SAM(params, base_optimizer, lr=1e-3, momentum=0.9)
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

    for iter_, (inputs, labels) in enumerate(tqdm(fisher_loader, desc='Processed test images: ')):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}.") for c in fisher_dataset.classes]).to(device)

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





def run_dataset(args, model, device, teloader, teset):
    correct = 0
    for batch_idx, (inputs, labels) in enumerate(tqdm(teloader, desc='Processed test images: ')):
        
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in teset.classes]).to(device)

        try:
            model.reset()
        except:
            print("not resetting model!")

        if args.adapt:
            if args.method in ['clipartt', 'tent']:
                Y = model(inputs, text_inputs, teset, device, K = args.K, target_method = args.target_method)  # infer and adapt
            elif args.method in ['sar', 'eata']:
                Y = model(inputs, text_inputs)

        if args.method in ['clipartt', 'tent', 'eata', 'sar'] or not args.adapt:
            # Calculate features
            with torch.no_grad():
                image_features = model.model.encode_image(inputs)
                text_features = model.model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            _, pred = similarity.topk(1, 1, True, True)
            
        elif args.method == 'lame':
            pred = Y.argmax(1)
            pred = pred.unsqueeze(1)

        pred = pred.t()
        correctness = pred.eq(labels.view(1, -1).expand_as(pred))
        correct += correctness.sum().item()

    return (round(correct / len(teloader.dataset), 4))    





def main():
    # Argues
    args = configuration.argparser()

    # Set random seed
    random.seed(1)
    torch.manual_seed(1)

    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_model, preprocess = clip.load(args.model, device)
    if args.method == 'eata':
        base_model.visual = eata.configure_model(base_model.visual, args.model)
        params, param_names = eata.collect_params(base_model.visual, args.model)
    elif args.method == 'sar':
        model = setup_sar(base_model, args.model)
    elif args.method in ['clipartt','tent']: 
        model = setup_tent(base_model, args.model, niter=args.niter, method = args.method)

    # datasets = ['caltech101', 'dtd', 'oxford_pets', 'ucf101', 'imagenet-a', 'imagenet-v']
    datasets = ['cifar10']

    # common_corruptions = ['original', 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
    #                       'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    #                       'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
    common_corruptions = ['gaussian_noise']
                          
    for dataset in datasets:
        if dataset in ['cifar10', 'cifar100']:
            if args.method == 'eata':
                fisher_loader, _, fisher_dataset = prepare_dataset.prepare_eata_data(args, dataset, 'original')
                fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
                model = setup_eata(base_model, args.model, fishers)

            for corruption in common_corruptions:
                teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset, corruption)
                print(run_dataset(args, model, device, teloader, teset))
        else:
            if args.method == 'eata':
                fisher_loader, _, fisher_dataset = prepare_dataset.prepare_eata_data(args, dataset)
                fishers = calc_fisher(base_model, device, params, fisher_loader, fisher_dataset)
                model = setup_eata(base_model, args.model, fishers)

            teloader, _, teset = prepare_dataset.prepare_test_data(args, dataset)
            print(run_dataset(args, model, device, teloader, teset))


if __name__ == "__main__":
    main()