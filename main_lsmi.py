import numpy as np
from utils import *
import torch
from entropy_estimation import MargKernel
import hydra



def RUS_adjustment(rus):
    """
    Adjusts the input tensors (r, u1, u2, s) while preserving certain sums
    and the original device of the tensors. The adjustment aims to make the
    means of these components non-negative based on a specific priority:

    1. If the mean of 'r' (R_mean) or 's' (S_mean) is negative, an adjustment
       factor is calculated to make both R_mean and S_mean non-negative.
       This adjustment might consequently alter the means of 'u1' (U1_mean)
       and 'u2' (U2_mean), potentially making them negative.

    2. If R_mean and S_mean are already non-negative, but U1_mean or U2_mean
       is negative, the adjustment factor is calculated to make both U1_mean
       and U2_mean non-negative. This adjustment might, in turn, make
       R_mean or S_mean negative if they were small positive values.

    The adjustment maintains the following sum properties for the means:
    - (R_mean + U1_mean + U2_mean + S_mean) remains unchanged.
    - (R_mean + U1_mean) remains unchanged.
    - (R_mean + U2_mean) remains unchanged.

    Args:
        rus (tuple or list): A collection of four PyTorch tensors (r, u1, u2, s).

    Returns:
        tuple: A tuple of four adjusted PyTorch tensors (r_adjusted, u1_adjusted,
               u2_adjusted, s_adjusted), on the same device as the input tensors.
    """
    r_orig, u_1_orig, u_2_orig, s_orig = rus

    R_mean = r_orig.detach().mean()
    U1_mean = u_1_orig.detach().mean()
    U2_mean = u_2_orig.detach().mean()
    S_mean = s_orig.detach().mean()

    adj_factor = torch.tensor(0.0, dtype=R_mean.dtype, device=R_mean.device)

    # Priority 1: Address negative mean of r or s
    if R_mean < 0 or S_mean < 0:
        adj_factor = -torch.min(R_mean, S_mean)
          
    # Priority 2: If means of r and s are non-negative, address negative mean of u1 or u2
    elif U1_mean < 0 or U2_mean < 0:
        adj_factor = torch.min(U1_mean, U2_mean)

    r_adjusted = r_orig + adj_factor
    u_1_adjusted = u_1_orig - adj_factor
    u_2_adjusted = u_2_orig - adj_factor
    s_adjusted = s_orig + adj_factor
    
    return r_adjusted, u_1_adjusted, u_2_adjusted, s_adjusted

def obtain_feature_input(batch, device):
    modal_1 = batch[0].to(device)
    modal_2 = batch[1].to(device)
    labels = batch[2].to(device)
    return modal_1, modal_2, labels

def get_entropy(dataloader, model, modality = 'modality_1', cfg = None):
    model.eval()
    info = []
    with torch.no_grad():
        losses = 0.0
        for batch in dataloader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device = cfg.device)
            if modality == "modality_1":
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            batch_size = input_data.shape[0]
            loss = model(input_data)
            info.append(loss)
            losses = losses + torch.mean(loss).item() * batch_size
    info = torch.cat(info, dim = 0).detach()
    return info
def get_mutual_info(dataloader, model, modality = 'modality_1', cfg = None):
    model.eval()
    info = []
    with torch.no_grad():
        infos = 0.0
        for batch in dataloader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device = cfg.device)
            if modality == "modality_1":    
                input_data = modal_1
            elif modality == "modality_2":
                input_data = modal_2
            elif modality == "modality_12":
                input_data = torch.cat([modal_1, modal_2], dim = 1)
            batch_size = input_data.shape[0]
            rows = torch.arange(batch_size)
            out = model(input_data)
            info_cur = np.log(cfg.n_classes) + torch.nn.Softmax(dim=1)(out)[rows, labels].log()
            info.append(info_cur)
            infos = infos + torch.mean(info_cur).item() * batch_size
    info = torch.cat(info, dim = 0).detach()
    return info
                
def LSMI_estimation(dataloader, discriminator, entropy_estimator, cfg = None):

    I_X1Y = get_mutual_info(dataloader, discriminator[0], modality = 'modality_1', cfg = cfg)
    I_X2Y = get_mutual_info(dataloader, discriminator[1], modality = 'modality_2', cfg = cfg)
    I_X1X2Y = get_mutual_info(dataloader, discriminator[2], modality = 'modality_12', cfg = cfg)
    H_X1 = get_entropy(dataloader, entropy_estimator[0], modality = 'modality_1', cfg = cfg)
    H_X2 = get_entropy(dataloader, entropy_estimator[1], modality = 'modality_2', cfg = cfg)

    r_plus = torch.minimum(H_X1, H_X2)
    r_minus = torch.minimum(H_X1 - I_X1Y, H_X2 - I_X2Y)
    r = r_plus - r_minus

    r =  r_plus - r_minus
    u_1 = I_X1Y - r
    u_2 = I_X2Y - r
    s = I_X1X2Y - r - u_1 - u_2
    r, u_1, u_2, s = RUS_adjustment([r, u_1, u_2, s])
    
    R = torch.mean(r)
    U_1 = torch.mean(u_1)
    U_2 = torch.mean(u_2)
    S = torch.mean(s)

    print(f"R: {R.item():.4f}, U1: {U_1.item():.4f}, U2: {U_2.item():.4f}, S: {S.item():.4f}")

    
    return R, U_1, U_2, S
          
def obtain_discriminator(cfg, train_loader):
    model_1 = cls_network(input_dim=cfg.input_size_1, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(cfg.device)
    model_2 = cls_network(input_dim=cfg.input_size_2, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(cfg.device) 
    model_j = cls_network(input_dim=cfg.input_size_1 + cfg.input_size_2, hidden_dim=cfg.embed_size, output_dim=cfg.n_classes).to(cfg.device)
    models = [model_1, model_2, model_j]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    num_epochs = cfg.num_epochs_discriminator
    for epoch in range(num_epochs):   
        losses = 0.0
        num_samples = 0
        for batch in train_loader:
            modal_1, modal_2, labels = obtain_feature_input(batch, device = cfg.device)
            batch_size = modal_1.shape[0]
            out_1 = models[0](modal_1) 
            out_2 = models[1](modal_2)
            out_j = models[2](torch.cat([modal_1, modal_2], dim = 1))
            optimizer.zero_grad()
            loss_1 = criterion(out_1, labels)
            loss_2 = criterion(out_2, labels)
            loss_j = criterion(out_j, labels)
            loss = loss_1 + loss_2 + loss_j
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
            num_samples += batch_size
        scheduler.step()
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / num_samples:.4f}')

    return models

def obtain_entropy_estimator(cfg, train_loader):
    model_1 = MargKernel(dim = cfg.input_size_1).to(cfg.device)
    model_2 = MargKernel(dim = cfg.input_size_2).to(cfg.device)
    models = [model_1, model_2]
    optimizer = torch.optim.Adam([p for model in models for p in model.parameters()], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    num_epochs = cfg.num_epochs_entropy_estimator
    for epoch in range(num_epochs):
        for model in models:
            model.train()
        losses = 0.0
        for batch in train_loader:
            modal_1, modal_2, _ = obtain_feature_input(batch, device = cfg.device)
            batch_size = modal_1.shape[0]
            loss_1 = model_1(modal_1)
            loss_2 = model_2(modal_2)
            loss = loss_1 + loss_2
            loss.backward()
            optimizer.step()
            losses += loss.item() * batch_size
        scheduler.step()
        if (epoch + 1) % 5 == 0:    
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {losses / len(train_loader.dataset):.4f}')

    return models

def estimation_main(cfg, feature_dir = None):
    train_loader, val_loader = get_loader(cfg, feature_dir)
    discriminator = obtain_discriminator(cfg, train_loader=train_loader)
    entropy_estimator = obtain_entropy_estimator(cfg, train_loader=train_loader)
    LSMI_estimation(train_loader, discriminator, entropy_estimator, cfg)
    LSMI_estimation(val_loader, discriminator, entropy_estimator, cfg)
    
def data_generate(cfg)  :
    from gaussian_data import generate_gaussian_data
    num_samples = 2000
    outputs = generate_gaussian_data(int(0.8 * num_samples), int(0.2 * num_samples), 0.5, 0.5, cfg.n_classes)
    train_list, test_list, dims = outputs["train_data"], outputs["test_data"], outputs["dims"]
    data = {
        'train_modal_1_features': torch.from_numpy(train_list[0]).float(),
        'train_modal_2_features': torch.from_numpy(train_list[1]).float(),
        'train_targets': torch.from_numpy(train_list[2]).long(),
        'val_modal_1_features': torch.from_numpy(test_list[0]).float(),
        'val_modal_2_features': torch.from_numpy(test_list[1]).float(),
        'val_targets': torch.from_numpy(test_list[2]).long()
    }
    cfg.input_size_1 = dims[0]
    cfg.input_size_2 = dims[1]
    
    torch.save(data, 'gaussian_data.pt')
    return 'gaussian_data.pt'
@hydra.main(config_path='cfgs', config_name='train', version_base=None)

def main(cfg):
    setup_seed(cfg.random_seed)
    data_path = data_generate(cfg)
    estimation_main(cfg, data_path)

        


if __name__ == '__main__':
	main()
