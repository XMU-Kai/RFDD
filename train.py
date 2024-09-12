import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
import torch
import torch.utils.data as Data
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from denoising_diffusion_pytorch_1d import UnetCon1D,GaussianDiffusionCon1D
from RFDD_utils import *
from parameters import *


def teacher_train():
   
    # Network
    model = UnetCon1D(
        dim=64,
        dim_mults=(1, 2, 4,8),
        num_classes=num_classes,
        cond_drop_prob=cond_drop_prob,
        channels=1,
        sinusoidal_pos_emb_theta= 10000
    )
    # Diffusion
    diffusion = GaussianDiffusionCon1D(
        model,
        seq_length = signal_len,
        timesteps=total_step
    ).to(device)

    print(f'time_step:{total_step},  cond_drop_prob:{cond_drop_prob}')
    # Labeled data
    labeled_data = np.load("your_dataset_labeled_x")[:sample_size,start_index:end_index]
    label = np.load("your_dataset_y")[:sample_size]
    # Unlabeled data
    unlabeled_data = np.load("your_dataset_unlabeled_x")[:sample_size,start_index:end_index]
    pseudo_label = np.full(sample_size, -1)
    print(labeled_data.shape)
    print(unlabeled_data.shape)
    # Train data
    combined_data = np.concatenate((labeled_data, unlabeled_data), axis=0)
    combined_label = np.concatenate((label, pseudo_label), axis=0)
    print(combined_data.shape)
    print(combined_label.shape)
    x = torch.Tensor(combined_data.reshape(sample_size*2, 1, signal_len))
    y = torch.Tensor(combined_label).type(torch.LongTensor)
    # Data Loader  
    dataset = Data.TensorDataset(x, y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    losses = []
    epoch_loss = []
    
    data_dir = 'your_base_dir'
    sample_path = f'{data_dir}/train_sample/'
    checkpoint_path = f'{data_dir}/checkpoint/'
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # -----------------------------Training----------------------------
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            # Traing step
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = diffusion(x, classes=y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = sum(losses[-100:]) / 100
        save_model = checkpoint_path + '/RFF_ckpt_' + f'{avg_loss:05f}' + '_' + str(epoch) + '_.pt'
        epoch_loss.append(avg_loss)
        torch.save(model.state_dict(), save_model)
        print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        # -----------------------------Sampling-------------------------
        sample_class = 0
        sample_num = 10
        dim = 192
        con_scale = 1.0
        # Save original data
        load_and_save_samples(combined_data, label, epoch=epoch, sample_path=sample_path, sample_class=sample_class, sample_num=sample_num, signal_len=signal_len)
        # Save samples
        sample_and_save_data(diffusion, epoch, sample_path, device, signal_len, combined_label, label,sample_class,sample_num,dim,con_scale)

    # Save epoch_loss plot
    save_path = os.path.join(data_dir, 'epoch_loss_total.npy')
    np.save(save_path, epoch_loss)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_loss)
    plot_save_path = os.path.join(data_dir, f'epoch_loss_total_plot_{epoch}.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f'Saved epoch_loss at  fianl epoch {epoch}')

# Distill_loss
def distill_loss(image, label,tlist, student_diffusion, teacher_diffusion,device):    
    t, t_1, t_2 = tlist
    batch= image.shape[0]
    # Forward diffusion
    alpha_t = teacher_diffusion.sqrt_alphas_cumprod[t].reshape([batch,1,1])
    sigma_t = teacher_diffusion.sqrt_one_minus_alphas_cumprod[t].reshape([batch,1,1])
    eps = torch.randn_like(image, device=device)
    img_T = alpha_t * image.to(device) + sigma_t * eps

    cond_scale = 1.0    
    rescaled_phi = 0.7  
    
    classes = label.to(device)
    # 2 steps of reverse diffusion using teacher model
    # t_1 --> t_2
    
    time_cond = t
    alpha_t_1 = teacher_diffusion.sqrt_alphas_cumprod[t_1].reshape([batch,1,1]).to(device)
    sigma_t_1 = teacher_diffusion.sqrt_one_minus_alphas_cumprod[t_1].reshape([batch,1,1]).to(device)
    # Predict x
    pred_noise, x_start, *_ = teacher_diffusion.model_predictions(img_T, time_cond, classes, cond_scale, rescaled_phi, x_self_cond=None, clip_x_start = True)
    img_T_1 = alpha_t_1 * x_start + (sigma_t_1) * pred_noise

    # t_2 --> t_3
    time_cond = t_1
    alpha_t_2 = teacher_diffusion.sqrt_alphas_cumprod[t_1].reshape([batch,1,1]).to(device)
    sigma_t_2 = teacher_diffusion.sqrt_one_minus_alphas_cumprod[t_2].reshape([batch,1,1]).to(device)
    pred_noise, x_start, *_ = teacher_diffusion.model_predictions(img_T_1, time_cond, classes, cond_scale, rescaled_phi, x_self_cond=None, clip_x_start = True)
    img_T_2 = alpha_t_2 * x_start + (sigma_t_2) * pred_noise
    
    # Target for student model
    img_target = (img_T_2 - (sigma_t_2/sigma_t) * img_T) / (alpha_t_2 - (sigma_t_2/sigma_t) * alpha_t)

    # 1 step of reverse diffusion using student model
    t_s = t
    time_cond = t_s 
    pred_noise, x_start, *_ = student_diffusion.model_predictions(img_T, time_cond, classes, cond_scale, rescaled_phi, x_self_cond=None, clip_x_start = True)
    img_pred = x_start.to(device)
    
    # Compute loss
    k = alpha_t**2 / sigma_t**2 
    lambda_t = torch.clamp(k, max=1, min=0.01).to(device)
    loss = F.mse_loss(img_pred * lambda_t, img_target.to(device) * lambda_t)
    return loss

def student_train(cond_drop_prob,teacher_checkpoint,student_checkpoint,sampling_steps):

    print(f'time_step:{total_step},  cond_drop_prob:{cond_drop_prob}')
    # Dataset
    data = np.load("your_dataset_x")[:sample_size,start_index:end_index]
    label = np.load("your_dataset_y")[:sample_size]
    print(data.shape)
    print(label.shape)
    x = torch.Tensor(data.reshape(sample_size, 1, signal_len))
    y = torch.Tensor(label).type(torch.LongTensor)
    # Data Loader  
    dataset = Data.TensorDataset(x, y)
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Teacher model
    teacher_model = UnetCon1D(
        dim=64,
        dim_mults=(1, 2, 4,8),
        num_classes=num_classes,
        cond_drop_prob=cond_drop_prob,
        channels=1,
        sinusoidal_pos_emb_theta= 10000
    )
    teacher_diffusion = GaussianDiffusionCon1D(
        teacher_model,
        seq_length = signal_len,
        timesteps=total_step
    ).to(device)
    
    # Student model
    student_model = UnetCon1D(
        dim=64,
        dim_mults=(1, 2, 4,8),
        num_classes=num_classes,
        cond_drop_prob=cond_drop_prob,
        channels=1,
        sinusoidal_pos_emb_theta= 10000
    )
    student_diffusion = GaussianDiffusionCon1D(
        student_model,
        seq_length = signal_len,
        timesteps=total_step
    ).to(device)
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)
    print(f"Initializing from teacher model......")
    student_model.load_state_dict(torch.load(student_checkpoint))
    teacher_model.load_state_dict(torch.load(teacher_checkpoint))

    print(f"Training student, student sampling steps = {sampling_steps}")

    print(f'time_step:{total_step},  cond_drop_prob:{cond_drop_prob}')

    data_dir = 'your_base_dir'
    losses = []
    epoch_loss = []
    sample_path = f'{data_dir}/train_sample/'
    checkpoint_path = f'{data_dir}/checkpoint/'
    save_model= ''
    # ------------------------Student Training----------------------------------------------
    for epoch in range(n_epochs):
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            f = total_step // sampling_steps
            idx = torch.randint(1, sampling_steps, (batch_size,), device=device).long()
            t = f * idx
            t1 = f * (idx - 0.5)
            t1 = t1.long()
            t2 = f * (idx - 1)
            x = x.to(device)
            y = y.to(device)
            # Compute distill loss
            loss = distill_loss(x,y, [t, t1, t2], student_diffusion,  teacher_diffusion,device)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        avg_loss = sum(losses[-100:]) / 100
        save_model = checkpoint_path + '/RFF_ckpt_' + f'{avg_loss:05f}' + '_' + str(epoch) + '_.pt'
        epoch_loss.append(avg_loss)
        torch.save(student_model.state_dict(), save_model)
        print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')
        
    # -----------------------------Sampling------------------------------------------------
    sample_class = 0
    sample_num = 10
    dim = 192
    con_scale = 1.0
    # Save original data
    load_and_save_samples(data, label, epoch=epoch, sample_path=sample_path, sample_class=sample_class, sample_num=sample_num, signal_len=signal_len)
    # Save samples
    sample_and_save_data(student_diffusion, epoch, sample_path, device, signal_len, data, label,sample_class,sample_num,dim,con_scale)
    save_path = os.path.join(data_dir, 'epoch_loss_total.npy')
    np.save(save_path, epoch_loss)
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_loss)
    plot_save_path = os.path.join(data_dir, f'epoch_loss_total_plot_{epoch}.png')
    plt.savefig(plot_save_path)
    plt.close()
    print(f'Saved epoch_loss at  fianl epoch {epoch}')
    return save_model
    

    
if __name__ == '__main__':


    # Teacher training
    teacher_train()
    steps = [2,4,8,16,32,64,128,256]
    teacher_point = 'your pre-trained teacher model path'
    student_point =  teacher_point
    # Sturdent training-distillation
    for sampling_steps in reversed(steps):
        print(f"Sampling steps = {sampling_steps}")
        student_point = teacher_point
        teacher_point = student_train(0,teacher_point,student_point,sampling_steps)
        
  