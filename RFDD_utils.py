import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import pytorch_ssim
import random
def save_plot_images(output_images,save_dir,signal_len):
    os.makedirs(save_dir, exist_ok=True)
    for j, image in enumerate(output_images):
        save_path = os.path.join(save_dir, f"{j}.png")
        plt.figure(figsize=(16, 9))
        plt.plot(image.reshape(signal_len), color='#1f77b4')
        plt.axis('off') 
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return save_path

def save_data(output_images, data_save_dir):
    for j, image in enumerate(output_images):
        file_path = f"{data_save_dir}/{j}.npy"
        np.save(file_path, image)
        
def sample_and_save_data(diffusion, epoch,  device, signal_len, sample_class, sample_num,dim,con_scale,data_save_dir,sampling_steps):
        comp_img_dir = data_save_dir + '/comp_sample/figs'
        comp_data_dir = data_save_dir + '/comp_sample/data'
        os.makedirs(comp_img_dir, exist_ok=True)
        os.makedirs(comp_data_dir, exist_ok=True)
        data_img_dir = data_save_dir + '/generated_sample/figs'
        generated_data_dir = data_save_dir + '/generated_sample/data'
        os.makedirs(data_img_dir, exist_ok=True)
        os.makedirs(generated_data_dir, exist_ok=True)

        classes = torch.tensor([sample_class]*sample_num).to(device)  
        shape = (sample_num, 1, signal_len ) 
        cond_scale = con_scale  
        rescaled_phi = 0.7 
        # DDIM
        output_images = diffusion.ddim_sample(classes, shape, cond_scale, rescaled_phi).cpu().numpy()
        # Save generated data
        save_data(output_images, generated_data_dir)
        save_plot_images(output_images,data_img_dir,signal_len)

        # SSIM
        ssim_list = []
        for j in range(0, sample_num):
            data_path = f"{generated_data_dir}/{j}.npy"
            comp_data_path = f"{comp_data_dir}/{j}.npy"
            data_np = np.load(data_path)
            comp_data_np = np.load(comp_data_path)
            data_tensor = torch.tensor(data_np).reshape(1,1,1,signal_len).double() 
            comp_data_tensor = torch.tensor(comp_data_np).reshape(1,1,1,signal_len).double()
            ssim_value = pytorch_ssim.ssim(data_tensor, comp_data_tensor)
            ssim_list.append(ssim_value.item())
            
        # Save SSIM
        with open(f"{data_save_dir}/ssim_info.log", "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write (f"con_scale: { con_scale}\n")
            f.write(f" SSIM: { sum(ssim_list) / len(ssim_list)}\n")
            f.write("------------------------------------\n")
        
        # FID
        from pytorch_fid import fid_score
        fid_value = fid_score.calculate_fid_given_paths([comp_img_dir, data_img_dir ], sample_num, device, dims=dim)
        fid_value = round(fid_value*10, 2)
        print(f"device {sample_class}, FID: {fid_value}")
        
        # Save FID
        with open(f"{data_save_dir}/fid_info.log", "a") as f:
            f.write(f"Epoch {epoch}\n")
            f.write (f"con_scale: { con_scale}\n")
            f.write(f" FID: {fid_value}\n")
            f.write("------------------------------------\n")

def load_and_save_samples(data, label, class_number, sample_num, signal_len=None,data_save_dir=None):

    # Save original data
    comp_img_dir = data_save_dir + '/comp_sample/figs'
    comp_data_dir = data_save_dir + '/comp_sample/data'
    os.makedirs(comp_img_dir, exist_ok=True)
    os.makedirs(comp_data_dir, exist_ok=True)
    
    random_sample_classes = random.sample(range(0, class_number), sample_num)
    comp_data = []
    for cls in random_sample_classes:
        indices = [j for j, lbl in enumerate(label) if lbl == cls]
        if indices:
            selected_index = random.choice(indices)
            comp_data.append(data[selected_index])
            
    # Save_data and save_plot_images
    save_data(comp_data, comp_data_dir)
    save_plot_images(comp_data, comp_img_dir, signal_len)