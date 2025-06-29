# mini-project
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
 
def make_layer(block, num_layers): 
    return nn.Sequential(*[block() for _ in range(num_layers)]) 
 
class ResidualDenseBlock(nn.Module): 
    def __init__(self, nf=64, gc=32): 
        super(ResidualDenseBlock, self).__init__() 
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1) 
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1) 
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1) 
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1) 
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1) 
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True) 
 
    def forward(self, x): 
        x1 = self.lrelu(self.conv1(x)) 
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1))) 
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1))) 
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1))) 
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) 
        return x + x5 * 0.2 
 
class RRDB(nn.Module): 
    def __init__(self, nf, gc=32): 
        super(RRDB, self).__init__() 
        self.rdb1 = ResidualDenseBlock(nf, gc) 
        self.rdb2 = ResidualDenseBlock(nf, gc) 
        self.rdb3 = ResidualDenseBlock(nf, gc) 
 
    def forward(self, x): 
        out = self.rdb1(x) 
        out = self.rdb2(out) 
        out = self.rdb3(out) 
        return x + out * 0.2 
 
class RRDN(nn.Module): 
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=4): 
        super(RRDN, self).__init__() 
        self.scale = scale 
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1) 
 
        # Residual Dense Blocks 
        self.rrdb_trunk = make_layer(lambda: RRDB(nf, gc), nb) 
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1) 
 
        # Upsampling layers 
        self.up_layers = nn.Sequential() 
        for _ in range(int(scale // 2)):  # Upsample 2x for each step 
            self.up_layers.add_module( 
                'upsample_conv', 
                nn.Conv2d(nf, nf * 4, 3, 1, 1) 
            ) 
            self.up_layers.add_module('pixel_shuffle', nn.PixelShuffle(2)) 
            self.up_layers.add_module('lrelu', nn.LeakyReLU(0.2, inplace=True)) 
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1) 
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1) 
 
    def forward(self, x): 
        fea = self.conv_first(x) 
        trunk = self.trunk_conv(self.rrdb_trunk(fea)) 
        fea = fea + trunk 
        out = self.up_layers(fea) 
        out = self.hr_conv(out) 
        out = self.conv_last(out) 
        return out 
 
train_rrdn.py 
import os 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Dataset 
from torchvision import transforms 
from PIL import Image 
from tqdm import tqdm 
import torch.nn.functional as F  # <-- required for resizing 
from models.RRDN_arch import RRDN 
 
 
# Dataset 
class SRDataset(Dataset): 
    def __init__(self, lr_dir, hr_dir): 
        super(SRDataset, self).__init__() 
        self.lr_dir = lr_dir 
        self.hr_dir = hr_dir 
        self.lr_images = sorted(os.listdir(lr_dir)) 
        self.hr_images = sorted(os.listdir(hr_dir)) 
        self.transform = transforms.ToTensor() 
        self.paired_files = [ 
            (os.path.join(self.lr_dir, lr), os.path.join(self.hr_dir, hr)) 
            for lr, hr in zip(self.lr_images, self.hr_images) 
            if os.path.splitext(lr)[0] == os.path.splitext(hr)[0] 
        ] 
        assert len(self.paired_files) > 0, "No matching LR/HR image pairs found!" 
 
    def __len__(self): 
        return len(self.paired_files) 
 
    def __getitem__(self, idx): 
        lr_path, hr_path = self.paired_files[idx] 
        lr = Image.open(lr_path).convert('RGB') 
        hr = Image.open(hr_path).convert('RGB') 
        lr = self.transform(lr) 
        hr = self.transform(hr) 
        return lr, hr 
 
# Training Script 
def train_rrdn_model(): 
    # Paths 
    lr_dir = '.\dataset\LR' 
    hr_dir = '.\dataset\HR' 
    save_path = './models/rrdn_model.pth' 
 
    # Hyperparameters 
    batch_size = 4 
    num_epochs = 100 
    learning_rate = 1e-4 
    scale = 4 
    # Dataset & Dataloader 
    dataset = SRDataset(lr_dir, hr_dir) 
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2) 
 
    # Model 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    model = RRDN(in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=scale).to(device) 
 
    # Optimizer & Loss 
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate) 
 
    # Training Loop 
    model.train() 
    for epoch in range(num_epochs): 
        epoch_loss = 0 
        for lr, hr in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"): 
            lr = lr.to(device) 
            hr = hr.to(device) 
            sr = model(lr) 
 
 
 
            # Resize HR to match SR if sizes mismatch 
            if sr.shape != hr.shape: 
                hr = F.interpolate(hr, size=sr.shape[2:], mode='bicubic', align_corners=False) 
            loss = criterion(sr, hr) 
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step() 
            epoch_loss += loss.item() 
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss/len(dataloader):.4f}") 
 
        # Save checkpoint 
        if (epoch + 1) % 10 == 0: 
            torch.save(model.state_dict(), save_path) 
            print(f"\u2705 Model checkpoint saved at: {save_path}") 
 
    # Final model 
    torch.save(model.state_dict(), save_path) 
    print(f"\n\u2705 Training complete. Final model saved at: {save_path}") 
 
if __name__ == '__main__': 
    train_rrdn_model() 
 
Final_enhance.py 
 
import streamlit as st 
import torch 
import torchvision.transforms as transforms 
from PIL import Image 
import numpy as np 
from skimage.metrics import peak_signal_noise_ratio, structural_similarity 
import matplotlib.pyplot as plt 
import seaborn as sns 
from RRDN_arch import RRDN 
# Page setup 
st.set_page_config(page_title="RRDN Super-Resolution", layout="centered") 
st.title(" RRDN Image Super-Resolution ") 
# Load RRDN model 
@st.cache_resource 
def load_model(): 
    model = RRDN(in_nc=3, out_nc=3, nf=64, nb=16, gc=32, scale=4) 
    model.load_state_dict(torch.load("rrdn_model.pth", map_location=torch.device("cpu"))) 
    model.eval() 
    return model 
model = load_model() 
 
# Image preprocessing/postprocessing 
def preprocess_image(image): 
    transform = transforms.ToTensor() 
    return transform(image).unsqueeze(0) 
 
def postprocess_image(tensor): 
    tensor = tensor.squeeze(0).detach().cpu().clamp(0, 1) 
    return transforms.ToPILImage()(tensor) 
 
# File upload 
uploaded_file = st.file_uploader("Upload a low-resolution image", type=["jpg", "jpeg", "png"]) 
if uploaded_file is not None: 
    input_image = Image.open(uploaded_file).convert("RGB") 
    st.image(input_image, caption="Input Image", use_container_width=True) 
    enhance = st.button(" Enhance ") 
 
    if enhance: 
        with st.spinner("Enhancing image..."): 
            input_tensor = preprocess_image(input_image) 
            with torch.no_grad(): 
                output_tensor = model(input_tensor) 
            output_image = postprocess_image(output_tensor) 
 
        # Save enhanced image 
        output_path = "enhanced_output.png" 
        output_image.save(output_path) 
 
        # Prepare images for comparison 
        original_resized = input_image.resize(output_image.size) 
        original_np = np.array(original_resized).astype(np.float32) / 255.0 
        enhanced_np = np.array(output_image).astype(np.float32) / 255.0 
 
 
        # Compute metrics 
        psnr_val = peak_signal_noise_ratio(original_np, enhanced_np, data_range=1.0) 
        ssim_val = structural_similarity(original_np, enhanced_np, channel_axis=-1, data_range=1.0) 
 
        # Quality metrics 
        st.subheader(" Image Quality Metrics ") 
        st.markdown(f"**PSNR:** {psnr_val:.2f} dB") 
        st.markdown(f"**SSIM:** {ssim_val:.4f}") 
 
        # Side-by-side image display 
        col1, col2 = st.columns(2) 
        with col1: 
            st.image(original_resized, caption="Original (Resized)",use_container_width=True) 
        with col2: 
            st.image(output_image, caption="Enhanced", use_container_width=True) 
 
        # Download button 
        with open(output_path, "rb") as f: 
            st.download_button( 
                label=" Download Enhanced Image ", 
                data=f, 
                file_name="enhanced_output.png", 
                mime="image/png" 
            ) 
# Set plot style 
sns.set(style="whitegrid") 
# Sample metric values (replace with real values if available) 
metrics = { 
    "PSNR (dB)": {"RRDN": 32.5, "SRGAN": 28.9, "ESRGAN": 30.2}, 
    "SSIM": {"RRDN": 0.945, "SRGAN": 0.880, "ESRGAN": 0.912}, 
    "MSE": {"RRDN": 0.0021, "SRGAN": 0.0058, "ESRGAN": 0.0040}, 
    "RMSE": {"RRDN": 0.0458, "SRGAN": 0.0762, "ESRGAN": 0.0632}, 
    "Accuracy": {"RRDN": 93.2, "SRGAN": 85.6, "ESRGAN": 89.4}  # Simulated accuracy as % 
} 
def plot_metric(metric_name, values): 
models = list(values.keys()) 
scores = list(values.values()) 
plt.figure(figsize=(6, 4)) 
sns.barplot(x=models, y=scores, palette="viridis") 
plt.title(f"{metric_name} Comparison") 
plt.xlabel("Super-Resolution Models") 
plt.ylabel(metric_name) 
plt.ylim(0, max(scores) * 1.2) 
for i, v in enumerate(scores): 
plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontweight='bold') 
plt.tight_layout() 
plt.savefig(f"{metric_name.replace(' ', '_').replace('(', '').replace(')', '')}_Comparison.png") 
plt.show() 
# Generate and save all graphs 
for metric, values in metrics.items(): 
plot_metric(metric, values) 
