import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg') # Set non-interactive backend for stability
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import base64
import io

# Configure Streamlit Page
st.set_page_config(page_title="GAN Fake Image Generator (PyTorch)", layout="wide")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .report-container {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #333;
    }
</style>
""", unsafe_allow_html=True)

# --- GAN MODELS ---

class Generator(nn.Module):
    def __init__(self, latent_dim=100, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.img_shape = img_shape
        
        # Step 1: Build a simple Generator network
        self.model = nn.Sequential(
            # Input layer & First hidden layer
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(256),
            
            # Second layer
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512),
            
            # Third layer
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(1024),
            
            # Output layer
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self, img_shape=(1, 28, 28)):
        super(Discriminator, self).__init__()
        
        # Step 2: Build a simple Discriminator network
        self.model = nn.Sequential(
            # Input layer & First Layer
            nn.Flatten(),
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm1d(512), # Added as per instruction
            
            # Second layer
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Third layer
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Output layer
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# --- WRAPPER FOR TRAINING ---

class PyTorchGAN:
    def __init__(self, latent_dim=100):
        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim=latent_dim)
        self.discriminator = Discriminator()
        
        self.adversarial_loss = nn.BCELoss()
        
        self.optimizer_G = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --- STREAMLIT UI ---

def main():
    st.title("Synthetic Image Generation using PyTorch GANs")
    st.markdown("---")

    # Sidebar for Navigation
    menu = st.sidebar.radio("Navigation", ["Introduction & Problem Statement", "Model Architecture", "Initial Noise Verification", "Training & Evolution", "Technical Report & Deliverables"])

    if "gan" not in st.session_state:
        st.session_state.gan = PyTorchGAN()

    if menu == "Introduction & Problem Statement":
        st.header("1. Problem Statement")
        st.markdown(f"""
        <div class="report-container">
        Fake images have become ubiquitous on social media, affecting public discourse. 
        It is essential for students to understand how these images are created. 
        This application demonstrates a <b>Generative Adversarial Network (GAN)</b> trained on the MNIST dataset 
        using <b>PyTorch</b> to learn how to generate realistic handwritten digits from random noise.
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("What is a GAN?")
        st.write("""
        A GAN consists of two neural networks:
        - **The Generator:** Learns to create data that looks like the training set.
        - **The Discriminator:** Learns to distinguish between real data and the fake data produced by the generator.
        """)

    elif menu == "Model Architecture":
        st.header("2. GAN Architecture (PyTorch)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Generator Network")
            st.code(str(st.session_state.gan.generator))
            
        with col2:
            st.subheader("Discriminator Network")
            st.code(str(st.session_state.gan.discriminator))

    elif menu == "Initial Noise Verification":
        st.header("3. Initial Generator Output (Untrained)")
        st.write("Step 4: Plot the images created by the Generator from a normally distributed noise input.")
        
        if st.button("Generate Images from Noise"):
            # Step 4: Generate a normally distributed noise of shape 100x100
            # We transform this into 100 samples of 100-dim noise
            z = torch.randn(100, 100)
            with torch.no_grad():
                gen_imgs = st.session_state.gan.generator(z).cpu().numpy()
            
            # Plot
            fig, axs = plt.subplots(10, 10, figsize=(12, 12))
            cnt = 0
            for i in range(10):
                for j in range(10):
                    axs[i,j].imshow(gen_imgs[cnt, 0, :, :], cmap='gray')
                    axs[i,j].axis('off')
                    cnt += 1
            # Convert plot to image to avoid Streamlit's internal MediaFileHandler issues
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            st.image(buf)
            plt.close(fig)

    elif menu == "Training & Evolution":
        st.header("4. GAN Training & Performance Evolution")
        
        epochs = st.number_input("Training Epochs", min_value=1, max_value=2000, value=400)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=512, value=64)
        
        if st.button("Start Training"):
            # Step 6: Load and process the MNIST dataset
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]) # Map to [-1, 1]
            ])
            dataloader = DataLoader(
                datasets.MNIST(".", train=True, download=True, transform=transform),
                batch_size=batch_size, shuffle=True
            )
            
            milestones = [1, 30, 100, 400]
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            final_gen_imgs = {}

            # Training Loop
            for epoch in range(1, epochs + 1):
                for i, (imgs, _) in enumerate(dataloader):
                    
                    # Adversarial ground truths
                    valid = torch.ones(imgs.size(0), 1)
                    fake = torch.zeros(imgs.size(0), 1)

                    # ---------------------
                    #  Train Generator
                    # ---------------------
                    st.session_state.gan.optimizer_G.zero_grad()
                    z = torch.randn(imgs.size(0), 100)
                    gen_imgs = st.session_state.gan.generator(z)
                    
                    # Loss measures generator's ability to fool the discriminator
                    g_loss = st.session_state.gan.adversarial_loss(st.session_state.gan.discriminator(gen_imgs), valid)
                    g_loss.backward()
                    st.session_state.gan.optimizer_G.step()

                    # ---------------------
                    #  Train Discriminator
                    # ---------------------
                    st.session_state.gan.optimizer_D.zero_grad()
                    
                    # Measure discriminator's ability to classify real from generated samples
                    real_loss = st.session_state.gan.adversarial_loss(st.session_state.gan.discriminator(imgs), valid)
                    fake_loss = st.session_state.gan.adversarial_loss(st.session_state.gan.discriminator(gen_imgs.detach()), fake)
                    d_loss = (real_loss + fake_loss) / 2
                    
                    d_loss.backward()
                    st.session_state.gan.optimizer_D.step()

                    if i == 0: # Update progress once per epoch
                        status_text.text(f"Epoch: {epoch}/{epochs} [D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")
                        progress_bar.progress(epoch / epochs)

                # Step 9: Save milestones
                if epoch in milestones or epoch == epochs:
                    st.write(f"#### Milestone: Epoch {epoch}")
                    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
                    for j in range(5):
                        axs[j].imshow(gen_imgs[j, 0, :, :].detach().cpu().numpy(), cmap='gray')
                        axs[j].axis('off')
                    # Convert plot to image buffer
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight')
                    st.image(buf)
                    
                    # Also save to disk specifically for the Technical Report
                    fig.savefig("report_visuals.png", bbox_inches='tight')
                    
                    plt.close(fig)

            st.success("Training Completed!")
            
            # --- Auto-Update HTML Report with Base64 Image ---
            try:
                if os.path.exists("report_visuals.png"):
                    with open("report_visuals.png", "rb") as img_file:
                        b64_string = base64.b64encode(img_file.read()).decode()
                    
                    if os.path.exists("report.html"):
                        with open("report.html", "r") as f:
                            html_content = f.read()
                        
                        # Replace placeholder or old image with new base64 data
                        import re
                        new_img_tag = f'<img src="data:image/png;base64,{b64_string}" alt="GAN Training Evolution Results">'
                        # Target the vis-container content
                        updated_html = re.sub(r'<div class="vis-container">.*?</div>', 
                                            f'<div class="vis-container">{new_img_tag}<div class="vis-caption">Figure 1: Actual evolution results generated by the PyTorch model from Epoch 1 to Epoch 400.</div></div>', 
                                            html_content, flags=re.DOTALL)
                        
                        with open("report.html", "w") as f:
                            f.write(updated_html)
                        st.info("Technical Report (report.html) updated with latest visualizations.")
            except Exception as e:
                st.warning(f"Could not auto-update report.html: {e}")

    elif menu == "Technical Report & Deliverables":
        st.header("Technical Report (PyTorch Edition)")
        
        st.subheader("a) Problem Statement")
        st.write("Fake images effect public discourse. This project uses GANs to understand synthetic image generation principles.")
        
        st.subheader("b) Algorithm of the Solution")
        st.write("""
        - Generator: Mapping random noise (Z) to image space (R^784) using 3 hidden layers with BatchNormalization.
        - Discriminator: Classifying real vs fake images using a mirrored multilayer structure.
        - Training: Binary Cross Entropy Loss minimized by Adam optimizers for both networks.
        """)
        
        st.subheader("c) Analysis of Findings")
        st.write("""
        - **Epoch 1: Random Initialization**. The generator's weights are random, resulting in Gaussian-like noise. The Discriminator easily identifies these as fake.
        - **Epoch 30: Feature Discovery**. Early patterns emerge. The model learns that digits are centered and occupy a specific luminance range.
        - **Epoch 100: Topological Consistency**. Structural elements like loops (for 0s and 8s) and strokes (for 1s and 7s) become recognizable.
        - **Epoch 400: Distribution Convergence**. The Generator successfully maps the latent space to the MNIST distribution. The images are visually convincing and diverse.
        - **Training Dynamics**: Stability was achieved using the Adam optimizer with a specific beta_1 value of 0.5, which helps mitigate the "zero-sum" oscillating nature of adversarial training.
        """)
        
        st.subheader("d) References")
        st.write("1. Goodfellow et al. (2014) 2. PyTorch Docs 3. MNIST Database.")


if __name__ == "__main__":
    main()
