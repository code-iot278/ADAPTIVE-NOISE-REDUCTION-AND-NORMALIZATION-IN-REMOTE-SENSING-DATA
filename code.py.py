import numpy as np
import cv2
import matplotlib.pyplot as plt
from curvelops import FDCT2D

# ============================================
# 1. Load Image (Grayscale)
# ============================================
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found!")

img = img.astype(np.float32) / 255.0

# ============================================
# 2. Morphological Filtering
# ============================================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

# Opening (removes small noise)
opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Closing (fills small holes)
morph_filtered = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# ============================================
# 3. Fast Discrete Curvelet Transform (FDCT)
# ============================================
# Initialize FDCT
fdct = FDCT2D(morph_filtered.shape, nbscales=4, nbangles_coarse=16)

# Forward transform
coeffs = fdct.fwd(morph_filtered)

# ============================================
# 4. Thresholding (Denoising in Curvelet Domain)
# ============================================
threshold = 0.05

coeffs_thresh = []
for scale in coeffs:
    scale_thresh = []
    for arr in scale:
        scale_thresh.append(np.where(np.abs(arr) > threshold, arr, 0))
    coeffs_thresh.append(scale_thresh)

# ============================================
# 5. Inverse FDCT (Reconstruction)
# ============================================
reconstructed = fdct.inv(coeffs_thresh)

# Normalize
reconstructed = np.clip(reconstructed, 0, 1)

# ============================================
# 6. Visualization
# ============================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1, 3, 2)
plt.imshow(morph_filtered, cmap='gray')
plt.title("Morphological Filtered")

plt.subplot(1, 3, 3)
plt.imshow(reconstructed, cmap='gray')
plt.title("FDCT Reconstructed")

plt.tight_layout()
plt.show()



import numpy as np
import cv2
import matplotlib.pyplot as plt

# ============================================
# 1. Load Image (Grayscale)
# ============================================
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found!")

img = img.astype(np.float32) / 255.0

# ============================================
# 2. Anisotropic Diffusion (Perona-Malik)
# ============================================
def anisotropic_diffusion(img, num_iter=15, kappa=30, gamma=0.2):
    img = img.copy()

    for _ in range(num_iter):
        # Gradients
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east  = np.roll(img, -1, axis=1) - img
        west  = np.roll(img, 1, axis=1) - img

        # Diffusion coefficients (Perona-Malik)
        cN = np.exp(-(north/kappa)**2)
        cS = np.exp(-(south/kappa)**2)
        cE = np.exp(-(east/kappa)**2)
        cW = np.exp(-(west/kappa)**2)

        # Update
        img += gamma * (cN*north + cS*south + cE*east + cW*west)

    return img

diffused = anisotropic_diffusion(img, num_iter=15, kappa=25, gamma=0.2)

# ============================================
# 3. Kuwahara Filter
# ============================================
def kuwahara_filter(img, window_size=5):
    pad = window_size // 2
    padded = np.pad(img, pad, mode='reflect')
    output = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            x, y = i + pad, j + pad

            # Four subregions
            regions = [
                padded[x-pad:x+1, y-pad:y+1],
                padded[x-pad:x+1, y:y+pad+1],
                padded[x:x+pad+1, y-pad:y+1],
                padded[x:x+pad+1, y:y+pad+1]
            ]

            means = [np.mean(r) for r in regions]
            vars_ = [np.var(r) for r in regions]

            # Choose region with minimum variance
            output[i, j] = means[np.argmin(vars_)]

    return output

kuwahara = kuwahara_filter(diffused, window_size=5)

# ============================================
# 4. Visualization
# ============================================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(diffused, cmap='gray')
plt.title("Anisotropic Diffusion")

plt.subplot(1,3,3)
plt.imshow(kuwahara, cmap='gray')
plt.title("Kuwahara Filter Output")

plt.tight_layout()
plt.show()




import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# 1. Hypergraph Construction (Incidence Matrix)
# ============================================
def build_hypergraph(X, k=5):
    """
    X: Node features (N x F)
    k: neighbors per hyperedge
    Returns: H (N x E)
    """
    N = X.shape[0]
    dist = torch.cdist(X, X)  # pairwise distance

    H = torch.zeros((N, N))

    for i in range(N):
        idx = torch.topk(dist[i], k=k, largest=False).indices
        H[idx, i] = 1.0

    return H


# ============================================
# 2. Semantic Similarity
# ============================================
def semantic_similarity(X):
    X_norm = F.normalize(X, p=2, dim=1)
    return torch.mm(X_norm, X_norm.t())  # cosine similarity


# ============================================
# 3. Hypergraph Attention Layer
# ============================================
class HypergraphAttentionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.attn = nn.Parameter(torch.Tensor(out_features, 1))

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.attn)

    def forward(self, X, H):
        # Linear projection
        Xw = self.W(X)  # (N, F')

        # Semantic similarity
        S = semantic_similarity(Xw)

        # Hyperedge aggregation
        H_T = H.t()
        E_feat = torch.matmul(H_T, Xw)  # (E, F')

        # Attention scores
        attn_scores = torch.matmul(E_feat, self.attn)  # (E, 1)
        attn_scores = F.leaky_relu(attn_scores)

        # Normalize attention
        alpha = torch.softmax(attn_scores, dim=0)

        # Apply semantic adaptation
        S_mean = torch.mean(S, dim=1, keepdim=True)
        alpha = alpha * S_mean[:alpha.shape[0]]

        # Normalize again
        alpha = alpha / (alpha.sum() + 1e-8)

        # Hypergraph filtering
        E_weighted = E_feat * alpha
        X_new = torch.matmul(H, E_weighted)

        return X_new


# ============================================
# 4. Full Model
# ============================================
class SAHAF(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.hg1 = HypergraphAttentionLayer(in_dim, hidden_dim)
        self.hg2 = HypergraphAttentionLayer(hidden_dim, out_dim)

    def forward(self, X, H):
        X = F.relu(self.hg1(X, H))
        X = self.hg2(X, H)
        return X


# ============================================
# 5. Example Usage
# ============================================
if __name__ == "__main__":
    torch.manual_seed(0)

    # Dummy node features (N nodes, F features)
    N, F_dim = 50, 32
    X = torch.randn(N, F_dim)

    # Build hypergraph
    H = build_hypergraph(X, k=5)

    # Model
    model = SAHAF(in_dim=F_dim, hidden_dim=64, out_dim=16)

    # Forward pass
    output = model(X, H)

    print("Input shape :", X.shape)
    print("Output shape:", output.shape)


import numpy as np
import cv2
from skimage.filters.rank import entropy
from skimage.morphology import disk
import matplotlib.pyplot as plt

# ============================================
# 1. Load Image
# ============================================
img = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    raise ValueError("Image not found!")

img = img.astype(np.float32) / 255.0

# ============================================
# 2. Compute Local Entropy Map
# ============================================
# Convert to uint8 for entropy function
img_uint8 = (img * 255).astype(np.uint8)

entropy_map = entropy(img_uint8, disk(5))  # neighborhood radius=5
entropy_map = entropy_map.astype(np.float32)

# Normalize entropy map to [0,1]
entropy_norm = (entropy_map - entropy_map.min()) / (entropy_map.max() - entropy_map.min() + 1e-8)

# ============================================
# 3. Entropy-Guided Normalization
# ============================================
def entropy_guided_normalization(img, entropy_map):
    # Global normalization
    img_min, img_max = img.min(), img.max()
    global_norm = (img - img_min) / (img_max - img_min + 1e-8)

    # Local adaptive normalization
    mean = cv2.GaussianBlur(img, (7,7), 0)
    std  = cv2.GaussianBlur((img - mean)**2, (7,7), 0)
    std  = np.sqrt(std + 1e-8)

    local_norm = (img - mean) / (std + 1e-8)

    # Blend using entropy
    # High entropy → keep original/local
    # Low entropy → apply stronger normalization
    alpha = entropy_map  # weight map

    output = alpha * local_norm + (1 - alpha) * global_norm

    # Normalize final output
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    return output

egn_output = entropy_guided_normalization(img, entropy_norm)

# ============================================
# 4. Visualization
# ============================================
plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.imshow(img, cmap='gray')
plt.title("Original")

plt.subplot(1,3,2)
plt.imshow(entropy_norm, cmap='jet')
plt.title("Entropy Map")

plt.subplot(1,3,3)
plt.imshow(egn_output, cmap='gray')
plt.title("EGN Output")

plt.tight_layout()
plt.show()





import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# 1. Generator (Refinement Network)
# ============================================
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_channels, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)


# ============================================
# 2. Discriminator (PatchGAN Style)
# ============================================
class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


# ============================================
# 3. GAN Loss Functions
# ============================================
adversarial_loss = nn.BCEWithLogitsLoss()
reconstruction_loss = nn.L1Loss()   # better than MSE for images


# ============================================
# 4. Training Step
# ============================================
def train_step(generator, discriminator, real_img, optimizer_G, optimizer_D, device):
    
    batch_size = real_img.size(0)

    # Ground truth labels
    valid = torch.ones((batch_size, 1, 30, 30), device=device)
    fake  = torch.zeros((batch_size, 1, 30, 30), device=device)

    # ====================================
    # Train Generator
    # ====================================
    optimizer_G.zero_grad()

    # Input = noisy or degraded version
    noise = real_img + 0.05 * torch.randn_like(real_img)
    fake_img = generator(noise)

    pred_fake = discriminator(fake_img)

    loss_G_adv = adversarial_loss(pred_fake, valid)
    loss_G_rec = reconstruction_loss(fake_img, real_img)

    loss_G = loss_G_adv + 100 * loss_G_rec
    loss_G.backward()
    optimizer_G.step()

    # ====================================
    # Train Discriminator
    # ====================================
    optimizer_D.zero_grad()

    pred_real = discriminator(real_img)
    loss_real = adversarial_loss(pred_real, valid)

    pred_fake = discriminator(fake_img.detach())
    loss_fake = adversarial_loss(pred_fake, fake)

    loss_D = (loss_real + loss_fake) / 2
    loss_D.backward()
    optimizer_D.step()

    return loss_G.item(), loss_D.item()


# ============================================
# 5. Example Usage
# ============================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = Generator().to(device)
    D = Discriminator().to(device)

    optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

    # Dummy data (replace with real images)
    real_images = torch.randn(8, 1, 128, 128).to(device)

    for epoch in range(5):
        loss_G, loss_D = train_step(G, D, real_images, optimizer_G, optimizer_D, device)

        print(f"Epoch {epoch+1} | G Loss: {loss_G:.4f} | D Loss: {loss_D:.4f}")