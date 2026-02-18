"""
Streamlit App - Final Version For deplotment
1. Static Grad calculation that changes a little bit per image (saves image hash in computation)
2. REAL animated GIF showing transformation stages to make it easier to understand 
3. Horizontal layout with visual connections
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import io
import hashlib
import time  
from train_model import AlienCNN

st.set_page_config(page_title="IA Classificateur", page_icon="🛸", layout="wide")

st.markdown("""
<style>
    .main { padding: 0.5rem; }
    .prediction-correct { background: linear-gradient(135deg, #d1fae5, #a7f3d0); 
                         border: 3px solid #10b981; padding: 20px; border-radius: 15px; margin: 10px 0; }
    .prediction-wrong { background: linear-gradient(135deg, #fee2e2, #fecaca); 
                       border: 3px solid #ef4444; padding: 20px; border-radius: 15px; margin: 10px 0; }
    .species-card { padding: 15px; border-radius: 12px; margin: 8px 0; border-left: 5px solid; }
</style>
""", unsafe_allow_html=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_model():
    try:
        with open('models/metadata.json', 'r') as f:
            metadata = json.load(f)
        model = AlienCNN(num_classes=metadata['num_classes'])
        state_dict = torch.load('models/alien_classifier_best.pth', map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, metadata
    except Exception as e:
        st.error(f"❌ {e}")
        return None, None

def compute_gradcam_with_image_dependency(model, image_tensor, target_class, image_hash):
    """
    Static Grad with proper gradient computation
    Creates truly unique heatmaps for each image
    """
    model.eval()
    
    # Clone tensor and enable gradients
    x = image_tensor.clone().requires_grad_(True)
    
    # Storage for gradients and activations
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks on the features layer
    forward_handle = model.features.register_forward_hook(forward_hook)
    backward_handle = model.features.register_backward_hook(backward_hook)
    
    # Forward pass
    output, _ = model(x)
    
    # Get target score
    target_score = output[0, target_class]
    
    # Backward pass
    model.zero_grad()
    target_score.backward(retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Process gradients and activations
    grads = gradients[0][0].cpu().data.numpy()  # [C, H, W]
    acts = activations[0][0].cpu().data.numpy()  # [C, H, W]
    
    # Compute weights (global average pooling of gradients)
    weights = np.mean(grads, axis=(1, 2))  # [C]
    
    # Create CAM
    cam = np.zeros(acts.shape[1:], dtype=np.float32)  # [H, W]
    for i in range(len(weights)):
        cam += weights[i] * acts[i]
    
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam

def create_advanced_heatmap_visualizations(image, heatmap, predicted_class, metadata):
    """
    Create multiple advanced heatmap visualizations:
    1. Classic overlay
    2. Contour map
    3. Side-by-side comparison
    4. Highlighted regions
    """
    
    # Convert image to numpy
    image_np = np.array(image)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    height, width = image_np.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    visualizations = []
    
    # 1. CLASSIC OVERLAY with better colormap
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay_classic = cv2.addWeighted(image_np, 0.5, heatmap_colored, 0.5, 0)
    visualizations.append(("Overlay Classique", Image.fromarray(overlay_classic)))
    
    # 2. INTENSITY MAP - Shows heatmap strength
    intensity_map = (heatmap_resized * 255).astype(np.uint8)
    intensity_colored = cv2.applyColorMap(intensity_map, cv2.COLORMAP_TURBO)
    intensity_colored = cv2.cvtColor(intensity_colored, cv2.COLOR_BGR2RGB)
    visualizations.append(("Carte d'Intensité", Image.fromarray(intensity_colored)))
    
    # 3. CONTOUR MAP - Shows decision boundaries
    contour_img = image_np.copy()
    # Create contours at different levels
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        contour = (heatmap_resized > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Color based on threshold (hot to cold)
        if threshold > 0.7:
            color = (255, 0, 0)  # Red for high importance
        elif threshold > 0.5:
            color = (255, 165, 0)  # Orange
        else:
            color = (255, 255, 0)  # Yellow
        
        cv2.drawContours(contour_img, contours, -1, color, 2)
    
    visualizations.append(("Contours de Décision", Image.fromarray(contour_img)))
    
    # 4. HIGHLIGHTED REGIONS - Only show important parts
    mask = (heatmap_resized > 0.5).astype(np.uint8)
    mask_3d = np.stack([mask] * 3, axis=-1)
    highlighted = image_np * mask_3d
    
    # Add red tint to highlighted areas
    highlighted_tinted = highlighted.copy()
    highlighted_tinted[:, :, 0] = np.clip(highlighted_tinted[:, :, 0] + mask * 50, 0, 255)
    
    visualizations.append(("Régions Importantes", Image.fromarray(highlighted_tinted)))
    
    # 5. TRANSPARENCY OVERLAY - Variable transparency based on heatmap
    alpha_channel = (heatmap_resized * 0.7).astype(np.float32)
    overlay_alpha = image_np.copy().astype(np.float32)
    
    for i in range(3):
        overlay_alpha[:, :, i] = (
            image_np[:, :, i] * (1 - alpha_channel) + 
            heatmap_colored[:, :, i] * alpha_channel
        )
    
    overlay_alpha = np.clip(overlay_alpha, 0, 255).astype(np.uint8)
    visualizations.append(("Overlay Adaptatif", Image.fromarray(overlay_alpha)))
    
    # 6. FOCUS MAP - Blur unimportant regions
    blurred = cv2.GaussianBlur(image_np, (21, 21), 0)
    focus_mask = heatmap_resized[:, :, np.newaxis]
    focus_img = (image_np * focus_mask + blurred * (1 - focus_mask)).astype(np.uint8)
    visualizations.append(("Focus Intelligent", Image.fromarray(focus_img)))
    
    return visualizations

def create_heatmap_comparison_figure(image, visualizations, predicted_class, true_class, confidence, metadata):
    """
    Create a comprehensive comparison figure with all heatmap views
    """
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Title
    species_info = get_species_info()
    emoji_pred = species_info[predicted_class]['emoji']
    emoji_true = species_info[true_class]['emoji']
    
    fig.suptitle(
        f'Analyse Grad-CAM Avancée\n{emoji_true} Vrai: {true_class} → {emoji_pred} Prédit: {predicted_class} ({confidence*100:.1f}%)',
        fontsize=20, fontweight='bold', y=0.98
    )
    
    # Original image (larger)
    ax_orig = fig.add_subplot(gs[0:2, 0])
    ax_orig.imshow(image)
    ax_orig.set_title('📸 Image Originale', fontsize=16, fontweight='bold', pad=10)
    ax_orig.axis('off')
    ax_orig.add_patch(plt.Rectangle((0, 0), image.width-1, image.height-1, 
                                     fill=False, edgecolor='#3b82f6', linewidth=4))
    
    # Display all visualizations
    positions = [
        (0, 1), (0, 2),  # Top row
        (1, 1), (1, 2),  # Middle row
        (2, 0), (2, 1), (2, 2)  # Bottom row
    ]
    
    for idx, ((title, vis_img), pos) in enumerate(zip(visualizations, positions)):
        ax = fig.add_subplot(gs[pos[0], pos[1]])
        ax.imshow(vis_img)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=8)
        ax.axis('off')
        
        # Add colored border based on importance
        colors = ['#ef4444', '#f59e0b', '#10b981', '#3b82f6', '#8b5cf6', '#ec4899']
        ax.add_patch(plt.Rectangle((0, 0), vis_img.width-1, vis_img.height-1,
                                   fill=False, edgecolor=colors[idx % len(colors)], linewidth=3))
    
    # Add legend
    legend_ax = fig.add_subplot(gs[2, 0])
    legend_ax.axis('off')
    legend_text = """
🔴 Rouge: Zones critiques (>70%)
🟠 Orange: Zones importantes (>50%)
🟡 Jaune: Zones secondaires (>30%)
    
💡 Plus la zone est rouge/chaude,
   plus elle influence la décision
    """
    legend_ax.text(0.5, 0.5, legend_text, transform=legend_ax.transAxes,
                  fontsize=12, verticalalignment='center', horizontalalignment='center',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_moving_filter_animation(image, image_tensor, model, pred_class, metadata):
    """Create ULTIMATE animation with spectacular visual effects"""
    frames = []
    W, H = 2400, 1000  # Larger resolution for better text visibility
    
    # Denormalize
    img_np = image_tensor[0].cpu().detach().numpy().transpose(1, 2, 0)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_np = np.clip(std * img_np + mean, 0, 1)
    img_np = (img_np * 255).astype(np.uint8)
    
    # Get outputs
    with torch.no_grad():
        layers = list(model.features.children())
        conv1_out = layers[0](image_tensor)
        features_out = model.features(image_tensor)
        pooled = model.avgpool(features_out)
        logits = model.classifier(torch.flatten(pooled, 1))
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
    
    # Fonts
    try:
        font_huge = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 52)
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 40)
        font_large = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_med = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 26)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except:
        font_huge = font_title = font_large = font_med = font_small = None
    
    img_pil = Image.fromarray(img_np).resize((650, 650), Image.Resampling.LANCZOS)
    
    # ===== FRAME 1: Spectacular Intro with Gradient =====
    frame1 = Image.new('RGB', (W, H), 'white')
    draw1 = ImageDraw.Draw(frame1)
    
    # Animated gradient header
    for y in range(120):
        alpha = y / 120
        r = int(79 + (99 - 79) * alpha)
        g = int(70 + (102 - 70) * alpha)
        b = int(229 + (241 - 229) * alpha)
        draw1.rectangle([(0, y), (W, y+1)], fill=(r, g, b))
    
    # Glowing title
    draw1.text((W//2, 60), "ÉTAPE 1/10 • IMAGE D'ENTRÉE", fill='white', font=font_huge, anchor="mm")
    
    # Main image with shadow and glow
    shadow_offset = 15
    for offset in range(shadow_offset, 0, -2):
        alpha = int(50 * (1 - offset/shadow_offset))
        draw1.rectangle([
            (775-offset, 200-offset), 
            (1425+offset, 850+offset)
        ], outline=(0, 0, 0, alpha), width=2)
    
    frame1.paste(img_pil, (775, 200))
    
    # Triple border effect
    draw1.rectangle([(770, 195), (1430, 855)], outline='#4f46e5', width=8)
    draw1.rectangle([(765, 190), (1435, 860)], outline='#6366f1', width=3)
    draw1.rectangle([(760, 185), (1440, 865)], outline='#818cf8', width=1)
    
    # Info panel with gradient background
    panel_x, panel_y = 150, 300
    for py in range(450):
        shade = int(255 * (0.95 + 0.05 * (py/450)))
        draw1.rectangle([(panel_x, panel_y+py), (panel_x+500, panel_y+py+1)], fill=(shade, shade, shade))
    
    draw1.rectangle([(panel_x, panel_y), (panel_x+500, panel_y+450)], outline='#6366f1', width=4)
    
    # Icons and text
    draw1.text((panel_x+250, panel_y+80), "📸", font=font_huge, anchor="mm")
    draw1.text((panel_x+250, panel_y+160), "IMAGE RGB", fill='#1f2937', font=font_title, anchor="mm")
    draw1.text((panel_x+250, panel_y+240), "224 × 224 pixels", fill='#4b5563', font=font_large, anchor="mm")
    draw1.text((panel_x+250, panel_y+300), "3 canaux (R, G, B)", fill='#4b5563', font=font_large, anchor="mm")
    draw1.text((panel_x+250, panel_y+380), "⚡ Le voyage commence!", fill='#4f46e5', font=font_large, anchor="mm")
    
    # Progress indicator
    draw1.rectangle([(100, H-50), (2100, H-30)], outline='#e5e7eb', width=2)
    draw1.rectangle([(100, H-50), (310, H-30)], fill='#4f46e5')
    draw1.text((W//2, H-60), "1/10", fill='#6b7280', font=font_small, anchor="mm")
    
    frames.append(frame1)
    
    # ===== FRAME 2: RGB with Animated Split =====
    frame2 = Image.new('RGB', (W, H), 'white')
    draw2 = ImageDraw.Draw(frame2)
    
    # Gradient header
    for y in range(120):
        alpha = y / 120
        r = int(79 + (99 - 79) * alpha)
        g = int(70 + (102 - 70) * alpha)
        b = int(229 + (241 - 229) * alpha)
        draw2.rectangle([(0, y), (W, y+1)], fill=(r, g, b))
    
    draw2.text((W//2, 60), "ÉTAPE 2/10 • SÉPARATION RGB", fill='white', font=font_huge, anchor="mm")
    
    # Original image
    img_display = img_pil.resize((450, 450), Image.Resampling.LANCZOS)
    frame2.paste(img_display, (250, 250))
    draw2.rectangle([(245, 245), (705, 705)], outline='#4f46e5', width=6)
    draw2.text((475, 730), "Original", fill='#1f2937', font=font_large, anchor="mm")
    
    # Animated arrow with glow
    arrow_x = 850
    for i in range(5):
        size = 60 - i*8
        alpha_val = 255 - i*40
        color = (79, 70, 229, alpha_val)
        draw2.text((arrow_x + i*15, 450), "→", fill=color[:3], font=font_huge, anchor="mm")
    
    # RGB channels with enhanced visuals
    channels_data = [
        (img_np[:,:,0], np.zeros_like(img_np[:,:,0]), np.zeros_like(img_np[:,:,0])),
        (np.zeros_like(img_np[:,:,1]), img_np[:,:,1], np.zeros_like(img_np[:,:,1])),
        (np.zeros_like(img_np[:,:,2]), np.zeros_like(img_np[:,:,2]), img_np[:,:,2])
    ]
    colors_hex = ['#ef4444', '#10b981', '#3b82f6']
    labels = ['🔴 ROUGE', '🟢 VERT', '🔵 BLEU']
    explanations = ['Tons rouges', 'Tons verts', 'Tons bleus']
    
    for i, (ch_data, color, label, expl) in enumerate(zip(channels_data, colors_hex, labels, explanations)):
        x, y = 1200, 220 + i * 220
        ch_img = np.stack(ch_data, axis=-1)
        ch_pil = Image.fromarray((ch_img * 255).astype(np.uint8)).resize((220, 220), Image.Resampling.LANCZOS)
        
        # Glow effect
        for offset in range(8, 0, -2):
            alpha_glow = int(30 * (1 - offset/8))
            rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
            draw2.rectangle([
                (x-offset, y-offset), 
                (x+220+offset, y+220+offset)
            ], outline=rgb + (alpha_glow,), width=2)
        
        frame2.paste(ch_pil, (x, y))
        draw2.rectangle([(x-4, y-4), (x+224, y+224)], outline=color, width=6)
        draw2.text((x+110, y+240), label, fill=color, font=font_large, anchor="mm")
        draw2.text((x+110, y+270), expl, fill='#6b7280', font=font_small, anchor="mm")
    
    # Progress
    draw2.rectangle([(100, H-50), (2100, H-30)], outline='#e5e7eb', width=2)
    draw2.rectangle([(100, H-50), (520, H-30)], fill='#4f46e5')
    draw2.text((W//2, H-60), "2/10", fill='#6b7280', font=font_small, anchor="mm")
    
    frames.append(frame2)
    
    # ===== FRAMES 3-8: Moving Convolution with Enhanced Effects =====
    conv1_np = conv1_out[0].cpu().numpy()
    positions = [(40,40), (110,40), (180,110), (40,180), (110,180), (180,180)]
    names = ["Bords Verticaux", "Bords Horizontaux", "Textures", "Coins", "Formes", "Patterns"]
    colors_conv = ['#ec4899', '#f59e0b', '#8b5cf6', '#06b6d4', '#10b981', '#f43f5e']
    
    img_small = img_pil.resize((450, 450), Image.Resampling.LANCZOS)
    
    for filter_idx in range(6):
        frame = Image.new('RGB', (W, H), 'white')
        draw_f = ImageDraw.Draw(frame)
        
        # Gradient header
        color_main = colors_conv[filter_idx]
        rgb_main = tuple(int(color_main[i:i+2], 16) for i in (1, 3, 5))
        for y in range(120):
            alpha = y / 120
            r = int(rgb_main[0] * (0.6 + 0.4 * alpha))
            g = int(rgb_main[1] * (0.6 + 0.4 * alpha))
            b = int(rgb_main[2] * (0.6 + 0.4 * alpha))
            draw_f.rectangle([(0, y), (W, y+1)], fill=(r, g, b))
        
        draw_f.text((W//2, 60), f"ÉTAPE {filter_idx+3}/10 • {names[filter_idx].upper()}", 
                   fill='white', font=font_huge, anchor="mm")
        
        # Input image
        frame.paste(img_small, (200, 250))
        draw_f.rectangle([(195, 245), (655, 705)], outline=color_main, width=6)
        draw_f.text((425, 730), "Entrée", fill='#6b7280', font=font_med, anchor="mm")
        
        # Moving kernel box with trail effect
        kx, ky = positions[filter_idx]
        px, py = 200 + int(kx*1.8), 250 + int(ky*1.8)
        
        # Trail effect
        for trail in range(3):
            offset = trail * 15
            alpha_trail = 100 - trail * 30
            draw_f.rectangle([
                (px-offset, py-offset), 
                (px+100-offset, py+100-offset)
            ], outline=(245, 158, 11, alpha_trail), width=3)
        
        # Main kernel box
        draw_f.rectangle([(px, py), (px+100, py+100)], outline='#f59e0b', width=8)
        draw_f.text((px+50, py+110), "Kernel", fill='#f59e0b', font=font_small, anchor="mm")
        
        # Show what pixels the kernel is looking at with HEATMAP - MOVED TO RIGHT
        zoom_x, zoom_y = 1850, 250  # Moved to right side, higher up
        draw_f.rectangle([(zoom_x-10, zoom_y-40), (zoom_x+210, zoom_y-10)], 
                        fill='white', outline='#f59e0b', width=2)
        draw_f.text((zoom_x+100, zoom_y-25), "🔍 Pixels analysés", 
                   fill='#f59e0b', font=font_small, anchor="mm")
        
        # Extract the 3x3 region from image that kernel is looking at
        img_region = img_small.crop((int(kx*1.8), int(ky*1.8), 
                                     int(kx*1.8)+100, int(ky*1.8)+100))
        img_region_resized = img_region.resize((150, 150), Image.Resampling.LANCZOS)
        
        # Convert to grayscale intensity for heatmap visualization
        img_gray = np.array(img_region_resized.convert('L'))
        
        # Draw 3x3 grid with heatmap colors
        for i in range(3):
            for j in range(3):
                cell_x, cell_y = zoom_x + j*70, zoom_y + i*70
                
                # Get pixel intensity (average of 50x50 region)
                region_y1, region_y2 = i*50, (i+1)*50
                region_x1, region_x2 = j*50, (j+1)*50
                intensity = img_gray[region_y1:region_y2, region_x1:region_x2].mean()
                
                # Map intensity to color (blue=dark, red=bright)
                normalized = intensity / 255
                if normalized < 0.5:
                    # Dark pixels: Blue
                    t = normalized * 2
                    r, g, b = int(30 + 100*t), int(60 + 150*t), 255
                else:
                    # Bright pixels: Yellow to Red
                    t = (normalized - 0.5) * 2
                    r, g, b = 255, int(255 - 100*t), int(50 - 50*t)
                
                # Draw cell
                draw_f.rectangle([(cell_x, cell_y), (cell_x+70, cell_y+70)], 
                                fill=(r, g, b), outline='#374151', width=2)
                
                # Show intensity value
                draw_f.text((cell_x+35, cell_y+35), f"{int(intensity)}", 
                           fill='white' if normalized < 0.7 else 'black', 
                           font=font_small, anchor="mm")
        
        draw_f.text((zoom_x+105, zoom_y+220), "Valeurs pixel (0-255)", 
                   fill='#6b7280', font=font_small, anchor="mm")

        
        # Convolution operation display
        op_x, op_y = 800, 350
        
        # Kernel grid with COLORFUL HEATMAP + values
        draw_f.rectangle([(op_x, op_y), (op_x+300, op_y+300)], fill='white', outline='#f59e0b', width=5)
        draw_f.text((op_x+150, op_y-50), "Kernel 3×3", fill='#f59e0b', font=font_title, anchor="mm")
        draw_f.text((op_x+150, op_y-80), "🎨 Filtre Coloré", fill='#6b7280', font=font_small, anchor="mm")
        
        # 6 DIFFERENT kernels for each filter type
        all_kernels = [
            # Filter 0: Vertical edges (Sobel X)
            [['1', '0', '-1'],
             ['2', '0', '-2'],
             ['1', '0', '-1']],
            
            # Filter 1: Horizontal edges (Sobel Y)
            [['1', '2', '1'],
             ['0', '0', '0'],
             ['-1', '-2', '-1']],
            
            # Filter 2: Diagonal edges
            [['2', '1', '0'],
             ['1', '0', '-1'],
             ['0', '-1', '-2']],
            
            # Filter 3: Corner detection (Laplacian)
            [['0', '-1', '0'],
             ['-1', '2', '-1'],
             ['0', '-1', '0']],
            
            # Filter 4: High-pass filter
            [['-1', '-1', '-1'],
             ['-1', '2', '-1'],
             ['-1', '-1', '-1']],
            
            # Filter 5: Emboss filter
            [['-2', '-1', '0'],
             ['-1', '1', '1'],
             ['0', '1', '2']]
        ]
        
        kernel_values = all_kernels[filter_idx]
        
        # Create color mapping: -2 to +2 mapped to blue→white→red
        def value_to_color(val_str):
            val = float(val_str)
            # Normalize from -2,+2 to 0,1
            normalized = (val + 2) / 4
            
            if normalized < 0.5:
                # Blue to White (cold)
                t = normalized * 2
                r = int(30 + (255 - 30) * t)
                g = int(144 + (255 - 144) * t)
                b = int(255)
                temp_emoji = "❄️"
            else:
                # White to Red (hot)
                t = (normalized - 0.5) * 2
                r = int(255)
                g = int(255 - 100 * t)
                b = int(255 - 255 * t)
                temp_emoji = "🔥" if t > 0.5 else "☀️"
            
            return (r, g, b), temp_emoji
        
        for i in range(3):
            for j in range(3):
                cx, cy = op_x + j*100, op_y + i*100
                val_str = kernel_values[i][j]
                
                # Get color based on value
                bg_color, temp_emoji = value_to_color(val_str)
                
                # Draw cell with gradient background
                for grad_y in range(100):
                    alpha = grad_y / 100
                    # Subtle gradient for depth
                    shade_r = int(bg_color[0] * (0.85 + 0.15 * alpha))
                    shade_g = int(bg_color[1] * (0.85 + 0.15 * alpha))
                    shade_b = int(bg_color[2] * (0.85 + 0.15 * alpha))
                    draw_f.rectangle([
                        (cx, cy + grad_y), 
                        (cx + 100, cy + grad_y + 1)
                    ], fill=(shade_r, shade_g, shade_b))
                
                # Border
                draw_f.rectangle([(cx, cy), (cx+100, cy+100)], outline='#374151', width=3)
                
                # Temperature emoji (small, top corner)
                draw_f.text((cx+15, cy+15), temp_emoji, font=font_small, anchor="mm")
                
                # Large number in center
                val_color = '#1f2937' if val_str == '0' else 'white'
                draw_f.text((cx+50, cy+50), val_str, fill=val_color, 
                           font=font_title, anchor="mm")
                
                # Label below (for kids)
                if val_str == '2':
                    label = "Fort+"
                elif val_str == '1':
                    label = "Moyen+"
                elif val_str == '0':
                    label = "Neutre"
                elif val_str == '-1':
                    label = "Moyen-"
                else:
                    label = "Fort-"
                
                draw_f.text((cx+50, cy+80), label, fill=val_color if val_str == '0' else 'white', 
                           font=font_small, anchor="mm")
        
        # Legend for kids
        legend_x, legend_y = op_x, op_y + 320
        draw_f.rectangle([(legend_x, legend_y), (legend_x+300, legend_y+100)], 
                        fill='#f9fafb', outline='#d1d5db', width=2)
        draw_f.text((legend_x+150, legend_y+15), "🎓 Légende:", fill='#1f2937', font=font_small, anchor="mm")
        draw_f.text((legend_x+150, legend_y+40), "🔥 Rouge = Renforce", fill='#ef4444', font=font_small, anchor="mm")
        draw_f.text((legend_x+150, legend_y+60), "⚪ Blanc = Neutre", fill='#6b7280', font=font_small, anchor="mm")
        draw_f.text((legend_x+150, legend_y+80), "❄️ Bleu = Diminue", fill='#3b82f6', font=font_small, anchor="mm")
        
        # Convolution symbol
        draw_f.text((1200, 450), "⊛", fill='#f59e0b', font=font_huge, anchor="mm")
        draw_f.text((1200, 520), "=", fill='#6b7280', font=font_title, anchor="mm")
        
        # Output feature map with enhanced visualization
        fmap = conv1_np[filter_idx*2 % conv1_np.shape[0]]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        fmap = (fmap * 255).astype(np.uint8)
        fmap_colored = cv2.applyColorMap(fmap, cv2.COLORMAP_VIRIDIS)
        fmap_colored = cv2.cvtColor(fmap_colored, cv2.COLOR_BGR2RGB)
        fmap_pil = Image.fromarray(fmap_colored).resize((400, 400), Image.Resampling.LANCZOS)
        
        # Glow around feature map
        for offset in range(10, 0, -2):
            alpha_glow = int(40 * (1 - offset/10))
            draw_f.rectangle([
                (1400-offset, 300-offset),
                (1800+offset, 700+offset)
            ], outline=rgb_main + (alpha_glow,), width=2)
        
        frame.paste(fmap_pil, (1400, 300))
        draw_f.rectangle([(1395, 295), (1805, 705)], outline=color_main, width=6)
        draw_f.text((1600, 730), "Feature Map", fill=color_main, font=font_large, anchor="mm")
        
        # Explanation box FOR KIDS - adjusted for 2400px width
        expl_y = H - 160
        
        # Background with kid-friendly colors
        draw_f.rectangle([(150, expl_y), (W-150, expl_y+120)], 
                        fill='#fef3c7', outline='#f59e0b', width=4)
        
        # Icon
        draw_f.text((220, expl_y+60), "👶", font=font_huge, anchor="mm")
        
        # Simple explanation for kids
        kid_explanations = [
            "Le filtre cherche les lignes verticales | (debout)",
            "Le filtre cherche les lignes horizontales — (couchées)",  
            "Le filtre trouve les diagonales / \\ (en biais)",
            "Le filtre détecte les coins ⌞⌟ (angles)",
            "Le filtre repère les contrastes forts (différences de couleur)",
            "Le filtre crée des effets de relief (3D)"
        ]
        
        draw_f.text((380, expl_y+35), "💡 Pour les enfants:", 
                   fill='#92400e', font=font_large, anchor="lm")
        draw_f.text((380, expl_y+80), kid_explanations[filter_idx], 
                   fill='#78350f', font=font_med, anchor="lm")
        
        # Color explanation on the right
        draw_f.text((1700, expl_y+35), "🎨 Code couleur:", 
                   fill='#92400e', font=font_large, anchor="lm")
        draw_f.text((1700, expl_y+80), "🔥 Rouge=Fort • ⚪ Blanc=Moyen • ❄️ Bleu=Faible", 
                   fill='#78350f', font=font_med, anchor="lm")
        
        # Progress bar - adjusted for new width
        draw_f.rectangle([(100, H-50), (W-100, H-30)], outline='#e5e7eb', width=2)
        progress_w = 100 + ((filter_idx+3)/10) * (W-200)
        draw_f.rectangle([(100, H-50), (progress_w, H-30)], fill=color_main)
        draw_f.text((W//2, H-60), f"{filter_idx+3}/10", fill='#6b7280', font=font_small, anchor="mm")
        
        frames.append(frame)
    
    # ===== FRAME 9: Deep Features with Neon Effects =====
    frame9 = Image.new('RGB', (W, H), '#0f172a')
    draw9 = ImageDraw.Draw(frame9)
    
    # Neon header
    draw9.rectangle([(0, 0), (W, 120)], fill='#1e293b')
    draw9.text((W//2, 60), "ÉTAPE 9/10 • 512 FEATURES ABSTRAITES", 
              fill='#60a5fa', font=font_huge, anchor="mm")
    
    # Feature grid with neon glow
    feat_np = features_out[0].cpu().numpy()
    grid_colors = ['#3b82f6', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981', '#06b6d4']
    
    for i in range(36):  # 6x6 grid
        fmap = feat_np[i]
        fmap = (fmap - fmap.min()) / (fmap.max() - fmap.min() + 1e-8)
        fmap_pil = Image.fromarray((fmap*255).astype(np.uint8)).resize((130, 130), Image.Resampling.LANCZOS)
        
        x = 150 + (i % 6) * 150
        y = 200 + (i // 6) * 150
        
        # Neon glow
        glow_color = grid_colors[i % len(grid_colors)]
        rgb_glow = tuple(int(glow_color[j:j+2], 16) for j in (1, 3, 5))
        for offset in range(6, 0, -1):
            alpha = int(60 * (1 - offset/6))
            draw9.rectangle([
                (x-offset, y-offset),
                (x+130+offset, y+130+offset)
            ], outline=rgb_glow + (alpha,), width=2)
        
        frame9.paste(fmap_pil, (x, y))
        draw9.rectangle([(x-3, y-3), (x+133, y+133)], outline=glow_color, width=3)
    
    # Info panel
    panel_x = 1250
    draw9.rectangle([(panel_x, 250), (panel_x+800, 700)], fill='#1e293b', outline='#60a5fa', width=5)
    draw9.text((panel_x+400, 330), "🧩", font=font_huge, anchor="mm")
    draw9.text((panel_x+400, 420), "ABSTRACTION", fill='#60a5fa', font=font_title, anchor="mm")
    draw9.text((panel_x+400, 480), "MAXIMALE", fill='#60a5fa', font=font_title, anchor="mm")
    draw9.text((panel_x+400, 560), "512 features × 7×7", fill='#cbd5e1', font=font_large, anchor="mm")
    draw9.text((panel_x+400, 620), "= 25,088 valeurs", fill='#4ade80', font=font_large, anchor="mm")
    
    # Progress
    draw9.rectangle([(100, H-50), (2100, H-30)], outline='#60a5fa', width=2)
    draw9.rectangle([(100, H-50), (1900, H-30)], fill='#60a5fa')
    draw9.text((W//2, H-60), "9/10", fill='#cbd5e1', font=font_small, anchor="mm")
    
    frames.append(frame9)
    
    # ===== FRAME 10: Victory Screen with Animations =====
    frame10 = Image.new('RGB', (W, H), 'white')
    draw10 = ImageDraw.Draw(frame10)
    
    # Gradient victory header
    for y in range(120):
        alpha = y / 120
        r = int(5 + (16 - 5) * alpha)
        g = int(150 + (185 - 150) * alpha)
        b = int(105 + (129 - 105) * alpha)
        draw10.rectangle([(0, y), (W, y+1)], fill=(r, g, b))
    
    draw10.text((W//2, 60), "🎯 ÉTAPE 10/10 • PRÉDICTION FINALE", 
               fill='white', font=font_huge, anchor="mm")
    
    # Softmax bars
    classes = metadata['classes']
    colors_map = {'Krythik': '#10b981', 'Abyssal': '#3b82f6', 'Anthroide': '#8b5cf6', 'Fluffony': '#f59e0b'}
    emojis = {'Krythik': '🦎', 'Abyssal': '🐙', 'Anthroide': '🤖', 'Fluffony': '☁️'}
    
    for i, (cls, prob) in enumerate(zip(classes, probs.cpu().numpy())):
        y = 220 + i*140
        bar_w = int(prob*1500)
        color = colors_map[cls]
        rgb = tuple(int(color[j:j+2], 16) for j in (1, 3, 5))
        
        # 3D shadow
        for shadow_y in range(5):
            shadow_alpha = 40 - shadow_y*8
            draw10.rectangle([
                (405, y+5+shadow_y),
                (405+bar_w, y+95+shadow_y)
            ], fill=(0, 0, 0, shadow_alpha))
        
        # Gradient bar
        for bx in range(bar_w):
            shade = 0.5 + 0.5 * (bx/max(bar_w,1))
            bar_color = tuple(int(c * shade) for c in rgb)
            draw10.rectangle([(400+bx, y), (401+bx, y+90)], fill=bar_color)
        
        # Border
        draw10.rectangle([(400, y), (400+bar_w, y+90)], outline=color, width=6)
        
        # Label with emoji
        draw10.text((120, y+45), emojis[cls], font=font_huge, anchor="mm")
        draw10.text((250, y+45), cls, fill='#1f2937', font=font_title, anchor="mm")
        
        # Percentage
        draw10.text((1950, y+45), f"{prob*100:.1f}%", fill=color, font=font_title, anchor="mm")
        
        # Winner effects
        if i == pred_class:
            # Triple border
            draw10.rectangle([(395, y-5), (405+bar_w, y+95)], outline='#fbbf24', width=10)
            draw10.rectangle([(390, y-10), (410+bar_w, y+100)], outline='#fcd34d', width=3)
            
            # Crown with glow
            for glow in range(4):
                draw10.text((400+bar_w+80-glow*2, y+45-glow*2), "👑", font=font_huge, anchor="mm")
            draw10.text((400+bar_w+80, y+45), "👑", font=font_huge, anchor="mm")
            
            # Winner text
            draw10.text((2100, y+45), "GAGNANT!", fill='#f59e0b', font=font_large, anchor="mm")
    
    # Final message with animation
    predicted_species = classes[pred_class]
    msg_y = H - 130
    
    # Background with gradient
    for py in range(90):
        shade = int(255 * (0.95 - 0.15 * (py/90)))
        draw10.rectangle([(200, msg_y+py), (2000, msg_y+py+1)], fill=(shade, shade, 200))
    
    draw10.rectangle([(200, msg_y), (2000, msg_y+90)], outline='#f59e0b', width=6)
    draw10.text((W//2, msg_y+45), 
               f"🎉 C'EST UN {emojis[predicted_species]} {predicted_species.upper()} ! Confiance: {probs[pred_class].item()*100:.1f}%",
               fill='#92400e', font=font_title, anchor="mm")
    
    # Progress (complete)
    draw10.rectangle([(100, H-50), (2100, H-30)], outline='#10b981', width=2)
    draw10.rectangle([(100, H-50), (2100, H-30)], fill='#10b981')
    draw10.text((W//2, H-60), "10/10 ✓", fill='#6b7280', font=font_small, anchor="mm")
    
    frames.append(frame10)
    
    return frames


def overlay_heatmap(image, heatmap, alpha=0.5):
    """Overlay heatmap - with proper dimension handling"""
    # Convert image to numpy array
    image_np = np.array(image)
    
    # Ensure image is RGB (3 channels)
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
    
    # Get dimensions
    height, width = image_np.shape[:2]
    
    # Resize heatmap to match image dimensions
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Ensure both arrays have the same dtype
    image_np = image_np.astype(np.uint8)
    heatmap_colored = heatmap_colored.astype(np.uint8)
    
    # Verify dimensions match
    if image_np.shape != heatmap_colored.shape:
        # Force resize if dimensions don't match
        heatmap_colored = cv2.resize(heatmap_colored, (width, height))
    
    # Overlay
    overlayed = cv2.addWeighted(image_np, 1-alpha, heatmap_colored, alpha, 0)
    
    return Image.fromarray(overlayed)

def classify_image(model, image, metadata, image_num):
    """Classify with unique Grad-CAM"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Create unique hash for this image
    image_bytes = image.tobytes()
    image_hash = hashlib.md5(image_bytes).hexdigest()
    
    # Prediction
    with torch.no_grad():
        outputs, _ = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        pred_class = torch.argmax(probabilities).item()
        confidence = probabilities[pred_class].item()
    
    all_probs = [probabilities[i].item() for i in range(len(metadata['classes']))]
    
    # Grad-CAM with image dependency
    heatmap = compute_gradcam_with_image_dependency(model, image_tensor, pred_class, image_hash)
    
    # Create animation frames
    animation_frames = create_moving_filter_animation(image, image_tensor, model, pred_class, metadata)
    
    return pred_class, confidence, all_probs, heatmap, animation_frames

def load_random_images(num=12):
    images_dir = Path('images/aliens')
    csv_path = Path('images/class/classification.csv')
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    samples = df.sample(min(num, len(df)))
    image_data = []
    for _, row in samples.iterrows():
        img_num = str(row['Image']).zfill(3)
        for ext in ['.jpg', '.jpeg', '.png', '.webp']:
            img_path = images_dir / f"{img_num}{ext}"
            if img_path.exists():
                image_data.append({'path': str(img_path), 'label': row['Label'], 'number': row['Image']})
                break
    return image_data

def get_species_info():
    return {
        'Krythik': {'emoji': '🦎', 'color': '#10b981'},
        'Abyssal': {'emoji': '🐙', 'color': '#3b82f6'},
        'Anthroide': {'emoji': '🤖', 'color': '#8b5cf6'},
        'Fluffony': {'emoji': '☁️', 'color': '#f59e0b'}
    }

def main():
    # Ultimate header with gradient
    st.markdown("""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 20px; color: white; text-align: center; 
                margin-bottom: 30px; box-shadow: 0 10px 40px rgba(102, 126, 234, 0.4);'>
        <h1 style='margin: 0; color: white; font-size: 3em;'>🛸 Classificateur IA d'Espèces Aliens</h1>
        <p style='margin: 15px 0 0 0; font-size: 1.3em; opacity: 0.95;'>
            🎓 Exploration Interactive de l'Intelligence Artificielle
        </p>
        <p style='margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.85;'>
            Bienvenue dans la planète Zorbalia 🦎 🐙 🤖 ☁️
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    model, metadata = load_model()
    if model is None:
        st.stop()
    
    species_info = get_species_info()
    
    # Stats banner
    st.markdown("### 📊 Informations du Système")
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); 
                    padding: 20px; border-radius: 15px; text-align: center; 
                    border: 3px solid #3b82f6; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);'>
            <div style='font-size: 30px; margin-bottom: 10px;'>🧠</div>
            <div style='font-weight: bold; color: #1e40af; font-size: 1.4em;'>CNN Custom</div>
            <div style='font-size: 0.9em; color: #60a5fa; margin-top: 5px;'>Architecture</div>
            <div style='font-size: 0.8em; color: #93c5fd; margin-top: 5px;'>Basé sur ResNet18</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;
                    border: 3px solid #10b981; box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);'>
            <div style='font-size: 30px; margin-bottom: 10px;'>🎯</div>
            <div style='font-weight: bold; color: #166534; font-size: 1.4em;'>4 Espèces</div>
            <div style='font-size: 0.9em; color: #22c55e; margin-top: 5px;'>Classification</div>
            <div style='font-size: 0.8em; color: #4ade80; margin-top: 5px;'>Multi-classe</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat3:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;
                    border: 3px solid #f59e0b; box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);'>
            <div style='font-size: 30px; margin-bottom: 10px;'>⚡</div>
            <div style='font-weight: bold; color: #92400e; font-size: 1.4em;'>~100ms</div>
            <div style='font-size: 0.9em; color: #f59e0b; margin-top: 5px;'>Inférence</div>
            <div style='font-size: 0.8em; color: #fbbf24; margin-top: 5px;'>CPU optimisé</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat4:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #fce7f3 0%, #fbcfe8 100%); 
                    padding: 20px; border-radius: 15px; text-align: center;
                    border: 3px solid #ec4899; box-shadow: 0 4px 15px rgba(236, 72, 153, 0.3);'>
            <div style='font-size: 30px; margin-bottom: 10px;'>🎬</div>
            <div style='font-weight: bold; color: #9f1239; font-size: 1.4em;'>10 Étapes</div>
            <div style='font-size: 0.9em; color: #ec4899; margin-top: 5px;'>Animation</div>
            <div style='font-size: 0.8em; color: #f472b6; margin-top: 5px;'>2200×900px</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick guide
    with st.expander("🚀 Guide Rapide - Comment utiliser cette application", expanded=False):
        col_guide1, col_guide2, col_guide3 = st.columns(3)
        
        with col_guide1:
            st.markdown("""
            #### 🎯 Onglet 1: Classification
            
            1. **Sélectionnez** une image dans la grille
            2. **Observez** la prédiction du réseau de Neurone
            3. **Explorez** les 6 visualisations de Grad
            4. **Ajustez** les sliders de contrôle
            5. **Téléchargez** vos visualisations préférées
            
            💡 **Astuce:** Changez d'image pour voir comment la heatmap évolue!
            """)
        
        with col_guide2:
            st.markdown("""
            #### 🎬 Onglet 2: Animation
            
            1. **Mode Manuel:** Utilisez le slider pour naviguer
            2. **Mode Auto:** Cliquez "▶️ Lancer l'Animation"
            3. **Vitesse:** Ajustez entre 0.5s et 2.5s par frame
            4. **Export:** Créez un GIF téléchargeable
            5. **Observez:** Les filtres en mouvement!
            
            💡 **Astuce:** Effacer le cache en cas de problème!
            """)
        
        with col_guide3:
            st.markdown("""
            #### 📊 Fonctionnalités Avancées
            
            - **6 types de Grad** différents
            - **Heatmap colorée** sur les filtres
            - **Pixels analysés** en temps réel
            - **Métriques** d'activation
            - **Explications** multi-niveaux
            
            🎓 **Éducatif:** C'est cool et fun!
            """)
    
    tab1, tab2 = st.tabs([
        "🎯 Classification & Grad Avancé", 
        "🎬 Animation Pédagogique Interactive"
    ])
    
    if 'random_images' not in st.session_state:
        st.session_state.random_images = load_random_images(12)
    
    with tab1:
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.subheader("🎲 Images")
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🔄 Nouvelles", width='stretch', key="btn_new_images"):
                    st.session_state.random_images = load_random_images(12)
                    # Clear cache to avoid file errors
                    if 'selected_image' in st.session_state:
                        del st.session_state.selected_image
                    if 'last_analyzed' in st.session_state:
                        del st.session_state.last_analyzed
                    st.cache_data.clear()
                    st.rerun()
            
            with col_btn2:
                if st.button("🧹 Nettoyer", width='stretch', key="btn_clear_cache"):
                    # Clear all cache
                    st.cache_data.clear()
                    st.cache_resource.clear()
                    st.success("Cache nettoyé!")
                    st.rerun()
            
            if st.session_state.random_images:
                cols = st.columns(3)
                for i, img_data in enumerate(st.session_state.random_images):
                    with cols[i % 3]:
                        try:
                            img = Image.open(img_data['path'])
                            st.image(img, width=150)
                            emoji = species_info[img_data['label']]['emoji']
                            if st.button(f"{emoji} #{img_data['number']}", key=f"img_{i}", width='stretch'):
                                st.session_state.selected_image = img_data
                                st.session_state.last_analyzed = None
                                st.rerun()
                        except Exception as e:
                            st.error(f"Erreur image #{img_data.get('number', '?')}")
                            if st.button(f"🔄 Recharger", key=f"reload_{i}", width='stretch'):
                                st.cache_data.clear()
                                st.rerun()
        
        with col_right:
            st.subheader("🔍 Analyse")
            
            if 'selected_image' in st.session_state:
                img_data = st.session_state.selected_image
                image = Image.open(img_data['path'])
                
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.image(image, width=400)
                
                if st.session_state.get('last_analyzed') != img_data['number']:
                    with st.spinner(f'🧠 Analyse de l\'image #{img_data["number"]}...'):
                        pred_class, confidence, all_probs, heatmap, anim_frames = classify_image(
                            model, image, metadata, img_data['number'])
                        predicted = metadata['classes'][pred_class]
                        
                        st.session_state.pred = predicted
                        st.session_state.conf = confidence
                        st.session_state.probs = all_probs
                        st.session_state.heatmap = heatmap
                        st.session_state.anim_frames = anim_frames
                        st.session_state.last_analyzed = img_data['number']
                        st.session_state.current_image = image
                
                predicted = st.session_state.pred
                confidence = st.session_state.conf
                heatmap = st.session_state.heatmap
                all_probs = st.session_state.probs
                
                st.markdown("---")
                st.markdown("### 📊 Résultats de l'Analyse")
                
                # REORGANIZED: All info at same level with 3 columns
                res_col1, res_col2, res_col3 = st.columns(3)
                
                # Column 1: AI Prediction + Confidence
                with res_col1:
                    st.markdown("#### 🤖 Prédiction IA")
                    emoji = species_info[predicted]['emoji']
                    color = species_info[predicted]['color']
                    st.markdown(f"""
                    <div style='padding: 20px; background: linear-gradient(135deg, {color}15, {color}30); 
                                border-radius: 10px; border-left: 5px solid {color}; text-align: center;'>
                        <div style='font-size: 3em;'>{emoji}</div>
                        <div style='font-size: 1.5em; font-weight: bold; color: {color}; margin-top: 10px;'>
                            {predicted}
                        </div>
                        <div style='font-size: 2em; font-weight: bold; color: #f2c13a; margin-top: 10px;'>
                            {confidence*100:.1f}%
                        </div>
                        <div style='font-size: 0.9em; color: #b8a427; margin-top: 5px;'>
                            Confiance
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Column 2: All Class Probabilities
                with res_col2:
                    st.markdown("#### 📈 Toutes les Probabilités")
                    for cls, prob in zip(metadata['classes'], all_probs):
                        cls_color = species_info[cls]['color']
                        cls_emoji = species_info[cls]['emoji']
                        is_predicted = (cls == predicted)
                        
                        # Bar with gradient
                        bar_width = int(prob * 100)
                        border = "3px solid #fbbf24" if is_predicted else "1px solid #e5e7eb"
                        
                        st.markdown(f"""
                        <div style='margin: 8px 0; padding: 8px; background: grey; 
                                    border-radius: 8px; border: {border};'>
                            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;'>
                                <span style='font-weight: 600;'>{cls_emoji} {cls}</span>
                                <span style='font-weight: bold; color: {cls_color};'>{prob*100:.1f}%</span>
                            </div>
                            <div style='width: 100%; height: 10px; background: #f3f4f6; border-radius: 5px; overflow: hidden;'>
                                <div style='width: {bar_width}%; height: 100%; background: {cls_color};'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Column 3: Real Class
                with res_col3:
                    st.markdown("#### ✅ Espèce Réelle")
                    real_emoji = species_info[img_data['label']]['emoji']
                    real_color = species_info[img_data['label']]['color']
                    is_correct = (predicted == img_data['label'])
                    
                    st.markdown(f"""
                    <div style='padding: 20px; background: linear-gradient(135deg, {real_color}15, {real_color}30); 
                                border-radius: 10px; border-left: 5px solid {real_color}; text-align: center;'>
                        <div style='font-size: 3em;'>{real_emoji}</div>
                        <div style='font-size: 1.5em; font-weight: bold; color: {real_color}; margin-top: 10px;'>
                            {img_data['label']}
                        </div>
                        <div style='font-size: 1.2em; margin-top: 15px;'>
                            {'✅ CORRECT' if is_correct else '❌ INCORRECT'}
                        </div>
                        <div style='font-size: 0.9em; color: #b8a427; margin-top: 5px;'>
                            Image #{img_data['number']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # GRAD-CAM SECTION - ULTIMATE VERSION
                st.markdown("#### 🎨 Analyse Grad-CAM Avancée")
                st.markdown("**Grad-CAM** (Gradient-weighted Class Activation Mapping) montre quelles parties de l'image influencent la décision du réseau.")
                
                # Create advanced visualizations
                with st.spinner('🔬 Génération des visualisations avancées...'):
                    visualizations = create_advanced_heatmap_visualizations(
                        image, heatmap, predicted, metadata
                    )
                
                # Tabs for different visualizations
                viz_tabs = st.tabs([
                    "🔥 Vue d'ensemble",
                    "🎯 Overlay Classique", 
                    "📊 Carte d'Intensité",
                    "🗺️ Contours",
                    "✨ Régions Clés",
                    "🔍 Focus"
                ])
                
                # Tab 1: Overview with all visualizations
                with viz_tabs[0]:
                    st.markdown(f"### Toutes les Visualisations - Image #{img_data['number']}")
                    
                    # Create comparison figure
                    comparison_fig = create_heatmap_comparison_figure(
                        image, visualizations, predicted, img_data['label'], confidence, metadata
                    )
                    st.pyplot(comparison_fig)
                    plt.close()
                    
                    # Statistics
                    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                    with col_stat1:
                        max_activation = heatmap.max()
                        st.metric("Activation Max", f"{max_activation:.3f}")
                    with col_stat2:
                        mean_activation = heatmap.mean()
                        st.metric("Activation Moyenne", f"{mean_activation:.3f}")
                    with col_stat3:
                        important_pixels = (heatmap > 0.5).sum() / heatmap.size * 100
                        st.metric("Pixels Importants", f"{important_pixels:.1f}%")
                    with col_stat4:
                        critical_pixels = (heatmap > 0.7).sum() / heatmap.size * 100
                        st.metric("Pixels Critiques", f"{critical_pixels:.1f}%")
                
                # Tab 2: Classic Overlay
                with viz_tabs[1]:
                    st.markdown("### Overlay Classique (50/50)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Originale", use_column_width=True)
                    with col2:
                        st.image(visualizations[0][1], caption=visualizations[0][0], use_column_width=True)
                    
                    st.info("🎨 Les zones rouges/jaunes indiquent les régions que le réseau examine le plus attentivement.")
                
                # Tab 3: Intensity Map
                with viz_tabs[2]:
                    st.markdown("### Carte d'Intensité (Colormap Turbo)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Originale", use_column_width=True)
                    with col2:
                        st.image(visualizations[1][1], caption=visualizations[1][0], use_column_width=True)
                    
                    st.info("🌡️ Colormap Turbo : Violet (faible) → Bleu → Vert → Jaune → Rouge (fort)")
                
                # Tab 4: Contours
                with viz_tabs[3]:
                    st.markdown("### Contours de Décision (Multi-niveaux)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Originale", use_column_width=True)
                    with col2:
                        st.image(visualizations[2][1], caption=visualizations[2][0], use_column_width=True)
                    
                    st.markdown("""
                    **Légende des contours:**
                    - 🔴 Rouge: Zones critiques (>90% importance)
                    - 🟠 Orange: Zones très importantes (>70%)
                    - 🟡 Jaune: Zones importantes (>50%)
                    - ⚪ Non marqué: Zones secondaires (<30%)
                    """)
                
                # Tab 5: Important Regions
                with viz_tabs[4]:
                    st.markdown("### Régions Clés Isolées")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Originale", use_column_width=True)
                    with col2:
                        st.image(visualizations[3][1], caption=visualizations[3][0], use_column_width=True)
                    
                    st.warning("⚡ Seules les zones avec >50% d'activation sont affichées (avec teinte rouge)")
                
                # Tab 6: Smart Focus
                with viz_tabs[5]:
                    st.markdown("### Focus Intelligent (Blur Adaptatif)")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(image, caption="Originale", use_column_width=True)
                    with col2:
                        st.image(visualizations[5][1], caption=visualizations[5][0], use_column_width=True)
                    
                    st.success("✨ Les zones importantes restent nettes, le reste est flou - comme l'attention du réseau!")
                
                # Additional controls
                st.markdown("---")
                st.markdown("### 🛠️ Contrôles Avancés")
                
                col_ctrl1, col_ctrl2 = st.columns(2)
                
                with col_ctrl1:
                    overlay_strength = st.slider(
                        "Intensité de l'overlay",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.6,
                        step=0.1,
                        help="Ajustez la transparence de la heatmap"
                    )
                    
                    if overlay_strength != 0.6:
                        # Regenerate with custom strength
                        custom_overlay = overlay_heatmap(image, heatmap, alpha=overlay_strength)
                        st.image(custom_overlay, caption=f"Overlay personnalisé (α={overlay_strength})", use_column_width=True)
                
                with col_ctrl2:
                    threshold = st.slider(
                        "Seuil de détection",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="Seuil pour définir les zones 'importantes'"
                    )
                    
                    # Show binary mask
                    mask = (heatmap > threshold).astype(np.float32)
                    mask_resized = cv2.resize(mask, (image.width, image.height))
                    mask_colored = (mask_resized * 255).astype(np.uint8)
                    mask_pil = Image.fromarray(mask_colored)
                    st.image(mask_pil, caption=f"Masque binaire (seuil={threshold})", use_column_width=True)
                
                # Explanation box
                with st.expander("ℹ️ Comment interpréter le Grad-CAM", expanded=False):
                    st.markdown("""
                    ### 🧠 Comprendre le Grad-CAM
                    
                    **Grad-CAM** utilise les gradients du réseau pour identifier quelles régions de l'image 
                    sont les plus importantes pour la prédiction.
                    
                    #### 🔬 Méthode:
                    1. Le réseau fait une prédiction
                    2. On calcule les gradients par rapport aux features de la dernière couche convolutive
                    3. On pondère les feature maps par ces gradients
                    4. On agrège pour créer une heatmap de saillance
                    
                    #### 🎨 Interprétation des couleurs:
                    - **Rouge/Chaud** : Zones où le modèle "regarde" intensément
                    - **Bleu/Froid** : Zones ignorées par le modèle
                    - **Vert/Jaune** : Zones d'importance moyenne
                    
                    #### ⚠️ Limitations:
                    - La heatmap montre où le modèle regarde, pas si c'est correct
                    - Un modèle peut se concentrer sur les bonnes zones mais se tromper quand même
                    - Les zones sombres ne sont pas forcément sans importance pour d'autres classes
                    
                    #### 💡 Utilisation:
                    - **Debug** : Vérifier si le modèle regarde les bonnes features
                    - **Explication** : Rendre les décisions IA plus transparentes
                    - **Confiance** : Si les zones importantes sont cohérentes, la prédiction est plus fiable
                    """)
                
                # Download options
                st.markdown("---")
                st.markdown("### 💾 Télécharger les Visualisations")
                
                download_cols = st.columns(3)
                with download_cols[0]:
                    if st.button("📥 Télécharger Overlay", width='stretch', key="btn_dl_overlay"):
                        buf = io.BytesIO()
                        visualizations[0][1].save(buf, format='PNG')
                        st.download_button(
                            label="⬇️ Overlay.png",
                            data=buf.getvalue(),
                            file_name=f"gradcam_overlay_{img_data['number']}.png",
                            mime="image/png",
                            width='stretch'
                        )
                
                with download_cols[1]:
                    if st.button("📥 Télécharger Contours", width='stretch', key="btn_dl_contours"):
                        buf = io.BytesIO()
                        visualizations[2][1].save(buf, format='PNG')
                        st.download_button(
                            label="⬇️ Contours.png",
                            data=buf.getvalue(),
                            file_name=f"gradcam_contours_{img_data['number']}.png",
                            mime="image/png",
                            width='stretch'
                        )
                
                with download_cols[2]:
                    if st.button("📥 Télécharger Focus", width='stretch', key="btn_dl_focus"):
                        buf = io.BytesIO()
                        visualizations[5][1].save(buf, format='PNG')
                        st.download_button(
                            label="⬇️ Focus.png",
                            data=buf.getvalue(),
                            file_name=f"gradcam_focus_{img_data['number']}.png",
                            mime="image/png",
                            width='stretch'
                        )
                
                st.success("🎉 Changez d'image pour voir comment la heatmap varie!")
                
                st.markdown("---")
                st.markdown("#### 🎯 Probabilités")
                # Create dict from list for sorting
                probs_dict = {metadata['classes'][i]: st.session_state.probs[i] for i in range(len(metadata['classes']))}
                for species, prob in sorted(probs_dict.items(), key=lambda x: x[1], reverse=True):
                    emoji = species_info[species]['emoji']
                    color = species_info[species]['color']
                    badges = ""
                    if species == img_data['label']:
                        badges += " ✅"
                    if species == predicted:
                        badges += " 🎯"
                    st.markdown(f"""<div class="species-card" style="border-left-color: {color};">
                        <div style="display: flex; justify-content: space-between;">
                            <div><strong>{emoji} {species}</strong>{badges}</div>
                            <div style="font-size: 20px; font-weight: bold; color: {color};">{prob*100:.1f}%</div>
                        </div></div>""", unsafe_allow_html=True)
                    st.progress(prob)
            else:
                st.info("👈 Sélectionnez une image")
    
    with tab2:
        st.subheader("🎬 Animation: Propagation Forward Complète")
        
        if 'anim_frames' in st.session_state and 'selected_image' in st.session_state:
            img_data = st.session_state.selected_image
            st.success(f"🎯 Animation pour Image #{img_data['number']} - Espèce: {st.session_state.pred}")
            
            # Info box
            with st.expander("ℹ️ À propos de cette animation", expanded=False):
                st.markdown("""
                **Cette animation montre le parcours complet d'une image à travers le réseau de neurones:**
                
                1. 📸 **Image Originale** - L'image brute en entrée (224×224 pixels, 3 canaux RGB)
                2. 🎨 **Décomposition RGB** - Séparation en canaux Rouge, Vert, Bleu
                3. 🔬 **Filtres Convolutifs 1-4** - 4 filtres différents détectent bords, textures, motifs
                4. 🧩 **Features Profondes** - 512 features abstraites après 4 blocs ResNet
                5. 🎯 **Softmax Final** - Probabilités pour chaque espèce alien
                
                **Chaque étape transforme progressivement l'image en une représentation plus abstraite.**
                """)
            
            st.markdown("---")
            
            # Create two modes: Manual (slider) and Auto-play
            mode_col1, mode_col2 = st.columns(2)
            
            with mode_col1:
                st.markdown("### 🎮 Mode Manuel")
                frame_selector = st.slider(
                    "Naviguez dans les étapes", 
                    1, 
                    len(st.session_state.anim_frames), 
                    1, 
                    key="frame_slider",
                    help="Déplacez le curseur pour voir chaque étape"
                )
                
                # Display selected frame - FULL WIDTH
                st.image(st.session_state.anim_frames[frame_selector - 1], width='stretch')
                
                # Step description
                step_descriptions = [
                    "📸 **Étape 1/10:** Image d'entrée brute - Le réseau reçoit une image RGB de 224×224 pixels",
                    "🎨 **Étape 2/10:** Décomposition RGB - L'image est séparée en 3 canaux de couleur indépendants",
                    "🔬 **Étape 3/10:** Convolution #1 - Le kernel se déplace en position (50,50) - Détection bords verticaux",
                    "🔬 **Étape 4/10:** Convolution #2 - Le kernel se déplace en position (100,50) - Détection bords horizontaux",
                    "🔬 **Étape 5/10:** Convolution #3 - Le kernel se déplace en position (150,100) - Extraction de textures",
                    "🔬 **Étape 6/10:** Convolution #4 - Le kernel se déplace en position (50,150) - Détection de coins",
                    "🔬 **Étape 7/10:** Convolution #5 - Le kernel se déplace en position (100,150) - Reconnaissance de formes",
                    "🔬 **Étape 8/10:** Convolution #6 - Le kernel se déplace en position (150,150) - Analyse de patterns",
                    "🧩 **Étape 9/10:** Features Profondes - 512 représentations abstraites de haut niveau (affichage de 32)",
                    "🎯 **Étape 10/10:** Prédiction Finale Softmax - Le réseau décide: c'est un **{}**!".format(st.session_state.pred)
                ]
                
                st.info(step_descriptions[frame_selector - 1])
            
            with mode_col2:
                st.markdown("### ▶️ Mode Automatique")
                st.markdown("Regardez l'animation complète se dérouler automatiquement:")
                
                # Speed control
                speed = st.select_slider(
                    "Vitesse d'animation",
                    options=[0.5, 1.0, 1.5, 2.0, 2.5],
                    value=1.5,
                    format_func=lambda x: f"{x}s par étape",
                    help="Choisissez la durée d'affichage de chaque frame"
                )
                
                # Auto-play button
                if st.button("▶️ Lancer l'Animation", width='stretch', type="primary", key="btn_play_animation"):
                    # Animation container
                    animation_placeholder = st.empty()
                    
                    with animation_placeholder.container():
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        frame_display = st.empty()
                        
                        for i, frame in enumerate(st.session_state.anim_frames):
                            # Update progress
                            progress = (i + 1) / len(st.session_state.anim_frames)
                            progress_bar.progress(progress)
                            
                            # Display frame at FULL WIDTH
                            frame_display.image(frame, width='stretch')
                            
                            # Update status with description
                            status_text.info(step_descriptions[i])
                            
                            # Wait
                            time.sleep(speed)
                        
                        # Final message
                        status_text.success(f"✅ Animation terminée! Le réseau a prédit: **{st.session_state.pred}** avec {st.session_state.conf*100:.1f}% de confiance")
                        progress_bar.empty()
                        time.sleep(2)
                
                # Download frames option
                st.markdown("---")
                st.markdown("### 💾 Télécharger l'Animation")
                
                if st.button("📥 Créer GIF Animé", width='stretch', key="btn_create_gif"):
                    with st.spinner("Création du GIF..."):
                        # Create GIF
                        gif_path = f"animation_{img_data['number']}.gif"
                        st.session_state.anim_frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=st.session_state.anim_frames[1:],
                            duration=int(speed * 1000),
                            loop=0
                        )
                        
                        with open(gif_path, 'rb') as f:
                            st.download_button(
                                label="⬇️ Télécharger le GIF",
                                data=f,
                                file_name=f"alien_classification_{img_data['number']}.gif",
                                mime="image/gif",
                                width='stretch'
                            )
                        
                        st.success("GIF créé avec succès!")
        else:
            st.info("👈 Analysez une image dans l'onglet 1 pour voir l'animation")
            
            # Show attractive preview
            st.markdown("### 🎬 Aperçu de l'Animation")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **🔄 Transformation Progressive**
                
                L'animation montre comment le réseau:
                - Décompose l'image en canaux RGB
                - Applique des filtres convolutifs
                - Extrait des features abstraites
                - Prend une décision finale
                """)
            
            with col2:
                st.markdown("""
                **📊 8 Étapes Détaillées**
                
                1. Image originale
                2. Canaux RGB
                3-6. 4 filtres différents
                7. Feature maps
                8. Prédiction Softmax
                """)
            
            with col3:
                st.markdown("""
                **🎮 Contrôles Interactifs**
                
                - Slider pour navigation manuelle
                - Vitesse d'animation réglable
                - Mode auto-play
                - Export en GIF
                """)
    
    # Perfect footer - Simplified for Streamlit compatibility
    st.markdown("---")
    
    # Use columns for better layout
    st.markdown("### 🎓 À Propos de cette Application")
    st.markdown("**Classification d'Images par Deep Learning**")
    st.markdown("")
    
    # Technology cards using Streamlit columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='font-size: 3em;'>🧠</div>
            <div style='font-weight: bold; color: #1f2937; margin-top: 10px;'>CNN Custom</div>
            <div style='font-size: 0.9em; color: #6b7280; margin-top: 5px;'>Basé sur ResNet18</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='font-size: 3em;'>🎨</div>
            <div style='font-weight: bold; color: #1f2937; margin-top: 10px;'>Grad-CAM</div>
            <div style='font-size: 0.9em; color: #6b7280; margin-top: 5px;'>6 visualisations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='font-size: 3em;'>👶</div>
            <div style='font-weight: bold; color: #1f2937; margin-top: 10px;'>Multi-niveaux</div>
            <div style='font-size: 0.9em; color: #6b7280; margin-top: 5px;'>5 ans → Expert</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
            <div style='font-size: 3em;'>⚡</div>
            <div style='font-weight: bold; color: #1f2937; margin-top: 10px;'>Performance</div>
            <div style='font-size: 0.9em; color: #6b7280; margin-top: 5px;'>~100ms (CPU)</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("")
    st.markdown("### 🛠️ Technologies Utilisées")
    
    col_tech1, col_tech2, col_tech3, col_tech4, col_tech5 = st.columns(5)
    with col_tech1:
        st.markdown("**PyTorch**")
    with col_tech2:
        st.markdown("**Streamlit**")
    with col_tech3:
        st.markdown("**OpenCV**")
    with col_tech4:
        st.markdown("**Matplotlib**")
    with col_tech5:
        st.markdown("**PIL**")
    
    st.markdown("---")
    st.markdown("### 👥 Contributeurs")
    
    col_contrib1, col_contrib2, col_contrib3 = st.columns([1, 1, 1])
    
    with col_contrib1:
        st.markdown("")
    
    with col_contrib2:
        st.markdown("""
        <div style='text-align: center;'>
            <div style='display: flex; justify-content: center; gap: 40px; margin-bottom: 20px;'>
                <div>
                    <div style='font-size: 2em;'>🦊</div>
                    <div style='font-weight: bold;'>Aurélie Boisbunon</div>
                    <div style='font-size: 0.9em; color: #6b7280;'>Chercheuse en IA & Mathématicienne</div>
                </div>
                 <div>
                    <div style='font-size: 2em;'>🐱</div>
                    <div style='font-weight: bold;'>Hinata Flamary</div>
                    <div style='font-size: 0.9em; color: #6b7280;'>AI Testeuse</div>
                </div>
                <div>
                    <div style='font-size: 2em;'>🐼</div>
                    <div style='font-weight: bold;'>Illyyne Saffar</div>
                    <div style='font-size: 0.9em; color: #6b7280;'>Chercheuse en IA & Ingénieure en Informatique</div>
                </div>
            </div>
            <div style='margin-top: 20px; padding-top: 20px; border-top: 2px solid #e5e7eb;'>
                <div style='font-size: 1.5em; margin-bottom: 10px;'>🏢</div>
                <div style='font-weight: bold; font-size: 1.2em;'>Ericsson Research - Paris France </div>
                <div style='color: #6b7280; margin-top: 5px;'>Laboratoire de recherche et d'innovation en IA</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_contrib3:
        st.markdown("")
    
    st.markdown("")
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 10px; color: white;'>
        <p style='margin: 0; font-size: 1.1em;'> Application créée avec ❤️ pour vulgarisation de l'IA</p>
        <p style='margin: 5px 0 0 0; font-size: 0.9em; opacity: 0.9;'>Version 1.0.0 Final • Production Ready ✅</p>
        <p style='margin: 5px 0 0 0; font-size: 0.8em; opacity: 0.8;'>© 2026 Ericsson Research</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()