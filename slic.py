import torch
from pytorch_grad_cam import GradCAM, GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os


def preprocess_image(img_path):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    preprocessing = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=means, std=stds)
    ])
    
    img = Image.open(img_path).convert('RGB')
    input_tensor = preprocessing(img).unsqueeze(0)
    return input_tensor, np.float32(img.resize((224, 224))) / 255

def run_visualizations(model, target_layer, input_tensor, rgb_img, save_prefix):
    if target_layer == "layer4":
        layer_obj = model.layer4[-1]
    elif target_layer == "layer3":
        layer_obj = model.layer3[-1]
    else:
        layer_obj = model.layer4[-1]

    gcam = GradCAM(model=model, target_layers=[layer_obj])
    
    mask = gcam(input_tensor=input_tensor)[0, :]
    cam_image = show_cam_on_image(rgb_img, mask, use_rgb=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gb_model = GuidedBackpropReLUModel(model=model, device=device)
    
    vanilla_bp = gb_model(input_tensor, target_category=None)
    guided_gradcam = vanilla_bp * mask[..., np.newaxis]
    
    cv2.imwrite(f"{save_prefix}_gradcam.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{save_prefix}_vanilla_bp.jpg", np.uint8(255 * vanilla_bp))
    cv2.imwrite(f"{save_prefix}_guided_gradcam.jpg", np.uint8(255 * guided_gradcam))
    print(f"All 3 visualizations saved for: {save_prefix}")

def experiment_layers(model, input_tensor, rgb_img):
    layers = ["relu", "layer1", "layer2", "layer3", "layer4"]
    for layer_name in layers:
        layer_map = {
           "relu": model.relu,
           "layer1": model.layer1[-1],
           "layer2": model.layer2[-1],
           "layer3": model.layer3[-1],
           "layer4": model.layer4[-1]
        }
        gcam = GradCAM(model=model, target_layers=[layer_map[layer_name]])
        mask = gcam(input_tensor=input_tensor)[0, :]
        cam_image = show_cam_on_image(rgb_img, mask)
        cv2.imwrite(f"layer_comparison_{layer_name}.jpg", cam_image)
    print("Layer comparison experiments completed.")

def test_robustness(model, img_path):
    input_tensor, rgb_img = preprocess_image(img_path)
    
    rotated_tensor = torch.rot90(input_tensor, 1, [2, 3]) 
    noise = torch.randn_like(input_tensor) * 0.05
    perturbed_tensor = input_tensor + noise

    run_visualizations(model, "layer4", rotated_tensor, rgb_img, "robust_rotate")
    run_visualizations(model, "layer4", perturbed_tensor, rgb_img, "robust_noise")

if __name__ == "__main__":
    model = models.resnet50(pretrained=True)
    model.eval()
    
    img_path = "brandeis_campus.jpg" 
    input_tensor, rgb_img = preprocess_image(img_path)
    
    run_visualizations(model, "layer4", input_tensor, rgb_img, "task1_comparison")
    experiment_layers(model, input_tensor, rgb_img)
    test_robustness(model, img_path)