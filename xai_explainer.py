import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from captum.attr import IntegratedGradients
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

def denormalize_image(tensor, mean, std):
    mean = torch.tensor(mean, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, 3, 1, 1)
    denormalized_tensor = tensor * std + mean
    return torch.clamp(denormalized_tensor, 0, 1)

def preprocess_image_for_xai(image_np_array, transform):
    pil_image = Image.fromarray((image_np_array * 255).astype(np.uint8))
    return transform(pil_image).unsqueeze(0)

def get_target_layer(model, model_name):
    if "inception" not in model_name.lower():
        raise ValueError("Only InceptionV3 is supported in this configuration.")
    return model.model.Mixed_7c

def explain_integrated_gradients(model, input_tensor, target_class_idx, denormalized_image_np):
    ig = IntegratedGradients(model)
    model.eval()
    attributions, _ = ig.attribute(input_tensor, target=target_class_idx, return_convergence_delta=True)
    attributions = attributions.squeeze(0).cpu().detach().numpy()
    attributions = np.transpose(attributions, (1, 2, 0))
    attributions_sum = np.sum(np.abs(attributions), axis=2)
    if attributions_sum.max() - attributions_sum.min() > 0:
        attributions_sum = (attributions_sum - attributions_sum.min()) / (attributions_sum.max() - attributions_sum.min())
    else:
        attributions_sum = np.zeros_like(attributions_sum)
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(denormalized_image_np)
    plt.title("Original Image")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(denormalized_image_np)
    plt.imshow(attributions_sum, cmap='jet', alpha=0.5)
    plt.title(f"Integrated Gradients (Class {target_class_idx})")
    plt.axis('off')
    plt.show()

def explain_gradcam(model, input_tensor, target_class_idx, denormalized_image_np, model_name):
    target_layer = get_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class_idx)]

    class GradCAMModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    cam_model = GradCAMModelWrapper(model)
    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    original_requires_grad = {}
    for param in model.model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = True

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(denormalized_image_np, grayscale_cam, use_rgb=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(denormalized_image_np)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"GradCAM (Class {target_class_idx})")
        plt.axis('off')
        plt.show()
    finally:
        for param, state in original_requires_grad.items():
            param.requires_grad = state

def explain_gradcam_plus_plus(model, input_tensor, target_class_idx, denormalized_image_np, model_name):
    target_layer = get_target_layer(model, model_name)
    targets = [ClassifierOutputTarget(target_class_idx)]

    class GradCAMModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, x):
            return self.model(x)

    cam_model = GradCAMModelWrapper(model)
    cam = GradCAMPlusPlus(model=cam_model, target_layers=[target_layer])

    original_requires_grad = {}
    for param in model.model.parameters():
        original_requires_grad[param] = param.requires_grad
        param.requires_grad = True

    try:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(denormalized_image_np, grayscale_cam, use_rgb=True)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(denormalized_image_np)
        plt.title("Original Image")
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(visualization)
        plt.title(f"GradCAM++ (Class {target_class_idx})")
        plt.axis('off')
        plt.show()
    finally:
        for param, state in original_requires_grad.items():
            param.requires_grad = state