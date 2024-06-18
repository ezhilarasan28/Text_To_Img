import tkinter as tk
from PIL import ImageTk, Image
from diffusers import StableDiffusionPipeline
import torch
import cv2
from skimage.metrics import structural_similarity as ssim
import numpy as np


class CFG:
    device = "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 30
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12


def generate_image(prompt):
    image_gen_model = StableDiffusionPipeline.from_pretrained(
        CFG.image_gen_model_id, torch_dtype=torch.float32,
        revision="fp16", use_auth_token='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', guidance_scale=9)
    image = image_gen_model(prompt, num_inference_steps=CFG.image_gen_steps,
                            generator=CFG.generator,
                            guidance_scale=CFG.image_gen_guidance_scale).images[0]
    image = image.resize(CFG.image_gen_size)
    return image


def calculate_image_quality(original_image, generated_image):
    # Convert PIL images to numpy arrays
    original_array = np.array(original_image)
    generated_array = np.array(generated_image)

    # Convert images to grayscale
    original_gray = cv2.cvtColor(original_array, cv2.COLOR_RGB2GRAY)
    generated_gray = cv2.cvtColor(generated_array, cv2.COLOR_RGB2GRAY)

    # Calculate SSIM (Structural Similarity Index)
    ssim_score = ssim(original_gray, generated_gray)

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((original_gray - generated_gray) ** 2)
    if mse == 0:
        psnr = 100  # Maximum PSNR value when images are identical
    else:
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return ssim_score, psnr


def generate_and_display_image():
    prompt = prompt_entry.get()
    generated_image = generate_image(prompt)
    photo = ImageTk.PhotoImage(generated_image)
    image_label.config(image=photo)
    image_label.image = photo

    # Save the generated image
    generated_image.save("generated_image.jpg")

    # Load the original image
    original_image = Image.open(
        "generated_image.jpg")  # Replace "original_image.jpg" with the path to your original image

    # Calculate image quality metrics
    ssim_score, psnr = calculate_image_quality(original_image, generated_image)
    print("SSIM Score:", ssim_score)
    print("PSNR:", psnr)


# Create the main window
root = tk.Tk()
root.geometry("530x650")
root.title("Image Generation and Quality Evaluation")

# Prompt entry
prompt_label = tk.Label(root, text="Enter Prompt:")
prompt_label.pack()
prompt_entry = tk.Entry(root, width=500, font=("Arial", 20))
prompt_entry.pack()

# Generate button
generate_button = tk.Button(root, text="Generate and Evaluate Image", command=generate_and_display_image)
generate_button.pack()

# Image label
image_label = tk.Label(root)
image_label.pack()

root.mainloop()
