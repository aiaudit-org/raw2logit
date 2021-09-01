import gradio as gr
#import tensorflow as tf
import numpy as np
import json
from os.path import dirname, realpath, join
import processing.pipeline_numpy as ppn


# Load human-readable labels for ImageNet.
current_dir = dirname(realpath(__file__))

    
def process(RawImage, CameraParameters, Debayer, Sharpening, Denoising):
    raw_img = RawImage
    if CameraParameters == "Microscope":
        black_level = [9.834368023181512e-06, 9.834368023181512e-06, 9.834368023181512e-06, 9.834368023181512e-06]
        white_balance = [-0.6567, 1.9673, 3.5304]
        colour_matrix = [-2.0338, 0.0933, 0.4157, -0.0286, 2.6464, -0.0574, -0.5516, -0.0947, 2.9308]
    elif CameraParameters == "Drone":
        #drone
        black_level = [0.0625, 0.0626, 0.0625, 0.0626]
        white_balance = [2.86653646, 1., 1.73079425]
        colour_matrix = [1.50768983, -0.33571374, -0.17197604, -0.23048614,
                        1.70698738, -0.47650126, -0.03119153, -0.32803956, 1.35923111]
    else:
        print("No valid camera parameter")
    debayer = Debayer
    sharpening = Sharpening
    denoising = Denoising
    print(np.max(raw_img))
    raw_img = (raw_img[:,:,0].astype(np.float64)/255.)
    img = ppn.processing(raw_img, black_level, white_balance, colour_matrix,
                        debayer=debayer, sharpening=sharpening, denoising=denoising)
    print(np.max(img))
    return img


iface = gr.Interface(
    process, 
    [gr.inputs.Image(),gr.inputs.Radio(["Microscope", "Drone"]),gr.inputs.Dropdown(["bilinear", "malvar2004", "menon2007"]), 
    gr.inputs.Dropdown(["sharpening_filter", "unsharp_masking"]),
    gr.inputs.Dropdown(["gaussian_denoising", "median_denoising"])], 
    "image",
    capture_session=True,
    examples=[
        ["demo-files/car.png"],
        ["demo-files/micro.png"]
    ],
    title="static pipeline demo",
    description="You can select a sample raw image, the camera parameters and the pipeline configuration to process the raw image.")

if __name__ == "__main__":
    iface.launch(share=True)

