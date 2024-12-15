import time
import random

import gradio as gr
import numpy as np
from PIL import Image



dummy_images = {
    "minifigures" : [
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "0"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "1"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "2"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "3"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "4"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "5"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "6"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "7"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "8"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "9"),
    ],
    "bricks" : [
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "0"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "1"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "2"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "3"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "4"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "5"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "6"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "7"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "8"),
        (Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)), "9"),
    ]
}

dummy_additional_info = {
    "minifigures" : [
        "Minifigure 1",
        "Minifigure 2",
        "Minifigure 3",
        "Minifigure 4",
        "Minifigure 5",
        "Minifigure 6",
        "Minifigure 7",
        "Minifigure 8",
        "Minifigure 9",
        "Minifigure 10",
    ],
    "bricks" : [
        "Brick 1",
        "Brick 2",
        "Brick 3",
        "Brick 4",
        "Brick 5",
        "Brick 6",
        "Brick 7",
        "Brick 8",
        "Brick 9",
        "Brick 10",
    ]
}

def run_query(query, images, dataset):
    # Simulate a long running query

    time.sleep(1)
    if query == "":
        return None, None
    
    if dataset == "minifigures":
        imgs = dummy_images["minifigures"]
    else:
        imgs = dummy_images["bricks"]
    
    random.shuffle(imgs)
    imgs_sub = imgs[:5] # simulate limited results
    return imgs_sub, None # no selected image yet

def select_image(evt: gr.SelectData):
    select_image = evt.value
    image_id = int(select_image["caption"])
    dataset_id = dataset_radio.value
    return dummy_additional_info[dataset_id][image_id], dummy_images[dataset_id][image_id][0]


with gr.Blocks() as demo:
    with gr.Row():
        query_box = gr.Textbox(label="Query")
        upload_box = gr.UploadButton("Upload Images", type="filepath", file_types=["image/*"])
        dataset_radio = gr.Radio(["minifigures", "bricks"], value="minifigures", label="Model/Dataset")
        run_button = gr.Button("Run Query")
    
    with gr.Row():
        gallery = gr.Gallery(
            label="Search results",
            columns=3,
        )

    with gr.Row():
        with gr.Column():
            selected_info = gr.Textbox(label="Additional Information")
            selected_image_preview = gr.Image(value=None, label="Selected Image", interactive=False)
    
    run_button.click(
        run_query,
        inputs=[query_box, upload_box, dataset_radio],
        outputs=[gallery, selected_image_preview]
    )
    gallery.select(
        select_image,
        None, 
        [selected_info, selected_image_preview]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=8000)