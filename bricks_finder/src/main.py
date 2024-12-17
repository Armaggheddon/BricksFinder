from pathlib import Path

import gradio as gr

from query_helper import QueryHelper, QueryResult, IndexType


THIS_PATH = Path(__file__).parent.parent
VECTOR_INDEX_ROOT = THIS_PATH / "vector_indexes"


query_helper = QueryHelper(VECTOR_INDEX_ROOT)


def run_query(query, images, dataset):
    results: list[QueryResult] = query_helper.query(query, 3)
    return [(result.image, result.idx) for result in results], None # no selected image yet

def select_image(evt: gr.SelectData):
    select_image = evt.value
    image_id = int(select_image["caption"])
    additional_info, image = query_helper.get_image_info(image_id)
    return additional_info, image


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
            rows=3
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