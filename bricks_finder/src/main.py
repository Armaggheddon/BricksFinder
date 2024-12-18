from pathlib import Path
import argparse

from loguru import logger
import gradio as gr

from query_helper import QueryHelper, IndexType


THIS_PATH = Path(__file__).parent.parent
VECTOR_INDEX_ROOT = THIS_PATH / "vector_indexes"
query_helper = QueryHelper(VECTOR_INDEX_ROOT)

def search(text_query, image_query, index_type, result_count):
    results = query_helper.query(
        query=text_query if image_query is None else image_query,
        top_k=result_count,
        index_type=IndexType.from_str(index_type)
    )

    return [(result.image, str(result.idx)) for result in results]

def get_image_info(evt: gr.SelectData):
    select_image = evt.value
    image_id = int(select_image["caption"])
    additional_info, image = query_helper.get_image_info(image_id)
    return additional_info, image

css = """
h1 {
    text-align: center;
    display:block;
}
h4 {
    text-align: center;
    display:block;
}
"""

with gr.Blocks(css=css) as interface:
    with gr.Column():
        gr.Markdown("# BricksFinder")
        gr.Markdown("#### *For more information, visit the [GitHub repository](https://github.com/armaggheddon/BricksFinder)*")
        
    with gr.Row():
        with gr.Column():
            text_query = gr.Textbox(label="Text Query")
            index_radio = gr.Radio(label="Search for", choices=["minifigure", "brick"], value="minifigure", interactive=True)
            with gr.Accordion(label="More options", open=False):
                result_count_slider = gr.Slider(label="Number of Results", minimum=8, maximum=16, value=8, step=1, interactive=True)
            with gr.Row():
                clear_btn = gr.Button(value="Clear")
                search_btn = gr.Button(value="Search", variant="primary")
        with gr.Column():
            image_query = gr.Image(label="Image Query")
    with gr.Row():
        image_gallery = gr.Gallery(label="Results", columns=4, height="auto", interactive=False)
    with gr.Row():
        additional_information = gr.Textbox(label="Additional Information")
        selected_image = gr.Image(label="Selected Image", interactive=False)

    
    clear_btn.click(
        lambda: ["", None, None, "", None],
        outputs=[
            text_query, 
            image_query, 
            image_gallery, 
            additional_information, 
            selected_image
        ],
        show_progress=False
    )
    search_btn.click(
        fn=search,
        inputs=[
            text_query,
            image_query,
            index_radio,
            result_count_slider
        ],
        outputs=[
            image_gallery
        ]
    )
    image_gallery.select(
        get_image_info,
        inputs=None,
        outputs=[additional_information, selected_image]
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--share",
        action="store_true",
        default=False,
        help="Share the interface on a public URL"
    )
    args = parser.parse_args()
    logger.info(f"Share link: {args.share}")

    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=args.share
    )