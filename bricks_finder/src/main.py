from pathlib import Path
import argparse
from io import StringIO

from loguru import logger
import gradio as gr

from query_helper import QueryHelper, IndexType, QueryResult


THIS_PATH = Path(__file__).parent.parent
VECTOR_INDEX_ROOT = THIS_PATH / "vector_indexes"
STATIC_RESOURCES = THIS_PATH / "src" / "static"

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--share",
    action="store_true",
    default=False,
    help="Share the interface on a public URL"
)
parser.add_argument(
    "--startup_index",
    type=str,
    default="minifigure",
    choices=["minifigure", "brick"],
    help="Index to load on startup"
)
parser.add_argument(
    "--rebuild",
    action="store_true",
    default=False,
    help="Rebuild the indexes for both datasets"
)
parser.add_argument(
    "--invalidate_cache",
    action="store_true",
    default=False,
    help=(
        "Invalidate the cache for the datasets and the models. "
        "This option also rebuilds the indexes"
    )
)
args = parser.parse_args()
if args.invalidate_cache:
    args.rebuild = True

logger.info(f"Share link: {args.share}")
logger.info(f"Startup index: {args.startup_index}")
logger.info(f"Rebuild indexes: {args.rebuild}")
logger.info(f"Invalidate cache: {args.invalidate_cache}")

query_helper = QueryHelper(
    VECTOR_INDEX_ROOT, 
    startup_index=IndexType.from_str(args.startup_index),
    rebuild_indexes=args.rebuild,
    invalidate_cache=args.invalidate_cache
)

#############################################################
#############################################################
#############################################################

def search(text_query, image_query, index_type, result_count):
    if text_query == "" and image_query is None:
        raise gr.Error("Text or image query is required", duration=3)
    
    results: QueryResult = query_helper.query(
        query=text_query if image_query is None else image_query,
        top_k=result_count,
        index_type=IndexType.from_str(index_type)
    )

    return [(result.image, str(result.idx)) for result in results]

def dict_to_markdown(data: dict):
    markdown = StringIO()
    markdown.write("### Additional Information\n")

    # Experimental, add rebrickable link
    # brick dataset has "part_num" for reference to rebrickable website
    # while minifigure as fig_num
    _rebrickable_base_link = "https://rebrickable.com/"
    if "part_num" in data:
        _rebrickable_base_link += "parts/" + data["part_num"]
    elif "fig_num" in data:
        _rebrickable_base_link += "minifigs/" + data["fig_num"]
    
    markdown.write(
        f"- **Rebrickable Link**: [Check it out on Rebrickable]({_rebrickable_base_link})\n"
    )


    for key, value in data.items():
        if key == "extra":
            # The brick dataset has an extra field
            # with additional information
            if isinstance(value, list) and len(value) != 0:
                markdown.write("- **Extra**:\n")
                for idx, extra_item in enumerate(value):
                    markdown.write(f"  - **Item {idx}**:\n")
                    for k, v in extra_item.items():
                        markdown.write(f"    - **{k}**: {v}\n")
        else:
            markdown.write(f"- **{key}**: {value}\n")
    return markdown.getvalue()

def get_image_info(evt: gr.SelectData):
    select_image = evt.value
    image_id = int(select_image["caption"])
    image_info: dict = query_helper.get_image_info(image_id)
    return dict_to_markdown(image_info)

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
        gr.HTML(
            "<img "
            f"src='/gradio_api/file={str(STATIC_RESOURCES/'bricksfinder.png')}' "
            "alt='BrickFinder' max-height='100px' align='center'>"
        )
        gr.Markdown(
            "#### *For more information, visit the [GitHub repository]"
            "(https://github.com/armaggheddon/BricksFinder)*"
        )
        
    with gr.Row():
        with gr.Column():
            text_query = gr.Textbox(
                label="Text Query", 
                placeholder=(
                    "a woman wearing a yellow shirt with a pen holder, brown "
                    "trousers and red hair"
                )
            )
            index_radio = gr.Radio(
                label="Search for", 
                choices=[IndexType.MINIFIGURE.value, IndexType.BRICK.value], 
                value=IndexType.from_str(args.startup_index).value, 
                interactive=True
            )
            with gr.Accordion(label="More options", open=False):
                result_count_slider = gr.Slider(
                    label="Number of Results", 
                    minimum=8, 
                    maximum=16, 
                    value=8, 
                    step=1, 
                    interactive=True
                )
            with gr.Row():
                clear_btn = gr.Button(value="Clear")
                search_btn = gr.Button(value="Search", variant="primary")
        with gr.Column():
            image_query = gr.Image(label="Image Query", type="pil")
    with gr.Row():
        image_gallery = gr.Gallery(
            label="Results", 
            columns=4, 
            height="auto", 
            interactive=False
        )
    with gr.Row():
        additional_information = gr.Markdown(
            value="### Additional Information\n", 
            show_label=True, 
            container=True
        )

    
    clear_btn.click(
        lambda: ["", None, None, "### Additional Information\n"],
        outputs=[
            text_query, 
            image_query, 
            image_gallery, 
            additional_information,
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
        outputs=[additional_information]
    )


if __name__ == "__main__":

    interface.launch(
        server_name="0.0.0.0",
        server_port=8000,
        share=args.share,
        allowed_paths=[str(STATIC_RESOURCES)],
        max_file_size=50 * gr.FileSize.MB,
    )