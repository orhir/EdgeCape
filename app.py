# import os
# os.system('python setup.py develop')

import argparse
import json
from pathlib import Path

import gradio as gr
import matplotlib

from gradio_utils.utils import (process_img, get_select_coords, select_skeleton,
                                reset_skeleton, reset_kp, process, update_examples)

LENGTH = 480  # Length of the square area displaying/editing images

matplotlib.use('agg')
model_dir = Path('./checkpoints')
parser = argparse.ArgumentParser(description='EdgeCape Demo')
parser.add_argument('--checkpoint',
                    help='checkpoint path',
                    default='ckpt/1shot_split1.pth')
args = parser.parse_args()
checkpoint_path = args.checkpoint
device = 'cuda'
TIMEOUT = 80

with gr.Blocks() as demo:
    gr.Markdown('''
    # We introduce EdgeCape, a novel framework that overcomes these limitations by predicting the graph's edge weights which optimizes localization. 
    To further leverage structural priors, we propose integrating Markovian Structural Bias, which modulates the self-attention interaction between nodes based on the number of hops between them. 
    We show that this improves the model’s ability to capture global spatial dependencies. 
    Evaluated on the MP-100 benchmark, which includes 100 categories and over 20K images, 
    EdgeCape achieves state-of-the-art results in the 1-shot setting and leads among similar-sized methods in the 5-shot setting, significantly improving keypoint localization accuracy.
    ### [Paper](https://arxiv.org/pdf/2411.16665) | [Project Page](https://orhir.github.io/edge_cape/) 
    ## Instructions
    1. Upload an image of the object you want to pose.
    2. Mark keypoints on the image.
    3. Mark limbs on the image.
    4. Upload an image of the object you want to pose to the query image (**bottom**).
    5. Click **Evaluate** to pose the query image.
    ''')

    global_state = gr.State({
        "images": {},
        "points": [],
        "skeleton": [],
        "prev_point": None,
        "curr_type_point": "start",
        "load_example": False,
    })
    with gr.Row():
        # Upload & Preprocess Image Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Upload & Preprocess Image</p>"""
            )
            support_image = gr.Image(
                height=LENGTH,
                width=LENGTH,
                type="pil",
                image_mode="RGB",
                label="Support Image",
                show_label=True,
                interactive=True,
            )

        # Click Points Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Click Points</p>"""
            )
            kp_support_image = gr.Image(
                type="pil",
                label="Keypoints Image",
                show_label=True,
                height=LENGTH,
                width=LENGTH,
                interactive=False,
                show_fullscreen_button=False,
            )
            with gr.Row():
                confirm_kp_button = gr.Button("Confirm Clicked Points", scale=3)
            with gr.Row():
                undo_kp_button = gr.Button("Undo Clicked Points", scale=3)

        # Editing Results Column
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Click Skeleton</p>"""
            )
            skel_support_image = gr.Image(
                type="pil",
                label="Skeleton Image",
                show_label=True,
                height=LENGTH,
                width=LENGTH,
                interactive=False,
                show_fullscreen_button=False,
            )
            with gr.Row():
                pass
            with gr.Row():
                undo_skel_button = gr.Button("Undo Skeleton")

    with gr.Row():
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Query Image</p>"""
            )
            query_image = gr.Image(
                type="pil",
                image_mode="RGB",
                label="Query Image",
                show_label=True,
                interactive=True,
            )
        with gr.Column():
            gr.Markdown(
                """<p style="text-align: center; font-size: 20px">Output</p>"""
            )
            output_img = gr.Plot(label="Output Image", )
    with gr.Row():
        eval_btn = gr.Button(value="Evaluate")
    with gr.Row():
        gr.Markdown("## Examples")
    with gr.Row():
        example_null = gr.Textbox(type='text',
                                  visible=False
                                  )
    with gr.Row():
        examples = gr.Examples([
            ['examples/dog2.png',
             'examples/dog1.png',
             json.dumps({
                 'points': [(232, 200), (312, 204), (228, 264), (316, 472), (316, 616), (296, 868), (412, 872),
                            (416, 624), (604, 608), (648, 860), (764, 852), (696, 608), (684, 432)],
                 'skeleton': [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5),
                              (3, 7), (7, 6), (3, 12), (12, 8), (8, 9),
                              (12, 11), (11, 10)],
             })
             ],
            ['examples/sofa1.jpg',
             'examples/sofa2.png',
             json.dumps({'points': [[272, 561], [193, 482], [339, 460], [445, 530], [264, 369], [203, 318], [354, 300],
                                    [457, 341], [345, 63], [187, 68]],
                         'skeleton': [[0, 4], [1, 5], [2, 6], [3, 7], [7, 6], [6, 5],
                                      [5, 4], [4, 7], [5, 9], [9, 8], [8, 6]],
             })],
            ['examples/person1.jpeg',
             'examples/person2.jpeg',
             json.dumps({
                 'points': [[322, 488], [431, 486], [526, 644], [593, 486], [697, 492], [407, 728],
                            [522, 726], [625, 737], [515, 798]],
                 'skeleton': [[0, 1], [1, 3], [3, 4], [1, 2], [2, 3], [5, 6], [6, 7], [7, 8], [8, 5]],
             })]
        ],
            inputs=[support_image, query_image, example_null],
            outputs=[support_image, kp_support_image, skel_support_image, query_image, global_state],
            fn=update_examples,
            run_on_click=True,
            examples_per_page=5,
        )

    support_image.upload(process_img,
                         inputs=[support_image, global_state],
                         outputs=[kp_support_image, global_state])
    kp_support_image.select(get_select_coords,
                            [global_state],
                            [global_state, kp_support_image],
                            queue=False, )
    confirm_kp_button.click(reset_skeleton,
                            inputs=global_state,
                            outputs=skel_support_image)
    undo_kp_button.click(reset_kp,
                         inputs=global_state,
                         outputs=[kp_support_image, skel_support_image])
    undo_skel_button.click(reset_skeleton,
                           inputs=global_state,
                           outputs=skel_support_image)
    skel_support_image.select(select_skeleton,
                              inputs=[global_state],
                              outputs=[global_state, skel_support_image])
    eval_btn.click(fn=process,
                   inputs=[query_image, global_state],
                   outputs=[output_img])

if __name__ == "__main__":
    print("Start app", parser.parse_args())
    gr.close_all()
    demo.launch(show_api=False)
