import os
import argparse
import numpy as np
from PIL import Image
import cv2
import shutil
import json
import datetime

import torch
from torchvision import transforms
from enum import Enum

from safetensors.torch import load_file
from models.unet import UNet2DConditionModel
from models.controlnext import ControlNeXtModel
from pipeline.pipeline_controlnext import StableDiffusionControlNeXtPipeline
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from transformers import AutoTokenizer, PretrainedConfig

from utils.process_svg_and_mask import (
    read_modified_svg_and_produce_mask,
    gen_mask_vis,
    gen_sketch_vis,
)
from utils.svg_utils import parse_svg_path_count
from utils.prompt_utils import append_prompt

from flask import Flask, request
from flask_cors import CORS

from web_app_helper import soft, url2image, image2url


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    else:
        raise ValueError(f"{model_class} is not supported.")


def load_safetensors(
    model, safetensors_path, strict=True, load_weight_increasement=False
):
    if not load_weight_increasement:
        if safetensors_path.endswith(".safetensors"):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        model.load_state_dict(state_dict, strict=strict)
    else:
        if safetensors_path.endswith(".safetensors"):
            state_dict = load_file(safetensors_path)
        else:
            state_dict = torch.load(safetensors_path)
        pretrained_state_dict = model.state_dict()
        for k in state_dict.keys():
            state_dict[k] = state_dict[k] + pretrained_state_dict[k]
        model.load_state_dict(state_dict, strict=False)


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Simple example."
    )

    parser.add_argument(
        "--port", default=9000, help="Specify the port number (1-65535)."
    )
    parser.add_argument("--gpu", default="0", help="Specify the GPU number (0-3).")

    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="backbone/foolkatGODOF_v3",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        type=str,
        default="checkpoint/controlnext-48000.bin",
        help="Path to pretrained controlnext model or model identifier from huggingface.co/models."
        " If not specified controlnext weights are initialized from unet.",
    )
    parser.add_argument(
        "--save_data_base",
        type=str,
        default="web_app_output/",
        help="Path to store data",
    )
    parser.add_argument(
        "--unet_model_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained unet model or subset",
    )
    parser.add_argument(
        "--controlnext_scale",
        type=float,
        default=0.9,
        help="Control level for the controlnext",
    )
    parser.add_argument(
        "--train_controlnext_only",
        type=int,
        default=1,
        help=("whether to train controlnext only without optimizing unet"),
    )
    ## change paeameters above

    parser.add_argument("--lora_path", type=str, default=None, help="Path to lora")
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help=".",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention",
        action="store_true",
        help="Whether or not to use xformers.",
    )
    parser.add_argument(
        "--bg_img_path",
        type=str,
        default="utils/bg_image.png",
        help="The column of the dataset containing the inner mask.",
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the controlnext conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--save_load_weights_increaments",
        type=int,
        default=1,
        help=("whether to store the weights_increaments"),
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError(
            "`--validation_image` must be set if `--validation_prompt` is set"
        )

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError(
            "`--validation_prompt` must be set if `--validation_image` is set"
        )

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the controlnet encoder."
        )

    return args


args = parse_args()


vae = AutoencoderKL.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="vae",
    revision=args.revision,
    variant=args.variant,
)

text_encoder_cls = import_model_class_from_model_name_or_path(
    args.pretrained_model_name_or_path, args.revision
)
text_encoder = text_encoder_cls.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="text_encoder",
    revision=args.revision,
    variant=args.variant,
)

tokenizer = AutoTokenizer.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="tokenizer",
    revision=args.revision,
    use_fast=False,
)


controlnext = ControlNeXtModel(controlnext_scale=args.controlnext_scale)
if args.controlnet_model_name_or_path is not None:
    load_safetensors(controlnext, args.controlnet_model_name_or_path)
else:
    controlnext.scale = 0.0


unet = UNet2DConditionModel.from_pretrained(
    args.pretrained_model_name_or_path,
    subfolder="unet",
    revision=args.revision,
    variant=args.variant,
)
if not args.train_controlnext_only:
    if args.unet_model_name_or_path is not None:
        load_safetensors(
            unet,
            args.unet_model_name_or_path,
            strict=False,
            load_weight_increasement=args.save_load_weights_increaments,
        )


pipeline = StableDiffusionControlNeXtPipeline.from_pretrained(
    args.pretrained_model_name_or_path,
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    controlnext=controlnext,
    safety_checker=None,
    revision=args.revision,
    variant=args.variant,
)
pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
pipeline = pipeline.to(args.device)
pipeline.set_progress_bar_config()
if args.lora_path is not None:
    pipeline.load_lora_weights(args.lora_path)
if args.enable_xformers_memory_efficient_attention:
    pipeline.enable_xformers_memory_efficient_attention()

if args.seed is None:
    generator = None
else:
    generator = torch.Generator(device=args.device).manual_seed(args.seed)


inference_ctx = torch.autocast(args.device)

image_transforms = transforms.Compose(
    [
        transforms.Resize(
            args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.CenterCrop(args.resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

appending_words = [
    "looking at viewer",
    "solo",
    "lineart",
    "anime line art",
    "monochrome",
    "white background",
    "simple background",
]


class GenderSource(Enum):
    # NONE = "None"
    GIRL = "Girl"
    BOY = "Boy"


quick_prompts = [
    "looking at viewer",
    "close-up",
    # expression
    "smile",
    # hair
    "long hair",
    "short hair",
    # mouth
    "open mouth",
    "closed mouth",
    # ear
    "cat ears",
    "rabbit ears",
    # dress
    "shirt",
    "dress",
    "skirt",
    "school uniform",
    # tail
    "cat tail",
    "fox tail",
    # sleeves
    "long sleeves",
    "short sleeves",
    # decoration
    "bowtie",
    "hat",
    "gloves",
    "glasses",
    "boots",
    "ponytail",
]
quick_prompts = [[x] for x in quick_prompts]

quick_prompts_plus_expression = [
    # expression
    "face",
    "blush",
    "teeth",
    "happy",
    "expressionless",
    "tears",
    "closed eyes",
    "parted lips",
    "pointy ears",
]
quick_prompts_plus_action = [
    # action
    "outstretched arm",
    "outstretched hand",
    "pointing",
    "pointing at viewer",
    "reaching out",
]
quick_prompts_plus_hair = [
    # hair
    "very long hair",
    "bangs",
    "swept bangs",
    "blunt bangs",
    "braid",
    "single braid",
    "wavy hair",
    "hair between eyes",
    "hair over one eye",
    "hair ornament",
    "hair bow",
    "hair ribbon",
]
quick_prompts_plus_dress = [
    # dress
    "collared shirt",
    "jacket",
    "apron",
    "pants",
    "shorts",
    "pleated skirt",
    "maid",
    "hoodie",
    "belt",
    "vest",
    "shoes",
    # sleeves
    "puffy sleeves",
    "wide sleeves",
    "sleeveless",
]
quick_prompts_plus_decoration = [
    # decoration
    "bow",
    "ribbon",
    "jewelry",
    "necktie",
    "necklace",
    "choker",
    "earrings",
    "scarf",
    "headphones",
    "witch hat",
    "beret",
    "wings",
    "angel wings",
    "flower",
    "balloon",
    "book",
    "holding weapon",
    "sword",
]
quick_prompts_plus_expression = [[x] for x in quick_prompts_plus_expression]
quick_prompts_plus_action = [[x] for x in quick_prompts_plus_action]
quick_prompts_plus_hair = [[x] for x in quick_prompts_plus_hair]
quick_prompts_plus_dress = [[x] for x in quick_prompts_plus_dress]
quick_prompts_plus_decoration = [[x] for x in quick_prompts_plus_decoration]


prev_sketch_base = os.path.join(args.save_data_base, "prev-sketch")
prev_gen_base = os.path.join(args.save_data_base, "prev-gen")
prev_mask_base = os.path.join(args.save_data_base, "prev-mask")

def remove_existing_files():
    existing_dirs = [prev_sketch_base, prev_gen_base, prev_mask_base]
    for existing_dir in existing_dirs:
        if os.path.isdir(existing_dir):
            shutil.rmtree(existing_dir)

    existing_files = ['untitled.svg', 'untitled.png',
                      'untitled-mask.png', 'untitled_manual_mask.png', 'untitled-mask_vis.png',
                      'untitled_overlay.png',
                      'prompt.txt', 'operations.txt']
    for existing_file in existing_files:
        existing_file_path = os.path.join(args.save_data_base, existing_file)
        if os.path.exists(existing_file_path):
            os.remove(existing_file_path)

# Remove existing files
remove_existing_files()

os.makedirs(prev_sketch_base, exist_ok=True)
os.makedirs(prev_gen_base, exist_ok=True)
os.makedirs(prev_mask_base, exist_ok=True)


@torch.inference_mode()
def process_main_gen(input_prompt, mask_dilate_size, num_samples):
    assert len(input_prompt) != 0
    if args.negative_prompt is not None:
        negative_prompts = args.negative_prompt
    else:
        negative_prompts = None

    ############################## main process ##############################
    validation_prompt = append_prompt(input_prompt, appending_words)

    prev_files = os.listdir(prev_sketch_base)
    prev_files = [item for item in prev_files if ".svg" in item]
    prev_idx = len(prev_files) - 1

    prev_gen_image_path = (
        args.bg_img_path
        if len(prev_files) == 0
        else os.path.join(prev_gen_base, "prev-gen" + str(prev_idx) + ".png")
    )
    pixel_values_bg = Image.open(prev_gen_image_path).convert("RGB")
    prev_svg_name = (
        None
        if len(prev_files) == 0
        else "prev-sketch/prev-sketch" + str(prev_idx) + ".svg"
    )

    ## background latents
    pixel_values_bg = image_transforms(pixel_values_bg)  # (3, 512, 512)
    pixel_values_bg = pixel_values_bg.unsqueeze(dim=0)  # (1, 3, 512, 512)
    pixel_values_bg = pixel_values_bg.to(memory_format=torch.contiguous_format).float()

    ## generate modified mask
    manual_mask_path = os.path.join(
        args.save_data_base, "untitled_manual_mask.png"
    )
    suggest_mask_path = os.path.join(
        args.save_data_base, "untitled-mask.png"
    )
    if os.path.exists(manual_mask_path):
        inner_mask_img = Image.open(manual_mask_path).convert(
            "RGB"
        )  # [0-BG, 255-FG]
        inner_mask_img = np.array(inner_mask_img, dtype=np.float32)[:, :, 0]
    elif os.path.exists(suggest_mask_path):
        inner_mask_img = Image.open(suggest_mask_path).convert(
            "RGB"
        )  # [0-BG, 255-FG]
        inner_mask_img = np.array(inner_mask_img, dtype=np.float32)[:, :, 0]
    else:
        inner_mask_img, _ = read_modified_svg_and_produce_mask(
            args.save_data_base,
            "untitled.svg",
            prev_gen_image_path,
            previous_svg_name=prev_svg_name,
            mask_dilate_size=mask_dilate_size,
            save_mask_vis=True,
        )

    inner_mask_img_s = cv2.resize(
        inner_mask_img, (64, 64), interpolation=cv2.INTER_LINEAR
    )
    inner_mask_s = np.array(inner_mask_img_s, dtype=np.float32) / 255.0  # [0-BG, 1-FG]
    inner_mask_s = np.expand_dims(np.expand_dims(inner_mask_s, axis=0), axis=0)
    validation_inner_mask_s = torch.tensor(inner_mask_s).float()  # (1, 1, 64, 64)
    validation_inner_mask = None

    gen_images = []
    negative_prompt = negative_prompts[0] if negative_prompts is not None else None

    curr_sketch_image_path = os.path.join(args.save_data_base, "untitled.png")
    validation_sketch_image = Image.open(curr_sketch_image_path).convert("RGB")

    ## v1: one-by-one
    # for _ in range(num_samples):
    #     with inference_ctx:
    #         image = pipeline(
    #             validation_prompt, validation_sketch_image, mask=validation_inner_mask, mask_s=validation_inner_mask_s,
    #             image_bg=pixel_values_bg,
    #             num_inference_steps=20, generator=generator, negative_prompt=negative_prompt
    #         ).images[0]
    #         image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2GRAY)
    #         gen_images.append(image_np)

    ## v2: batch processing
    with inference_ctx:
        image = pipeline(
            validation_prompt,
            validation_sketch_image,
            mask=validation_inner_mask,
            mask_s=validation_inner_mask_s,
            image_bg=pixel_values_bg,
            num_images_per_prompt=num_samples,
            num_inference_steps=20,
            generator=generator,
            negative_prompt=negative_prompt,
        ).images
        gen_images += [
            cv2.cvtColor(np.asarray(item), cv2.COLOR_BGR2GRAY) for item in image
        ]

    return gen_images


@torch.inference_mode()
def process_mask_update_manual(mask):
    manual_mask = mask[:, :, 0]  # [0-BG, 255-FG]

    manual_mask_save_path = os.path.join(
        args.save_data_base, "untitled_manual_mask.png"
    )
    vis_mask_save_path = os.path.join(args.save_data_base, "untitled-mask_vis.png")

    prev_files = os.listdir(prev_sketch_base)
    prev_files = [item for item in prev_files if ".svg" in item]
    prev_idx = len(prev_files) - 1
    prev_gen_image_path = (
        args.bg_img_path
        if len(prev_files) == 0
        else os.path.join(prev_gen_base, "prev-gen" + str(prev_idx) + ".png")
    )

    if np.sum(manual_mask) == 0:
        if os.path.exists(manual_mask_save_path):  # clear mode
            os.remove(manual_mask_save_path)
        return gen_mask_vis(
            prev_gen_image_path, manual_mask, save_path=vis_mask_save_path
        )
    else:
        # Save mask image for loading during genration
        manual_mask_png = Image.fromarray(np.copy(manual_mask).astype(np.uint8), "L")
        manual_mask_png.save(manual_mask_save_path, "PNG")
        return gen_mask_vis(
            prev_gen_image_path, manual_mask, save_path=vis_mask_save_path
        )


@torch.inference_mode()
def process_mask_update(mask_dilate_size_):
    prev_files = os.listdir(prev_sketch_base)
    prev_files = [item for item in prev_files if ".svg" in item]
    prev_idx = len(prev_files) - 1
    prev_gen_image_path = (
        args.bg_img_path
        if len(prev_files) == 0
        else os.path.join(prev_gen_base, "prev-gen" + str(prev_idx) + ".png")
    )
    prev_svg_name = (
        None
        if len(prev_files) == 0
        else "prev-sketch/prev-sketch" + str(prev_idx) + ".svg"
    )

    ## generate modified mask
    pure_mask, inner_mask_img_vis = read_modified_svg_and_produce_mask(
        args.save_data_base,
        "untitled.svg",
        prev_gen_image_path,
        previous_svg_name=prev_svg_name,
        mask_dilate_size=mask_dilate_size_,
        save_mask_vis=True,
    )
    pure_mask = np.stack([pure_mask] * 3, axis=-1)

    return pure_mask


@torch.inference_mode()
def on_select_generation(select_idx, results_value, input_prompt):
    selected_result = results_value[
        int(select_idx)
    ]  # dict: ['name', 'data', 'is_file']

    prev_files = os.listdir(prev_gen_base)
    prev_files = [item for item in prev_files if ".png" in item]
    curr_idx = len(prev_files)
    save_gen_path = os.path.join(prev_gen_base, "prev-gen" + str(curr_idx) + ".png")

    print(selected_result.shape)
    Image.fromarray(
        cv2.cvtColor(selected_result[..., np.newaxis], cv2.COLOR_GRAY2RGB), "RGB"
    ).save(save_gen_path, "PNG")
    # shutil.copy(selected_result_path, save_gen_path)

    ori_svg_path = os.path.join(args.save_data_base, "untitled.svg")
    save_svg_path = os.path.join(
        prev_sketch_base, "prev-sketch" + str(curr_idx) + ".svg"
    )
    shutil.copy(ori_svg_path, save_svg_path)

    ori_mask_path = os.path.join(args.save_data_base, "untitled-mask.png")
    save_mask_path = os.path.join(prev_mask_base, "prev-mask" + str(curr_idx) + ".png")

    # Update sketch visualization window with the latest generated line art
    selected_result_with_sketch = gen_sketch_vis(
        args.save_data_base, "untitled.svg", save_gen_path
    )  # (H, W, 3)

    prompt_txt_path = os.path.join(args.save_data_base, "prompt.txt")
    with open(prompt_txt_path, "a") as f:
        f.write(input_prompt + "\n")

    manual_mask_save_path = os.path.join(
        args.save_data_base, "untitled_manual_mask.png"
    )
    vis_mask_save_path = os.path.join(
        args.save_data_base, "untitled-mask_vis.png"
    )
    if os.path.exists(manual_mask_save_path):
        selected_result_with_mask = gen_mask_vis(
            save_gen_path, manual_mask_save_path, save_path=vis_mask_save_path
        )
        shutil.copy(manual_mask_save_path, save_mask_path)
    else:
        selected_result_with_mask = gen_mask_vis(
            save_gen_path, None, save_path=vis_mask_save_path
        )
        shutil.copy(ori_mask_path, save_mask_path)
    return selected_result_with_sketch, selected_result_with_mask, save_gen_path


# instantiate the app
app = Flask(__name__)

# enable CORS
CORS(app)


def get_modified_operation(modified_path_id_dict):
    operations = []
    assert 'add' in modified_path_id_dict
    if len(modified_path_id_dict['add']) != 0:
        operations.append('add')
    assert 'edit' in modified_path_id_dict
    if len(modified_path_id_dict['edit']) != 0:
        operations.append('edit')
    assert 'delete' in modified_path_id_dict
    if len(modified_path_id_dict['delete']) != 0:
        operations.append('delete')
    return ','.join(operations)


def get_time_str():
    formatted_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    return formatted_time


@app.route("/modify_svg", methods=["POST"])
def modify_svg():
    svg = request.json.get("svg")
    modified_path_ids = request.json.get("modified_path_ids")  # dict, {'add': ['svg_1', 'svg_2'], 'edit': [], 'delete': ['svg_3']}

    # First, save svg file
    with open(os.path.join(args.save_data_base, "untitled.svg"), "w") as f:
        f.write(svg)

    operation_txt_path = os.path.join(args.save_data_base, "operations.txt")
    operation_prompt = get_modified_operation(modified_path_ids) + " | " + get_time_str() + " | path_count:" + str(parse_svg_path_count(os.path.join(args.save_data_base, "untitled.svg")))
    with open(operation_txt_path, "a") as f:
        f.write(operation_prompt + "\n")

    # # Then, get new_mask_update and remove previous manually-edited mask
    mask = process_mask_update(32)  # [0-BG, 255-FG]

    manual_mask_save_path = os.path.join(
        args.save_data_base, "untitled_manual_mask.png"
    )
    # Image.fromarray(mask, "RGB").save("vis.png", "PNG")
    if os.path.exists(manual_mask_save_path):
        os.remove(manual_mask_save_path)

    print(mask.shape)
    # mask = 255 - mask
    # np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Change the mask color same to web client
    c = np.any(mask != [0, 0, 0], axis=-1)

    mask[c] = [122, 122, 122]
    mask[~c] = [255, 255, 255]
    mask = mask.astype(np.uint8)

    mask_url = image2url(mask)
    return (
        json.dumps({"mask_url": mask_url}),
        # json.dumps({}),
        200,
        {"ContentType": "application/json"},
    )


last_prompt = ""


@app.route("/update_mask", methods=["POST"])
def update_mask():
    operation_txt_path = os.path.join(args.save_data_base, "operations.txt")
    operation_prompt = "update_mask" + " | " + get_time_str()
    with open(operation_txt_path, "a") as f:
        f.write(operation_prompt + "\n")

    mask_url = request.json.get("mask_url")
    if mask_url is not None:
        mask = url2image(mask_url)
        mask = 255 - cv2.resize(mask, (args.resolution, args.resolution))
        mask[mask != 0] = 255
        process_mask_update_manual(mask)

    return (
        json.dumps("Done!"),
        200,
        {"ContentType": "application/json"},
    )


@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt")

    operation_txt_path = os.path.join(args.save_data_base, "operations.txt")
    operation_prompt = "generate start" + " | " + get_time_str()
    with open(operation_txt_path, "a") as f:
        f.write(operation_prompt + "\n")

    global last_prompt
    last_prompt = prompt

    ips = [prompt, 32, 1]
    print(ips)
    global result_gallery
    result_gallery = process_main_gen(*ips)
    result_gallery_url = [image2url(_) for _ in result_gallery]

    operation_txt_path = os.path.join(args.save_data_base, "operations.txt")
    operation_prompt = "generate done" + " | " + get_time_str()
    with open(operation_txt_path, "a") as f:
        f.write(operation_prompt + "\n")

    return (
        json.dumps({"result_gallery": result_gallery_url}),
        200,
        {"ContentType": "application/json"},
    )


@app.route("/select_generation", methods=["POST"])
def select_generation():
    selected_idx = request.json.get("selected_idx")

    operation_txt_path = os.path.join(args.save_data_base, "operations.txt")
    operation_prompt = "select_generation" + " | " + get_time_str()
    with open(operation_txt_path, "a") as f:
        f.write(operation_prompt + "\n")

    global last_prompt
    sketch_overlay, masked_region, src_masked_region = on_select_generation(
        selected_idx, result_gallery, last_prompt
    )
    return (json.dumps("Done"), 200, {"ContentType": "application/json"})


if __name__ == "__main__":
    app.run(
        host="0.0.0.0", port=args.port, debug=True, threaded=True, use_reloader=False
    )
