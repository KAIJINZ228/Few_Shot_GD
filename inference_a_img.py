import os
import io
import time
import json
import torch
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple
import datasets.transforms as T
from util.slconfig import DictAction, SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.vl_utils import create_positive_map_from_span
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib import colors
import random
import json

color_names = list(colors.CSS4_COLORS.keys())
def write_json(data, fname):
    with open(fname, "w") as f:
        json.dump(data, f)
    print("finished writing")

def plot_boxes2image_show_and_save(image_pil, tgt, save_path=None):

    img_h, img_w = tgt["size"]
    assert len(tgt["boxes"]) == len(tgt["labels"]), "boxes and labels must have same length"
    fig, ax = plt.subplots(1)
    ax.imshow(image_pil)

    # draw boxes and masks
    for box, label in zip(tgt["boxes"], tgt["labels"]):

        box = box * torch.Tensor([img_w, img_h, img_w, img_h])  # from 0..1 to 0..W, 0..H
        # from cxcywh to xywh
        box[:2] -= box[2:] / 2
        x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = random.choice(color_names) 
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect)
        plt.text(x, y - 10, label, color='w', fontsize=8, backgroundcolor='black')

    plt.axis('off')
    plt.savefig(os.path.join(save_path, "pred.jpg"))
    plt.show()
    plt.close()


def build_model(args):

    # we use register to maintain models from catdet6 on.
    from models.registry import MODULE_BUILD_FUNCS

    assert args.modelname in MODULE_BUILD_FUNCS._module_dict
    build_func = MODULE_BUILD_FUNCS.get(args.modelname)
    model = build_func(args)

    return model


def load_model(args):

    model = build_model(args)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()    #

    return model


def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w

    return image_pil, image


def caption_clean_and_span_build(caption):

    caption = caption.replace('，', ' . ')

    import re
    caption = caption.strip()
    caption = re.sub(" +", " ", caption)
    if caption[-1] != '.' and caption[-2] != ' ':
        caption = caption + ' .'
    elif caption[-1] == '.' and caption[-2] != ' ':
        caption = caption[:-1]
        caption = caption + ' .'

    caption_ = caption.split('.')
    token_spans = []
    count = 0
    for cap in caption_:
        if len(cap) == 0:
            continue
        token_span = []
        caps = cap.strip().split(' ')
        for i, item in enumerate(caps):
            if i:
                count += + 1
            token_span.append([count, count + len(item)])
            count += len(item)
        count += 3
        token_spans.append(token_span)

    return caption, str(token_spans)


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):

    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        if image.dim() < 4:
            image = image[None] 
        outputs = model(image, captions=[caption])

    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]    # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]      # num_filt, 4

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"{str(logit.max().item())[:4]}")
            else:
                pred_phrases.append(pred_phrase)
    else:
        # given-phrase mode
        temp = model.tokenizer(caption)
        positive_maps = create_positive_map_from_span(model.tokenizer(caption), token_span=token_spans).to(image.device)
        logits_for_phrases = positive_maps @ logits.T   # n_phrase, nq

        all_logits = []
        logit_ret = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            filt_mask = logit_phr > box_threshold
            all_boxes.append(boxes[filt_mask])
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + "|" +  f"{str(logit.item())[:4]}" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])

        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases

    return boxes_filt, pred_phrases


script_path = os.path.abspath(__file__)
project_path = script_path.rsplit('/', 1)[0] + '/'


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument('--options', nargs='+', action=DictAction,
                        default={'text_encoder_type': 'bert-base-uncased/'},
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
    parser.add_argument("--config_file", "-c", type=str, default=project_path + 'config/config_cfg_swinb.py',
                        help="path to config file")
    parser.add_argument("--checkpoint_path", "-p", type=str,
                        help="path to checkpoint file")
    parser.add_argument("--image_path", "-i", type=str,
                        help="path to image file")
    parser.add_argument("--text_prompt", "-t", type=str,,
                        help="text prompt")
    parser.add_argument("--output_dir", "-o", type=str,
                        help="output directory")
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--rank', default=0, type=int, help='number of distributed processes')
    parser.add_argument("--cpu-only", default=False, help="running on cpu only!, default=False")

    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")

    parser.add_argument("--inference", type=bool, default=True, help="inference flag")

    args = parser.parse_args()

    # 加载配置文件参数 并更新 args
    print("Loading config file from {}".format(args.config_file))
    time.sleep(args.rank * 0.02)
    cfg = SLConfig.fromfile(args.config_file)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
    if args.rank == 0:
        save_cfg_path = os.path.join(args.output_dir, args.config_file.split('/')[-1])
        cfg.dump(save_cfg_path)
        save_json_path = os.path.join(args.output_dir, "config_args_raw.json")
        with open(save_json_path, 'w') as f:
            json.dump(vars(args), f, indent=2)
    cfg_dict = cfg._cfg_dict.to_dict()
    args_vars = vars(args)
    for k, v in cfg_dict.items():
        if k not in args_vars:
            setattr(args, k, v)
        else:
            raise ValueError("Key {} can used by args only".format(k))

    # cfg
    config_file = args.config_file  # change the path of the model config file
    checkpoint_path = args.checkpoint_path  # change the path of the model
    image_path = args.image_path
    text_prompt = args.text_prompt
    output_dir = args.output_dir
    box_threshold = args.box_threshold
    text_threshold = args.text_threshold

    # load model
    model = load_model(args=args)


    # import json
    from tqdm import tqdm

    text_prompt = 'feed by yourself'
    output_json_dir = "feed by yourself"

    TEST_ANN_PATH = "feed by yourself"
    TEST_IMG_PATH = "feed by yourself"

    ret_lst = []

    # READ TEST ANN
    with open(TEST_ANN_PATH, "r") as f:
        data = json.load(f)

    images = data['images']
    categories = data['categories']
    annotations = data['annotations']

    categories_dict = {}
    for cat in categories:
        categories_dict[cat['name']] = cat['id']

    text_prompt, token_spans = caption_clean_and_span_build(caption=text_prompt)

    for img_info in tqdm(images):
        image_pil, image = load_image(TEST_IMG_PATH + img_info['file_name'])

        # set the text_threshold to None if token_spans is set.
        if token_spans is not None:
            text_threshold = None
            print("Using token_spans. Set the text_threshold to None.")

        # run model
        if token_spans is None:
            boxes_filt, pred_phrases = get_grounding_output(
                model,
                image,
                text_prompt,
                box_threshold,
                text_threshold,
                cpu_only=args.cpu_only,
            )
        else:
            boxes_filt, pred_phrases = get_grounding_output(
                model,
                image,
                text_prompt,
                box_threshold,
                text_threshold,
                cpu_only=args.cpu_only,
                token_spans=eval(token_spans),
            )

        # visualize pred
        size = image_pil.size
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H, W
            "labels": pred_phrases,
        }
        # plot_boxes2image_show_and_save(image_pil, pred_dict, save_path=args.output_dir)
        img_h, img_w = pred_dict["size"]

        for box, label in zip(pred_dict["boxes"], pred_dict["labels"]):
            box = box * torch.Tensor([img_w, img_h, img_w, img_h])  # from 0..1 to 0..W, 0..H
            # from cxcywh to xywh
            box[:2] -= box[2:] / 2
            x, y, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            label, score = label.split("|")
            score = float(score)

            category_id = categories_dict[label]

            ret_lst.append({
                "image_id": img_info['id'],
                "category_id": category_id,
                "bbox": [
                    x,
                    y,
                    w,
                    h
                ],
                "score": score
            })

    write_json(ret_lst, output_json_dir)

