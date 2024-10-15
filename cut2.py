import os
import threading

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# if using Apple MPS, fall back to CPU for unsupported ops
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
from flask import Blueprint, request, render_template
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


click_handler_bp = Blueprint('click_handler', __name__)


@click_handler_bp.route('/get-coordinates', methods=['GET'])
def get_coordinates():
    x = int(request.args.get('x'))
    y = int(request.args.get('y'))
    filename = request.args.get('filename')
    print(f"Click coordinates: X={x}, Y={y}")
    print_current_thread()
    matplotlib.use('agg')

    cut(x, y, initDevice(), filename)

    return render_template('res.html')
    # return 'Click coordinates received successfully!!!'


def print_current_thread():
    current_thread = threading.current_thread()
    print(f"Current Thread: {current_thread.name}")


def initDevice():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )
    np.random.seed(3)
    return device


def cut(x, y, device, filename):
    print("Cutting>>>>")

    image = Image.open('static/uploads/' + filename)
    # image = Image.open('images/truck.jpg')
    image = np.array(image.convert("RGB"))

    sam2_checkpoint = "sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    # sam2_checkpoint = "sam2_hiera_large.pt"
    # model_cfg = "sam2_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

    predictor = SAM2ImagePredictor(sam2_model)
    predictor.set_image(image)
    input_point = np.array([[x, y]])
    # input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # plt.axis('on')
    # plt.show()

    # plt.figure(figsize=(10, 10))
    # plt.imshow(image)
    # show_points(input_point, input_label, plt.gca())
    # plt.axis('on')
    # plt.show()

    print(predictor._features["image_embed"].shape, predictor._features["image_embed"][-1].shape)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    masks.shape  # (number_of_masks) x H x W
    save_masks(image, masks, scores, point_coords=input_point, input_labels=input_label, borders=True)




def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]

    # TODO: green star
    # ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
    #            linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


# def save_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
#     output_dir = os.path.expanduser("static")  # Hardcoded output directory
#     os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it does not exist
#
#     for i, (mask, score) in enumerate(zip(masks, scores)):
#         fig = plt.figure(figsize=(10, 10))
#         plt.imshow(image)
#         show_mask(mask, plt.gca(), borders=borders)
#         if point_coords is not None:
#             assert input_labels is not None
#             show_points(point_coords, input_labels, plt.gca())
#         if box_coords is not None:
#             show_box(box_coords, plt.gca())
#         if len(scores) > 1:
#             plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
#         plt.axis('off')
#         output_path = f'{output_dir}/mask_{i}.png'
#         plt.savefig(output_path)
#         plt.close(fig)


def save_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    output_dir = os.path.expanduser("static")  # Hardcoded output directory
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it does not exist

    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, ax = plt.subplots(figsize=(image.shape[1] / 100, image.shape[0] / 100))
        
        # !!
        ax.imshow(image)
        ax.set_aspect('auto')  # Adjust aspect ratio of the plot to fill the space
        show_mask(mask, ax, borders=borders)

        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, ax)

        if box_coords is not None:
            show_box(box_coords, ax)

        # if len(scores) > 1:
        #     ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

        ax.axis('off')
        output_path = f'{output_dir}/mask_{i}.png'

        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
                            hspace=0, wspace=0)
        plt.margins(0, 0)

        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        # plt.savefig(output_path, pad_inches=0)
        plt.close(fig)



