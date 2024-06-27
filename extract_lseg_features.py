import os
import argparse
import torch
import torchvision.transforms as transforms
from PIL import Image
from modules.lseg_module import LSegModule
from additional_utils.models import LSeg_MultiEvalModule
from tqdm import tqdm
import time
import gradio as gr
class Options:
    def __init__(self):
        parser = argparse.ArgumentParser(description="PyTorch Segmentation")
        # model and dataset
        parser.add_argument("--model", type=str, default="encnet", help="model name (default: encnet)")
        parser.add_argument("--backbone", type=str, default="clip_vitl16_384", help="backbone name (default: resnet50)")
        parser.add_argument("--dataset", type=str, default="ade20k", help="dataset name (default: pascal12)")
        parser.add_argument("--workers", type=int, default=16, metavar="N", help="dataloader threads")
        parser.add_argument("--base-size", type=int, default=520, help="base image size")
        parser.add_argument("--crop-size", type=int, default=480, help="crop image size")
        parser.add_argument("--train-split", type=str, default="train", help="dataset train split (default: train)")
        parser.add_argument("--aux", action="store_true", default=False, help="Auxiliary Loss")
        parser.add_argument("--se-loss", action="store_true", default=False, help="Semantic Encoding Loss SE-loss")
        parser.add_argument("--se-weight", type=float, default=0.2, help="SE-loss weight (default: 0.2)")
        parser.add_argument("--batch-size", type=int, default=16, metavar="N", help="input batch size for training (default: auto)")
        parser.add_argument("--test-batch-size", type=int, default=16, metavar="N", help="input batch size for testing (default: same as batch size)")
        # cuda, seed and logging
        parser.add_argument("--no-cuda", action="store_true", default=False, help="disables CUDA training")
        parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
        # checkpoint
        parser.add_argument("--weights", type=str, default='', help="checkpoint to test")
        # evaluation option
        parser.add_argument("--eval", action="store_true", default=False, help="evaluating mIoU")
        parser.add_argument("--export", type=str, default=None, help="put the path to resuming file if needed")
        parser.add_argument("--acc-bn", action="store_true", default=False, help="Re-accumulate BN statistics")
        parser.add_argument("--test-val", action="store_true", default=False, help="generate masks on val set")
        parser.add_argument("--no-val", action="store_true", default=False, help="skip validation during training")
        parser.add_argument("--module", default='lseg', help="select model definition")
        # test option
        parser.add_argument("--data-path", type=str, default='../datasets/', help="path to test image folder")
        parser.add_argument("--no-scaleinv", dest="scale_inv", default=True, action="store_false", help="turn off scaleinv layers")
        parser.add_argument("--widehead", default=False, action="store_true", help="wider output head")
        parser.add_argument("--widehead_hr", default=False, action="store_true", help="wider output head")
        parser.add_argument("--ignore_index", type=int, default=-1, help="numeric value of ignore label in gt")
        parser.add_argument("--label_src", type=str, default="default", help="how to get the labels")
        parser.add_argument("--arch_option", type=int, default=0, help="which kind of architecture to be used")
        parser.add_argument("--block_depth", type=int, default=0, help="how many blocks should be used")
        parser.add_argument("--activation", choices=['lrelu', 'tanh'], default="lrelu", help="use which activation to activate the block")
        parser.add_argument("--image_devide", type=int, default=2, help="resize the image to a smaller size")

        self.parser = parser

    def parse(self):
        args = self.parser.parse_args(args=[])
        args.cuda = not args.no_cuda and torch.cuda.is_available()
        return args

def extract_lseg_img_feature(image, evaluator):
    with torch.no_grad():
        outputs = evaluator.parallel_forward(image, '')
        feat_2d = outputs[0][0].half().to("cuda")
    return feat_2d

def extract_save_lseg_features(inputfiles, progress, devide=2):
    num_files = len(inputfiles)
    progress(0 / num_files)
    args = Options().parse()

    torch.manual_seed(args.seed)
    args.scale_inv = False
    args.widehead = True
    args.dataset = 'ade20k'
    args.backbone = 'clip_vitl16_384'
    args.weights = 'lang-seg/checkpoints/demo_e200.ckpt'
    args.ignore_index = 255
    args.image_devide = devide

    model = LSegModule.load_from_checkpoint(
        checkpoint_path=args.weights,
        data_path=args.data_path,
        dataset=args.dataset,
        backbone=args.backbone,
        aux=args.aux,
        num_features=256,
        aux_weight=0,
        se_loss=False,
        se_weight=0,
        base_lr=0,
        batch_size=1,
        max_epochs=0,
        ignore_index=args.ignore_index,
        dropout=0.0,
        scale_inv=args.scale_inv,
        augment=False,
        no_batchnorm=False,
        widehead=args.widehead,
        widehead_hr=args.widehead_hr,
        map_location="cpu",
        arch_option=0,
        block_depth=0,
        activation='lrelu',
    )


    # Define image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    for i, img_path in enumerate(inputfiles):
        # Update progress
        progress((i + 1) / num_files, desc=f"Processing file {i+1}/{num_files}")

        # Save features
        file_name = os.path.basename(img_path).split('.')[0]
        out_dir = './temp/feat_lseg'
        os.makedirs(out_dir, exist_ok=True)
        feature_path = os.path.join(out_dir, f'{file_name}.pt')
        
        if os.path.exists(feature_path):
            continue
        # Load and resize image
        #img_path = '/home/fangj1/Code/go_vocation/data/scene_example/color/576.jpg' #'/home/fangj1/Code/go_vocation/data/scene_example/color/392.jpg'
        image = Image.open(img_path)
        width, height = image.size
        image = image.resize((int(width / devide), int(height / devide)), Image.NEAREST)

        # Ensure correct model settings
        model.crop_size = 2 * max(image.size)
        model.base_size = 2 * max(image.size)

        # Prepare evaluator
        evaluator = LSeg_MultiEvalModule(model, scales=[1], flip=True).cuda()
        evaluator.eval()

        # Apply transformation
        image = transform(image).unsqueeze(0).cuda()

        # Extract features
        feat = extract_lseg_img_feature(image, evaluator)
        feat = feat[:, :image.shape[2], :image.shape[3]]#select correct length
        # Save features
        torch.save(feat, feature_path)   

        # Clear variables to free memory
        del image, feat
        torch.cuda.empty_cache()
        time.sleep(0.1)
    print("Features are extracted features saved at ./temp/feat_lseg")
    return "Extracted features saved at ./temp/feat_lseg"
if __name__ == '__main__':
    inputfiles = ['/home/fangj1/Code/go_vocation/data/scene_example/color/576.jpg', '/home/fangj1/Code/go_vocation/data/scene_example/color/392.jpg']
    extract_save_lseg_features(inputfiles, devide=2)