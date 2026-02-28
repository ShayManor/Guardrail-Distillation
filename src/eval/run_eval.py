from eval import run_eval
from analysis import get_worst_k, plot_results
from data import CITYSCAPES_LABELID_TO_TRAINID, CITYSCAPES_COLOR_TO_TRAINID
from torchvision import transforms as T

img_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
run_eval("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
         "/workspace/data/cityscapes", images_subdir="leftImg8bit/val", labels_subdir="gtFine/val", output_csv="results/b5_dz.csv", label_map=CITYSCAPES_LABELID_TO_TRAINID, label_transform=T.Compose([]))
plot_results("results/b5_dz.csv", save_dir="results/figures")
worst = get_worst_k("results/b5_dz.csv", k_percent=5)
