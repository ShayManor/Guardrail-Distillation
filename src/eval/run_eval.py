from eval import run_eval
from analysis import get_worst_k, plot_results
from data import CITYSCAPES_LABELID_TO_TRAINID, CITYSCAPES_COLOR_TO_TRAINID

run_eval("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
         "hf://ShayManor/leftImg8bit_trainvaltest/validation", "results/b5_dz.csv")
plot_results("results/b5_dz.csv", save_dir="results/figures")
worst = get_worst_k("results/b5_dz.csv", k_percent=5)
