from eval import run_eval
from analysis import get_worst_k, plot_results

run_eval("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
         "kaggle://dansbecker/cityscapes-image-pairs", "results/b5_dz.csv")
plot_results("results/b5_dz.csv", save_dir="results/figures")
worst = get_worst_k("results/b5_dz.csv", k_percent=5)
