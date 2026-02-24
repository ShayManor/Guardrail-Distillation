from eval import run_eval
from analysis import get_worst_k, plot_results

run_eval("nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
         "hf://ethz-asl/dark_zurich/val", "results/b5_dz.csv")

worst = get_worst_k("results/b5_dz.csv", k_percent=5)
plot_results(["results/b5_dz.csv", "results/b0_dz.csv"], save_dir="results/figures")