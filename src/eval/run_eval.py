from eval import run_eval, plot_results

# KD student on Dark Zurich
run_eval(
    model_path="./checkpoints/student_kd.pth",
    dataset_path="hf://danniccs/dark_zurich/val",
    output_csv="results/kd_darkzurich.csv",
    num_classes=19,
    model_name="student_kd",
    dataset_name="dark_zurich",
)

# Compare multiple models
plot_results(
    ["results/kd_darkzurich.csv", "results/teacher_darkzurich.csv"],
    save_dir="results/figures",
)
