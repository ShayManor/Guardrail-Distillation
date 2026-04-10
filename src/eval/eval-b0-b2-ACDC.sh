cd ~/Guardrail-Distillation

# Checkpoint directories — set these to match your training output dirs
B0_DIR="${B0_DIR:-outputs-mit-b0-v3}"
B1_DIR="${B1_DIR:-outputs-mit-b1}"
B2_DIR="${B2_DIR:-outputs-mit-b2}"
EVAL_DIR="${EVAL_DIR:-acdc_b0_b2_eval}"

PYTHONPATH=/root/Guardrail-Distillation:/root/Guardrail-Distillation/src:/root/Guardrail-Distillation/src/train \

python -m src.eval.full_eval eval \
  --run-id acdc_b0_sup \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b0 \
  --student-ckpt "$B0_DIR/student_sup.ckpt" \
  --seeds 42,137,256 \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b0_kd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b0 \
  --student-ckpt "$B0_DIR/student_kd.ckpt" \
  --seeds 42,137,256 \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b0_skd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b0 \
  --student-ckpt "$B0_DIR/student_skd.ckpt" \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt "$B0_DIR/guardrail.ckpt" \
  --seeds 42,137,256 \
  --guardrail-student-name student_skd \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir "$EVAL_DIR"

  python -m src.eval.full_eval eval \
  --run-id acdc_b1_sup \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b1 \
  --student-ckpt "$B1_DIR/student_sup.ckpt" \
  --seeds 42,137,256 \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b1_kd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b1 \
  --student-ckpt "$B1_DIR/student_kd.ckpt" \
  --seeds 42,137,256 \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b1_skd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b1 \
  --student-ckpt "$B1_DIR/student_skd.ckpt" \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt "$B1_DIR/guardrail.ckpt" \
  --guardrail-student-name student_skd \
  --seeds 42,137,256 \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir "$EVAL_DIR"

  python -m src.eval.full_eval eval \
  --run-id acdc_b2_sup \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b2 \
  --student-ckpt "$B2_DIR/student_sup.ckpt" \
  --seeds 42,137,256 \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b2_kd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b2 \
  --student-ckpt "$B2_DIR/student_kd.ckpt" \
  --seeds 42,137,256 \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval eval \
  --run-id acdc_b2_skd \
  --dataset-name acdc --dataset-path /root/Guardrail-Distillation/data/acdc \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b2 \
  --student-ckpt "$B2_DIR/student_skd.ckpt" \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt "$B2_DIR/guardrail.ckpt" \
  --guardrail-student-name student_skd \
  --seeds 42,137,256 \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir "$EVAL_DIR"

python -m src.eval.full_eval plots --output-dir "$EVAL_DIR"
