cd ~/Guardrail-Distillation

PYTHONPATH=/root/Guardrail-Distillation:/root/Guardrail-Distillation/src:/root/Guardrail-Distillation/src/train \

python -m src.eval.full_eval eval \
  --run-id city_b0_sup \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b0 \
  --student-ckpt outputs-mit-b0-v3/student_sup.ckpt \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_kd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b0 \
  --student-ckpt outputs-mit-b0-v3/student_kd.ckpt \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_skd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b0 \
  --student-ckpt outputs-mit-b0-v3/student_skd.ckpt \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt outputs-mit-b0/guardrail.ckpt \
  --guardrail-student-name student_skd \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir cs_b0_b2_eval

  python -m src.eval.full_eval eval \
  --run-id city_b0_sup \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b1 \
  --student-ckpt outputs-mit-b1/student_sup.ckpt \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_kd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b1 \
  --student-ckpt outputs-mit-b1/student_kd.ckpt \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_skd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b1 \
  --student-ckpt outputs-mit-b1/student_skd.ckpt \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt outputs-mit-b1/guardrail.ckpt \
  --guardrail-student-name student_skd \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir cs_b0_b2_eval
  
  python -m src.eval.full_eval eval \
  --run-id city_b0_sup \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_sup --student-backbone nvidia/mit-b2 \
  --student-ckpt outputs-mit-b2/student_sup.ckpt \
  --train-method sup \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_kd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_kd --student-backbone nvidia/mit-b2 \
  --student-ckpt outputs-mit-b2/student_kd.ckpt \
  --train-method kd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval eval \
  --run-id city_b0_skd \
  --dataset-name cityscapes --dataset-path /root/Guardrail-Distillation/data/cityscapes \
  --split val --domain in_domain \
  --student-name student_skd --student-backbone nvidia/mit-b2 \
  --student-ckpt outputs-mit-b2/student_skd.ckpt \
  --train-method skd \
  --teacher-backbone nvidia/segformer-b5-finetuned-cityscapes-1024-1024 \
  --guardrail-ckpt outputs-mit-b2/guardrail.ckpt \
  --guardrail-student-name student_skd \
  --temperature 2.0 \
  --mc-dropout-passes 8 \
  --output-dir cs_b0_b2_eval

python -m src.eval.full_eval plots --output-dir cs_b0_b2_eval