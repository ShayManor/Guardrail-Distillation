# Guardrail-Distillation
Implementation for Guardrail Distillation of a Large Vision Teacher to an Edge-Deployed Student
Training:
1) Pull Datasets + models - done
   2) Requires kaggle api key as KAGGLE_API_KEY=...
2) Build eval pipeline
3) Distill models
    - Student: nvidia/segformer-b0-finetuned-ade-512-512
    - Teacher: nvidia/segformer-b5-finetuned-cityscapes-1024-1024
4) Train guardrail
5) Evals

Eval:
