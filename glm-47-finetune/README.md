# GLM-4.7 Fine-tuning on RTX 4090 (24GB)

QLoRA를 사용한 GLM-4.7 Fine-tuning 프로젝트입니다.

## 요구사항

- GPU: NVIDIA RTX 4090 (24GB VRAM)
- Python: 3.10+
- CUDA: 12.1+

## 설치

```bash
# 1. 의존성 설치
pip install -r requirements.txt

# 2. W&B API Key 설정
export WANDB_API_KEY=your_wandb_api_key
```

## 사용법

```bash
#.training 실행
bash train.sh

#또는 Python으로 직접 실행
python -m src.train configs/train.yaml
```

## 주요 설정

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| Model | THUDM/GLM-4-7b-chat | GLM-4.7.chat 모델 |
| Max Seq Length | 2048 | 최대 시퀀스 길이 |
| LoRA Rank | 16 | LoRA 랭크 |
| Batch Size | 2 | 배치 크기 |
| Gradient Accumulation | 4 | Effective batch: 2×4=8 |
| Learning Rate | 2e-4 | 학습률 |

## 메모리 최적화

- 4-bit 양자화 (QLoRA)
- Gradient checkpointing
- Mixed precision (FP16/BF16)
- Paged optimizer

## W&B 로깅

모든 학습 지표가 W&B에 기록됩니다:
- Loss
- Learning rate
- Gradient norm
- Training speed (tokens/sec)
