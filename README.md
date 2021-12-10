# 모델 최적화 대회

## Project 개요
분리수거 로봇의 핵심 기술인 쓰레기 분류기를 만들면서 실제로 로봇에 탑재될 만큼 작고 계산량이 적은 모델을 만들어봅니다.
+ Input : COCO format의 TACO Dataset 쓰레기의 이미지
+ Output : 입력된 쓰레기 이미지에 대한 분류 성능 (f1 score)과 추론 속도 (submit time)

## Data
### TACO (Trash Annotations in Context Dataset)
* 총 6개의 category (COCO format)
class: Metal, Paper, Paperpack, Plastic, Plasticbag, Styrofoam
*	Train + Valid : 20851
*	Test : 5217
    *	Public  : 2606
    *	Private : 2611


## 평가 방법
- **Score (f1-score)** : 분류 성능 지표로, 기준이 되는 모델의 f1 score에서 제출한 모델의 f1 score의 차이를 구한 뒤, 상수를 곱하고 sigmoid 함수를 적용한 값 (낮을수록 좋음)
- **Score (submit time)** : 기준이 되는 모델의 추론하는 시간으로 제출한 모델의 추론하는 시간을 나눈 값 (낮을수록 좋음)
![](https://i.imgur.com/djKuklt.png)

## Model
### MnasNet
<img src='https://i.imgur.com/6T8TTtZ.png' width='400'>

* Latency 를 주 목표에 포함시켜 Accuracy 와 Latency 의 좋은 균형을 이루는 최적의 모델
* CNN 모델을 고유한 블록으로 분해한 다음 블록 당 작업 및 연결을 개별적으로 검색하는 새로운 분해된 계층적 검색 공간이 사용되어 서로 다른 블록에서 서로 다른 계층 아키텍처를 허용

## Optimization Apply

- **Pruning** : 네트워크에서 중요도가 낮은 파라미터들을 제거하여 사이즈를 줄이는 접근법
- **Quantization** : 더 작은 Data Type으로 Mapping하는 기법 

## AutoML(Optuna)
- [tune.py](https://github.com/boostcampaitech2/model-optimization-level3-cv-17/blob/main/tune.py) 
- **Optuna** : 하이퍼 파라미터 자동 최적화 라이브러리

### Pretrained Model
- [src/MnasNet.py](https://github.com/boostcampaitech2/model-optimization-level3-cv-17/blob/main/src/MnasNet.py)
## Baseline Structure
```
.
├── configs
│   ├── data
│   └── model
├── src
│   ├── MnasNet.py
│   ├── MoGA_A.py
│   ├── VBMF
│   ├── augmentation
│   ├── dataloader.py
│   ├── decompositions.py
│   ├── loss.py
│   ├── mobilenetv2.py
│   ├── mobilenetv3.py
│   ├── model.py
│   ├── modules
│   ├── robnet.py
│   ├── trainer.py
│   └── utils
├── tests
│   ├── test_model_conversion.py
│   └── test_model_parser.py
├── requirements.txt
├── inference.py
├── train.py
└── tune.py
```

## 🏆Result
### Public Score!
![](https://i.imgur.com/ilLjI26.png)

### Private Score!
![](https://i.imgur.com/rP03uV2.png)

<br/>
  
---
## Members

|   <div align="center">김주영 </div>	|  <div align="center">오현세 </div> 	|  <div align="center">채유리 </div> 	|  <div align="center">배상우 </div> 	|  <div align="center">최세화 </div>  | <div align="center">송정현 </div> |
|---	|---	|---	|---	|---	|---	|
| <img src="https://avatars.githubusercontent.com/u/61103343?s=120&v=4" alt="0" width="200"/>	|  <img src="https://avatars.githubusercontent.com/u/79178335?s=120&v=4" alt="1" width="200"/> 	|  <img src="https://avatars.githubusercontent.com/u/78344298?s=120&v=4" alt="1" width="200"/> 	|   <img src="https://avatars.githubusercontent.com/u/42166742?s=120&v=4" alt="1" width="200"/>	| <img src="https://avatars.githubusercontent.com/u/43446451?s=120&v=4" alt="1" width="200"/> | <img src="https://avatars.githubusercontent.com/u/68193636?v=4" alt="1" width="200"/> |
|   <div align="center">[Github](https://github.com/JadeKim042386)</div>	|   <div align="center">[Github](https://github.com/5Hyeons)</div>	|   <div align="center">[Github](https://github.com/yoorichae)</div>	|   <div align="center">[Github](https://github.com/wSangbae)</div>	| <div align="center">[Github](https://github.com/choisaywhy)</div> | <div align="center">[Github](https://github.com/pirate-turtle)</div>|
