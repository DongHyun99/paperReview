# SDD: Single Shot MultiBox Detector

## 1. Intrduction

최근 Object detection 시스템은 기본적으로 다음과 같은 과정을 변형하여 사용되곤 했다.

    - Bonding Box 특정
    - 각 Bonding Box에 대한 픽셀이나 특징을 리샘플링
    - 고품질의 Classifier에 집어넣음

성능은 좋지만 계산이 너무 복잡하고 좋은 하드웨어가 필요하며 시간이 오래걸렸다.  
그러다보니 제일 빠른 모델의 경우에도 7FPS 밖에 안나오는 등 문제가 크