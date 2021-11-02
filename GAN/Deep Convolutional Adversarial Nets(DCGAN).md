# Deep ConVolutional Adversarial Nets(DCGAN)

## Abstract  

최근 컴퓨터 비젼 분야에서는 CNN을 이용한 Supervised learning이 대세였고, Unsupervised learning은 덜 주목 받았다.  
논문에서는 강력한 비지도 학습인 DCGAN에 대해서 소개한다.  
적대적 생성망에 Convolution layer를 결합해 계층적으로 학습하여 좋은 성능을 보여주었다.  

## 1 Introduction

요약하자면 GAN은 굉장히 매력적이라고 한다.  
그리고 논문의 기여에 대해 설명하자면  

    - Convolutional GAN에 몇가지 제약조건을 통해 학습을 안정화 시킴. 이것을 DCGAN이라고 부르기로 했다.  
    - 학습된 Discriminator를 통해 이미지 Classification을 수행했고, 다른 비지도학습과 경쟁했을 때 좋은 성능을 보임
    - GAN을 통해 학습 된 필터를 시각화 할 수 있음
    - 여러가지의 생성 샘플의 semantic qualities를 쉽게 조절하기위해 생성자는 흥미로운 벡터 산술 특성을 가지고 있음

## 3 Approach and Model Architecture  

지금까지의 여러 시도들을 보면 GAN에 CNN을 적용한 결과물은 성공적이지 못했다.  
논문의 저자 또한 이전에 사용되었던 CNN 아키텍처를 이용하여 GAN을 시도하는데에 어려움을 겪었다.  
그러나 다양한 모델과 다양한 데이터 셋으로 실험해본 결과 고해상도의 CNN 모델의 학습이 가능한 아키텍쳐를 설계할 수 있었다고 한다.  
핵심 수정사항은 총 3가지이다.

1. Generator의 spatial pooling을 (max pooling과 같은) stride convolution으로 대체하였다. 
    - 이를통해 자체적인 spatial upsampling과 discriminator를 학습시킬 수 있었다.  

2. Convolution feature에서 연결되어있는 FC Layer를 전부 제거하였다.  
    - Classification에서 GAP (Global Average Pooling)을 사용하는것과 같은 이치
    - 다만 GAP은 안정적인 만큼 수렴속도를 낮춘다.
    - 그래서 Convolution 된 feature를 Discriminator와 Generator에 그냥 연결해 봤는데 잘 작동했다고 한다.  

3. 