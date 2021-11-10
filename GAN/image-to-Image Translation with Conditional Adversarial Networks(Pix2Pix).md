# image-to-Image Translation with Conditional Adversarial Networks(Pix2Pix)  

## Abstract  

image-to-image translation 문제에서 conditional adversarial network를 사용해보았다.  

(여기서 image-to-image translation이란?어떤 도메인의 이미지를 가져와서 다른 도메인에서 이미지의 특성을 가지도록 변환하는 작업 -> https://paperswithcode.com/task/image-to-image-translation)  

여기에서 입력이미지 -> 출력 이미지 간의 매핑을 학습할 뿐 아니라 이에대한 손실함수의 학습도 가능하다.  
이에따라 다양한 문제애 대해서 따로 손실 함수에 대한 설정을 수동적으로 해야 할 필요없이 좋은 결과를 얻을 수 있다는 것을 강조하고 있다.  

## 1 Introduction  

논문의 목적은 전체적으로 적용가능한 일반화된 프레임워크를 개발하는 것이라고 함.

CNN은 손실 함수를 최소화할 때 수동적으로 조절해줘야 하고 많은 자원을 낭비한다.  
그리고 Naive 하게 픽셀에 대한 Euclidean distance를 최소화 하도록 모델을 제작하면 흐릿한 결과를 생성하는 경향이 있다.  
만약에 출력물을 GT와 비슷하게 만들수 있는 손실 함수를 자동으로 학습이 가능하다면 좋을것이다. -> GAN  

CGAN(Conditional GAN)은 입력 이미지를 조건화 하고, 출력 이미지를 생성하는 image-to-image translation에 적합한 모델이다. 즉 광범위한 모델에 적합하다.  

