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

요약하자면 CGAN을 통해 광범위한 image-to-image translation에 효과적이고 간단한 프레임워크를 제시하는 것이 이 논문의 목적이다.  

Pix2Pix 논문 저자의 코드(https://github.com/phillipi/pix2pix)  

## 2 Related work  

GAN 및 CGAN에 대한 내용이므로 생략  

## 3 Method  

### 3.1 Objective  

![img](./Asset/3.png)  
CGAN Loss를 설명하는 공식인데 조금 햇갈리게 써놨다.  
y가 GT인 real image이고 z가 random noize vector, x는 CGAN의 조건에 해당한다. (CGAN의 원 논문의 y가 x로 되어있다..)  

또한 Unconditional한 경우를 고려하는 Loss의 공식도 설명했다.  
![img](./Asset/4.png)  

여기서 GAN loss는 L2 distance와 같은 전통적 손실함수들과 혼합하는 편이 더 효과적이다. 왜냐하면, Generator의 목적은 오로지 Discrimonator를 속이는 것이기 때문에 (Loss 값을 minimize) 실제 이미지와 직접 distance를 비교해주는 loss와 혼합해주는 편이 좋다.  
논문에서는 L2에 비해 L1을 사용하는 편이 덜 흐려져 L1을 사용했다고 한다.  
![img](./Asset/5.png)  

그렇게 만들어진 최종 Loss Function은 다음과 같다.  
![img](./Asset/6.png)  

또한 이전의 GAN에서는 Gaussian Noise Z를 넣었던 것에 반해, 오히려 Noise를 무시하는 방향으로 학습하기 때문에 Z를 넣지 않았다고 한다.  

대신 train 및 test 시 생성자의 몇몇 layer에 dropout을 넣어줌으로서 노이즈를 제공해 줬다.  

### 3.2 Network architectures  

DCGAN과 다르게, Generator와 Discriminator에 Convolution-BatchNorm-ReLU의 구조를 사용했다고 한다.  

#### 3.2.1 Generator with skips