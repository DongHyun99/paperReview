# Shadow Removal via Shadow Image Decomposition (SID)  

2019년 ICCV에서 제안된 그림자 제거 논문이다.  

## Abstract  

linear illumination transformation (선형 조명 변환)을 사용하여 shadow-free image, shadow parameter, matte layer를 조합하는 그림자 제거 방식을 제안한다.  
이를 위해서 SP-Net, M-Net이라는 2개의 모델을 통해 각각 shadow parameter와 shadow matte respectively(그림자 무광도)를 예측한다.  

이전의 ISTD dataset SOTA 모델과 비교할 때 무려 RMSE를 40%(13.3->7.9) 가량 감소시켰다.  
또한 shadow parameter를 수정하여 image decomposition system을 기반으로 ISTD dataset을 증강시켰다. (증강한 ISTD dataset을 통해 모델을 학습시키면 RMSE가 7.4로 더 낮아진다!!)  

## 1. Introduction  

그림자의 가장자리(edge)는 객체가 변화(여러물체에 그림자가 지는 등)하기 때문에 구별하기 어렵다. 그래서 그림자를 식별하고 제거하기위한 많은 연구들이 제안되었다.  

초기에는 source-occluder system을 통해 parameter를 측정해 그림자를 제거했지만 (물리적인 방식으로 그림자를 제거했다.) 이러한 방식은 상당한 시간과 자원을 필요로 하는 방식이였다.  

한편, 최근에는 방대한 양의 shadow dataset들이 탄생했기 때문에 그림자 제거에 딥러닝을 이용한 방식을 사용할 수 있게 되었다. 이러한 방식은 network가 shadow image를 shadow-free image로 mapping 하기위해 end-to-end 방식으로 훈련된다.  
그러나 딥러닝을 통한 그림자 제거는 물리적인 특성을 무시하기때문에, 그럴듯하게 그림자가 제거된다는 보장이 없다. 또한 결과가 blurry하고 자연스럽게 보이지 않는다.  
따라서 생성된 영상의 품질을 향상시키는 방법들이 활발히 연구되고 있다.  

**따라서 본 연구에서는 그림자의 조명과 딥러닝을 모두 활용하는 새로운 그림자 제거 방식을 제안한다.**  

제안한 illumination model(조명 모델)은 그림자 영역에 대한 scaling factor와 색상 채널당 추가 상수로 구성된 선형 변환이다. parameter 추정치를 통해 그림자를 제거하는데에 중요한 역할을 한다.  
![img](./Asset/20.png)  
  
이는 SP-Net 딥러닝 모델이 담당하는데, SP-Net은 그림자 이미지에서 illumination model parameter까지 mapping을 학습한다.  

또한 저자는 shadow matting 기술을 사용해 그림자의 음영 영역을 처리한다고 한다. illumination model을 image decomposition formulation에 통합하여, shadow image, parameter, shadow density matte (그림자 밀도 matte)를 조합하여 shadow-free image를 생성한다. (위의 그림과 같다.)  

