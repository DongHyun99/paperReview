# Deep Residual Learning for Image Recognition

> ## ABSTRACT

깊은 신경망은 훈련하기가 어렵다.  
따라서 이 논문에서는 잔차학습(residual learning)을 통해 더 깊은 네트워크의 학습을 쉽게 할 수 있는 방법을 제시했다.  

ImageNet에서 vggNet보다 depth가 8배 깊은 152 layer를 사용했지만 더 낮은 complexity를 가졌다고 한다.
결과적으로 ImageNet testSet에서 3.57%의 error를 달성하며 ILSVRC 2015 Classification 1위를 차지했다. 

> ## 1. Introduction

네트워크의 depth는 매우 중요한 요소이고, ImageNet에서 깊은 네트워크들은 좋은 결과를 보여주었다.  
ResNet 연구팀은 'depth를 늘리기만 해도 쉽게 성능이 향상될까?'라는 의문이 생겼고 
