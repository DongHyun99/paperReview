# From Shadow Segmentation to Shadow Removal

## Abstract  

Pair 데이터셋은 구하기 어렵고 사이즈나 기타 다양한 조건들을 맞춰야 하기 때문에 학습을 할 때 제한적일수 밖에 없다. 이는 그림자 제거 분야에서도 마찬가지다.  
따라서 논문에서는 그림자 이미지에서 shadow patch와 shadow-free patch를 잘라내어 학습하는 방법을 제안한다.  
Pair 데이터셋을 통한 학습방식과 비슷한 정도의 결과물을 낼수 있고 특히 동영상의 그림자 제거에 효과가 탁월하다. 

> KeyWords: Shadow Removal, GAN, Weakly-supervised, Illumination model, Unpaired, Image-to-Image.  

(Weakly-supervised learning이란?: https://nuguziii.github.io/survey/S-004/)  

## 1. Introduction  

현재 DNN을 이용한 Shadow Removal task는 거의 대부분 완전한 Supervised Learning을 이용한다. 이같은 모델은 Pair image 데이터셋을 요구하는데 Pair image는 데이터를 만들기 힘들고 다양성이 부족하다는 단점이 있다.  
거기다가 Shadow image와 Shadow-free image는 서로 색상 불일치를 보이기 때문에 Shadow Removal task에서는 단점이 많은 방식이다.  