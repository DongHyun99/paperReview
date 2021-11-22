# Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (CycleGAN)  

CycleGAN은 Pix2Pix 저자들의 후속연구라고 한다.  

## Abstract  

기본적으로 Image-to-image translation은 짝을 이루는 pair 이미지로 훈련하는게 필수적이지만, CycleGAN은 짝이 없는 도메인 X에서 도메인 Y로의 변환하는 방법을 제시한다.  
즉 짝이 없어도 학습이 가능하다.  

이게 가능하기 위해서는 Cycle Consistency loss를 사용해야 한다.  

## 1. Introduction  

논문에서는 pair 데이터 없이도 이미지의 특성을 배울 수 있는 방법에 대해 소개한다.  
이 방법이 필요한 이유는 pair 데이터는 비싸고 구하기 어렵기 때문에 image-to-image translation에 좋은 해결책이 될 것이라고 한다.  

앞서 언급한 **cycle consistent**란 특성이 사용된다. 이것은 영어에서 프랑스어로 번역한 문장을 다시 영어로 번역했을 때 원래의 문장으로 돌아가는 것과 같다.  

## 3. Formulation  

목표는 도메인 X와 도메인 Y가 있을 때, mapping function을 학습시키는 것이다.  
(G: X->Y, F: Y->X)  
또한 x와 F(y)를 구별하고, y와 G(x)를 구분할 수 있어야 한다.  
즉, Discriminator와 Generator가 2개씩 있다는 소리다.  

CycleGAN의 모델은 이를 만족하기 위한 ***adversarial Loss***와 G와 F가 서로 모순되는 것을 방지하는 ***cycle consistency Loss***가 존재한다.  

### 3.1 Adversarial Loss  

