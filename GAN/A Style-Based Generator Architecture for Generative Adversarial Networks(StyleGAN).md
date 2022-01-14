# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)  

## Abstract  

기존 GAN 기반의 Style tranfer task에서 Generator의 구조를 변형하여 좋은 결과를 보여줬다.  
(포즈, 얼굴의 특징 같은 것을 잘 학습한다.)  
latent factor의 변화 분석, distribution quality metrics에 대한 Generator의 성능 향상이 가능했다.  
논문에서는 이러한 성능향상을 위해, 모든 Generator Architecture에 적용할 수 있는 두가지 방법을 제안한다.  

## 1. Introduction  

최근들어 image generation task는 Generative Adversarial Network를 통해 큰 발전을 했지만, GAN의 Generator는 BlackBox 모델이며 Stochastic feature에 대한 이해가 부족하다.  
Latent Space에 대한 이해도 부족하며 이에 따라 서로 다른 Generator 모델을 비교하기도 힘들다.  

따라서 Style Tranfer를 제어할 수 있도록 Generator를 새롭게 설계했다. Discriminator와 loss function은 따로 건들이지 않있다.  
또한 논문에서는 input latent space는 train data에 대한 확률 밀도를 따르고 피할 수 없이 얽히게 된다고 주장한다.  
이전의 latent space 상의 관계를 추정하는 벙법은 쓸수 없기 때문에, 이를 측정하기 위한 metric인 Paerceptual path length와 linear separability를 제안한다.  
이를통해, 기존의 Generator 구조와 비교하여 더 선형적이고 더 분리된 표현이 가능하다.  

그리고 기존의 고해상도의 얼굴 데이터셋보다 더 높은 품질과 다양성을 제공해주는 데이터셋인 FFHQ를 제시한다.  

### 요약  
> 새로운 Generator구조와 latent space에 대한 정확한 이해를 통해 각 특징에 대한 분리를 더 잘 하기 위한 2가지의 metric을 제시하고 이전보다 좋은 품질의 고해상도 얼굴 데이터셋 FFHQ에 대한 소개  

## 2. Style-based Generator  

일반적으로 latent code는 Generator(feed-froward network)의 입력 계층에 전달된다.  
