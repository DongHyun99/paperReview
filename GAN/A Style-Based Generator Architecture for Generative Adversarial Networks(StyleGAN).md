# A Style-Based Generator Architecture for Generative Adversarial Networks (StyleGAN)  

## Abstract  

기존 GAN 기반의 Style tranfer task에서 Generator의 구조를 변형하여 좋은 결과를 보여줬다.  
(포즈, 얼굴의 특징 같은 것을 잘 학습한다.)  
latent factor의 변화 분석, distribution quality metrics에 대한 Generator의 성능 향상이 가능했다.  
논문에서는 이러한 성능향상을 위해, 모든 Generator Architecture에 적용할 수 있는 두가지 방법을 제안한다.  

## 1. Introduction  

최근들어 image generation task는 Generative Adversarial Network를 통해 큰 발전을 했지만, GAN의 Generator는 BlackBox 모델이며 Stochastic feature에 대한 이해가 부족하다.  

따라서 Style Tranfer를 제어할 수 있도록 Generator를 새롭게 설계했다. Discriminator와 loss function은 따로 건들이지 않있다.  
또한 논문에서는 input latent space는 train data에 대한 확률 밀도를 따르고 피할 수 없이 얽히게 된다고 주장한다.  
이전의 latent space 상의 관계를 추정하는 벙법은 쓸수 없기 때문에, 이를 측정하기 위한 metric인 Paerceptual path length와 linear separability를 제안한다.  
이를통해, 기존의 Generator 구조와 비교하여 더 선형적이고 더 분리된 표현이 가능하다.  

그리고 기존의 고해상도의 얼굴 데이터셋보다 더 높은 품질과 다양성을 제공해주는 데이터셋인 FFHQ를 제시한다.  

### 요약  
> 새로운 Generator구조와 latent space에 대한 정확한 이해를 통해 각 특징에 대한 분리를 더 잘 하기 위한 2가지의 metric을 제시하고 이전보다 좋은 품질의 고해상도 얼굴 데이터셋 FFHQ에 대한 소개  

## 2. Style-based Generator  

일반적으로 latent code는 Generator(feed-froward network)의 입력 계층에 전달된다.  
StyleGAN은 여기서 입력 레이어를 없애버리고 아래 그림처럼 학습된 Constant layer에서 시작할수 있도록 만들었다.(Constant layer란 상수로 이루어진 레이어를 뜻한다.)  

<img src = "Asset/31.png" width = "60%">  

입력 latent code인 z가 latent space Z에서 주어지면, non-linear mapping newtork인 f: Z->W가 w를 생성해낸다.  
각각에 대한 차원수는 512개로, mapping network f는 8개의 multi-layer perceptron으로 구성된다.  
이렇게 만들어진 intermediate latent space W는 Synthesis network g의 중간중간에 삽입될 latent space로써 AdaIN을 통해 각 convolution layer에 들어간다.  
Gaussian noise는 convolution 이후에 네트워크에 삽입된다.  
f와 g는 각 8개, 18개의 네트워크로 구성되어 있는데 block 단위로 4^2 ~ 1024^2 까지의 resolution에 대해 관여한다.  

다시 w로 돌아가서, 학습된 아핀 transformation은 (그림에서의 A부분) adaptive instance normalization(AdaIN)을 조정해 $y = (y_s, y_b)$의 스타일로 w를 가공한다.  
이것을 Synthesis network g의 각 convolution layer의 뒤에 삽입한다.  

![img](./Asset/32.png)  

feature map $x_i$는 정규화된 style y의 구성요소인 $y_s, y_b$를 통해 변형된다. 따라서 y의 차원수는 해당 layer의 feature map 크기의 2배이다.  

정리하자면 latent vector w에서 공간상 변화하지 않는 style y를 뽑아내 convolution된 feature에 적용하는 것이다.  
다른 일반적인 transforms task와 비교했을 때 adaIN이 효율적이고 간결하다고 한다.  

마지막으로 noise를 통해 stochastic detail을 generator에 적용할 수 있도록 한다. noise는 서로 상관 관계가 없는 가우시안 noise로 구성된 single-channel 이미지이며 각 계층마다 전용 noise를 공급한다.  
noise는 학습된 scaling factor를 통해 각 feature map에 브로드캐스트되며, convolution의 출력에 추가된다.  

### 요약
> 새로운 Generator는 Mapping network f와 Synthesis network g로 구분되고 g는 기존처럼 latent space를 입력으로 두지 않는 대신 constant tensor를 입력으로 받는다. latent space는 f를 통해 intermediate latent space w로 변환되고 adaIN을 통해 g의 중간 중간 layer에 입력된다. 마찬가지로 noise를 중간중간 입력으로 받아온다. 여기서 w는 생성하는 이미지에 스타일을, noise는 스타일에 대한 디테일을 관장한다.  

### 2.1 Quality of generated images  

Generator에 대한 여러 실험 끝에 generator의 구조를 재설계하는 것은 이미지 quality를 개선하는데에 도움이 된다는 것을 알게 되었다.  

![img](./Asset/33.png)  

먼저 bilinear up/downsampling, 더 긴 학습, 하이퍼파라미터 튜닝을 사용했다.  그리고 mapping network, AdaIN을 추가했다. 여기서 첫번째 convolution layer에 latent code를 넣는 것이 효과적이지 못하다는 것을 알게되었다.  
따라서 기존 입력 계층을 제거하고 4x4x512의 constant tensor를 시작으로 이미지를 생성하도록 했다. 네트워크가 AdaIN을 통해서만 style을 제어함에도 불구하고 의미있는 결과를 생성이 가능했다.  

noise의 입력을 통해 결과를 더 개선하고, novel mixing regularization을 통해 세밀하게 제어가 가능해졌다.  

Style 기반의 Generator는 기존에 비해서 FID를 거의 20% 향상시켰다.  

(여기서 FID란? https://m.blog.naver.com/chrhdhkd/222013835684) -> GAN의 품질 평가지표  

latent space z대신 w를 사용하기 위해 truncation trick을 사용하였다. truncation trick은 low resolution에서만 적용하므로 high resolution에서는 세부적인 영향을 끼치지 않는다.  

### 요약
> bilinear up/downsampling, 긴 학습시간, 하이퍼 파라미터 튜닝, 그리고 앞서 말한 것들을 통해 FID를 이전보다 20% 가량 향상 시켰고, truncation trick을 low resolution에 적용해 z에서 w로의 변환을 성공적으로 이뤄냈다.  

### 2.2. Prior art  

대부분의 GAN architecture 관련 연구들은 Discriminator를 개선하는 것에 초점이 맞춰져있다.  
Generator 관련 연구들은 주로 input latent space를 어떻게 형성할 것인가를 제안했다.  
거기다가 GAN architecture의 중간에 latent code를 삽입하려는 연구는 거의 없었다.  

## 3. Properties of the style-based generator  

Style GAN의 Generator는 여러 style에 대한 sacle 별 수정을 통해 이미지의 합성이 가능하다. 
각 style은 네트워크의 지역 (block)에서만 작용하기 때문에, 스타일들은 각 특정한 부분에 대한 영향만을 끼칠수 있다. (한마디로, 스타일들은 서로 엮이지 않고 style 고유의 표현할 부분을 표현한다는 것 같다.)  
 
### 3.1. Style mixing  

style localize를 위해 학습중에 latent code를 한개가 아닌 두개를 사용해서 특정한 비율만큼 이미지를 생성하는 mixing regularization을 사용한다.  
즉 두 latent code z1, z2를 통해 w1, w2가 만들어지고 배치한다.  

![img](./Asset/34.png)  
위의 그림은 두개의 latent code를 통해 다양한 척도로 혼합했을 때 생성된 이미지를 보여준다.  

![img](./Asset/35.png)  
훈련 샘플에서 mixing regularization의 비율을 조정했을 때 FFHQ 데이터 셋의 FID이다.  

### 3.2. Stochastic variation  

사람의 얼굴에는 머리카락, 기미, 주근깨, 피부 모공 등의 stochastic한 특징들이 존재한다.  
만약 이러한 특징들이 정확한 분포를 따른다면 이미지에 손실을 입히지 않고 특징들을 추출할 수 있을 것이다.  

기존의 Generator는 z라는 입력 하나에만 의존하기 때문에 디테일한 특징을 바꿔줄수 없었지만, StyleGAN은 Noise를 추가하여 이러한 점을 개선했다.  

![img](./Asset/37.png)  

noise를 적용했을 때 머리카락과 배경, 피부 모공 등에서 더 디테일한 표현이 가능해졌다.  
또한 noise는 stochastic한 측면에만 영향을 끼치고 high-level의 측면은 그대로인 것을 알수 있다.  

![img](./Asset/36.png)  

(a)는 noise를 전체 layer에서 적용했을때, (b)는 noise를 적용하지 않았을 때이다.  
(c)는 64~1024의 특정 layer에 noise를 적용했을 때, (d)는 4~32의 특정 layer에 noise를 적용했을 때이다.  

위 그림처럼 논문은 noise에 따른 효과는 layer 마다 다르게 나타나고, stochastic variation을 생성하는 가장 쉬운 방법이 noise라고 주장한다.  

### 요약  
> noise는 각 layer마다 적용하고, 사람 얼굴의 여러 스타일(머리카락, 모공, 배경 등)의 디테일한 요소들의 표현을 해주는 역할을 한다.  

### 3.3. Separation of global effects from stochasticity  

Style-based generator의 feature map은 동일 값으로 scale과 biase가 조정되기 때문에 style이 전체 이미지에 영향을 끼친다.  
따라서 포즈, 조명, 배경 스타일 등의 전역적인 특징들은 일관성있게 제어된다.  
한편 noise는 독립적으로 추가되어 stochastic variation을 제어하는데에 이상적으로 사용된다.  
포즈와 같은 전역적인 특징들에 대해서noise가 제어 혹은 간섭하려하면 공간적으로 일관되지 않아 Discriminator가 패널티를 부여한다.  

## 4. Disentanglement studies  

![img](./Asset/38.png)  

Disentanglement에 대해 여러가지 정의가 있기는 하지만, 기본적으로 linear subspace(선형 하위공간)로 구성된 latent space이다. 각 space는 하나의 variation factor을 제어한다.  

각 factor가 조합된 Z에서 표본추출을 하기 위해서는 training data의 밀도와 일치해야 하는데 위의 그림과 같이 입력 잠재분포와 일반적인 dataset으로 disentangle되는것을 방지한다.  

이게 무슨 의미냐면, training dataset은 (a)와 같은 분포로 feature에 대한 space가 형성이 되어 있을 때, (b)와 같은 형태의 feature를 가지고는 (a)의 고정된 분포를 non linear하게 mapping 할 수 밖에 없다. 즉 분포가 어그러져 원하는 feature에 대한 mapping이 힘들다. 따라서 mapping network를 통해 Z를 W로 변화시켜, 학습 데이터셋의 확률 분포와 비슷한 형태를 mapping 시키게 된다.
이를 통해 latent space w는 이전보다 disentangle 하게 된다. (style에 대한 분리가 가능하여 어떠한 feature에 대해 명확하게 이미지를 생성하기 쉽다는 것 같다.)  

다시 본론으로 돌아가서 Generator 구조의 이점중 하나로 latent space는 (b)처럼 고정된 분포로 있을 필요 없이 학습된 f(z)에 의해 분포가 유도된다. 이렇게 disentangled 된 W는 mapping하기 좋고, realistic한 이미지의 생성이 쉽다.  
기존에 disentanglement를 정량화 하기 위해서는 encoder가 필요 했다.  그러나 StyleGAN과 같은 task에서 encoder를 사용하는 것은 피하고 싶었기 때문에 encoder없이 disentanglement에 대해 정량화 하는 두가지 방법을 아래에 제안한다.  

### 4.1. Perceptual path length  

