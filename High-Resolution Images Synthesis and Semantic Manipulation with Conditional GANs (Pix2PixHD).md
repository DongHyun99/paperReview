# High-Resolution Images Synthesis and Semantic Manipulation with Conditional GANs (Pix2PixHD)  

Image-to-Image Translation with Conditional Adversarial Networks(Pix2Pix)의 후속논문으로 UC Berkeley와 NVIDIA에서 작성한 논문이다.  

## Abstract  

이 논문은 conditional GAN을 통해 높은 해상도(high resolution)의 semantic label map을 realistic한 이미지로 생성했다고 한다.  
conditional GAN은 높은 해상도에서 사용하기 힘들었는데 새로운 multi-scale Generator와 Discriminator + 새로운 Adversarial loss를 사용해 2048 x 1024의 사이즈로 괜찮은 결과를 뽑아냈다.  

그리고 두가지의 추가적인 기능을 통해 **interactive visual manipulation**으로 확장하는데 이는 다음과 같다.  

    1. object manipulation이 가능한 객체 segmentation information을 통합한다.  
    2. 동일 입력에 대한 다양한 결과를 생성해 사용자가 객채의 모양을 편집할 수 있도록 한다.  

이를 통해 image synthesis 품질과 해상도를 향상시켰다.  

## 1. Introduction  

데이터 학습 모델(DL)을 통해 realistic한 이미지를 렌더링 할수 있으면 가상 셰계를 생성하는 프로세스를 간소화 할수 있을 것이다.  
왜냐하면, 조명이라던가 환경적인 요소를 사용자가 직접 모델링 해야 하는데 그런 것들을 자동화해주기 때문이다.  

논문에서는 앞서 설명했듯 sementic label map을 통해 높은 해상도의 realistic 한 이미지를 생성했다. 그리고 여기서 사용된 메소드는 응용할 수 있는 범위가 넓다.  

이를 위해서는 image-to-image translation에서 사용되는 (conditional GAN을 활용하는) Pix2Pix 방법을 사용한다. 다만 Pix2Pix는 높은 해상도의 작업에서는 좋은 결과물을 보여주지 못했다고 한다.  
따라서 질감과 미세한 디테일을 살리기 위해 Perceptual Loss를 사용한다.

일단 처음에는 따로 Perceptual loss 등에 