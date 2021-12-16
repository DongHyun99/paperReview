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

