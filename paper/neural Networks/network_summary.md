# Network Summary

## SENet (Squeeze and Excitation Networks)

SENet은 feature map의 각 channel마다 가중치를 부여하여 feature map의 각 channel에 곱한다.
(즉 가중치가 큰 채널은 중요한 특징을 담고있다고 생각)  

SE block의 개념이 있다.  
SE block의 장점  
	- 다른 네트워크에서도 사용가능  
	- 파라미터의 증가량에 비해 모델 성능 향상도가 매우 크다.  

SE block은 squeeze와 excitation의 두가지 과정으로 구성된다.

(1) squeeze (압축)  

	채널을 1차원으로 압축한다.  
	H x W x C 크기의 feature map을 GAP 연산을 통해 (1 x 1 x c)로 압축한다.  

(2) Excitation(재조정)  

	squeeze에서 생성된 벡터를 정규화 하여 가중치를 부여한다.  
	FC1 - ReLU - FC2 - Sigmoid로 구성되며  
	FC 1을 통과하며 채널을 축소하고 ReLU를 통과한뒤  
	다시 FC 2롤 통과하며 채널 수를 되돌린다.  
	그 후 Sigmoid를 거쳐서 0~1의 범위의 값을 지니게 된다.  
	마지막으로 피쳐맵과 곱해져 패쳐맵의 채널에 가중치를 가한다.  

## DenseNet

Growth Rate 각 feature map끼리 densely connected 구조  
이전 layer들의 feature map을 계속해서 다음 layer의 입력에 연결한다.  
ResNet과 다른점: feature map끼리 더하는것이 아닌 Concat하는 구조다.  

DenseNet에서는 각 layer의 feature map을 계속 연결하다 보면 concat하면서 channel이 매우 많아질 수 있기 때문에
channel의 개수를 작은 값을 사용한다. 이  channel의 개수를 growth rate라고 부른다.  

### Bottleneck layer  
(Tensor의 Depth를 줄이기 위해서 1x1 Conv 층을 사용해 Depth를 줄이는 것) -> 연산량을 줄여줌  

3x3 conv를 거치기 전 1 x 1를 쓰는 것이 ResNet과 같지만 그 뒤로 다시 입력 feature map의 channel 개수 만큼 생성하지 않고
growth rate 만큼 feature map을 생성한다.  

## ResNet
마이크로소프트에서 개발한 알고리즘  
굉장히 deep한 depth를 가짐 (googleNet에 비해 약 7배정도)  
무조건 깊다고 해서 능사는 아니지만 resNet은 residual block을 사용해 성능을 높였다.  
(입력값을 출력값에 더해주는 길을 하나 더 주는 것)  
VGGNet을 토대로 conv layer를 추가해 더 깊게 만든 뒤 위의 방식대로 입력 값을 출력 값에 더해주는 것이 구조이다.  

## ResNext  

같은 block을 반복적으로 구축하여 더적은 파라미터로 이미지를 classification  
더 깊은 차원보다, cardimality를 높이는 것이 분류의 정확도를 높임  
(cardinality: 똑같은 형태의 building block의 개수)  
Inception의 주요 아이디어인 split- transform - merge의 구조를 지님  
Inception resnet과 1개의 입력이 여러 방향으로 쪼개진다는 것은 같고, 각 path 별로 같은 layer 구성을 가진다는 점은 다르다.  
같은 layer의 구성: group convolution이라고 한다.  
resnet이나 inception보다 더 간단한 구조이지만 더 나은 성능을 가진다.  
즉 resnet을 더 깊고 넓게 만드는 것보다 cardinality를 높이는 것이 error를 더 낮춘다.  