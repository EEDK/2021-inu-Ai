# Project 1: implementation of backpropagation /wo torch library

torch를 사용하지 않은 딥러닝 예시

Training을 위한 파일 
- 네트워크 구성 및 weight initialization 함수
- weight summation 및 activation 함수
- 순방향 전파 함수
- 각 layer에서의 error 계산 및 저장 함수
- weight update 함수 (Learning rate는 자유)
- Epoch (시행횟수)를 입력 중 하나로 받는 전체 training 함수 (시행횟수 자유)


Test 를 위한 파일
- Training의 결과를 사용 (학습된 weight 값 사용)
- train dataset을 이용한 예측 값 구하는 함수
- 예측 값과 정답을 비교하여 출력하는 함수

평가항목:
 - 2개의 파일 및 파일 내 해당 함수 구현하였는가?
 - 에러 없이 동작하는가?
 - 정상적으로 동작하는가? 

가산점:
 - 각 변수를 입력 dataset과 output class가 변함에 따라 적용될 수 있도록 구현하였는가?
 - ReLu로 activation을 바꾸고 동작하는가? 
 - ReLu로 바꿨을 경우 성능이 차이난다면 그 이유는 무엇인가?
