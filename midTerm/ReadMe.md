# 2021년도 1학기 인공지능과 딥러닝

##Project 1: implementation of backpropagation /wo torch library

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
