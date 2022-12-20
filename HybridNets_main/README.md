### this code is combined with yolov5 and hybridnet.
### hybridnet is functionalized to use in yolov5 code.
### hybridnet is used to detect lane detection.
### yolov5 is used to detect car, trafficsign ...
### when you use hybridnet, you can use weights file that provided by default in hybridNet
### and finally, you can detect the jaywalking by using test_ui.py with weihts file, is project.pt
### this pt file made by my team !
### this pt file inclue information of trafficlights, person.

### you have to file in this path >>> ..\HybridNets-main\yolov5-master\HybridNets_main\weights\hybridnets.pth
### you can download hybridnets.pth from https://github.com/alibaba/hybridnet

### you have to file in this path >>> ..\HybridNets-main\yolov5-master\project_files\project.pt<<<
### you can download project.pt from https://mecoj0170.tistory.com/11

<div>파이썬 버전 3.8 버전
cuda 11.7v 버전 맞추기


./HybridNets-main\yolov5-master\HybridNets_main
의 requirements 먼저 설치
pip install -r requirements

./yolov5-master\HybridNets_main
pip install -r requirements


uninstall 로 pytorch 관련 다 지우고
 pytorch <- cuda 버전에 맞춰서 설치하기~</div>
