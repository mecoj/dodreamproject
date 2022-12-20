### this code is combined with yolov5 and hybridnet.<br>
### hybridnet is functionalized to use in yolov5 code.<br>
### hybridnet is used to detect lane detection.<br>
### yolov5 is used to detect car, trafficsign ...<br>
### when you use hybridnet, you can use weights file that provided by default in hybridNet<br>
### and finally, you can detect the jaywalking by using test_ui.py with weihts file, is project.pt<br>
### this pt file made by my team !<br>
### this pt file inclue information of trafficlights, person.<br>

### you have to file in this path >>> ..\HybridNets-main\yolov5-master\HybridNets_main\weights\hybridnets.pth <<< <br>
### you can download hybridnets.pth from https://github.com/alibaba/hybridnet

### you have to file in this path >>> ..\HybridNets-main\yolov5-master\project_files\project.pt<<< <br>
### you can download project.pt from https://mecoj0170.tistory.com/11

<div>python 3.8 have to match cuda 11.7v


first you have to install requirements of [./HybridNets-main\yolov5-master\HybridNets_main] <br>
>>> pip install -r requirements
 <br><br>
 

2nd you have to install requirements of [./yolov5-master\HybridNets_main]<br>
pip install -r requirements

you may need to uninstall modules related to pytorch
 and then.. you can install pytorch that is matched cuda 11.7v  
</div>
