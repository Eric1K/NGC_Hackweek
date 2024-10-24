#parser.add_argument('--source', default = ROOT / 'data/images', help = 'file/dir/URL/glob, 0 for webcam')
# python yolov5/detect.py --source 1
# python yolov5/detect.py --weights yolov5x.pt --source 1
# parser.add_argument("--view-img", action="store_true", help="show results")
# python yolov5/detect.py --weights yolov5x.pt --source ~/file directory --view-img
# --conf-thres 0.9
# --iou-thres (overlap between objects)


"""
Running on video data set
yolov5x.pt is the file of trained data

Train data:
python train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5s.pt

python yolov5/train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5x.pt --device 0

python yolov5/train.py --img 640 --epochs 3 --data coco128.yaml --weights yolov5x.pt --device 0


python train.py --img 640 --epochs 3 --data dataset.yaml --weights yolov5s.pt

python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --cfg models/yolov5x.yaml --weights yolov5x.pt --name custom_yolov5x


python yolov5/train.py --img 640 --epochs 3 --data yolov5/office-2/data.yaml --weights yolov5/yolov5s.pt --device 0

python yolov5/train.py --img 640 --epochs 3 --data yolov5/office-2/data.yaml --weights yolov5yolov5s.pt --device 0

python yolov5/train.py --img 640 --epochs 10 --data yolov5/office-2/data.yaml --weights yolov5/yolov5s.pt --device 0


python yolov5/detect.py --weights yolov5/runs/train/exp14/weights/best.pt --img 640 --conf 0.01 --source prenotedimage/Image10.jpg

python yolov5/detect.py --weights yolov5/yolov5x.pt --img 640 --conf 0.5 --source prenotedimage/Image7.jpg

"""


from roboflow import Roboflow
rf = Roboflow(api_key="5xiGW78XB4P0sC8q2CN5")
project = rf.workspace("ngc-gizra").project("office-jaiku")
version = project.version(2)
dataset = version.download("yolov5")
                


