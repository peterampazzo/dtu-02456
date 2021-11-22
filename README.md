# dtu-02456

### How to set up local env:

```
virtualenv env
source env/bin/activate # MacOS
.\env\Scripts\activate # Windows
pip install --upgrade pip
pip -r requirements.txt
```

### How to run Jupyter locally:
```
# cd into project folder
source env/bin/activate # MacOS
.\env\Scripts\activate # Windows

jupyter lab
```

### How to run bike detection script:
```
git clone https://github.com/eriklindernoren/PyTorch-YOLOv3.git pytorchyolo
bash ./pytorchyolo/weights/download_weights.sh
mv -t pytorchyolo/weights/ yolov3-tiny.weights yolov3.weights darknet53.conv.74

# Then run:
python src/bike_detections.py
```


[Guide to HPC](https://docs.google.com/document/d/1pBBmoLTj_JPWiCSFYzfHj646bb8uUCh8lMetJxnE68c/edit)