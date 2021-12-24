# 02456 Deep Learning Project

### TRANSFER LEARNING: PASSENGERS AND HELMET USE ON MOTORCYCLES IN YANMAR AND NEPAL


## Project Structure 

```
├── app.conf                           <- Shared settings across the repo
├── README.md                          <- The top-level README for developers using this project
├── notebooks                          <- All notebooks
|   ├── inspect_model.csv              <- Inspect the achitecture of the save model 
│   └── ...
├── src
|   ├── bike_detection.py            <- Bike detection using YOLOv3
│   ├── classification_encode.py     <- Classification Encoding
│   ├── data_split.py                <- Data Splitting
│   ├── merge_annotations.py         <- Merge annotation files
│   └── utils.py                     <- Utility functions
├── scripts
|   ├── cp_nepal.sh                   <- Copy Nepal data set from shared folder
|   ├── dl_myanmar.sh                 <- Download Myanmar data set from the web
│   └── run_on_hpc.sh                 <- Run classification_encode.py on HPC with GPU
│
└── requirements.txt                  <- The requirements file for reproducing the analysis environment
```

## Development

Find below all the instructions to set up your enviroment, run Jupyter Lab and use DTU HPC with GPU.

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

## Run on HPC
[Guide to HPC](https://docs.google.com/document/d/1pBBmoLTj_JPWiCSFYzfHj646bb8uUCh8lMetJxnE68c/edit)

The model can be trained running `python src/classification_encode.py`.
The script requires some arguments to be executed:

```
usage: classification_encode.py [-h] [--load LOAD] [--train | --no-train] [--tuning | --no-tuning]
                                [--save SAVE]
                                project

positional arguments:
  project               Project to run.

optional arguments:
  -h, --help            show this help message and exit
  --load LOAD           (Optional) Saved model to load.
  --train, --no-train   (Optional) if model needs to be trained.
  --tuning, --no-tuning
                        (Optional) if model needs to be fine tuned.
  --save SAVE           (Optional) Filename for saving the model.
```

Be sure to adapt the bash script accordingly for the desired output.

```
# Make sure a logs file exists
mkdir logs

# Adapt the bash script according to the model you want to run, then run
bsub < ./scripts/run_on_hpc.sh 
```

## Data set

Extra storage has been requested to DTU for this project. Since the provided data for Myanmar wasn't completed, we created our own copy. Moreover, to facilitate the preprocessing we made a copy of the Nepal dataset as well.

#### MYANMAR

```
bash ./scripts/dl_myanmar.sh
python src/merge_annotations.py Myanmar # check
python src/data_split.py Myanmar
```

#### NEPAL

```
bash ./scripts/cp_nepal.sh
python src/merge_annotations.py Nepal
python src/data_split.py Nepal
```

## Authors 

* Christina Nørgaard Bartozzi
* Erla Hrafnkelsdóttir 
* Pietro Rampazzo