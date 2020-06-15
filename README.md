# HOI-detection-TIN

I used the "Transferable Interactiveness Knowledge for Human-Object Interaction Detection"[[website]](https://github.com/DirtyHarryLYL/HOI-Learning-List)
And based on this method, developed a visual interactive system of Hoi detection.

### Installation
1.Download dataset and setup evaluation and API. (The detection results (person and object boudning boxes) are collected from: iCAN: Instance-Centric Attention Network for Human-Object Interaction Detection [[website]](http://chengao.vision/iCAN/).)

```
chmod +x ./script/Dataset_download.sh 
./script/Dataset_download.sh
```
2.Install Python dependencies.

```
pip install -r requirements.txt
```
3.Download our pre-trained weight (Optional)

```
python script/Download_data.py 1f_w7HQxTfXGxOPrkriu7jTyCTC-KPEH3 Weights/TIN_HICO.zip
python script/Download_data.py 1iU9dN9rLtekcHX2MT_zU_df3Yf0paL9s Weights/TIN_VCOCO.zip
```

### Training
Train on V-COCO dataset

```
python tools/Train_TIN_VCOCO.py --num_iteration 20000 --model TIN_VCOCO_test
```

### Testing

Test on V-COCO dataset

```
python tools/Test_TIN_VCOCO.py --num_iteration 6000 --model TIN_VCOCO
```

### GUI Visualization System

```
python GUI/GUI_demo.py
```
