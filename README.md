#### demo_1 Dewei’s old code 
[https://drive.google.com/drive/folders/1UBhen5lmalaFdH8CmSnuMtHbKxyRhbP5](https://drive.google.com/drive/folders/1UBhen5lmalaFdH8CmSnuMtHbKxyRhbP5)
uses only contrastive learning
* TO-DO: add inference code

#### demo_2 
cloves + cumin + oregano with transformer，highest accuracy ≈ 0.626

#### deploy
Realtime Data Collection and Prediction for Live Demo
* run.py: collect data
* monitor.py: show realtime collection visualization
  * TO-DO: improve realtime visualization
* infer.py: add correct model directory to checkpoints and the realtime data collected, run infer.py to predict
  * TO-DO: add realtime prediction code and improve readme


## TO-DOs
### Data Collection
- cumin_glass, cloves_glass, oregano_glass
- new glass box + new 4-channel sensor
- 60min for each: 6 rounds * 10min (cumin + cloves + oregano) (at lease 30 min for each)
- reset the sensor to the lowest point so nothing is dropping

### Model
- model1 contrastive learning: try dewei's old code: demo_1/demo_smell_contrastive_learning_gradient_method.ipynb + write a new inference code
    - try to find the saved model in the christouml laptop
- model2 transformer:
    - put the collected data into data/train
    - run train.py
    - run to_csv.py
    - the most stabel parameter: long_overlap	60	30	[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11]	['highpass_fft_batch']

### Interface
- Write a code for the realtime collection of the sensor data (like in monitor.py)
- Write a code for the realtime prediction