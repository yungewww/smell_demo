demo_1: Dewei's old code https://drive.google.com/drive/folders/1UBhen5lmalaFdH8CmSnuMtHbKxyRhbP5
demo_2: cloves + cumin + oregano, highest accuracy = about 0.626
demo_3: cloves + cumin + almond, highest accuracy = about 0.279
demo_4:  cloves + cumin + nutmeg, highest accuracy = about 0.435
demo_5: cloves + cumin + cinammon, hightest accuracy = 



1. change config.py CLASS_LABELS
2. put new online real data into data/online, change _ to .
3. run train.py
4. go to to_csv.py, change LOG_PATH to the newest log.txt, run to_csv.py

when copy and paste a new demo folder:
1. delete logs and checkpoints

every time after trianing
1. put all your checkpoints into a new folder