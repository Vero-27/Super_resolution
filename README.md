# Super_resolution

installer le data set et glisser les .png dans le dossier '/data/HR' :  https://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip 

Dans un terminal, à la racine du projet, lancer : 
        python -m training.train_sr --data data/HR --model espcn --scale 2 --epochs 50 --batch 8 --out Core/checkpoints/espcn_x2.pth
