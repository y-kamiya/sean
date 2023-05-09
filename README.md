# sean
## setup
```
poetry install
```
or if you don't have poetry
```
pip install -r requirements.txt
```

## train model
in case that you will train with CelebA-HQ dataset put on original SEAN repo, download dataset according to this
https://github.com/ZPdesu/SEAN#dataset-preparation

```
accelerate config
accelerate launch src/train.py --dataroot datasets/CelebA-HQ --load-size 256 --crop-size 256 --name example
```

see logs on tensorboard
```
tensorboard --logdir output/runs
```
and open http://127.0.0.1:6006 

### restart training
trainer creates checkpoints by epoch in output/checkpoints/example
you can restart training from it like this
```
accelerate launch src/train.py --dataroot datasets/CelebA-HQ --load-size 256 --crop-size 256 --name example --from-checkpoint
```

## generate image
you need to put image and segmentation map like
https://github.com/y-kamiya/sean/blob/main/src/generate.py#L15-L17

and generate from it and trained model
```
python src/generate.py --load-size 256 --crop-size 256 --model-path output/models/example/netG_<epoch>.pth
```
