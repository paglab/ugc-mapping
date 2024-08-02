# ground-cover-mapping
This branch trains the semantic segmentation model based on Semantic Segmentation on PyTorch（https://github.com/Tramac/awesome-semantic-segmentation-pytorch/tree/master） and images downloaded from the Internet.
## First Step: Installation
    # semantic-segmentation-pytorch dependencies
    pip install ninja tqdm

    # follow PyTorch installation in https://pytorch.org/get-started/locally/

    conda install pytorch torchvision -c pytorch

    # install PyTorch Segmentation
    git clone https://github.com/paglab/ground-cover-mapping.git

## Train
    # for example, train fcn32_vgg16_pascal_voc:
    python train.py --model fcn32s --backbone vgg16 --dataset pascal_voc --lr 0.0001 --epochs 50
