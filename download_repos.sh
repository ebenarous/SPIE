# Clone the DreamSim repository
git clone https://github.com/ssundaram21/dreamsim.git

# Clone the Depth-Anything repository and download the model weights
git clone https://github.com/DepthAnything/Depth-Anything-V2.git
cd Depth-Anything-V2
mkdir checkpoints
cd Depth-Anything-V2/checkpoints
wget https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true
cd ../
mv Depth-Anything-V2 Depth_Anything_V2 # rename for imports

# Clone the SAM 2 repository
git clone https://github.com/facebookresearch/sam2.git

# For evaluation, clone HPSv2 and ImageReward repositories
pip install hpsv2x==1.2.0 # install HPSv2 through pip (see https://github.com/tgxs002/HPSv2/pull/38)
git clone https://github.com/THUDM/ImageReward.git

# add directories to Python's module search path to resolve imports
export PYTHONPATH="$PYTHONPATH:$(realpath ./dreamsim)"
export PYTHONPATH="$PYTHONPATH:$(realpath ./sam2)"
export PYTHONPATH="$PYTHONPATH:$(realpath ./ImageReward)" 
