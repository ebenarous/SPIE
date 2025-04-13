from setuptools import setup, find_packages

SAM2_reqs = [
        "hydra-core>=1.3.2",
        "iopath>=0.1.10",
]

dreamsim_reqs = [
    "open-clip-torch"
]

imagereward_reqs = [
    "clip@https://github.com/openai/CLIP/archive/refs/heads/main.zip#egg=clip-1.0",
    "fairscale>=0.4.13",
    "datasets>=2.11.0"
]

general_reqs = [
        "absl-py",
        "accelerate==0.33.0",
        "diffusers==0.29.2",
        "inflect==7.3.1",
        "ml-collections==0.1.1",
        "opencv-python==4.10.0.84",
        "peft==0.12.0",
        "pydantic==2.9.1",
        "torch==2.4.0",
        "torchvision==0.19.0",
        "transformers==4.45.2",
        "wandb",
]

setup(
    name="EditSpecialists",
    version="0.0.1",
    packages=["EditSpecialists"],
    python_requires=">=3.10",
    install_requires=general_reqs+dreamsim_reqs+SAM2_reqs+imagereward_reqs,
)