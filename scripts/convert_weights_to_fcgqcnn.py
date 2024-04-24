"""
Takes in gqcnn checkpoint. Converts and saves an FCGQCNN checkpoint
"""
from dexnet.grasp_model import DexNet3 as model
from dexnet.grasp_model import BaseFCGQCNN as FCGQCNN
from pathlib import Path
import torch

def convert_model(path: Path):
    model_w = torch.load(path)
    gqcnn = model().load_state_dict(model_w)
    fcgqcnn = FCGQCNN()

    model_w['conv5.weight'] = model_w['fc.weight'].reshape(1024, 64, 16, 16)
    model_w['conv5.bias'] = model_w['fc.bias']

    model_w['conv6.weight'] = model_w['fc2.weight'].reshape(1024, 1024, 1, 1)
    model_w['conv6.bias'] = model_w['fc2.bias']

    model_w['conv7.weight'] = model_w['fc3.weight'].reshape(1, 1024, 1, 1)
    model_w['conv7.bias'] = model_w['fc3.bias']

    del model_w['fc.weight'], model_w['fc.bias']
    del model_w['fc2.weight'], model_w['fc2.bias']
    del model_w['fc3.weight'], model_w['fc3.bias']

    fcgqcnn.load_state_dict(model_w)
    torch.save(fcgqcnn.state_dict(), "model_zoo/fcgqcnn_conversion.pt")
    # torch.save(fcgqcnn.state_dict(), f"model_zoo/{Path(path).fname}_conversion.pt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process configuration file.")
    parser.add_argument(
        "--model_path",
        dest="model_path",
        metavar="MODEL_FILE_PATH",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    args = parser.parse_args()
    convert_model(Path(args.model_path))
