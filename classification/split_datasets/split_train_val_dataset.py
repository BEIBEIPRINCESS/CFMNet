import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.tif', '.tiff'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a reproducible train/validation image split.')
    parser.add_argument('source', type=Path,
                        help='Dataset root containing one directory per class.')
    parser.add_argument('output', type=Path,
                        help='Output root for train/ and val/ directories.')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                        help='Fraction of each class assigned to validation.')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


def split_dataset(source, output, val_ratio, seed):
    if not 0.0 < val_ratio < 1.0:
        raise ValueError('--val-ratio must be between 0 and 1.')
    if not source.is_dir():
        raise FileNotFoundError(f'Dataset directory not found: {source}')

    rng = random.Random(seed)
    class_dirs = sorted(path for path in source.iterdir() if path.is_dir())
    if not class_dirs:
        raise ValueError(f'No class directories found in {source}')

    for class_dir in class_dirs:
        images = sorted(
            path for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        rng.shuffle(images)
        val_count = int(len(images) * val_ratio)
        split_index = len(images) - val_count

        for split, split_images in (
                ('train', images[:split_index]),
                ('val', images[split_index:])):
            target_dir = output / split / class_dir.name
            target_dir.mkdir(parents=True, exist_ok=True)
            for image_path in split_images:
                shutil.copy2(image_path, target_dir / image_path.name)

        print(f'{class_dir.name}: {split_index} train, {val_count} val')


if __name__ == '__main__':
    args = parse_args()
    split_dataset(args.source, args.output, args.val_ratio, args.seed)
