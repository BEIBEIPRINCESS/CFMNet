import importlib
import os
import torch
import random
import datetime
import argparse
import numpy as np
import torch.optim as optim

importlib.import_module('distutils.version')
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import RSDataSet, load_model, resolve_dataset
from ptflops import get_model_complexity_info
from utils import train_one_epoch, evaluate, create_lr_scheduler, read_train_data, read_val_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '--datasets', dest='dataset',
                        type=str, default='UCM-82',
                        help='RESISC45-82 | UCM-82 | AID-82')
    parser.add_argument('--data-path', type=str, default='',
                        help='dataset root containing train/ and val/; overrides the built-in dataset paths')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--img-size', '--img_size', dest='img_size',
                        type=int, default=224)
    # batchsize and lr
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--wd', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--pretrained', type=str, default='',
                        help='optional ImageNet-pretrained CFMNet checkpoint')
    parser.add_argument('--resume', type=str, default='',
                        help='optional training checkpoint to resume')
    # device
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')
    args = parser.parse_args()
    return args


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")
    # create dataset
    dataset, data_path, num_classes = resolve_dataset(args)
    # create model
    model, version = load_model(num_classes, device)

    img_size = args.img_size
    # flops and params
    flops, params = get_model_complexity_info(
        model, (3, img_size, img_size), as_strings=True,
        print_per_layer_stat=False)
    print("params: ", params)
    print("flops: ", flops)

    time = "results_{}.txt".format(datetime.datetime.now().strftime("%y%m%d-%H%M"))
    aaa = datetime.datetime.now().strftime("%y%m%d-%H%M")
    mode = 'pretrained' if args.pretrained else 'scratch'
    output = os.path.join('./outputs', mode, dataset, version, aaa)
    results_file = os.path.join(output, time)
    if os.path.exists(output) is False:
        os.makedirs(output)

    tensorboard = os.path.join(output, 'runs')
    tb_writer = SummaryWriter(tensorboard)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    train_data_path = data_path + "train/"
    val_data_path = data_path + "val/"
    train_images_path, train_images_label = read_train_data(train_data_path)
    val_images_path, val_images_label = read_val_data(val_data_path)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandAugment(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # train_dataset
    train_dataset = RSDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # val_dataset
    val_dataset = RSDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               pin_memory=True, num_workers=nw, collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                             pin_memory=True, num_workers=nw, collate_fn=val_dataset.collate_fn)

    with open(results_file, "a") as f:
        info = f"args: {args}\n"
        f.write(info + "\n")
    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(
                f'Pretrained checkpoint not found: {args.pretrained}')
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        state_dict = checkpoint.get(
            'model', checkpoint.get('state_dict', checkpoint))
        model_state = model.state_dict()
        state_dict = {
            key.removeprefix('module.'): value
            for key, value in state_dict.items()
            if key.removeprefix('module.') in model_state
            and value.shape == model_state[key.removeprefix('module.')].shape
        }
        incompatible = model.load_state_dict(state_dict, strict=False)
        print(f'Loaded pretrained checkpoint: {args.pretrained}')
        print(f'Missing keys: {incompatible.missing_keys}')
        print(f'Unexpected keys: {incompatible.unexpected_keys}')

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=args.wd)
    scheduler = create_lr_scheduler(optimizer, len(train_loader), args.epochs, warmup=True, warmup_epochs=2)

    best_acc = 0.0
    best_epoch = 0
    start_epoch = -1
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']

    with open(results_file, "a") as f:
        info = f"params: {params}\n" \
               f"flops: {flops}\n"
        f.write(info + "\n\n")

    # train
    start_time = datetime.datetime.now()
    print("start_time:", start_time)
    for epoch in range(start_epoch + 1, args.epochs):

        model.train()
        train_loss, train_acc = train_one_epoch(model=model, optimizer=optimizer, scheduler=scheduler,
                                                data_loader=train_loader, device=device, epoch=epoch)

        # validate
        model.eval()
        val_loss, val_acc = evaluate(model=model, data_loader=val_loader,
                                     device=device, epoch=epoch)

        # save checkpoint
        save_path = os.path.join(output, "weights")
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        if epoch >= 0:
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            torch.save(checkpoint, os.path.join(save_path, 'last-val_acc.pth'))

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            checkpoint = {
                "model": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch}
            torch.save(checkpoint, os.path.join(save_path, 'best-val_acc.pth'))

        with open(results_file, "a") as f:
            info = f"[epoch: {epoch}]  "\
                   f"train_acc: {train_acc:.4f}  " \
                   f"train_loss: {train_loss:.4f}  " \
                   f"val_acc: {val_acc:.4f}  "\
                   f"val_loss: {val_loss:.4f}  "
            f.write(info + "\n")

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "best_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], best_acc, epoch)
        tb_writer.add_scalar(tags[5], optimizer.param_groups[0]["lr"], epoch)

    # train time
    end_time = datetime.datetime.now()
    print("start_time:", start_time)
    print("end_time:", end_time)
    all_time = end_time - start_time
    print("all_time:", all_time)

    with open(results_file, "a") as f:
        info = f"best_epoch: {best_epoch}\n" \
               f"best_acc: {best_acc:.4f}\n" \
               f"start_time: {start_time}\n" \
               f"end_time: {end_time}\n" \
               f"train_and_val all_time: {all_time}\n" \
               f"params: {params}\n" \
               f"flops: {flops}"
        f.write(info + "\n\n")
    tb_writer.close()


if __name__ == '__main__':
    args = parse_args()

    main(args)
