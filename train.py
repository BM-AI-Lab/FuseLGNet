import os
import argparse
import csv

import pandas as pd
from pandas.core.frame import DataFrame
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from my_dataset import MyDataSet
from model import swin_tiny_patch4_window7_224 as create_model
from utils import read_split_data, train_one_epoch, evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from scipy.special import softmax

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data(args.data_path)

    img_size = 224
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(img_size),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(int(img_size * 1.143)),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=data_transform["train"])

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=data_transform["val"])

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 4])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=args.num_classes).to(device)

    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)["model"]
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if "head" in k:
                del weights_dict[k]
        # print(model.load_state_dict(weights_dict, strict=False))

    if args.freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr=args.lr, weight_decay=5E-2)

    for epoch in range(args.epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)
        print('epoch{}：train_loss{}，train_acc{}'.format(epoch, train_loss, train_acc))



        # tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        # tb_writer.add_scalar(tags[0], train_loss, epoch)
        # tb_writer.add_scalar(tags[1], train_acc, epoch)
        # tb_writer.add_scalar(tags[2], val_loss, epoch)
        # tb_writer.add_scalar(tags[3], val_acc, epoch)
        # tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

       # torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))
    model.eval()
    y_true, y_pred = [], []
    p0, p1= [], []
    for images, labels in val_dataset:
        images = torch.unsqueeze(images, dim=0)
        labels = torch.tensor(labels)
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1].detach().cpu().numpy()
        y_true.append(labels)
        y_pred.append(pred_classes)

        pred = softmax(pred.detach().cpu().numpy())
        p0.append(pred[0][0].item())
        p1.append(pred[0][1].item())


    acc = accuracy_score(y_true, y_pred,)
    recall = recall_score(y_true, y_pred,)
    f1 = f1_score(y_true, y_pred,)
    precision = precision_score(y_true, y_pred,)
    zidian = {'准确度': acc, '召回率': recall, 'F1分数': f1, '精确率': precision}
    print('准确度{}，召回率{}，F1分数{}，精确率{}'.format(acc, recall, f1, precision))
    y_true = [y.item() for y in y_true]
    y_pred = [y.item() for y in y_pred]
    name = [value.split('\\')[-1] for value in val_images_path]
    data = {"name": name, "parkinson": p0, "nonparkinson": p1, 'Actual label': y_true, 'Predict labels': y_pred}
    data = DataFrame(data)
    if os.path.exists('./swin_transformer') is False:
        os.mkdir('./swin_transformer')
    data.to_csv('./swin_transformer/IT-sil-front.csv')

    f = open('./swin_transformer/IT-sil-frontzhibiao.csv', 'w', encoding='utf_8_sig', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["准确度", "召回率", "F1分数", "精确率"])
    csv_writer.writerow([zidian['准确度'], zidian['召回率'], zidian['F1分数'], zidian['精确率']])
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.0001)

    # 数据集所在根目录
    # https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz
    parser.add_argument('--data-path', type=str,
                        default="./GAIT-IT/silhouettes/parkinson/front")

    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='./data/swin_tiny_patch4_window7_224.pth',
                        help='initial weights path')
    # 是否冻结权重
    parser.add_argument('--freeze-layers', type=bool, default=False)
    parser.add_argument('--device', default=0, help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()
    main(opt)
