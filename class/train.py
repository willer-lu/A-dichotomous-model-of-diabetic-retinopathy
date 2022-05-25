import os
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from my_dataset import MyDataSet
from model import swin_base_patch4_window12_384_in22k as create_model
from utils import read_split_data, train_one_epoch, evaluate
from lr_scheduler import build_scheduler
import transforms as T

if __name__ == "__main__":

    num_classes = 2
    epochs = 50
    batch_size = 8
    learning_rate = 1e-5

    model_path = 'models/swin_base_patch4_window12_384_22k.pth'

    freeze_layers = False

    device = 'cuda:0'


    if os.path.exists("./models") is False:
        os.makedirs("./models")

    tb_writer = SummaryWriter()

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_data('train')


    # 实例化训练数据集
    train_dataset = MyDataSet(images_path=train_images_path,
                              images_class=train_images_label,
                              transform=T.get_transform(train=True))

    # 实例化验证数据集
    val_dataset = MyDataSet(images_path=val_images_path,
                            images_class=val_images_label,
                            transform=T.get_transform(train=False))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=0,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,
                                             collate_fn=val_dataset.collate_fn)

    model = create_model(num_classes=num_classes).to(device)


    weights_dict = torch.load(model_path, map_location=device)["model"]
        # 删除有关分类类别的权重
    for k in list(weights_dict.keys()):
        if "head" in k:
            del weights_dict[k]
    model.load_state_dict(weights_dict, strict=False)

    if freeze_layers:
        for name, para in model.named_parameters():
            # 除head外，其他权重全部冻结
            if "head" not in name:
                para.requires_grad_(False)
            else:
                print("training {}".format(name))

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(pg, lr = learning_rate,eps=1e-8,betas=(0.9,0.999),weight_decay=0.05)
    scheduler = build_scheduler(epochs,WARMUP_EPOCHS=5, optimizer=optimizer, n_iter_per_epoch=len(train_loader))

    max_acc = 0
    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                scheduler=scheduler,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)
        print('epoch:',epoch)
        tags = ["train_loss:", "train_acc:", "val_loss:", "val_acc:", "learning_rate:"]
        print(tags[0], train_loss)
        print(tags[1], train_acc)
        print(tags[2], val_loss)
        print(tags[3], val_acc)
        print(tags[4], optimizer.param_groups[0]["lr"])
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)
        if val_acc > max_acc:
            max_acc = val_acc
            torch.save(model.state_dict(), "./models/model.pth")
            print("save model at epoch{}".format(epoch))



