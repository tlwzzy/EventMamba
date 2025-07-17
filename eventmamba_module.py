import torch
import pytorch_lightning as pl
from torchmetrics.classification import MulticlassAccuracy
from collections import Counter

class EventMambaLitModule(pl.LightningModule):
    def __init__(self, net, num_classes, learning_rate=0.001, optimizer_name="AdamW", weight_decay=1e-4, smoothing=True):
        super().__init__()
        self.save_hyperparameters(ignore=['net'])
        self.net = net
        self.criterion = self.cal_loss
        self.smoothing = smoothing

        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

        # 用于acc_seq缓存
        self.val_true = []
        self.val_pred = []
        self.test_true = []
        self.test_pred = []

    def cal_loss(self, pred, gold, smoothing=True):
        gold = gold.contiguous().view(-1)
        if smoothing:
            eps = 0.2
            n_class = pred.size(1)
            one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = torch.nn.functional.log_softmax(pred, dim=1)
            loss = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss = torch.nn.functional.cross_entropy(pred, gold, reduction='mean')
        return loss

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        data, label = batch
        data = data.permute(0, 2, 1)
        logits = self(data)
        loss = self.criterion(logits, label, self.smoothing)
        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, label)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data = data.permute(0, 2, 1)
        logits = self(data)
        loss = self.criterion(logits, label, self.smoothing)
        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, label)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.val_true.append(label.cpu())
        self.val_pred.append(preds.cpu())

    def on_validation_epoch_end(self):
        if self.val_true and self.val_pred:
            val_true = torch.cat(self.val_true).numpy()
            val_pred = torch.cat(self.val_pred).numpy()
            acc_seq = self.calc_acc_seq(val_pred, val_true)
            self.log("val/acc_seq", acc_seq, prog_bar=True)
            self.val_true.clear()
            self.val_pred.clear()

    def test_step(self, batch, batch_idx):
        data, label = batch
        data = data.permute(0, 2, 1)
        logits = self(data)
        loss = self.criterion(logits, label, self.smoothing)
        preds = torch.argmax(logits, dim=1)
        acc = self.test_acc(preds, label)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.test_true.append(label.cpu())
        self.test_pred.append(preds.cpu())

    def on_test_epoch_end(self):
        if self.test_true and self.test_pred:
            test_true = torch.cat(self.test_true).numpy()
            test_pred = torch.cat(self.test_pred).numpy()
            acc_seq = self.calc_acc_seq(test_pred, test_true)
            self.log("test/acc_seq", acc_seq, prog_bar=True)
            self.test_true.clear()
            self.test_pred.clear()

    @staticmethod
    def calc_acc_seq(preds, labels):
        count = 0
        correct_seq = 0
        index = 0
        for i in range(len(labels)-2):
            if (labels[i] != labels[i+1]) or (i == len(labels)-2):
                tar = Counter(preds[index:i+1])
                tar = tar.most_common(1)[0][0]
                if tar == labels[i]:
                    correct_seq += 1
                index = i+1
                count += 1
        if count == 0:
            return 0.0
        return correct_seq / count

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler} 