import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
import numpy as np
import logging
import heapq

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Running:
    def __init__(self, model, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.result = None
        self.best_acc = 0.0
        self.best_coef = []
        self.model = model
        # Data parallel
        self.gpu_num = torch.cuda.device_count()
        # if self.gpu_num > 1:
        #     # logger.info("Model has been loaded, using {} GPUs!".format(torch.cuda.device_count()))
        #     self.model = nn.DataParallel(self.model)
        # else:
        #     logger.info('Model has been loaded, using {}!'.format(self.device))
        self.model.to(self.device)
    def train(self, train_loader, eval_loader, epoch):
        model_save_path = self.args.model_path
        learning_rate = self.args.learning_rate
        # Define optimizer and loss function
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = F.cross_entropy
        # Set learning rate reduction strategy
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=0.5)

        best_valid_acc = float('-inf')
        for epoch in range(self.args.epochs):
            loss_one_epoch = 0.0
            correct_num = 0.0
            total_num = 0.0
            scheduler.step()

            for i, batch in enumerate(train_loader):
                self.model.train()
                label, content = batch.label2, batch.text
                label = label
                # backward(), Update weight
                optimizer.zero_grad()
                pred = self.model(content)
                loss = criterion(pred, label)
                loss.backward()
                optimizer.step()

                # Statistical forecast information
                total_num += label.size(0)
                correct_num += (torch.argmax(pred, dim=1) == label).sum().float().item()
                loss_one_epoch += loss.item()

                if i % 10 == 9:
                    loss_avg = loss_one_epoch / 10
                    loss_one_epoch = 0.0
                    logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%}".format(epoch + 1, self.args.epochs, i + 1, len(train_loader), loss_avg, correct_num / total_num))

            if epoch % 2 == 0:
                loss_one_epoch = 0.0
                classes_num = self.args.num_classes
                conf_mat = np.zeros([classes_num, classes_num])
                self.model.eval()
                for i, batch in enumerate(eval_loader):
                    label, content = batch.label2, batch.text
                    label = label
                    pred = self.model(content)
                    pred.detach()
                    # Calculate loss
                    loss = criterion(pred, label)
                    loss_one_epoch += loss.item()

                    # Statistical forecast information
                    total_num += label.size(0)
                    correct_num += (torch.argmax(pred, dim=1) == label).sum().float().item()
                    loss_one_epoch += loss.item()

                    # Confusion matrix
                    for j in range(len(label)):
                        cate_i = label[j].item()
                    pre_i = torch.argmax(pred, dim=1)[j].item()
                    conf_mat[cate_i, pre_i] += 1.0

                # Print the accuracy of the validation set
                logger.info('{} set Accuracy:{:.2%}'.format('Valid', conf_mat.trace() / conf_mat.sum()))

            #  Save model
            if (conf_mat.trace() / conf_mat.sum()) > best_valid_acc:
                logger.info("***Save model***")
                best_valid_acc = (conf_mat.trace() / conf_mat.sum())
                torch.save(self.model.state_dict(), model_save_path)
            else:
                logger.info('The model do not get better')

    def predict(self,test_loader):
        loss_one_epoch = 0.0
        correct_num = 0.0
        total_num = 0.0
        classes_num = self.args.num_classes
        conf_mat = np.zeros([classes_num, classes_num])
        self.model.eval()
        criterion = F.cross_entropy
        for i, batch in enumerate(test_loader):
            label, content = batch.label2, batch.text
            label = label
            pred = self.model(content)
            pred.detach()
            # Calculate loss
            loss = criterion(pred, label)
            loss_one_epoch += loss.item()
            total_num += label.size(0)
            correct_num += (torch.argmax(pred, dim=1) == label).sum().float().item()
            loss_one_epoch += loss.item()

            # Confusion matrix
            for j in range(len(label)):
                cate_i = label[j].item()
            pre_i = torch.argmax(pred, dim=1)[j].item()
            conf_mat[cate_i, pre_i] += 1.0

        # Print the accuracy of the test set
        logger.info('{} set Accuracy:{:.2%}'.format('test', conf_mat.trace() / conf_mat.sum()))

    def softmax(self, x):
        exp_x = np.exp(x)
        softmax_x = exp_x / np.sum(exp_x)
        return softmax_x

    def predict_single(self,tensor):
        # Load language name
        language = np.load(self.args.data_path + 'language.npy')
        prediction = self.model(tensor)
        prediction = self.softmax(prediction.cpu().detach().numpy()[0])
        # label = np.argmax(prediction)
        p = heapq.nlargest(self.args.topk, prediction)
        # Probability
        label = list(map(list(prediction).index, heapq.nlargest(self.args.topk, list(prediction))))
        lan = language[label]
        # Probability
        return lan, p
