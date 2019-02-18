##########################################################
# pytorch-qnn v1.0                                     
# Titouan Parcollet
# LIA, Université d'Avignon et des Pays du Vaucluse
# ORKIS, Aix-en-provence
# October 2018
##########################################################

import torch
import torch.nn as nn
import torch.optim

from recurrent_models import LSTM, QLSTM

import numpy as np

import sys
import os
import shutil

# PARAMETERS #
CUDA = False
N_BATCH_TRAIN = 10
SEQ_LENGTH = 10
FEAT_SIZE = 8
EPOCHS = 100
RNN_HIDDEN_SIZE = 40
QRNN_HIDDEN_SIZE = 80


def get_task(n_batch, seq_length, feat_size, blank_size, embedding):
    data = []
    lab = []
    seq = []
    target = []

    for i in range(n_batch):

        # Target values of blank and delim
        blank = feat_size
        delim = feat_size + 1

        # Embedding
        blank_emb = feat_size
        blank_emb = torch.tensor(blank_emb, dtype=torch.long)
        blank_emb = embedding(blank_emb).data.numpy()
        delim_emb = feat_size + 1
        delim_emb = torch.tensor(delim_emb, dtype=torch.long)
        delim_emb = embedding(delim_emb).data.numpy()

        random_index_list = []

        for j in range(seq_length):
            random = np.random.randint(feat_size, size=1)
            feat = torch.tensor(random, dtype=torch.long)
            feat = embedding(feat).data.numpy()[0]
            random_index_list.append(random)

            seq.append(feat)
            target.append(blank)

            # BLANK
        for j in range(blank_size - 1):
            seq.append(blank_emb)
            target.append(blank)

        # Append a last blank to target for delimiter in input
        target.append(blank)

        # DELIMITER
        seq.append(delim_emb)

        # Append input to target
        for j in random_index_list:
            target.append(j)
            seq.append(delim_emb)

        data.append(seq)
        lab.append(target)

        seq = []
        target = []

    return np.array(data), np.array(lab)


#
# DEFINING THE TASK
#

def main(argv):

    if len(argv) > 1:
        BLANK_SIZE = int(argv[1])
    else:
        BLANK_SIZE = 25

    losses_r = []
    losses_q = []
    accs_r = []
    accs_q = []
    accs_test = []

    net_r = LSTM(FEAT_SIZE, RNN_HIDDEN_SIZE, CUDA).cpu()
    net_q = QLSTM(FEAT_SIZE, QRNN_HIDDEN_SIZE, CUDA).cpu()

    emb = nn.Embedding(FEAT_SIZE + 2, FEAT_SIZE, max_norm=1.0)

    nb_param_r = sum(p.numel() for p in net_r.parameters() if p.requires_grad)
    nb_param_q = sum(p.numel() for p in net_q.parameters() if p.requires_grad)

    print("QRNN & RNN Copy Task - Titouan Parcollet - LIA, ORKIS")
    print("Models Infos --------------------")
    print("(RNN)  Number of trainable parameters : " + str(nb_param_r))
    print("(QRNN) Number of trainable parameters : " + str(nb_param_q))

    # TRAINING LOOP #

    for epoch in range(EPOCHS):

        #
        # The input sequence size is 2 times the sequence length + number_of_blank - 1 + 1
        # (+ 1 for the delimiter). We generate N_BATCH_TRAIN new sequences each epoch
        #
        train, train_target = get_task(N_BATCH_TRAIN, SEQ_LENGTH, FEAT_SIZE, BLANK_SIZE, emb)

        # Train shape must be (SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE) for QLSTM and LSTM
        train = train.reshape((BLANK_SIZE + (2 * SEQ_LENGTH), N_BATCH_TRAIN, FEAT_SIZE))

        train_var = torch.FloatTensor(train).cpu()
        train_target_var = torch.FloatTensor(train_target.astype(np.float32))

        # NN Training
        net_r.zero_grad()
        p = net_r.forward(train_var)

        # Pred. shape : (SEQ_LENGTH, BATCH_SIZE, FEATURE_SIZE) to (SEQ_LENGTH * BATCH_SIZE, FEATURE_SIZE)
        predictions = p.view(-1, FEAT_SIZE + 1)

        # Target shape to (BATCH_SIZE)
        targets = train_target_var.view(-1)
        loss = nn.CrossEntropyLoss()
        val_loss = loss(predictions, targets.long())

        val_loss.backward()
        net_r.adam.step()

        # Train ACC and LOSS
        p = p.cpu().data.numpy()
        shape = np.argmax(p, axis=2).shape
        p = np.reshape(np.argmax(p, axis=2), shape[0] * shape[1])
        targets = targets.cpu().data.numpy()
        acc = np.sum(p == targets) / train_target.size

        if (epoch % 5) == 0:
            accs_r.append(acc)
            losses_r.append(float(val_loss.data))
        if (epoch % 10) == 0:
            string = " (NN) It : " + str(epoch) + " | Train Loss = " + str(
                float(val_loss.data)) + " | Train Acc = " + str(
                acc)
            print(string)

        # QNN Training
        net_q.zero_grad()
        p = net_q.forward(train_var)
        predictions = p.view(-1, FEAT_SIZE + 1)
        targets = train_target_var.view(-1)
        loss = nn.CrossEntropyLoss()
        val_loss = loss(predictions, targets.long())

        val_loss.backward()
        net_q.adam.step()

        p = p.cpu().data.numpy()
        shape = np.argmax(p, axis=2).shape
        p = np.reshape(np.argmax(p, axis=2), shape[0] * shape[1])
        targets = targets.cpu().data.numpy()
        acc = np.sum(p == targets) / train_target.size

        if (epoch % 5) == 0:
            losses_q.append(float(val_loss.data))
            accs_q.append(acc)
        if (epoch % 10) == 0:
            string = "(QNN) It : " + str(epoch) + " | Train Loss = " + str(
                float(val_loss.data)) + " | Train Acc = " + str(
                acc)
            print(string)

    if os.path.exists('RES'):
        shutil.rmtree('RES')
    os.mkdir('RES')

    print("Training Ended - Saving Acc and losses in RES")

    np.savetxt("RES/memory_task_acc_r_" + str(BLANK_SIZE) + ".txt", accs_r)
    np.savetxt("RES/memory_task_acc_q_" + str(BLANK_SIZE) + ".txt", accs_q)
    np.savetxt("RES/memory_task_loss_r_" + str(BLANK_SIZE) + ".txt", losses_r)
    np.savetxt("RES/memory_task_loss_q_" + str(BLANK_SIZE) + ".txt", losses_q)

    print("Done ! That's All Folks ;) !")


if __name__ == "__main__":
    main(sys.argv)
