import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from ASTGCN_r import make_model
import util
import random
import numpy as np
from baseline_methods import test_error, StandardScaler
from tqdm import trange
import json
import argparse


def store_result(yhat, label, i):
    metrics = test_error(yhat[:, :, i], label[:, :, i])
    return metrics[0], metrics[2], metrics[1]


def main(delay_index=1):
    # Read the configuration file
    with open('configs_u.json', 'r') as f:
        config = json.load(f)

    # Create an empty argparse Namespace object to store the configuration settings
    args = argparse.Namespace()
    for key, value in config.items():
        setattr(args, key, value)
        device = torch.device(args.device)

    device = torch.device(args.device)

    # Load your dataset
    adj, training_data, val_data, training_w, val_w = util.load_data(args.data)
    adj = adj[0]

    model = make_model('cuda:0', nb_block=2, in_channels=2, K=3, nb_chev_filter=3, nb_time_filter=3,
                       time_strides=3, adj_mx=adj, num_for_predict=args.out_len, len_input=args.in_len, num_of_vertices=70)
    optimizer, scheduler, scaler, training_data, batch_index, val_index = util.model_preprocess(
        model, args.lr, args.gamma, args.step_size, training_data, val_data, args.in_len, args.out_len)
    label = util.label_loader(val_index, args.in_len,
                              args.out_len, delay_index, val_data)
    n_epochs = args.episode

    amae3, amape3, armse3, amae6, amape6, armse6, amae12, amape12, armse12 = [
    ], [], [], [], [], [], [], [], []

    # Train the model
    print("start training...", flush=True)
    best_ep, best_loss = -1, np.inf
    for ep in trange(n_epochs):
        model.train()
        random.shuffle(batch_index)
        for j in range(len(batch_index) // args.batch - 1):
            trainx, trainy, trainw = util.train_dataloader(
                batch_index, args.batch, training_data, training_w, j, args.in_len, args.out_len)
            # print(trainx.shape) [32, 50, 36, 2]
            trainw = torch.LongTensor(trainw).to(device)
            trainx = torch.index_select(torch.Tensor(
                trainx), -1, torch.tensor([delay_index]))  # Select which feature to be used
            # print(trainx.shape) [32, 50, 36, 1]
            trainx = trainx.to(device)
            trainw = trainw.unsqueeze(-1)
            train = torch.cat((trainw, trainx), dim=-1).permute(0, 1, 3, 2)
            # print(train.shape) [32, 50, 36, 2]
            trainy = torch.index_select(torch.Tensor(
                trainy), -1, torch.tensor([delay_index]))
            trainy = trainy.to(device)
            trainy = trainy.permute(0, 3, 1, 2)[:, 0, :, :]
            # print(trainy.shape) [32, 50, 12]
            optimizer.zero_grad()
            output = model(train)
            loss = util.masked_mae(output, trainy, 0.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()

        scheduler.step()
        outputs = []

        # Evaluate the model
        model.eval()
        for i in range(len(val_index)):
            testx, testw = util.test_dataloader(
                val_index, val_data, val_w, i, args.in_len, args.out_len)
            testx = scaler.transform(testx)
            testw = torch.LongTensor(testw).to(device)
            testx[np.isnan(testx)] = 0
            testx = torch.index_select(torch.Tensor(
                testx), -1, torch.tensor([delay_index]))
            testx = testx.to(device)
            testw = testw.unsqueeze(-1)
            test = torch.cat((testw, testx), dim=-1).permute(0, 1, 3, 2)

            output = model(test)
            output = output.detach().cpu().numpy()
            output = scaler.inverse_transform(output)
            outputs.append(output)
        yhat = np.concatenate(outputs)
        loss = test_error(yhat, label)[0]  # Cal based on MAE
        if (loss < best_loss):
            best_loss, best_ep = loss, ep

        prediction2 = store_result(yhat, label, 2)
        amae3.append(prediction2[0])
        amape3.append(prediction2[1])
        armse3.append(prediction2[2])
        prediction5 = store_result(yhat, label, 5)
        amae6.append(prediction5[0])
        amape6.append(prediction5[1])
        armse6.append(prediction5[2])
        prediction11 = store_result(yhat, label, 11)
        amae12.append(prediction11[0])
        amape12.append(prediction11[1])
        armse12.append(prediction11[2])

    print("Best epoch:", best_ep)
    print(
        f"Error of ASTGCN in 3-step: ({round(amae3[best_ep],3)}, {round(armse3[best_ep],3)}, {round(amape3[best_ep],3)})")
    print(
        f"Error of ASTGCN in 6-step: ({round(amae6[best_ep],3)}, {round(armse6[best_ep],3)}, {round(amape6[best_ep],3)})")
    print(
        f"Error of ASTGCN in 12-step: ({round(amae12[best_ep],3)}, {round(armse12[best_ep],3)}, {round(amape12[best_ep],3)})")


if __name__ == "__main__":
    print("ASTGCN:")
    main()
