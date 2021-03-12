import math
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

rgb_map = [
    [255, 0, 0],  # cap
    [181,70,174],  # cg
    [61,184,102],  # pz
    [0,0,0]  # background
]

import matplotlib.pyplot as plt

def vizualize_labels(true,pred):
    maxes = torch.argmax(true, dim=0)
    rgb_values = [rgb_map[p] for p in maxes.numpy().flatten()]
    matlib_true = np.array(rgb_values).reshape(true.shape[1], true.shape[2], 3)

    maxes = torch.argmax(pred, dim=0)
    rgb_values = [rgb_map[p] for p in maxes.numpy().flatten()]
    matlib_pred = np.array(rgb_values).reshape(true.shape[1], true.shape[2], 3)

    f, axarr = plt.subplots(1, 2)
    axarr[0].set_title('True')
    axarr[0].imshow(matlib_true)
    axarr[1].set_title('Pred')
    axarr[1].imshow(matlib_pred)
    plt.show()


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          epochs=20,
          device="cpu",
          save_best_model=True,
          best_model_name='best-model-parameters',
          early_stopping=True,
          early_stopping_patience=5):
    best_model_loss = math.inf
    epochs_since_last_improvement = 0
    for epoch in range(1, epochs + 1):
        start = time.time()
        print('-------Epoch %d-------' % epoch)
        epochs_since_last_improvement += 1
        valid_loss = 0.0
        model.train()
        for i, batch in enumerate(train_loader, 0):
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            print(f'Training Loss: {loss.data.item():.3f} (Progress: {((i*inputs.shape[0]) / len(train_loader.dataset)) * 100:.3f})', end='\r')
            combined_classes_last = targets[0]
            pred_last = output[0]

        model.eval()
        with torch.no_grad():
            true_pos = []
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                output = model(inputs)
                loss = loss_fn(output, targets)
                valid_loss += loss.data.item() * inputs.size(0)
                combined_classes_last = targets[0]
                pred_last = output[0]
                tt = []
                one_hot_pred = torch.nn.functional.one_hot(torch.argmax(pred_last, dim=0)).permute(2, 0, 1)
                for i in range(len(one_hot_pred)):
                    tt.append((torch.sum(one_hot_pred[i] * combined_classes_last[i]) / torch.sum(combined_classes_last[i])).cpu().numpy())
                true_pos.append(tt)
            # calc acc.
            true_pos = np.where(np.isnan(true_pos), np.ma.array(true_pos, mask=np.isnan(true_pos)).mean(axis=0), true_pos)
            true_pos = np.mean(true_pos,axis=0)
            #end cals acc.
            vizualize_labels(combined_classes_last.cpu(), pred_last.cpu())
            valid_loss /= len(val_loader.dataset)
        print(f' CaP: {true_pos[0]} | CG {true_pos[1]} | PZ {true_pos[2]} | BG {true_pos[3]}')
        print(f'Validation Loss: {valid_loss:.2f} (Took {time.time() - start} seconds)')
        if valid_loss < best_model_loss:
            epochs_since_last_improvement = 0
            if save_best_model:
                print('Saving best validation loss %.2f' % valid_loss)
                best_model_loss = valid_loss
                torch.save(model.state_dict(), f'{best_model_name}.pt')
        if early_stopping and epochs_since_last_improvement >= early_stopping_patience:
            print('Thats enough, early stopping after %d epochs.' % early_stopping_patience)
            break
