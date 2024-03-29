import math
import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from pathlib import Path

rgb_map = [
    [255, 0, 0],  # cap
    [181,70,174],  # cg
    [61,184,102],  # pz
    [0,0,0]  # background
]

def vizualize_labels(true,pred,save_dir):
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
    plt.savefig(save_dir)
    plt.show()


def train(model,
          optimizer,
          loss_fn,
          train_loader,
          val_loader,
          epochs=80,
          device="cpu",
          save_best_model=True,
          best_model_dir='.',
          early_stopping=True,
          early_stopping_patience=20,
          step_size_decay=None):
    best_model_loss = math.inf
    epochs_since_last_improvement = 0
    train_loss = []
    val_loss = []
    for epoch in range(1, epochs + 1):
        Path(f'{best_model_dir}/imgs').mkdir(parents=True, exist_ok=True)
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
            loss = loss_fn(targets,output)
            train_loss.append(loss.cpu().detach().numpy())
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
                val_loss.append(loss.cpu().detach().numpy())
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
            save_name = f'{best_model_dir}/imgs/{epoch}.png'
            vizualize_labels(combined_classes_last.cpu(), pred_last.cpu(),save_name)
            valid_loss /= len(val_loader.dataset)
        if len(true_pos) == 4:
            print(f' CaP: {true_pos[0]} | CG {true_pos[1]} | PZ {true_pos[2]} | BG {true_pos[3]}')
        if len(true_pos) == 3:
            print(f' CG {true_pos[0]} | PZ {true_pos[1]} | BG {true_pos[2]}')
        print(f'Validation Loss: {valid_loss:.2f} (Took {time.time() - start} seconds)')
        if valid_loss < best_model_loss:
            epochs_since_last_improvement = 0
            if save_best_model:
                print('Saving best validation loss %.2f' % valid_loss)
                best_model_loss = valid_loss
                save_name = f'{best_model_dir}/{best_model_loss}_{epoch}.pt'
                torch.save(model.state_dict(), save_name)
        if early_stopping and epochs_since_last_improvement >= early_stopping_patience:
            print('Thats enough, early stopping after %d epochs.' % early_stopping_patience)
            break
        if step_size_decay != None:
            step_size_decay.step()
    val_loss_name = f'{best_model_dir}/val_loss.pickle'
    train_loss_name = f'{best_model_dir}/train_loss.pickle'
    with open(val_loss_name, 'wb') as f:
        pickle.dump(val_loss, f)
    with open(train_loss_name, 'wb') as f:
        pickle.dump(train_loss, f)