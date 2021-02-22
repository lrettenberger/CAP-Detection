import math
import torch
import time


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
        training_loss = 0.0
        valid_loss = 0.0
        model.train()
        progress = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            loss.backward()
            optimizer.step()
            training_loss += loss.data.item() * inputs.size(0)
            print(f'Training Loss: {loss.data.item():.3f} (Progress: {(progress/len(train_loader.dataset))*100:.3f})', end='\r')
            progress += len(inputs)
        training_loss /= len(train_loader.dataset)

        model.eval()
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            output = model(inputs)
            loss = loss_fn(output, targets)
            valid_loss += loss.data.item() * inputs.size(0)
        valid_loss /= len(val_loader.dataset)

        print(f'Training Loss: {training_loss:.2f}, Validation Loss: {valid_loss:.2f} (Took {time.time()-start} seconds)')
        if valid_loss < best_model_loss:
            epochs_since_last_improvement = 0
            if save_best_model:
                print('Saving best validation loss %.2f' % valid_loss)
                best_model_loss = valid_loss
                torch.save(model.state_dict(), f'{best_model_name}.pt')
        if early_stopping and epochs_since_last_improvement >= early_stopping_patience:
            print('Thats enough, early stopping after %d epochs.' % early_stopping_patience)
            break
