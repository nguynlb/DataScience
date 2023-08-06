
import torch
import torch.nn as nn
from typing import Dict


def train_step(model: nn.Module,
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = args.device) -> Dict[str, float]:
    
    model.train()
    total_loss, total_acc = 0, 0
    for idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y_pred = model(X)
        
        loss = loss_fn(y_pred, y)
        acc = accuracy_fn(y_pred.argmax(dim = 1), y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss
        total_acc += acc
        
        if idx % 4 == 0:
            print(f"Trained on {idx * len(X)}/{len(dataloader.dataset)}")
            
    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    
    print(f"Train loss {total_loss:.4f} | Train accuracy {total_acc:.4f}")
    
    return {"loss_score": total_loss, "acc_score" :total_acc}
    
    
    
def test_step(model: nn.Module, 
              dataloader: torch.utils.data.DataLoader,
              loss_fn: nn.Module,
              accuracy_fn,
              device: torch.device = args.device) -> Dict[str, float]:
    model.eval()
    total_loss, total_acc = 0, 0 
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            acc = accuracy_fn(y_pred.argmax(dim = 1), y)
            
            total_loss += loss
            total_acc += acc
            
        total_loss /= len(dataloader)
        total_acc /= len(dataloader)
    
    print(f"Test loss {total_loss:.4f} | Test accuracy {total_acc:.4f}")
    
    return {"loss_score": total_loss, "acc_score": total_acc}



def train_loop(model: nn.Module, 
               train_loader: torch.utils.data.DataLoader,
               test_loader: torch.utils.data.DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               epochs: int = args.epochs) -> Dict[str, float]:

    result = {
        "loss_train": [],
        "acc_train": [],
        "loss_test": [],
        "acc_test": []
    }
    start_train = timer()

    for epoch in tqdm(range(epochs)):
        print(f"Epoch {epoch}:\n-----------\n")

        train_score = train_step(model=model, 
                   dataloader=train_loader,
                   loss_fn=loss_fn,
                   accuracy_fn=accuracy_fn,
                   optimizer=optimizer)

        test_score = test_step(model=model,
                  dataloader=test_loader,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn)
        result["loss_train"].append(train_score["loss_score"].to("cpu").detach().numpy())
        result["acc_train"].append(train_score["acc_score"].to("cpu").detach().numpy())
        result["loss_test"].append(test_score["loss_score"].to("cpu").detach().numpy())
        result["acc_test"].append(test_score["acc_score"].to("cpu").detach().numpy())
        
    end_train = timer()

    time_train = print_train_time(start=start_train,
                                     end=end_train)    
    result["total_time"] = time_train
    return result
    
