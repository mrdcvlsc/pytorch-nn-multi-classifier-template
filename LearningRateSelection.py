import math

def find_lr(model, dataset_loader, loss_fn, optimizer, init_value=1e-8, final_value=10.0):
    
    number_in_epoch = len(dataset_loader) - 1
    update_step = (final_value / init_value) ** (1 / number_in_epoch)
    
    lr = init_value
    optimizer.param_groups[0]["lr"] = lr
    
    best_loss = 0.0
    batch_num = 0
    
    losses = []
    log_lrs = []

    outputs = None
    labels = None

    for data in dataset_loader:
        batch_num += 1
        inputs, labels = data
        inputs, labels = inputs, labels
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = loss_fn(outputs, labels)

        # Crash out if loss explodes

        if batch_num > 1 and loss > 4 * best_loss:
            return log_lrs[10:-5], losses[10:-5]

        # Record the best loss

        if loss < best_loss or batch_num == 1:
            best_loss = loss
        
        # store the values

        losses.append(loss)
        log_lrs.append(math.log10(lr))

        # do the backward pass and optimize

        loss.backward()
        optimizer.step()

        # update the lr for the next step and store

        lr *= update_step
        optimizer.param_groups[0]["lr"] = lr
    
    return log_lrs[10:-5], losses[10:-5]



    