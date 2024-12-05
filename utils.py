from tqdm import tqdm

def prediction_accuracy(model, data, device, flatten=True):
    correct = 0
    total = 0
    if device.type == "cuda":
        model = model.to(device)
    for image, label in data:
        image, label = image.to(device), label.to(device)
        
        if flatten:
            image = torch.reshape(image, (-1, 3*32*32))  
        
        output = model(image)
        _, pred_label = torch.max(output, dim=1)
        
        # Update total and correct counts
        total += label.size(0)
        correct += (pred_label == label).sum().item()
    
    accuracy = correct / total
    print(f'Correct: {correct}, Total: {total}, Accuracy: {accuracy:.2f}')
    return accuracy

def fit_one_cycle(model, train_loader, optimizer, eindex, writer, device, flatten=True, size=None, log_freq=20):
    running_loss = 0.0
    last_loss = 0.0
    i = 0
    
    data = train_loader
    if size is not None:
        print(f"Truncating dataset to {size} samples")
        data = []
        for batch in train_loader:
            if len(data) == size:
                break
            data.append(batch)
            
    num_batches = len(train_loader)
    log_interval = max(1, num_batches // log_freq)

    model.train(True)  # Set the model to training mode
    
    for image, label in tqdm(data, desc="Training", leave=True):
        optimizer.zero_grad()
        
        image, label = image.to(device), label.to(device)
        print("moved batch to ", device)
        if flatten:
            image = torch.reshape(image, (-1, 3*32*32))
        
        output = model(image)
        
        probs = output.float()

        loss = F.cross_entropy(probs, label)
                
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        
        if i % log_interval == (log_interval - 1):
            last_loss = running_loss / log_interval
            #print(print('  batch {} loss: {}'.format(i + 1, last_loss)))
            tb_x = eindex * len(train_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.0
        i += 1

    return last_loss
        
def validation_loop(model, data, epochs, writer, device, flatten=True):
    best_vloss = 1_000_000.
    
    if device.type == "cuda":
        model = model.to(device)
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        
        model.train(True)
        avg_loss = fit_one_cycle(model, opt, epoch, writer, device, flatten)
        i = 0
        
        running_vloss = 0.0
        model.eval()
        
        with torch.no_grad():
            for validation_data in tqdm(data, desc="Validation", leave=True):
                vimages, vlabels = validation_data
                vimages, vlabels = vimages.to(device), vlabels.to(device)
                
                if flatten:
                    vimages = torch.reshape(vimages, (-1, 3*32*32))
                
                validation_output = model(vimages)
                validation_loss = F.cross_entropy(validation_output, vlabels)
                running_vloss += validation_loss
                
                i += 1
        
        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch + 1)
        
        writer.flush()
        
        if avg_loss < best_vloss:
            best_vloss = avg_loss
            print("Saving model")
            model_path = 'saved_models/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)