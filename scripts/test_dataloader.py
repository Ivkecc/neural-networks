from data.dataloaders import create_dataloaders

train_loader, test_loader = create_dataloaders()

X, y = next(iter(train_loader))

print(X.shape)
print(y.shape)

print(X.dtype)
print(y.dtype)

print(y[:10])
