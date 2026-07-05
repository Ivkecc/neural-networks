class EarlyStopping:

    def __init__(self, patience: int):
        self.patience = patience

        self.best_loss = float("inf")
        self.counter = 0

    def step(self, current_loss: float) -> bool:
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.counter = 0
            return False

        self.counter += 1

        return self.counter >= self.patience
