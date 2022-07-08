import torch
from torch.utils.data.dataloader import DataLoader


def to(xs, device):
    res = []
    for x in xs:
        if isinstance(x, dict):
            for k in x:
                x[k] = x[k].to(device)
            res.append(x)
        else:
            res.append(x.to(device))
    return res


class RLTrainer:

    def __init__(self, config):
        self.config = config
        self.device = config.device

        self.n_epochs = 0
        self.optimizers = None
        self.lr_schedulers = None

    def get_optimizer(self, model):
        if self.optimizers is None:
            print(f'[ utils/training ] Making optimizer at epoch {self.n_epochs}')
            self.optimizers = model.configure_optimizers(self.config)
        return self.optimizers

    def get_lr_scheduler(self, model):
        if self.lr_schedulers is None:
            print(f'[ utils/training ] Making LR scheduelr at epoch {self.n_epochs}')
            if hasattr(model, 'configure_lr_schedulers'):
                self.lr_schedulers = model.configure_lr_schedulers(self.config)
            else:
                self.lr_schedulers = [None] * len(self.optimizers)
        return self.lr_schedulers

    def train(self, model, dataset, n_epochs=1, log_freq=100):

        config = self.config
        optimizers = self.get_optimizer(model)
        lr_schedulers = self.get_lr_scheduler(model)
        model.train(True)

        loader = DataLoader(dataset, shuffle=True, pin_memory=True,
                            batch_size=config.batch_size,
                            num_workers=config.num_workers)

        for _ in range(n_epochs):

            for it, batch in enumerate(loader):

                batch = to(batch, self.device)

                # forward the model
                with torch.set_grad_enabled(True):
                    losses = model.compute_losses(batch[0])

                l = 0

                for loss, optimizer, lr_scheduler in zip(losses, optimizers, lr_schedulers):
                    # backprop and update the parameters
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    if lr_scheduler is not None:
                        lr_scheduler.step()

                    l = l + loss.item()

                if hasattr(model, 'target_updates'):
                    model.target_updates()

                # report progress
                if it % log_freq == 0:
                    print(
                        f'[ utils/training ] epoch {self.n_epochs} [ {it:4d} / {len(loader):4d} ] ',
                        f'train loss {l:.5f}')

            self.n_epochs += 1
