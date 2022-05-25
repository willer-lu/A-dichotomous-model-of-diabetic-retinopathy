from timm.scheduler.cosine_lr import CosineLRScheduler

def build_scheduler(epochS, WARMUP_EPOCHS,optimizer, n_iter_per_epoch):
    num_steps = int(epochS*n_iter_per_epoch)
    warmup_steps = int(WARMUP_EPOCHS*n_iter_per_epoch)

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=1e-7,
            warmup_lr_init=5e-8,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
    )

    return lr_scheduler
