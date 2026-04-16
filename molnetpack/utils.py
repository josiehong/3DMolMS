from tqdm import tqdm

import torch


def pred_step(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask, env = batch
            x = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            env = env.to(device=device, dtype=torch.float)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad(): 
                pred = model(x, mask, env, idx_base)
                pred = pred / torch.max(pred)  # normalize the output

            bar.set_description("Eval")
            bar.update(1)

            # recover sqrt spectra to original spectra
            pred = torch.pow(pred, 2)
            # post process
            pred = pred.detach().cpu().apply_(lambda x: x if x > 0.01 else 0)

            id_list += ids
            pred_list.append(pred)

    pred_list = torch.cat(pred_list, dim=0)
    return id_list, pred_list


def eval_step_oth(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask, env = batch
            x = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )
            env = env.to(device=device, dtype=torch.float)
            env = env[:, 1:]  # remove collision energy in `env`

            with torch.no_grad():
                pred = model(x, mask, env, idx_base)

            bar.set_description("Eval")
            bar.update(1)

            id_list += ids
            pred_list.append(pred)

    pred_list = torch.cat(pred_list, dim=0)
    return id_list, pred_list


def pred_feat(model, device, loader, batch_size, num_points):
    assert batch_size == 1, "batch_size should be 1 for prediction"
    model.eval()
    id_list = []
    pred_list = []
    with tqdm(total=len(loader)) as bar:
        for step, batch in enumerate(loader):
            ids, x, mask, _ = batch
            x = x.to(device=device, dtype=torch.float).permute(0, 2, 1)
            mask = mask.to(device=device)
            idx_base = (
                torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
            )

            with torch.no_grad():
                pred = model(x, mask, idx_base)

            bar.set_description("Eval")
            bar.update(1)

            id_list += ids
            pred_list.append(pred)

    pred_list = torch.cat(pred_list, dim=0)
    return id_list, pred_list
