import torch


def feature_loss(fmap_r, fmap_g):
    """Calculatie feature map loss.

    L1 loss between output of all layers in discriminator calculated by real sample and generated sample.

    Args:
        fmap_r: feature map of real sample.
        fmap_g: feature map of generated sample.
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):  # loop for sub-discriminators
        for rl, gl in zip(dr, dg):  # loop for layers
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_gen_outputs):
    """Discriminator loss.

    Args:
        disc_real_outputs: disc_outputs: output of all layers in discriminators calculated by real sample.
        disc_gen_outputs: disc_outputs: output of all layers in discriminators calculated by generated sample.
    """
    loss = 0
    real_losses = []
    gen_losses = []
    for dr, dg in zip(disc_real_outputs, disc_gen_outputs):
        dr = dr.float()
        dg = dg.float()
        real_loss = torch.mean((1 - dr) ** 2)
        gen_loss = torch.mean(dg**2)
        loss += real_loss + gen_loss
        real_losses.append(real_loss.item())
        gen_losses.append(gen_loss.item())

    return loss, real_losses, gen_losses


def generator_loss(disc_outputs):
    """Generator loss.

    Args:
        disc_outputs: output of all layers in discriminators calculated by generated sample.
    """
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        loss_tmp = torch.mean((1 - dg) ** 2)
        gen_losses.append(loss_tmp)
        loss += loss_tmp

    return loss, gen_losses


def kl_loss(z_theta, logs_linspec, mean_text, logs_text, z_mask):
    """KL Divergence.

    Args:
        z_theta: latent variable of flow.
        logs_linspec: stats of linspec calculated by linspec encoder(posterior encoder).
        mean_text: stats of text(phoneme) calculated by linspec encoder(text encoder).
        logs_text: stats of text(phoneme) calculated by linspec encoder(text encoder).
        z_mask: mask for z_theta
    """
    z_theta = z_theta.float()
    logs_linspec = logs_linspec.float()
    mean_text = mean_text.float()
    logs_text = logs_text.float()
    z_mask = z_mask.float()

    kl = logs_text - logs_linspec - 0.5
    kl += 0.5 * ((z_theta - mean_text) ** 2) * torch.exp(-2.0 * logs_text)
    kl = torch.sum(kl * z_mask)
    loss = kl / torch.sum(z_mask)
    return loss
