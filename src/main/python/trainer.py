import pickle

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

time_steps = 16
in_folder = "in/desert/"
out_file = f"in/desert_{time_steps}"

unet = Unet(
    dim=512,
    flash_attn=True,
    channels=1
)

model = GaussianDiffusion(
    unet,
    image_size=64,
    timesteps=time_steps,
    auto_normalize=False
)

trainer = Trainer(
    model,
    in_folder,
    train_batch_size=16,
    gradient_accumulate_every=1,
    train_num_steps=10000,
    amp=True,
    calculate_fid=False
)

# trainer.load(7)

trainer.train()

model.eval()

pickle.dump(model, open(out_file, "wb"))
