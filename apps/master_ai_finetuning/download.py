import wandb
run = wandb.init()
artifact = run.use_artifact('mail-analsarkar/AnotherGPT_kaggle/run-jdqfypfk-history:v0', type='wandb-history')
artifact_dir = artifact.download()