import wandb
run = wandb.init()
artifact = run.use_artifact('mail-analsarkar/my-colab-project/model-4pnawq9e:v0', type='model')
artifact_dir = artifact.download("E:/github_project/")