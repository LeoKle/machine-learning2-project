from models.gan.discriminator import Discriminator, DiscriminatorCNN
from models.gan.generator import Generator, GeneratorCNN
from pipeline.optuna_gan import OptunaStudy


if __name__ == "__main__":
    generator = Generator
    discriminator = Discriminator
    optuna_study = OptunaStudy(generator, discriminator, dataset="MNIST")
    optuna_study.optimize(n_trials=5)

    print("Best trials:")
    for trial in optuna_study.best_trials():
        print(f"Trial #{trial.number}")
        print(f"  Time: {trial.datetime_start} - {trial.datetime_complete}")
        print(f"  Values (IS ↑, FID ↓): {trial.values}")
        print(f"  Params: {trial.params}")
