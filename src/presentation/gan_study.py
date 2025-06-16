# from models.gan.discriminator import Discriminator
# from models.gan.discriminator_cnn import DiscriminatorCNN
# from models.gan.generator import Generator
# from models.gan.generator_cnn import GeneratorCNN, GeneratorCNN2
from models.gan.dcgan import DCGANDiscriminator, DCGANGenerator
from pipeline.optuna_gan import OptunaStudy


if __name__ == "__main__":
    generator = DCGANGenerator
    discriminator = DCGANDiscriminator
    optuna_study = OptunaStudy(generator, discriminator, dataset="CIFAR10")
    optuna_study.optimize(n_trials=1)

    print("Best trials:")
    for trial in optuna_study.best_trials():
        print(f"Trial #{trial.number}")
        print(f"  Time: {trial.datetime_start} - {trial.datetime_complete}")
        print(f"  Values (IS ↑, FID ↓): {trial.values}")
        print(f"  Params: {trial.params}")
