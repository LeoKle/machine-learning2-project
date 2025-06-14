import optuna


optuna.delete_study(
    study_name="gan_IS_FID_mnist_generator_discriminator",
    storage="sqlite:///gan_optuna_study_IS.db",
)

