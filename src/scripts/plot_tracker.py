from pathlib import Path
from classes.tracker import Tracker
from utils.plotter import Plotter

STUDY_NAMES = [
    "gan_IS_cifar10_generator_discriminator",
    "gan_IS_FID_cifar10_generator_discriminator"
    "gan_IS_FID_cifar10_generatorcnn_discriminator",
    "gan_IS_FID_cifar10_generatorcnn_discriminatorcnn",
    "gan_IS_FID_cifar10_generatorcnn2_discriminatorcnn",
    "gan_IS_FID_mnist_generator_discriminator",
    "gan_IS_FID_mnist_generatorcnn_discriminator",
    "gan_IS_FID_mnist_generatorcnn_discriminatorcnn",
    "gan_IS_mnist_generator_discriminator",
    "dcgan",
]
TRIAL_NUMBERS = range(0, 15)
FILE = "gan_metrics.json"


def test(path, title, output_folder):
    tracker = Tracker(path)
    data = tracker.import_data()
    Path.mkdir(Path(output_folder), exist_ok=True, parents=True)

    # Loss plot:
    Plotter.plot_tracker_dict(
        data,
        title=title,
        keys_to_plot=["discriminator_loss", "generator_loss"],
        marker="",
        save_path=output_folder + "loss_plot.svg",
    )

    # IS plot
    Plotter.plot_tracker_dict(
        data,
        title=title,
        keys_to_plot=["is_mean", "is_std"],
        marker="o",
        save_path=output_folder + "is_plot.svg",
    )

    # FID + IS plot
    Plotter.plot_tracker_dict(
        data,
        title=title,
        keys_to_plot=["is_mean", "fid_score"],
        save_path=output_folder + "is_fid_plot.svg",
    )

    # FID plot
    Plotter.plot_tracker_dict(
        data,
        title=title,
        keys_to_plot=["fid_score"],
        save_path=output_folder + "fid_plot.svg",
    )


if __name__ == "__main__":
    for study_name in STUDY_NAMES:
        for trial_number in TRIAL_NUMBERS:
            try:
                path = f"output/{study_name}/{trial_number}/{FILE}"
                title = f"{study_name}" + " " + f"Trial {trial_number}"

                test(
                    path,
                    title,
                    output_folder=f"results/gan/{study_name}/{trial_number}/",
                )

            except:
                print(f"Failed to export study {study_name} trial {trial_number}")
