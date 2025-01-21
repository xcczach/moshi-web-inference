import os
import subprocess


def install_requirements():
    if os.path.exists("installation_complete"):
        return

    subprocess.run(["pip", "install", "moshi"])

    with open("installation_complete", "w") as f:
        f.write("Installation complete.")


if __name__ == "__main__":
    install_requirements()
