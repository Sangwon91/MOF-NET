from setuptools import setup, find_packages

# Get dependencies from requirements.txt file.
with open("requirements.txt", "r") as f:
    install_requires = f.readlines()

setup(
    name="mofnet",
    version="0.0.1",
    description="Predict properties of MOFs from topology and"
                " building blocks.",
    install_requires=install_requires,
    author="Sangwon Lee",
    author_email="integratedmailsystem@gmail.com",
    packages=["mofnet"],
    python_requires=">=3.5",
    zip_safe=False
)
