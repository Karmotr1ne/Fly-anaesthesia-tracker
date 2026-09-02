from setuptools import setup, find_packages

setup(
    name="drosophila-anesthesia-tracker",
    version="0.1.0",
    description="Integrated Drosophila vision tracking, kinematic cleaning, and anesthesia kinetics testing workbench.",
    author="Drosophila Behavioral Phenotyping Lab",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.23.0",
        "scipy>=1.9.0",
        "pandas>=1.5.0",
        "opencv-python>=4.7.0",
        "matplotlib>=3.6.0",
        "PyQt6>=6.4.0",
        "pyyaml>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "anesthesia-gui = drosophila_suite.gui_app:run_gui",
            "anesthesia-cli = drosophila_suite.cli:main",
        ]
    },
)
