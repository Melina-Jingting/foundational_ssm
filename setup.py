from setuptools import setup, find_packages

def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
setup(
    name="foundational_ssm",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="State Space Models for neural data analysis",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/foundational_ssm",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)