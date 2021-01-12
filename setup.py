import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="IMNN",
    version="0.3dev",
    author="Tom Charnock",
    author_email="charnock@iap.fr",
    description="Using neural networks to extract sufficient statistics from \
        data by maximising the Fisher information",
    long_description=long_description,
    long_description_content_type="text/reStructuredText",
    url="https://bitbucket.org/tomcharnock/IMNN.git",
<<<<<<< HEAD
    packages=["IMNN", "IMNN.LFI", "IMNN.utils", "IMNN.experimental", "IMNN.experimental.jax", "IMNN.experimental.jax.imnn", "IMNN.experimental.jax.lfi", "IMNN.experimental.jax.utils"],
=======
    packages=["IMNN", "IMNN.LFI", "IMNN.utils", "IMNN.experimental",
              "IMNN.experimental.jax", "IMNN.experimental.jax.imnn",
              "IMNN.experimental.jax.lfi", "IMNN.experimental.jax.utils"],
>>>>>>> f41c676f86e48f2460f1436766d3c0653daf0aa9
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3",
    install_requires=[
          "tensorflow>=2.1.0",
<<<<<<< HEAD
=======
          "jaxlib==0.1.57",
>>>>>>> f41c676f86e48f2460f1436766d3c0653daf0aa9
          "jax>=0.2.5",
          "tqdm>=4.31.0",
          "numpy>=1.16.0",
          "scipy>=1.4.1",
          "matplotlib"],
)
