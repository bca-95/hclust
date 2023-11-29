from setuptools import setup, find_packages

setup(
    name='r_hclust',
    version='0.1',
    description='An updated version of the hierarchical clustering. Deep learning techniques were applied to optimize the cutoff distance, that is used to refine clusters.',
    author='Carene',
    author_email='your.email@example.com',
    url=' https://github.com/bca-95/hclust',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'scipy',
        'pandas',
        'matplotlib',
        'seaborn'
        ],

    classifiers=[
        'Development Status  3 - Alpha',
        'Intended Audience  Developers',
        'License  OSI Approved  MIT License',
        'Programming Language  Python  3.6',
        'Programming Language  Python  3.7',
        'Programming Language  Python  3.8',
        'Programming Language  Python  3.9',
        ],
)

