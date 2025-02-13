from setuptools import setup, find_packages

setup(
    name='expts',
    version='0.1.0',
    description='Códigos para gerar exemperimentos',
    long_description_content_type='text/markdown',
    author='Seu nome',
    author_email='seu_email@example.com',
    url='https://github.com/seu_usuario/meu_projeto',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    install_requires=[
        'requests',
        'numpy',
    ],
    python_requires='>=3.7',
)