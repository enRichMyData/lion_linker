from setuptools import setup, find_packages

setup(
    name='lion_linker',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'requests',
        'pandas',
        'ollama'
    ],
    entry_points={
        'console_scripts': [
            'lion_linker=cli:main'
        ]
    }
)