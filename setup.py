from setuptools import setup, find_packages

# Function to read the requirements from requirements.txt
def load_requirements(file_name):
    with open(file_name, 'r') as file:
        return file.read().splitlines()

setup(
    name='lion_linker',
    version='0.1.0',
    packages=find_packages(),
    install_requires=load_requirements('requirements.txt'),
    entry_points={
        'console_scripts': [
            'lion-linker=lion_linker.cli:main',
        ],
    },
    include_package_data=True,
    description='A package for entity linking using LionLinker.',
    author='Roberto Avogadro',
    author_email='roberto.avogadro@sintef.no',  # Replace with your actual email
    url='https://github.com/roby-avo/lion_linker',  # Replace with your repository URL
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)