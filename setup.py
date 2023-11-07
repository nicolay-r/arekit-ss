from setuptools import (
    setup,
    find_packages,
)


def get_requirements(filenames):
    r_total = []
    for filename in filenames:
        with open(filename) as f:
            r_local = f.read().splitlines()
            r_total.extend(r_local)
    return r_total

setup(
    name='arekit_ss',
    version='0.24.0',
    description='Low Resource Context Relation Sampler for contexts'
                ' with relations for fact-checking and fine-tuning your LLM models, powered by AREkit',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/nicolay-r/arekit-ss',
    author='Nicolay Rusnachenko',
    author_email='rusnicolay@gmail.com',
    license='MIT License',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Text Processing :: Linguistic',
    ],
    keywords='relation extraction, data processing',
    packages=find_packages(),
    package_dir={'src': 'src'},
    install_requires=get_requirements(['dependencies.txt']),
)