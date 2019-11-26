from setuptools import setup

setup(name='emonet',
      version='0.1',
      description='description',
      url='github.com/UBC-NLP/emonet',
      author='',
      author_email='',
      license='GNU',
      packages=['emonet'],
      install_requires=[
          'happiestfuntokenizing',
          'numpy', 'pandas',
          'torch',
          'transformers'
      ],

      entry_points={
          'console_scripts': [
              'emonet = emonet.emonet:main',
          ],
      },
      package_data={'emonet': ['resources/*']},
      include_package_data=True,
      zip_safe=False)
