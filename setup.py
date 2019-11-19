from setuptools import setup

setup(name='emotion',
      version='0.1',
      description='description',
      url='',
      author='',
      author_email='',
      license='GNU',
      packages=['emotion'],
      install_requires=[
          'numpy', 'pandas', 'keras', 'sklearn',
          'torch',
          'transformers',
          'gensim',
          'skimage'
      ],

      entry_points={
          'console_scripts': [
              'emotionnet = emotione.emotionnet:main',
          ],
      },
      package_data={'emotionnet': ['resources/*']},
      include_package_data=True,
      zip_safe=False)
