=====
Usage
=====

The `competency_maps` package can be used in 3 different ways:

1. Use inside another project as a dependency
2. Use the Interactive Command Line Interface
3. Use a non-interactive Script


Use in a Project
----------------

To use competency_maps in a project::

    import competency_maps


Use as a CLI
------------

This package provides 2 CLI options:

Use as an Interactive CLI
^^^^^^^^^^^^^^^^^^^^^^^^^
Once the package is installed, we can use the interactive CLI to perform various actions.
This can be triggered by the below command::

    $ competency_maps

Use as a non-interactive script
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can also use the competency_maps as a non-interactive script as below::

    $ cmaps -i <Input Path of the Learning Resources>
    -o < Output Directory to store the results>
    -d < Delimiter of the File if the input is a CSV>
    -f <Field Names that store the content if input is a CSV File>
    -c <Config File that store the map model parameters>


Example of a Config File
------------------------

While using the package in CLI mode, we can specify an external config file which contains the
properties for the map creation.

.. code-block::
  :linenos:
    [preprocessor]
    type: spacy
    [topic_model]
    type: lda
    enable_tfidf: true
    filter_extremes: true
    evaluation_metric: cv_score
    num_topic_clusters: 50
    num_topics: 10
    [embeddings]
    word_embedding_type: fasttext
    word_embedding_dimensions: 150
    document_embedding_type: doc2vec
    document_embedding_dimensions: 150
    document_embedding_model_path: <path to pretrained model>
    num_levels: 10
