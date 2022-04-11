.. highlight:: shell

=====================================
Approach to Construct Competency Maps
=====================================

High Level Design
-----------------

Competency Map organises a learning space in terms of basic units of learning, each of which is called a competency.
The competency map is organised as a 2-dimensional progression space. A progression space not only has a concept of
distance between any pairs of competencies, but also a partial ordering that indicates progress made by a learner
on reaching some competency. In addition to the idea of progression, a competency map needs to be organised such
that, learning resources mapped onto the space can create coherent sequences of learning pathways that can be
traversed on the space.

To create a competency map given a representative set of learning resources,
we follow the below 5 step approach:

1. Identify Topics to form the X-axis
2. Topic sequencing on X-axis
3. Sequencing on Y-axis
4. Identifying Suitable Locations for the Resources
5. Generate the competency map

Figure below shows the overall approach:

.. figure:: _static/images/competency_map_design.png
   :scale: 100 %
   :align: center
   :alt: Competency Map Design

Identify Topics to form the X-axis
----------------------------------

For identifying the topics that represent the corpus of learning resources, we use topic modelling techniques.
Topic models are algorithms that are used for discovering the main themes that describe unstructured collection of
documents. There are several algorithms that can be used to perform topic modelling. Here, we provide 3 choices:
1. LDA
2. LDAMallet
3. Hierarchical LDA

All of these are variations of the `Latent Dirichlet Allocation (LDA) Model`_ which is the
most popular and widely used topic modelling algorithm. LDA is a generative probabilistic model that assumes
each document is a mixture over a set of topic probabilities and each topic is a mixture over an underlying set of words.
We consider the underlying set of words from each of the topic clusters obtained from the LDA Model as the topics
for the competency map.

To determine the optimal number of topic clusters for a given corpus, we can use any one of the 3 below
different `metrics`_:
1. Coherence Value
2. UMASS Value
3. Log perplexity.

We train several LDA models each with different number of topic clusters. The LDA model with the highest metric score
is considered as the best model. Once the set of topic clusters are identified, we use the notion of
`relevance`_ to identify and extract the top $n$ significant words from each cluster. Each of these words
constitute the topics for the competency map. Relevance is a method of ranking terms within a topic cluster.

The number of topics $N_t$ that will be extracted from each cluster is provided as an input. We divide this number
by three to obtain the relevant, marker and generic topics in equal proportion respectively.
1. Relevant Topics: We set :math:`\\lambda = 0.6` to obtain relevant topics which provide the best interpretation to the topic
clusters.
2. We set :math:`\\lambda=0` to obtain marker topics which are topics that are specific to a cluster. Marker topics can be
considered as milestones that identify key competencies in the competency map
3. We set :math:`\\lambda=1` to get generic topics that are common to all the clusters. Generic topics are considered to be
connecting competencies between two marker topics.

The number of topics thus obtained from all the topic clusters form the X-axis range for the competency map.

.. _Latent Dirichlet Allocation (LDA) Model: https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation
.. _metrics: https://datascienceplus.com/evaluation-of-topic-modeling-topic-coherence/
.. _relevance: https://github.com/bmabey/pyLDAvis


Topic sequencing on X-axis
--------------------------
To sequence the topics on the X-axis, we compute the **volume** for each topic from the word embeddings and arrange
them in ascending order of this metric. Word embeddings create a numerical representation for words in
the form of vectors that capture their meanings, semantic relationships and the different types of contexts they are
used in. We provide 3 options for constructing a word embedding model:

1. `Word2vec`_
2. `Glove`_
3. `Fasttext`_

Typically, word embeddings are high dimensional vectors  where each dimension can be thought of as semantic
feature in the broader, higher-dimensional semantic space. To be able to capture the semantics and retain the
importance of the topic across all dimensions, we use the volume metric which is the log-sum of the absolute values
of the corresponding embedding vector. Formally, the volume *V_t* of a topic t is defined as follows:

.. math::
	V_{t} = \\sum_{i=1}^{N}\\log(abs(w_{t_i}))


where :math:`w_{t_i}` is the weight of the :math:`i^{th}` dimension of the word embedding vector :math:`w_t` for the
topic *t* obtained from the word embedding model that has *N* dimensions.

.. _Word2vec: https://arxiv.org/pdf/1301.3781.pdf
.. _Fasttext: https://arxiv.org/abs/1607.04606
.. _Glove: http://nlp.stanford.edu/projects/glove/

Sequencing on Y-axis
--------------------

To arrange the learning resources in increasing order of their complexity/importance, we use the **volume** metric
obtained from document embedding models. Document embedding methods reduce an entire document into a
single n-dimensional vector and is usually a function of the word vectors of the words contained in them.
We utilise the `doc2vec`_ model to generate resource embeddings. The resource volume $V_r$ of a resource $r$ is defined
as follows:

.. math::
	V_{r} = \\sum_{i=1}^{N}\\log(abs(w_{r_i}))

where :math:`w_{r_i}` is the weight of the :math:`i^{th}` dimension of the document embedding :math:`w_r` of the
resource *r* obtained from the document embedding model that has *N* dimensions.


.. _doc2vec: https://arxiv.org/abs/1405.4053v2

Identifying Suitable Locations for the Resources
------------------------------------------------

For all the learning resources in the corpus, the X and Y coordinates are calculated and the learning resource is
mapped to the competency map. Mapping a learning resource on the Y-axis is straight forward, the volume of the resource
$V_{r}$ is computed as described in `Sequencing on Y-axis`_ and is used as the Y coordinate for the
learning resource. To map learning resources on X-axis, we first find the probability of the learning resources
belonging to the set of topics that it represents and then take a weighted average of the same.

To determine the probability that the learning resource is mapped to the topics identified in Section~\ref{sec:topic_model},
we use the following two probabilities: Resource Cluster Probability *p(r|c)* which is the probability that a resource
*r* is mapped to a topic cluster *c* and Topic Cluster Probability *p(t|c)* which is the probability that a topic t
belongs to a topic cluster(c). Since a learning resource can be mapped to more than one topic cluster, we choose the
topic cluster that has the maximum probability. The probability that a learning resource *r* is mapped to a topic *t*
is thus defined as follows:

.. math::
    p(r | t) = \\arg \\max_{c \\in N_c} p(r | c)* p(t|c)

where $N_c$ is no of topic clusters in the LDA model.

We calculate the X coordinate for each learning resource by computing the weighted sum of the probabilities of all the
topics the learning resource represents. This weighted sum is computed by normalising the probabilities of all the
constituent topics. First, the \emph{resource probability} is calculated, which is the ratio of the probability of the
learning resource mapped to each topic $p(r|t)$ divided by the total probability of all the topics the learning
resource is mapped to $p(r)$. Let $\textit{topics(r)}$ represent all the topics that the resource r belongs to, then
the total probability is defined as:

.. math::
    p(r) = \\sum_{t \\in \\textit{topics(r)}} p(r|t)

The weighted X value $r_{x}$ for the learning resource is then obtained by multiplying each \emph{topic probability}
with the \emph{topic volume} $V_t$ for that topic and summing over all the topics. Formally, this is defined as follows:

.. math::
    r_{x} = \\sum_{t \\in topics(r)} V_{t} * \\frac{p(r|t)}{p(r)}



Generating the Competency Map
-----------------------------

The final step in the competency map generation process is to transform the coordinates of each learning resource into the competency map space. Each learning resource is characterised currently by two attributes - $r_x$ and $V_r$. We then transform these attributes to the competency map space.

Transform the X-coordinate
^^^^^^^^^^^^^^^^^^^^^^^^^^

The X-axis in the competency map space ranges from 0 to the number of topics($N_t$) that are identified in Section
\ref{sec:topic_model}. To identify the X-coordinate of the learning resource, we first define the topic intervals
$t_i$ as follows:

.. math::
    t_i = \\frac{V_{t_{\\text{max}}} - V_{t_{\\text{min}}}}{N_t}

where :math:`V_{t_{\\text{min}}}` and :math:`V_{t_{\\text{max}}}` are the minimum and maximum topic volumes respectively
for the topics obtained in Section \ref{sec:topic_model}. The X-coordinate for each resource `r` is then computed using:

.. math::
    R_x = \\frac{r_x - V_{t_{\\text{min}}}}{t_i}


Transform the Y-coordinate
^^^^^^^^^^^^^^^^^^^^^^^^^^

We define a metric called **level** for Y-axis which is a fixed interval for volumes. Logically this means that, the
learning resources that have similar importance or complexity can be grouped together in an interval.To obtain the
levels at which each resource will be placed, we first define the level intervals $l_i$ as follows:

.. math::
    l_i = \\frac{V_{r_{\\text{max}}} - V_{r_{\\text{min}}}}{N_L}

where :math:`V_{r_{\\text{min}}}` and :math:`V_{r_{\\text{max}}}` are the minimum and maximum learning resource volumes
respectively and `N_L` is the number of levels that are required in the map. It is currently modelled as a user input.
The level at which each resource r exists is defined as `R_y`

.. math::
    R_y = \\frac{V_r - V_{r_{\\text{min}}}}{l_i}

The `R_x` and `R_y` thus obtained for each learning resource corresponds to the location of the resource on the
competency map space.
