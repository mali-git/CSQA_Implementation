# csqa
csqa is a package for training a dialogue system for the Complex Sequential Question Answering (CSQA) task.

## Project Structure

* In 'src/csqa/src/deployer' all deployer scripts are contained.
* In 'src/neural_network_models' all neural networks are defined
* In 'src/utilities' all utilities such as preprocessing and postprocessing scripts are contained.
* In 'csqa/tests' the test scripts for all modules are defined

## Utilities

### Package 'tensor_embeddings_creation'

#### Module 'utterance_to_tensor_embeddings_creator.py':
Representation of an training instance: A dictionary containing following entries:
* Instance ID
* List of IDs of entity mentioned in question
* List of IDs of predicates mentioned in question
* Question utterance
* Question utterance embedded
* List of IDs for entity mentioned in question
* List of IDs of predicates mentioned in answer
* Answer utterance
* Answer utterance embedded 

Representation of a prediction instance: A dictionary containing following entries:
* Instance ID
* Dictionary [Question Entity ID: Entity text mention]
* Dictionary [Question Predicate ID: Predicate text mention]
* Question utterance
* Question utterance embedded


