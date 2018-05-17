import tensorflow as tf
from tensorflow.python.estimator.export import export_output

from utilities.constants import ADADELTA, ADAGRAD, ADAGRAD_DA, ADAM, \
    RMS_PROP, DEFAULT_SERVING_KEY, CLASSIFY_SERVING_KEY, PREDICT_SERVING_KEY


def get_estimator_specification(mode, predictions_dict, classifier_output, loss=None, train_op=None):
    """
    model_fct returns an EstimatorSpec object containing all the information needed for training,eval and pedict
    :param mode: Defines the mode (train,eval,predict) in which model_fct was called
    :param predictions_dict:
    :param classifier_output:
    :param loss:
    :param train_op:
    :rtype: EstimatorSpec
    """

    estimator_specification = None

    if mode == tf.estimator.ModeKeys.TRAIN:
        estimator_specification = tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op,
                                                             predictions=predictions_dict,
                                                             export_outputs={
                                                                 DEFAULT_SERVING_KEY: classifier_output,
                                                                 CLASSIFY_SERVING_KEY: classifier_output,
                                                                 PREDICT_SERVING_KEY: export_output.PredictOutput(
                                                                     predictions_dict)})

    elif mode == tf.estimator.ModeKeys.EVAL:
        estimator_specification = tf.estimator.EstimatorSpec(mode, loss=loss, predictions=predictions_dict,
                                                             export_outputs={
                                                                 DEFAULT_SERVING_KEY: classifier_output,
                                                                 CLASSIFY_SERVING_KEY: classifier_output,
                                                                 PREDICT_SERVING_KEY: export_output.PredictOutput(
                                                                     predictions_dict)})

    elif mode == tf.estimator.ModeKeys.PREDICT:
        estimator_specification = tf.estimator.EstimatorSpec(mode, predictions=predictions_dict, export_outputs={
            DEFAULT_SERVING_KEY: classifier_output,
            CLASSIFY_SERVING_KEY: classifier_output,
            PREDICT_SERVING_KEY: export_output.PredictOutput(predictions_dict)})

    return estimator_specification

def get_optimizer(optimizer, learning_rate=None):
    """

    :param optimizer:
    :param learning_rate:
    :return:
    """
    if optimizer == ADADELTA:
        if learning_rate != None:
            return tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
        else:
            return tf.train.AdadeltaOptimizer()

    elif optimizer == ADAGRAD:

        return tf.train.AdagradOptimizer(learning_rate=learning_rate)

    elif optimizer == ADAGRAD_DA:

        return tf.train.AdagradDAOptimizer(learning_rate=learning_rate)

    elif optimizer == ADAM:

        if learning_rate != None:
            return tf.train.AdamOptimizer(learning_rate=learning_rate)
        else:
            return tf.train.AdamOptimizer()

    elif optimizer == RMS_PROP:
        tf.train.RMSPropOptimizer(learning_rate=learning_rate)

    else:
        raise Exception("Optimizer %s isn't available. Choose one of following %s, %s, %s, %s or %s" % (optimizer,
                                                                                                        ADADELTA,
                                                                                                        ADAGRAD,
                                                                                                        ADAGRAD_DA,
                                                                                                        ADAM,
                                                                                                        RMS_PROP))
