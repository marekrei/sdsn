import sys
import theano
import numpy
import collections
import lasagne

try:
    import cPickle as pickle
except:
    import pickle

sys.setrecursionlimit(50000)
floatX=theano.config.floatX

class WordPairClassifier(object):
    def __init__(self, config):
        self.config = config
        self.params = collections.OrderedDict()
        self.rng = numpy.random.RandomState(config["random_seed"])

        word1_ids = theano.tensor.ivector('word1_ids')
        word2_ids = theano.tensor.ivector('word2_ids')
        features = theano.tensor.fmatrix('features')
        labels = theano.tensor.fvector('labels')
        learningrate = theano.tensor.fscalar('learningrate')
        is_training = theano.tensor.iscalar('is_training')

        self.word_embedding_matrix_A = self.create_parameter_matrix('word_embedding_matrix_a', (config["n_words"], config["word_embedding_size_a"]), init_type=("zero" if config["init_embeddings_zero"] == True else "rand"))
        scores = self.construct_network(word1_ids, word2_ids, self.word_embedding_matrix_A, config["word_embedding_size_a"], is_training, features, self.config, "A")

        if config["late_fusion"] == True:
            self.word_embedding_matrix_B = self.create_parameter_matrix('word_embedding_matrix_b', (config["n_words"], config["word_embedding_size_b"]))
            scoresB = self.construct_network(word1_ids, word2_ids, self.word_embedding_matrix_B, config["word_embedding_size_b"], is_training, features, self.config, "B")
            gamma = theano.tensor.nnet.sigmoid(self.create_parameter_matrix('late_fusion_gamma', (1,)))[0]
            scores = gamma * scores + (1.0 - gamma) * scoresB

        cost = 0.0
        if config["cost"] == "mse":
            cost += ((scores - labels)*(scores - labels)).sum()
        elif config["cost"] == "hinge":
            difference = theano.tensor.abs_(scores - labels)
            se = (scores - labels)*(scores - labels)
            cost += theano.tensor.switch(theano.tensor.gt(difference, 4.0), se, 0.0).sum()

        cost_pretrain = theano.tensor.maximum((scores - labels)*(scores - labels) - (5.0-config["pretrain_margin"])*(5.0-config["pretrain_margin"]), 0.0).sum()

        if config["update_embeddings"] == True:
            params_ = self.params
        else:
            params_ = self.params.copy()
            del params_['word_embedding_matrix_a']
            if 'word_embedding_matrix_b' in params_:
                del params_['word_embedding_matrix_b']

        if config["cost_l2"] > 0.0:
            for param in params_.values():
                cost += config["cost_l2"] * theano.tensor.sum(param ** 2)
                cost_pretrain += config["cost_l2"] * theano.tensor.sum(param ** 2)

        gradients = theano.tensor.grad(cost, list(params_.values()), disconnected_inputs='ignore')
        gradients_pretrain = theano.tensor.grad(cost_pretrain, list(params_.values()), disconnected_inputs='ignore')
        if hasattr(lasagne.updates, config["optimisation_strategy"]):
            update_method = getattr(lasagne.updates, config["optimisation_strategy"])
        else:
            raise ValueError("Optimisation strategy not implemented: " + str(config["optimisation_strategy"]))
        updates = update_method(gradients, params_.values(), learningrate)
        updates_pretrain = update_method(gradients_pretrain, params_.values(), learningrate)

        input_vars_test = [word1_ids, word2_ids, features, labels]
        input_vars_train = input_vars_test + [learningrate]
        output_vars = [cost, scores]
        self.train = theano.function(input_vars_train, output_vars, updates=updates, on_unused_input='ignore', allow_input_downcast = True, givens=({is_training: numpy.cast['int32'](1)}))
        self.test = theano.function(input_vars_test, output_vars, on_unused_input='ignore', allow_input_downcast = True, givens=({is_training: numpy.cast['int32'](0)}))
        self.pretrain = theano.function(input_vars_train, output_vars, updates=updates_pretrain, on_unused_input='ignore', allow_input_downcast = True, givens=({is_training: numpy.cast['int32'](1)}))

    def apply_dropout(self, m, p, is_training):
        if p > 0.0:
            trng = theano.tensor.shared_randomstreams.RandomStreams(seed=1)
            dropout_mask = trng.binomial(n=1, p=1-p, size=m.shape, dtype=floatX)
            input_train = dropout_mask * m
            input_test = (1 - p) * m
            m = theano.tensor.switch(theano.tensor.neq(is_training, 0), input_train, input_test)
        return m


    def construct_network(self, word1_ids, word2_ids, word_embedding_matrix, word_embedding_size, is_training, features, config, name):
        word1_embeddings = word_embedding_matrix[word1_ids]
        word2_embeddings = word_embedding_matrix[word2_ids]

        word1_embeddings = self.apply_dropout(word1_embeddings, config["embedding_dropout"], is_training)
        word2_embeddings = self.apply_dropout(word2_embeddings, config["embedding_dropout"], is_training)

        gating1 = theano.tensor.nnet.sigmoid(self.create_layer(word2_embeddings, word_embedding_size, word_embedding_size, name + "_gating_first"))
        gating2 = theano.tensor.nnet.sigmoid(self.create_layer(word1_embeddings, word_embedding_size, word_embedding_size, name + "_gating_second"))

        if config["gating"] in ["second", "both"]:
            word2_embeddings = word2_embeddings * gating2
        elif config["gating"] in ["first", "both"]:
            word1_embeddings = word1_embeddings * gating1

        if config["embedding_mapping_size"] > 0:
            word1_embeddings = self.create_layer(word1_embeddings, word_embedding_size, config["embedding_mapping_size"], "word1_mapping_"+name)
            word2_embeddings = self.create_layer(word2_embeddings, word_embedding_size, config["embedding_mapping_size"], "word2_mapping_"+name)
            word_embedding_size = config["embedding_mapping_size"]

        if config["tanh_location"] == "before":
            word1_embeddings = theano.tensor.tanh(word1_embeddings)
            word2_embeddings = theano.tensor.tanh(word2_embeddings)

        if config["embedding_combination"] == "concat":
            combination = theano.tensor.concatenate([word1_embeddings, word2_embeddings], axis=1)
            combination_size = 2*word_embedding_size
        elif config["embedding_combination"] == "multiply":
            combination = word1_embeddings * word2_embeddings
            combination_size = word_embedding_size
        elif config["embedding_combination"] == "add":
            combination = word1_embeddings + word2_embeddings
            combination_size = word_embedding_size
        else:
            raise ValueError("Unknown combination: " + config["embedding_combination"])

        if config["tanh_location"] == "after":
            combination = theano.tensor.tanh(combination)


        combination = self.apply_dropout(combination, config["combination_layer_dropout"], is_training)

        if config["feature_count"] > 0:
            combination = theano.tensor.concatenate([combination, features], axis=1)
            combination_size += config["feature_count"]

        if config["hidden_layer_size"] > 0:
            combination = theano.tensor.tanh(self.create_layer(combination, combination_size, config["hidden_layer_size"], "hidden_"+name))
            combination_size = config["hidden_layer_size"]

        combination = self.apply_dropout(combination, config["hidden_layer_dropout"], is_training)

        scores = self.create_layer(combination, combination_size, 1, "output_"+name).reshape((word1_ids.shape[0],))

        if config["output_method"] == "linear":
            scores = scores
        elif config["output_method"] == "scaled_sigmoid":
            a = self.create_parameter_matrix("output_ " + name + "_sigmoid_a", (1,), init_type="one")
            b = self.create_parameter_matrix("output_ " + name + "_sigmoid_b", (1,), init_type="zero")
            scores = 10.0 * theano.tensor.nnet.sigmoid(a[0] * (scores - b[0]))

        return scores


    def save(self, filename):
        dump = {}
        dump["config"] = self.config
        dump["params"] = {}
        for param_name in self.params:
            dump["params"][param_name] = self.params[param_name].get_value()
        with open(filename, 'wb') as f:
            pickle.dump(dump, f, protocol=pickle.HIGHEST_PROTOCOL)


    @staticmethod
    def load(filename, new_config=None, new_output_layer_size=None):
        with open(filename, 'rb') as f:
            dump = pickle.load(f)
        model = WordPairClassifier(dump["config"])
        for param_name in model.params:
            assert(param_name in dump["params"])
            model.params[param_name].set_value(dump["params"][param_name])
        return model


    def create_parameter_matrix(self, name, size, init_type="rand"):
        if init_type == "rand":
            param_vals = numpy.asarray(self.rng.normal(loc=0.0, scale=0.1, size=size), dtype=floatX)
        elif init_type == "one":
            param_vals = numpy.ones((size), dtype=floatX)
        elif init_type == "zero":
            param_vals = numpy.zeros((size), dtype=floatX)
        self.params[name] = theano.shared(param_vals, name)
        return self.params[name]


    def create_layer(self, input_matrix, input_size, output_size, name):
        W = self.create_parameter_matrix(name + 'W', (input_size, output_size))
        bias = self.create_parameter_matrix(name + 'bias', (output_size,))
        result = theano.tensor.dot(input_matrix, W) + bias
        return result

