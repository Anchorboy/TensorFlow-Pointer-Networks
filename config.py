from datetime import *

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

	Create test here.
    """
    encode_size = 2
    decode_size = 1
    min_length = 1
    max_length = 10 # longest sequence to parse
    rnn_size = 32
    dropout = 0.667
    hidden_size = 128
    batch_size = 32
    beam_size = 5
    n_epochs = 1000
    lr = 0.001
    lr_decay = 0.95
    max_grad_norm = 2.
    is_train = True

    def __init__(self, args):

        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{:%Y%m%d_%H%M%S}/".format(datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"