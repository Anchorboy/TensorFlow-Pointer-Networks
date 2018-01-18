from datetime import *

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.

	Create test here.
    """
    input_size = 2
    encode_size = 2
    decode_size = 1
    min_length = 1
    max_length = 5 # longest sequence to parse
    rnn_size = 32
    dropout = 0.667
    hidden_size = 512
    batch_size = 80
    beam_size = 5
    n_epochs = 1000
    lr = 1e-5
    lr_decay = 0.99
    max_grad_norm = 5.
    is_train = True

    def __init__(self, args):
        self.max_length = args.tsp_num
        if "model_path" in args:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/tsp{}/{:%Y%m%d_%H%M%S}/".format(self.max_length, datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"
