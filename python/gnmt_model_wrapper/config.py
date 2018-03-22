import json


class Config:
    def __init__(self, hparam_file_path=None):
        if hparam_file_path is None:
            self.num_encoder_layers = 2
            self.num_decoder_layers = 2
            self.encoder_vocab_size = 42456
            self.decoder_vocab_size = 42456
            self.num_units = 1024
            self.forget_bias = 1.
            self.embed_size = 1024
            self.sos_id = 1
            self.eos_id = 2
            self.beam_width = 5
        else:
            with open(hparam_file_path, 'r', encoding='utf-8') as hf:
                hparam_dict = json.loads(hf.readline().strip(), encoding='utf-8')
                self.num_decoder_layers = self.num_encoder_layers = hparam_dict['num_layers']
                self.encoder_vocab_size = hparam_dict['src_vocab_size']
                self.decoder_vocab_size = hparam_dict['tgt_vocab_size']
                self.num_units = hparam_dict['num_units']
                self.forget_bias = hparam_dict['forget_bias']
                self.embed_size = hparam_dict['num_units']
                self.sos_id = 1
                self.eos_id = 2
                self.beam_width = 5
