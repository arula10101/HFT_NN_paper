#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 18:43:50 2019
@author: laura
"""

import argparse
import sys
import json
import hashlib 
import shlex

#%%

class Config:
    
    def __init__(self):
        pass
    
    def get_meta_config(self, meta_config_dict):
        # get configuration from meta_config json file
        cli_parser = argparse.ArgumentParser(description='meta_config')
        for dest, v in meta_config_dict.items():
            cli_1letter = v.pop('cli')
            cli_ext = v.pop('ext-cli')
            arg_type = eval(v.pop('type'))
            v['dest'] = dest
            v['type'] = arg_type
            cli_parser.add_argument(cli_1letter, cli_ext, **v)
        return cli_parser

    def get_cli_args(self, cli_parser): 
        # get arguments from command line
        args = cli_parser.parse_args(sys.argv[1:])
        self.args_dict = args.__dict__
        return self.args_dict

    def update(self, args_dict):
        # update arguments using model-specific json file
        if 'json' in args_dict:
            # read json in args_json
            with open(args_dict.pop('json')) as json_file:
                self.final_args = json.load(json_file)
        else:
            self.final_args = {}
        self.final_args.update(args_dict)   
        return self.final_args    

    def get_config(self, meta_config_dict):
        # get all arguments/parameters into dictionary
        cli_parser = self.get_meta_config(meta_config_dict)
        args_dict = self.get_cli_args(cli_parser)
        final_args = self.update(args_dict)        
        return final_args
    
    
    # If we want to update from within code instead of command line:
    
    def get_cli_args_from_string(self, cli_parser, cli_string): 
        # get arguments from command line
        args = cli_parser.parse_args(shlex.split(cli_string))
        args_dict = args.__dict__  
        return args_dict
    
    def get_config_from_string(self, meta_config_dict, cli_string):
        cli_parser = self.get_meta_config(meta_config_dict)
        args_dict = self.get_cli_args_from_string(cli_parser, cli_string)
        final_args = self.update(args_dict)        
        return final_args
    
    
class hexConfig:
    
    def __init__(self):
        pass
                
    def sort_dict(self, config_dict):
        # sort config dictionary and save to string of parameters in alphabetical order
        sorted_config_str = ""
        for i in sorted(config_dict.keys()):
            sorted_config_str += i + "_"            
            sorted_config_str += str(config_dict[i]) + "_"            
        return sorted_config_str

    def get_hex(self, s):
        # gets hexadecimal number for string of parameters in the model
        s_ = hashlib.md5(s.encode('utf-8'))  
        s_hex = s_.hexdigest() 
        return s_hex 
     
    def hex_config(self, config_dict):
        sorted_config_str = self.sort_dict(config_dict)
        hex_configs = self.get_hex(sorted_config_str)
        return hex_configs
    