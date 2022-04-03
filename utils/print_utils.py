from colorama import Fore
# https://www.tecmint.com/boxes-draws-ascii-art-boxes-in-linux-terminal/
# https://www.w3resource.com/python/python-format.php
# ┌TITLE──────────────────── ─┐
# │                            │
# │                            │
# │                            │
# └──────────────────────(v1.0)┘


def print_msg(title:str, msg:str, *args):
    title = Fore.RED +title.strip()+' ' + Fore.RESET
    print('┌─ {:─<70}┐'.format(title))
    print('│', '{:>62}'.format('│'))
    print('│{:^62}│'.format(msg))
    print('│', '{:>62}'.format('│'))
    print('└{:─>61}─┘'.format('(v1.0)'))

def print_time(): ...
#TODO
# print_msg("Unfreeze all layers","Resnet")