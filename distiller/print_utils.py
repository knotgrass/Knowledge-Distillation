import math
from colorama import Fore
# https://www.tecmint.com/boxes-draws-ascii-art-boxes-in-linux-terminal/
# https://www.w3resource.com/python/python-format.php
# ┌TITLE──────────────────── ─┐
# │                            │
# │                            │
# │                            │
# └──────────────────────(v1.0)┘


def print_msg(title:str, msg:str, *args) -> None:
    title = Fore.RED +title.strip()+' ' + Fore.RESET
    print('┌─ {:─<70}┐'.format(title))
    print('│', '{:>62}'.format('│'))
    print('│{:^62}│'.format(msg))
    print('│', '{:>62}'.format('│'))
    print('└{:─>61}─┘'.format('(v1.0)'))

def print_time(title:str, time:float):
    minute = int(time // 60)
    milisec, sec = math.modf(time % 60)
    print('{}  {} :{:^3d}:{:>3d}'.format(
        title, minute, int(sec), int(1000*milisec)))
    

def print_metric(metric_name:str, result:float) -> None:
    ...
# time = 12 * 60 +  1.007
# print_time("CLASSIFIER TRAINING TIME", time)


def desc(epoch:int, n_epoch:int, phase:str, loss:float, acc:float) -> str:
    phase_str = '{:<5}'.format(phase.capitalize())
    epoch_str = Fore.RED +'Epoch'+ Fore.RESET +' {:>2d}/{:<2d}'.format(epoch, n_epoch)
    loss_str = Fore.LIGHTMAGENTA_EX + 'loss' + Fore.RESET + ' = {:.6f}'.format(loss)
    acc_str = Fore.LIGHTCYAN_EX + 'acc'+ Fore.RESET + ' = {:.3f}'.format(acc)
    
    return '{} - {} - {} - {}'.format(phase_str, epoch_str, loss_str, acc_str)
