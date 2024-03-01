#!/usr/bin/env python

# PA7, CS124, Stanford
# v.1.1.0
#
# Original Python code by Ignacio Cases (@cases)
# Update: 2024-01: Added the ability to run the chatbot as LLM interface (@mryan0)
#
# See: https://docs.python.org/3/library/cmd.html
######################################################################
import argparse
import cmd
import logging
import sys

logging.basicConfig()
logger = logging.getLogger(__name__)

# Uncomment me to see which REPL commands are being run!
# logger.setLevel(logging.DEBUG)

from chatbot import Chatbot
from util import load_together_client, stream_llm_to_console, DEFAULT_STOP

# Modular ASCII font from http://patorjk.com/software/taag/
HEADER = """Welcome to Stanford CS124's
 _______  _______       ___________
|       ||   _   |      \          \\
|    _  ||  |_|  | ____  ------    /
|   |_| ||       ||____|      /   /
|    ___||       |           /   /       
|   |    |   _   |          /   /      
|___|    |__| |__|         /___/

 _______  __   __  _______  _______  _______  _______  _______  __
|       ||  | |  ||   _   ||       ||  _    ||       ||       ||  |
|       ||  |_|  ||  |_|  ||_     _|| |_|   ||   _   ||_     _||  |
|       ||       ||       |  |   |  |       ||  | |  |  |   |  |  |
|      _||       ||       |  |   |  |  _   | |  |_|  |  |   |  |__|
|     |_ |   _   ||   _   |  |   |  | |_|   ||       |  |   |   __
|_______||__| |__||__| |__|  |___|  |_______||_______|  |___|  |__|
"""

description = 'Simple Read-Eval-Print-Loop that handles the input/output ' \
              'part of the conversational agent '


class REPL(cmd.Cmd):
    """Simple REPL to handle the conversation with the chatbot."""
    prompt = '> '
    doc_header = ''
    misc_header = ''
    undoc_header = ''
    ruler = '-'

    def __init__(self, llm_programming=False, llm_prompting=False):
        super().__init__()

        self.chatbot = Chatbot(llm_enabled=llm_programming)
        self.name = self.chatbot.name
        self.bot_prompt = '\001\033[96m\002%s> \001\033[0m\002' % self.name

        self.greeting = self.chatbot.greeting()
        self.intro = self.chatbot.intro() + '\n' + self.bot_prompt + \
                     self.greeting
        self.debug = False
        self.debug_chatbot = False

        self.llm_prompting = llm_prompting
        self.llm_history = [{
            "role": "system",
            "content": self.chatbot.llm_system_prompt(),
            },
            {
                "role": "assistant",
                "content": self.greeting,
            }]
        self.llm_client = load_together_client()

    def cmdloop(self, intro=None):
        logger.debug('cmdloop(%s)', intro)
        return super().cmdloop(intro)

    def preloop(self):
        logger.debug('preloop(); Chatbot %s created and loaded', self.chatbot)
        print(HEADER)
        self.debug_chatbot = False

    def postloop(self):
        goodbye = self.chatbot.goodbye()
        print(self.bot_says(goodbye))

    def onecmd(self, s):
        logger.debug('onecmd(%s)', s)
        if s:
            return super().onecmd(s)
        else:
            return False  # Continue processing special commands.

    def emptyline(self):
        logger.debug('emptyline()')
        return super().emptyline()

    def default(self, line):
        logger.debug('default(%s)', line)
        # Stop processing commands if the user enters :quit
        if line == ":quit":
            return True
        elif self.llm_prompting:
            self.process_llm(line)
        else:
            response = self.chatbot.process(line)
            print(self.bot_says(response))

    def precmd(self, line):
        logger.debug('precmd(%s)', line)
        return super().precmd(line)

    def postcmd(self, stop, line):
        logger.debug('postcmd(%s, %s)', stop, line)

        if line == ':quit':
            return True
        elif line.lower() == 'who are you?' and not self.llm_prompting:
            self.do_secret(line)
        elif ':debug on' in line.lower():
            print('enabling debug...')
            self.debug_chatbot = True
        elif ':debug off' in line.lower():
            print('disabling debug...')
            self.debug_chatbot = False

        # Debugging the chatbot
        if self.debug_chatbot:
            print(self.chatbot.debug(line))

        return super().postcmd(stop, line)

    def bot_says(self, response):
        return self.bot_prompt + response

    def do_prompt(self, line):
        """Set the interactive prompt."""
        self.prompt = line + ': '

    def do_secret(self, line):
        """Could it be... a secret message?"""
        story = """A long time ago, in a remote land, a young developer named 
        Alberto Caso managed to build an ingenious and mysterious chatbot... 
        Now it's your turn! """
        print(story)

    def process_llm(self, line):
        self.llm_history.append({
            "role": "user",
            "content": line,
        })
        print(self.bot_says(''), end="")
        response = stream_llm_to_console(
            messages=self.llm_history,
            client=self.llm_client,
            stop=DEFAULT_STOP,
        )
        self.llm_history.append({
            "role": "assistant",
            "content": response,
        })

def process_command_line():
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--llm_programming', dest='llm_programming', action='store_true',
                        default=False, help='Enables LLM programming mode')
    parser.add_argument('--llm_prompting', dest='llm_prompting', action='store_true',
                        default=False, help='Enables LLM prompting mode')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    #########################
    # ADDED FOR TESTING     #
    #########################
    class Tee(object):
        # Modified from
        # https://stackoverflow.com/questions/34366763/input-redirection-with-python
        def __init__(self, input_handle, output_handle):
            self.input = input_handle
            self.output = output_handle

        def readline(self):
            result = self.input.readline()
            self.output.write(result)
            self.output.write('\n')
            self.output.flush()

            return result

        # Forward all other attribute references to the input object.
        def __getattr__(self, attr):
            return getattr(self.input, attr)


    if not sys.stdin.isatty():
        sys.stdin = Tee(input_handle=sys.stdin, output_handle=sys.stdout)

    #########################
    # END TESTING CODE      #
    #########################
    args = process_command_line()
    repl = REPL(llm_prompting=args.llm_prompting, llm_programming=args.llm_programming)
    repl.cmdloop()
